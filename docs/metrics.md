# Metrics Reference

This document provides a complete reference for every metric computed in the
comparative analysis pipeline (`metrics_analysis.py`).  Each section states
the formula, inputs, source module, and — critically — whether the metric
can be evaluated **before** Monte Carlo (predictive) or only **after**
(a posteriori).

> **Why this distinction matters.**  The whole point of the stability radius
> is to predict which lines are at risk of thermal overload *without*
> running expensive Monte Carlo.  Metrics that consume MC output cannot
> serve as predictors — they can only be used as post-hoc summaries.

Source code pointers use the pattern `module:function`.

---

## Table of Contents

1. [Classification of Metrics](#1-classification-of-metrics)
2. [Stability-Radius Family (Predictive)](#2-stability-radius-family-predictive)
   - 2.1 [AC L2 Radius](#21-ac-l2-radius-radius_ac_l2)
   - 2.2 [AC Sigma-Radius](#22-ac-sigma-radius-radius_ac_sigma)
   - 2.3 [AC Metric Radius](#23-ac-metric-radius-radius_ac_metric)
3. [Probabilistic Bounds (Predictive)](#3-probabilistic-bounds-predictive)
   - 3.1 [Analytic Overload Probability](#31-analytic-overload-probability-overload_probability_ac)
   - 3.2 [Cantelli / Chebyshev Upper Bound](#32-cantelli--chebyshev-upper-bound-cheb_prob_upper)
4. [Conventional Metrics (Predictive)](#4-conventional-metrics-predictive)
   - 4.1 [Loading Ratio](#41-loading-ratio-loading_ratio)
   - 4.2 [Headroom](#42-headroom-headroom_mva)
   - 4.3 [Performance Index](#43-performance-index-performance_index)
5. [Sensitivity Metrics (Predictive)](#5-sensitivity-metrics-predictive)
   - 5.1 [Directional Sensitivity](#51-directional-sensitivity-dir_sens_name)
   - 5.2 [Linearized Transfer Margin](#52-linearized-transfer-margin)
6. [A Posteriori Metrics (Require MC)](#6-a-posteriori-metrics-require-mc)
   - 6.1 [Empirical Overload Probability](#61-empirical-overload-probability-empirical_overload_prob)
   - 6.2 [Thermal Risk Index](#62-thermal-risk-index-thermal_risk_index)
7. [Intermediate Quantities](#7-intermediate-quantities)
   - 7.1 [Sigma-Flow](#71-sigma-flow-sigma_flow_mva)
   - 7.2 [Sensitivity Norm](#72-sensitivity-norm-h2)
8. [Evaluation Methodology](#8-evaluation-methodology)
9. [Summary Table](#9-summary-table)

---

## 1. Classification of Metrics

Every metric falls into one of three categories:

| Category | Computed from | Purpose |
|----------|--------------|---------|
| **Predictive** | Base-point power flow + network Jacobian | Rank lines by danger *without* MC |
| **Ground truth** | Monte Carlo samples | Reference ranking to evaluate predictors |
| **A posteriori** | Combines predictive data with MC output | Post-hoc summary; cannot serve as a predictor |

The analysis pipeline (`metrics_analysis.py`) evaluates predictive metrics
by their **Spearman rank correlation** with the ground truth column
`empirical_overload_prob` (see [Section 8](#8-evaluation-methodology)).

---

## 2. Stability-Radius Family (Predictive)

All three radius variants share the same structure:

```
r_l = margin_l / ||h_l||
```

where `margin_l` is the thermal headroom and `||h_l||` is measured in a
norm that depends on the variant.  **Smaller radius = more dangerous.**

### 2.1 AC L2 Radius (`radius_ac_l2`)

**Source:** `radii.ac_l2:compute_ac_l2_radius`

**Formula:**

```
r_l = (c_l - |S_0_l|) / ||h_l||_2
```

| Symbol | Meaning | Units |
|--------|---------|-------|
| `c_l` | Thermal limit (RATE_A) of line *l* | MVA |
| `\|S_0_l\|` | Apparent power flow at the binding end in the base point | MVA |
| `h_l` | Adjoint sensitivity vector: the gradient of `\|S_l\|` w.r.t. all bus injections `(P, Q)` | MVA/MW |
| `\|\|h_l\|\|_2` | Euclidean norm of `h_l` | dimensionless (MVA/MW ~ 1) |

Both the "from" and "to" ends of each line are evaluated; the binding end
(smaller radius) is selected.

With `balance=True` (default), the h-vector is projected onto the
kernel of the all-ones vector (enforcing `sum(dP) = 0` and `sum(dQ) = 0`).

**Interpretation:** The smallest L2-norm perturbation `(dP, dQ)` that
causes line *l* to reach its thermal limit has norm exactly `r_l`.

**Uses MC:** No.

### 2.2 AC Sigma-Radius (`radius_ac_sigma`)

**Source:** `radii.ac_sigma_radius:compute_ac_sigma_radius`

**Formula:**

```
r_sigma_l = margin_l / sigma_flow_l
```

where `sigma_flow_l` is the standard deviation of the linearized flow
under Gaussian perturbations (see [Section 7.1](#71-sigma-flow-sigma_flow_mva)):

```
sigma_flow_l = || diag(sigma) * h_l ||_2
             = sqrt( sum_i (sigma_P_i * h_l^P_i)^2
                   + sum_j (sigma_Q_j * h_l^Q_j)^2 )
```

**Interpretation:** The number of standard deviations of flow
fluctuation that fit within the thermal margin.  `r_sigma = 2` means
the margin is 2 sigma wide.

**Uses MC:** No.

### 2.3 AC Metric Radius (`radius_ac_metric`)

**Source:** `radii.ac_metric_radius:compute_ac_metric_radius`

**Formula:**

```
r_l^M = margin_l / sqrt( h_l^T  M^{-1}  h_l )
```

where `M` is a user-supplied symmetric positive-definite weight matrix.
Special cases:

- `M = I` reduces to the L2 radius.
- `M = diag(1/sigma^2)` reduces to the sigma-radius.

For diagonal `M`, the denominator is computed in O(n) time.
For dense `M`, a Cholesky factorisation `M = L L^T` is used.

**Uses MC:** No.

---

## 3. Probabilistic Bounds (Predictive)

These metrics estimate the probability of overload *analytically*, without
running Monte Carlo.  They consume `sigma_flow_mva` and `margin_ac_mva`
which are already computed by the sigma-radius module.

### 3.1 Analytic Overload Probability (`overload_probability_ac`)

**Source:** `radii.ac_sigma_radius:_overload_probability_symmetric_limit`

**Formula (Gaussian Q-function):**

```
P(|S_l| > c_l) = Q( (c_l - |S_0_l|) / sigma_flow_l )
               + Q( (c_l + |S_0_l|) / sigma_flow_l )
```

where `Q(x) = 0.5 * erfc(x / sqrt(2))`.

The first term dominates; the second term accounts for the (rare)
possibility of flow reversal past `-c_l`.

**Relationship to sigma-radius:** The argument of the dominant Q-term
is exactly `r_sigma_l`.  Therefore `overload_probability_ac` is a
monotonic transformation of the sigma-radius and carries the same
rank ordering.

**Uses MC:** No.

### 3.2 Cantelli / Chebyshev Upper Bound (`cheb_prob_upper`)

**Source:** `metrics.ac_baselines:cantelli_upper_bound`

**Formula:**

```
P(|S_l| >= margin_l) <= sigma_flow_l^2 / (sigma_flow_l^2 + margin_l^2)
```

This is the one-sided Cantelli inequality, a tightened form of Chebyshev's
bound.  It is distribution-free — valid for any random variable with
finite variance, not just Gaussian.

**Uses MC:** No.  Consumes `sigma_flow_mva` (analytical) and
`margin_ac_mva`.

---

## 4. Conventional Metrics (Predictive)

These are standard power-engineering indicators that do **not** account
for perturbation sensitivity.

### 4.1 Loading Ratio (`loading_ratio`)

**Source:** `metrics.ac_baselines:loading_ratio`

**Formula:**

```
LR_l = |S_0_l| / c_l
```

Values near 1.0 indicate the line is near its limit at the base point.

**Limitation:** A line with `LR = 0.3` may have a small stability radius
(and thus be dangerous) if its sensitivity norm `||h_l||` is large.
This is the core failure mode that the stability radius addresses.

**Uses MC:** No.

### 4.2 Headroom (`headroom_mva`)

**Source:** `metrics.ac_baselines:headroom_mva`

**Formula:**

```
HR_l = c_l - |S_0_l|     [MVA]
```

This is the numerator of the stability radius.  Unlike the radius, it
does not account for *how easily* perturbations can consume the margin
(the denominator `||h||`).

**Uses MC:** No.

### 4.3 Performance Index (`performance_index`)

**Source:** `metrics.ac_baselines:performance_index_line`

**Formula:**

```
PI_l = (w / 2n) * (|S_l| / c_l)^{2n}
```

Default: `w = 1`, `n = 1` (quadratic penalty).  This is the classical
Overload Performance Index used in contingency screening.

**Uses MC:** No.

---

## 5. Sensitivity Metrics (Predictive)

### 5.1 Directional Sensitivity (`dir_sens_<name>`)

**Source:** `metrics.ac_baselines:directional_sensitivity`

**Formula:**

```
DS_l(d) = |h_l^T d| / margin_l
```

Higher values indicate that line *l* is more vulnerable to perturbations
along direction `d`.  Canonical directions include:

- `max_gen_to_max_load` — shift from the largest generator bus to the
  largest load bus.
- `second_gen_to_second_load` — shift between the second-largest
  generator and load.
- `uniform_stress` — uniform injection increase on all load buses.

**Uses MC:** No.

### 5.2 Linearized Transfer Margin

**Source:** `metrics.ac_baselines:transfer_margin_linearized`

**Formula:**

```
TM(d) = min_l { margin_l / |h_l^T d| }
```

The transfer margin is the maximum scalar multiple of direction `d` that
can be sustained before any line reaches its thermal limit.  This is the
system-wide bottleneck, not a per-line metric.

**Uses MC:** No.

---

## 6. A Posteriori Metrics (Require MC)

These metrics consume Monte Carlo output.  **They cannot be used as
predictors** in the Spearman correlation comparison because they
contain the ground-truth signal.

### 6.1 Empirical Overload Probability (`empirical_overload_prob`)

**Source:** `verification.monte_carlo:run_monte_carlo_verification`

**Definition:** For each line, the fraction of MC random samples in which
`|S_l| > c_l`.

This is the **ground truth** target column.  All predictive metrics are
evaluated by their rank correlation with this column.

**Uses MC:** This IS the MC result.

### 6.2 Thermal Risk Index (`thermal_risk_index`)

**Source:** `metrics.ac_baselines:thermal_risk_index`

**Formula:**

```
R_l = empirical_overload_prob_l * loading_ratio_l
```

This follows the Risk = Probability x Impact principle.  However, because
it multiplies the MC ground truth (`empirical_overload_prob`) by a
severity proxy (`loading_ratio`), it is **circular** when used in the
rank-correlation evaluation.

Its Spearman correlation with `empirical_overload_prob` is trivially
near 1.0 because it is a near-monotonic transformation of the ground
truth itself.

**Why it exists:** It was included for post-hoc risk ranking — to
produce a single "risk score" that combines the probability of overload
with its operational severity.  It is useful as a summary for operators
who already have MC results, but it should **not** be compared alongside
predictive metrics like the stability radius.

**Uses MC:** Yes.

---

## 7. Intermediate Quantities

### 7.1 Sigma-Flow (`sigma_flow_mva`)

**Source:** `radii.ac_sigma_radius:compute_ac_sigma_radius`

```
sigma_flow_l = || diag(sigma) * h_l ||_2
```

The standard deviation of the linearized apparent power flow on line *l*
under independent Gaussian perturbations with per-bus standard deviations
`sigma_P_i` and `sigma_Q_j`.

Consumed by: sigma-radius, Cantelli bound, analytic overload probability.

### 7.2 Sensitivity Norm (`||h||2`)

**Source:** `radii.ac_l2:compute_ac_l2_radius`

```
||h_l||_2 = Euclidean norm of the adjoint sensitivity vector at the binding end
```

The denominator of the L2 radius.  Captures how strongly the aggregate
injection perturbation projects onto line *l*'s apparent power flow.
Lines with large `||h||` are sensitive — even moderate perturbations
translate into large flow changes.

---

## 8. Evaluation Methodology

The analysis pipeline evaluates each predictive metric by:

1. **Spearman rank correlation** (`rho`) between the metric column and
   `empirical_overload_prob` across all lines.  Higher |rho| means the
   metric better predicts which lines will be overloaded.

2. **Precision-at-k**: Among the top-k lines ranked by each metric, what
   fraction of MC overload probability mass is captured?

3. **Hidden-danger detection**: Lines where a conventional metric (e.g.,
   loading ratio) says "safe" but the stability radius says "dangerous",
   confirmed by MC.  Formally:
   - `loading_ratio < 0.6` (conventional: safe)
   - `rank_gap_norm >= 0.1` (radius ranks the line significantly higher)
   - `empirical_overload_prob > 0` (MC confirms the danger)

---

## 9. Summary Table

| Metric | Column name | Formula | Predictive? | Source module |
|--------|------------|---------|-------------|---------------|
| AC L2 Radius | `radius_ac_l2` | `margin / \|\|h\|\|_2` | Yes | `radii.ac_l2` |
| AC Sigma-Radius | `radius_ac_sigma` | `margin / sigma_flow` | Yes | `radii.ac_sigma_radius` |
| AC Metric Radius | `radius_ac_metric` | `margin / sqrt(h^T M^{-1} h)` | Yes | `radii.ac_metric_radius` |
| Analytic Overload Prob | `overload_probability_ac` | `Q(margin / sigma_flow)` | Yes | `radii.ac_sigma_radius` |
| Cantelli Bound | `cheb_prob_upper` | `sigma^2 / (sigma^2 + margin^2)` | Yes | `metrics.ac_baselines` |
| Loading Ratio | `loading_ratio` | `\|S_0\| / c` | Yes | `metrics.ac_baselines` |
| Headroom | `headroom_mva` | `c - \|S_0\|` | Yes | `metrics.ac_baselines` |
| Performance Index | `performance_index` | `(w/2n)(\|S\|/c)^{2n}` | Yes | `metrics.ac_baselines` |
| Directional Sensitivity | `dir_sens_<name>` | `\|h^T d\| / margin` | Yes | `metrics.ac_baselines` |
| Empirical Overload Prob | `empirical_overload_prob` | MC fraction | Ground truth | `verification.monte_carlo` |
| Thermal Risk Index | `thermal_risk_index` | `emp_prob * LR` | **No (uses MC)** | `metrics.ac_baselines` |

---

*See also:*
[mathematical_foundations.md](mathematical_foundations.md) for full derivations,
[algorithms_and_models.md](algorithms_and_models.md) for algorithmic details,
[glossary.md](glossary.md) for term definitions.
