# Scientific Concepts and Research Motivation

This document presents the scientific ideas, research hypotheses, and conceptual
mechanisms behind the **Power Stability Radius** project. It is written for
researchers evaluating the methodology and for developers seeking to understand
*why* each computation exists, not just *how* it is implemented.

Cross-references to companion documents:

- **[mathematical_foundations.md](mathematical_foundations.md)** for formal
  definitions, equations, and proofs.
- **[algorithms_and_models.md](algorithms_and_models.md)** for step-by-step
  algorithmic descriptions.

---

## 1. Problem Motivation

### 1.1 Why Robustness Certificates for Power Systems?

Modern power grids operate under increasing uncertainty. Renewable generation
(wind, solar) introduces stochastic variability in bus injections, while
demand-side flexibility and electric vehicles create heterogeneous load patterns.
The classical deterministic security assessment -- running a fixed set of
contingencies and checking limits -- becomes inadequate when the perturbation
space is continuous and high-dimensional.

The central question this project addresses is:

> **How far can bus injections deviate from a dispatched operating point before
> any transmission line exceeds its thermal rating?**

This "distance to the nearest constraint violation" is the **stability radius**
(or robustness radius). It provides a single, interpretable number that
quantifies the security margin of an operating point.

### 1.2 Who Needs This?

The stability radius concept serves three distinct audiences:

1. **Grid operators** assessing real-time security margins. When renewable
   output is uncertain, the stability radius gives a worst-case guarantee: any
   injection perturbation within the radius ball is safe.

2. **Planning engineers** evaluating network adequacy. By computing stability
   radii across many operating points and network configurations, one can
   identify structural bottlenecks and justify reinforcement investments.

3. **Researchers** studying robustness metrics. The project provides a
   framework for comparing different notions of "how dangerous is this line?"
   and testing whether theoretically motivated metrics outperform simple
   heuristics.

### 1.3 Relationship to Existing Work

The stability radius concept originates from robust control theory, where it
measures the distance from a nominal system to the nearest unstable system.
This project adapts the idea to power systems by:

- Replacing dynamical stability with **static constraint satisfaction** (thermal
  limits on line flows).
- Using **linearized sensitivity models** (PTDF for DC, Jacobian adjoint for
  AC) as the mapping from injection perturbations to constraint violations.
- Introducing **probabilistic extensions** (sigma-radius) that connect the
  geometric certificate to overload probabilities under Gaussian uncertainty.

The key distinction from standard power systems security analysis is that the
stability radius considers **all possible perturbation directions simultaneously**,
not just a predefined set of scenarios. This provides a formal *certificate* of
safety rather than a finite spot-check.

---

## 2. The Stability Radius Concept

### 2.1 Geometric Intuition

Consider the space of all possible injection perturbations. Each line
constraint defines a half-space: the set of perturbations for which that line
remains within its thermal limit. The intersection of all these half-spaces
forms a polytope -- the **feasible region** in injection space.

The stability radius is the radius of the largest ball (in a chosen norm)
centered at the origin (i.e., at the base operating point) that fits entirely
inside this polytope. Under the linear model, this ball is fully contained
within the feasible region, meaning every perturbation within it is safe.

**Implemented in:**
- `src/stability_radius/radii/core_l2.py` -- pure mathematical L2 certificate
  (function `compute_l2_certificate_from_H`)
- `src/stability_radius/radii/l2.py` -- DC L2 radius with balanced projection

### 2.2 Per-Line Decomposition

Because line constraints are independent in the linear model, the global
certificate decomposes into per-line radii:

- For each line l, the per-line radius r_l measures how far injections can
  move (in the worst-case direction for that line) before line l overloads.
- The global radius r* = min_l r_l is determined by the **bottleneck line** --
  the line whose per-line radius is smallest.

This decomposition is scientifically important because it identifies which
lines are the weakest links in the system. The bottleneck line is returned
by all radius computations (field `argmin_pos` in `L2RadiusCertificate`).

### 2.3 Certificate vs Exact Distance

An important distinction: the stability radius computed here is a
**certificate** (lower bound), not the exact distance to the nearest
constraint violation. The gap arises from:

1. **Linearization error**: The actual power flow is nonlinear. The
   certificate guarantees safety within the linear model, which is only
   accurate near the operating point. This is the primary source of
   conservatism.

2. **Norm choice**: The L2 ball is the largest ball that fits inside the
   feasible polytope. An L-infinity or ellipsoidal certificate would give
   different (possibly tighter) bounds for specific perturbation structures.

The worst-case verification experiments (`src/stability_radius/experiments/run_worst_case_verify.py`)
are designed to quantify this gap empirically -- see Section 12.1.

---

## 3. DC vs AC Modeling: The Accuracy-Scalability Trade-off

### 3.1 The DC Linear Model

The DC power flow model makes three simplifying assumptions:

- Lossless lines (resistance r = 0)
- Flat voltage magnitudes (|V| = 1 pu at all buses)
- Small angle differences (sin(theta) ~ theta)

Under these assumptions, the flow on each line is a linear function of bus
injections: f = H * p, where H is the PTDF (Power Transfer Distribution
Factor) matrix. The stability radius then follows immediately from the
Cauchy-Schwarz inequality.

**Advantages:** The DC model is fast. Building the PTDF requires one sparse
LU factorization of the (n-1) x (n-1) B-matrix. Radius computation is O(m*n)
where m is the number of lines and n is the number of buses. The project
scales to 10,000+ bus networks in this mode.

**Limitations:** By ignoring reactive power and voltage magnitudes, the DC
model misses phenomena that can cause overloads in practice:
- Reactive power flows contribute to apparent power |S| = sqrt(P^2 + Q^2).
- Voltage-dependent line loading can differ significantly from the P-only
  approximation.
- Network losses redistribute flows relative to the lossless model.

**Implemented in:** `src/stability_radius/dc/dc_model.py` (classes
`DCOperator` and function `build_dc_operator`)

### 3.2 The Linearized AC Model

The AC model in this project uses a **first-order Taylor expansion** of the
full nonlinear AC power flow equations around an AC PF base point. This yields
a linear sensitivity mapping from injection perturbations [Delta_P; Delta_Q]
to state changes [delta_theta; delta_V], from which flow changes are computed.

The key object is the AC power flow **Jacobian** J, a sparse matrix relating
injection changes to state changes:

    J * [d_theta; d_V] = [d_P; d_Q]

The radius computation uses the **adjoint** of this system to compute
sensitivities of each line's apparent power to all injection changes
simultaneously -- see Section 8.

**Advantages:**
- Captures reactive power effects (Q contributes to |S|).
- Handles PV/PQ bus types correctly (generators control voltage, so their
  V is not a free variable).
- Linearization around the actual AC PF solution provides a locally accurate
  approximation.

**Limitations:**
- Requires solving an AC power flow as a prerequisite (the base point).
- The linearization is only accurate near the base point -- large perturbations
  may violate the linear model.
- Computationally more expensive than DC (but still fast due to sparse LU).

**Implemented in:** `src/stability_radius/ac/ac_model.py` (class `ACOperator`
and function `build_ac_operator`)

### 3.3 The Design Choice: When to Use Which?

The project deliberately supports both models to enable **comparative
analysis**. The research question (addressed by `src/stability_radius/experiments/run_pglib_sweep.py`)
is: *How different are DC and AC radii in practice?*

If they agree closely, the cheaper DC model suffices. If they disagree, the
AC model provides a more realistic (and generally smaller, more conservative)
radius. The AC/DC ratio across benchmark cases characterizes the modeling gap.

---

## 4. The Lossless Policy: A Deliberate Design Choice

### 4.1 Why the AC Certificate Uses r=0

A subtle but critical design decision: the AC certificate deliberately uses a
**lossless (r=0), series-only (no shunt elements)** network model. This is
controlled by the `lossless=True` parameter in `build_ac_operator()`.

The rationale involves consistency across three components:

1. **The DC model** is inherently lossless. Its radii assume r=0.
2. **The AC Jacobian** built without shunt elements produces sensitivities
   that match the verification model.
3. **The Monte Carlo verification** uses pandapower's `runpp`, which by
   default includes shunt admittances and voltage-dependent elements.

If the AC certificate included losses (r>0) but the verification PF included
shunt admittances the Jacobian does not model, there would be a systematic
mismatch between the certificate's linear model and the verification's
nonlinear model. The lossless policy ensures the certificate and verification
are consistent within the approximation's validity region.

**Where enforced:** `src/stability_radius/ac/ac_model.py`, function
`_line_z_total_ohm()` -- when `lossless=True`, resistance is set to zero.
Also in `_build_ybus_pu()`, which builds Ybus without shunt elements.

### 4.2 Implications

- The AC certificate is conservative relative to the full lossy AC model:
  it certifies safety under a simpler model, so the actual system may be safe
  for larger perturbations than the certificate predicts.
- This conservatism is quantified by the worst-case verification experiments,
  which measure the "crossing alpha" (the scale factor at which the nonlinear
  PF actually violates the limit). A crossing alpha near 1.0 indicates the
  linearization is tight; a crossing alpha significantly above 1.0 indicates
  conservatism.

---

## 5. Multiple Radius Variants: Different Questions, Different Answers

The project implements several radius variants, each answering a different
question about robustness. Understanding when to use each is central to
the scientific contribution.

### 5.1 L2 Radius -- Worst-Case Absolute Guarantee

**Question:** What is the largest Euclidean ball of injection perturbations
guaranteed to be safe?

**Formula:** r_l = margin_l / ||Proj(g_l)||_2

where margin_l = c_l - |f0_l| is the flow headroom, g_l is the sensitivity
row for line l, and Proj denotes the balanced-subspace projection.

**When to use:** When you need an absolute, worst-case guarantee with no
distributional assumptions. Suitable for conservative security assessment
where the perturbation direction is unknown.

**Limitation:** The L2 radius is isotropic -- it treats all perturbation
directions as equally likely. In practice, some buses (e.g., those with
large wind farms) have much more uncertainty than others. The L2 radius may
be overly conservative if the actual uncertainty is concentrated on a few
buses.

**Implemented in:**
- DC: `src/stability_radius/radii/core_l2.py` and
  `src/stability_radius/radii/l2.py`
- AC: `src/stability_radius/radii/ac_l2.py`

### 5.2 Sigma Radius -- Margin in Standard Deviations

**Question:** How many standard deviations of injection uncertainty separate
the operating point from the nearest line overload?

**Formula:** r_sigma_l = margin_l / sigma_flow_l

where sigma_flow_l = ||Sigma^{1/2} h_l||_2 is the standard deviation of the
linearized flow on line l under Gaussian injection perturbations with
covariance Sigma.

**When to use:** When you have per-bus uncertainty data (e.g., forecast error
standard deviations) and want a practically interpretable metric. A sigma
radius of 3.0 means the operating point is 3 standard deviations from
overload -- directly connecting to overload probability under the Gaussian
assumption.

**Key property:** The sigma radius is **anisotropic**. Buses with larger
sigma contribute more to the worst-case perturbation. This correctly reflects
the physical reality that uncertainty is heterogeneous.

**Implemented in:**
- DC: `src/stability_radius/radii/probabilistic.py`
- AC: `src/stability_radius/radii/ac_sigma_radius.py`

### 5.3 Metric Radius -- Generalized Weighted Certificate

**Question:** What is the largest ball in an arbitrary weighted norm that is
safe?

**Formula:** r_M_l = margin_l / sqrt(h_l^T M^{-1} h_l)

where M is any symmetric positive-definite (SPD) weight matrix.

**When to use:** As a unifying framework. Special cases:
- M = I recovers the L2 radius.
- M = diag(1/sigma^2) recovers the sigma radius.

This equivalence is a designed **cross-check** in the codebase: when the
metric radius module is given M = diag(1/sigma^2), its output must match the
sigma radius module. This verifies consistency between two independent
implementations.

**Implemented in:**
- DC: `src/stability_radius/radii/metric.py`
- AC: `src/stability_radius/radii/ac_metric_radius.py`

### 5.4 Probabilistic Overload Probability

**Question:** Under Gaussian injection uncertainty, what is the probability
that a given line overloads?

**Formula:** P(|f| > c) = Q((c - |f0|) / sigma) + Q((c + |f0|) / sigma)

where Q is the Gaussian Q-function (tail probability).

The second term accounts for the possibility of overload in the negative
direction (flow reversal). For typical operating points where |f0| is much
smaller than c, this second term is negligible, and the overload probability
is approximately Q(r_sigma).

**When to use:** When you need an explicit probability, e.g., for risk-based
operations or planning studies.

**Implemented in:**
- DC: `src/stability_radius/radii/probabilistic.py` (function
  `overload_probability_symmetric_limit`)
- AC: `src/stability_radius/radii/ac_sigma_radius.py` (function
  `_overload_probability_symmetric_limit`)

### 5.5 N-1 Radius -- Contingency-Aware Security

**Question:** How far can injections deviate before any line overloads,
accounting for single-line outages?

The N-1 radius extends the L2 radius by considering the post-contingency
state: after one line trips, remaining lines see redistributed flows (via
LODF -- Line Outage Distribution Factors) and modified sensitivities. The
effective N-1 radius per line m is the minimum of its post-contingency radii
across all possible single-line outages.

**When to use:** For N-1 secure operation, which is the standard security
criterion in most grid codes.

**Implemented in:** `src/stability_radius/radii/nminus1.py` (functions
`lodf_from_ptdf`, `effective_nminus1_l2_radii`). Requires the full H_full
matrix (dc.mode=materialize).

---

## 6. Balanced Disturbances: The Physics of Power Balance

### 6.1 Why Balance Matters

In a real power system, active power must be balanced: total generation equals
total load plus losses. This means physical injection perturbations satisfy
the constraint 1^T Delta_p = 0 (and 1^T Delta_Q = 0 in AC).

Ignoring this constraint would allow unphysical perturbations where, say, all
buses simultaneously increase injection. Such perturbations are artifacts of
the model, not real threats. Enforcing balance restricts the perturbation
space to the physically meaningful balanced subspace and generally
**increases** the stability radius (since fewer perturbation directions are
allowed).

### 6.2 Implementation for DC (Isotropic Projection)

In the DC model, balanced projections are implemented via mean subtraction:

    g_projected = g - mean(g) * 1

The projected sensitivity vector g_projected lives in the balanced subspace
{x : 1^T x = 0}. Its norm ||g_projected||_2 is the correct dual norm for
measuring the maximum flow change per unit balanced perturbation.

A key mathematical property: this projection makes the radius **invariant
to the choice of slack bus**. Different slack bus choices change g by adding
a constant to all entries, but the projected norm is unchanged. This is
tested explicitly in `tests/test_certificate_concept.py`.

**Implemented in:** `src/stability_radius/radii/core_l2.py`, function
`l2_norm_projected_ones_complement()`.

### 6.3 Implementation for AC Sigma (Anisotropic Projection)

For the sigma radius, the balanced projection is more subtle. The worst-case
perturbation under the sigma norm is:

    dp_i = r * sigma_i^2 * h_i / sigma_flow

To enforce sum(dp) = 0, we need sum(sigma_i^2 * h_i) = 0. This is achieved
by a **sigma-squared-weighted mean subtraction**:

    h_adjusted = h - [sum(sigma^2 * h) / sum(sigma^2)]

This differs from the standard mean subtraction used in the L2 case because
the perturbation ellipsoid is anisotropic (Sigma-weighted). The unweighted
mean subtraction would enforce balance in the wrong geometry.

**Implemented in:** `src/stability_radius/radii/ac_sigma_radius.py`, within
the `compute_ac_sigma_radius()` function (see the `if balance:` block).

---

## 7. Binding End Selection in AC Analysis

### 7.1 The Two-End Problem

In AC power flow, each line carries different apparent power at its two ends
due to losses (even in the lossless model, reactive power flows cause
|S_from| != |S_to|). The thermal limit constraint must be checked at both
ends:

    max(|S_from|, |S_to|) <= c

This means each line contributes **two** constraint functions to the radius
computation, not one as in the DC case.

### 7.2 The Binding End

For each line, the project computes radii for both the from-end and to-end
constraints, then takes the minimum:

    r_line = min(r_from, r_to)

The end that achieves this minimum is the **binding end** -- it is the end
where the line is more vulnerable to overload. The binding end is reported
in the results (field `binding_end: "from" | "to"`) so downstream analyses
(sigma radius, verification) use consistent constraint functions.

### 7.3 Scientific Significance

The binding end selection is not merely a technical detail. It captures the
physical fact that power flow direction and line charging determine which
end sees higher apparent power. For long, heavily loaded lines, the from
and to ends can differ substantially, and using the wrong end would
produce an unsound (overly optimistic) certificate.

**Implemented in:** `src/stability_radius/radii/ac_l2.py`, in the per-line
aggregation loop at the end of `compute_ac_l2_radius()`.

---

## 8. The Adjoint Method for Efficient Sensitivities

### 8.1 The Problem: Computing H is Expensive in AC

In the DC model, the sensitivity matrix H_full can be materialized explicitly
because it has a closed-form expression involving the sparse LU of the
B-matrix. In AC, the analogous computation would require solving the Jacobian
system once per bus to build the full inverse -- O(n) solves for an n-bus
network.

### 8.2 The Adjoint Trick

Instead of computing the full sensitivity matrix, the project uses the
**adjoint method**: for each constraint (line end), it solves a single
adjoint system:

    J^T a = b

where b encodes the gradient of the constraint function (|S| at the line end)
with respect to the state variables (theta, V). The solution vector a gives
the sensitivity of that constraint to **all** injection changes simultaneously.

This is mathematically equivalent to computing one row of J^{-1} (or more
precisely, one row of the composed sensitivity b^T J^{-1}), but avoids
building the full inverse. Since the number of constraints (2 * m_lines) is
typically much smaller than the number of buses, this is much cheaper.

### 8.3 Chunked Computation

The adjoint systems are solved in chunks (controlled by `chunk_size`). For
each chunk, a batch of right-hand sides is assembled and solved simultaneously
using the pre-factored LU decomposition of J. This amortizes the overhead of
the LU solve across multiple constraints.

**Implemented in:** `src/stability_radius/radii/ac_l2.py`, in the "chunked
adjoint solves" section of `compute_ac_l2_radius()`. The LU factorization is
done once in `build_ac_operator()` and stored as `ACOperator.J_lu`.

### 8.4 Gradient of Apparent Power

The constraint function is the apparent power magnitude |S| = sqrt(P^2 + Q^2)
at a line end. Its gradient with respect to the state variables is:

    d|S|/dx = (P/|S|) * dP/dx + (Q/|S|) * dQ/dx

The weights wP = P/|S| and wQ = Q/|S| determine the relative contribution of
active and reactive power changes. When |S| is near zero (a degenerate case
for lightly loaded lines), the gradient is undefined. The code handles this
by using equal weights wP = wQ = 1/sqrt(2) as a conservative, unbiased
fallback (see the `_FALLBACK_WP_WQ` constant in `ac_l2.py`).

---

## 9. Verification Philosophy: Trust but Verify

### 9.1 Multi-Level Verification

The project employs a rigorous multi-level verification strategy to ensure
the computed certificates are correct and meaningful:

**Level 1: Deterministic certificate check.** Apply the analytically computed
worst-case perturbation at scale factor 1.0 and verify that the linearized
flow exactly reaches the limit. This checks internal consistency of the
certificate computation (margin/norm = radius, and the worst-case direction
achieves equality).

**Level 2: Worst-case nonlinear verification.** Apply the worst-case
perturbation to the full nonlinear AC power flow solver (pandapower `runpp`).
At scale 1.0, the actual flow should be close to (but not necessarily equal
to) the limit. At scales below 1.0, the actual flow should be below the
limit. At scales above 1.0, violations may occur. The "crossing alpha" --
the scale factor at which the actual flow first reaches the limit -- measures
the tightness of the linear certificate.

**Implemented in:** `src/stability_radius/verification/verify_worst_case.py`
and `src/stability_radius/experiments/run_worst_case_verify.py`.

**Level 3: Monte Carlo simulation.** Sample random perturbations from a
specified distribution (typically Gaussian), apply each to the nonlinear PF
solver, and compute empirical overload rates. This provides a statistical
check of the analytic overload probability predictions.

**Implemented in:** `src/stability_radius/verification/monte_carlo.py` and
`src/stability_radius/verification/ac_monte_carlo_sigma.py`.

**Level 4: Comparative metrics analysis.** Compare stability radii against
naive metrics (loading ratio, headroom, Cantelli bound) using Spearman rank
correlation with empirical overload probabilities. This tests the research
hypothesis that stability radii are better predictors of danger.

**Implemented in:** `src/stability_radius/analysis/metrics_analysis.py`.

### 9.2 Soundness vs Usefulness

The project explicitly distinguishes two properties of a certificate:

- **Soundness**: "Within the certified ball, no violations occur." A sound
  certificate is correct; an unsound one is a bug.
- **Usefulness**: "The certified ball has non-trivial volume." A certificate
  with r* = 0 is trivially sound (the empty ball contains no perturbations)
  but useless in practice.

This distinction is formalized in `src/stability_radius/verification/verify_certificate.py`
(class `CertificateInterpretation`) and avoids conflating correctness with
informativeness in reporting.

---

## 10. The Sigma-Radius: Main Research Contribution

### 10.1 Motivation

The L2 radius treats all buses symmetrically. In reality, uncertainty is
heterogeneous: a bus with a large wind farm has much higher injection
variability than a bus with a stable industrial load. The **sigma-radius**
accounts for this by weighting each bus by its injection standard deviation.

### 10.2 Connection to the L2 Radius

The sigma-radius can be understood as an L2 radius in a transformed space.
If we define the transformed injection vector z = Sigma^{-1/2} Delta_u,
then ||z||_2 = 1 corresponds to a perturbation of one standard deviation.
The sigma-radius is the L2 radius in this z-space:

    r_sigma = margin / ||Sigma^{1/2} h||_2

This is also the metric radius with M = Sigma^{-1} (inverse covariance as
weight matrix). The project exploits this equivalence as a correctness check:
`compute_ac_metric_radius()` with M = diag(1/sigma^2) must produce the same
result as `compute_ac_sigma_radius()`.

### 10.3 Practical Interpretability

The sigma-radius has units of "standard deviations." This makes it directly
interpretable:

- r_sigma = 3.0 means the operating point is 3 sigma from overload.
- Under the Gaussian assumption, the overload probability is approximately
  Q(r_sigma) ~ 0.13% for r_sigma = 3.0.

This connects the geometric certificate to a probabilistic statement, making
it actionable for grid operators.

### 10.4 Sources of Per-Bus Uncertainty

The project supports two sources of per-bus injection standard deviations:

1. **Uniform**: sigma_p and sigma_q are constant across all buses. Useful
   for sensitivity analysis but not realistic.

2. **UnitCommitment.jl (UC.jl)**: Per-bus sigma is derived from hourly demand
   time series in UC.jl instance files. The parser
   (`src/stability_radius/parsers/uc_jl.py`) reads JSON instance files,
   extracts demand profiles per bus, and computes population standard
   deviations. This provides realistic, heterogeneous uncertainty data.

**Implemented in:** `src/stability_radius/parsers/uc_jl.py` for data loading,
and the `_build_sigma_arrays()` helper in `src/stability_radius/workflows.py`
for constructing the sigma vectors.

### 10.5 Worst-Case Perturbation Direction

For each line, the sigma-radius computation also produces the **worst-case
perturbation direction** -- the injection pattern that maximally increases
flow on that line per unit sigma-norm. This is:

    dp_i* = r_sigma * sigma_i^2 * h_i / sigma_flow

The worst-case perturbation is stored in the results (`worst_case_dp_mw`,
`worst_case_dq_mvar`) and can be fed into nonlinear verification to check
certificate tightness.

---

## 11. Comparative Metrics and the Precision-at-k Framework

### 11.1 Baseline Metrics

To evaluate whether stability radii provide meaningful information beyond
simple heuristics, the project computes three baseline metrics for each line:

1. **Loading ratio** = |S0| / c. The fraction of the thermal limit already
   consumed by the base flow. Higher = more dangerous.

2. **Headroom** = c - |S0|. The absolute margin in MVA. Lower = more
   dangerous.

3. **Cantelli (Chebyshev) upper bound** = sigma^2 / (sigma^2 + headroom^2).
   A distribution-free upper bound on overload probability. Higher = more
   dangerous.

These are implemented in `src/stability_radius/metrics/ac_baselines.py`.

### 11.2 Why These Baselines Are Incomplete

Loading ratio and headroom capture only the **margin** aspect of vulnerability
but ignore the **sensitivity** aspect. A line can have a large margin but be
extremely sensitive to injection changes (large ||h||), making it vulnerable
despite appearances. Conversely, a heavily loaded line with low sensitivity
may be safer than it looks.

The stability radius combines both aspects: r = margin / ||h||. The research
hypothesis is that this combination produces a better ranking of "dangerous"
lines.

### 11.3 Evaluation via Spearman Rank Correlation

The metrics analysis module computes Spearman rank correlation between each
metric and the empirical overload probability from Monte Carlo simulation.
A high correlation (positive for "higher=more dangerous" metrics like loading
ratio, negative for "lower=more dangerous" metrics like radii) indicates the
metric correctly identifies dangerous lines.

For "lower-is-more-dangerous" metrics (radii, headroom), the sign is flipped
before computing correlation so that a positive rho consistently means
"correctly identifies danger."

**Implemented in:** `src/stability_radius/analysis/metrics_analysis.py`, function
`compute_rank_correlations()`.

### 11.4 Precision-at-k

Beyond overall correlation, the project evaluates **precision-at-k**: when a
metric ranks lines by danger, how good are its top-k picks?

For each metric:
1. Rank lines by the metric (ascending for radii, descending for probabilities).
2. Take the top-k lines.
3. Compute the mean empirical overload probability of these top-k lines.
4. Higher mean probability = better metric (it correctly identified the
   actually dangerous lines).

This evaluation is more practically relevant than overall correlation because
grid operators typically focus on the most critical lines, not the full
ranking.

**Implemented in:** `src/stability_radius/analysis/metrics_analysis.py`, function
`compute_precision_at_k()`.

---

## 12. Experimental Research Questions

The `experiments/` directory contains scripts that address specific research
questions. Each experiment is designed to test a hypothesis about the stability
radius methodology.

### 12.1 Experiment 1: DC vs AC Radius Comparison (`run_pglib_sweep.py`)

**Research question:** How does the AC stability radius compare to the DC
radius across different network sizes and topologies?

**Hypothesis:** The AC radius is generally smaller than the DC radius because
the AC model captures additional physics (reactive power, voltage effects)
that can cause overloads the DC model misses.

**Design:** For each PGLib-OPF benchmark case, both DC and AC radii are
computed from the **same** OPF dispatch and base point. This shared base
point is critical for a fair comparison -- using different dispatches would
confound the model difference with the dispatch difference.

**Output:** Table 1 (case, n_buses, n_lines, r*_DC, r*_AC, AC/DC ratio,
time, bottleneck) and Figure 1 (bar chart).

### 12.2 Experiment 2: Sigma Radius with Realistic Uncertainty (`run_sigma_radius.py`)

**Research question:** How does the sigma-radius depend on the bus uncertainty
structure? Does realistic (UC.jl-derived) uncertainty produce qualitatively
different results than uniform uncertainty?

**Hypothesis:** With realistic per-bus sigma, the sigma-radius identifies
different bottleneck lines than the L2 radius, because high-uncertainty buses
have disproportionate influence on certain lines.

**Design:** Load hourly demand profiles from UC.jl instance files, compute
per-bus sigma as the population standard deviation across hours, then compute
sigma-radius at the average operating point. Includes Monte Carlo validation
of the analytic overload probability predictions.

**Output:** Table 2 (per-line sigma-radius), Figure 2 (L2 vs sigma scatter),
Figure 2b (per-bus sigma heatmap).

### 12.3 Experiment 3: Worst-Case Verification (`run_worst_case_verify.py`)

**Research question:** Does the worst-case perturbation actually cause the
predicted violation? How tight is the linear certificate relative to the
nonlinear AC model?

**Hypothesis:** At scale factor alpha = 1.0, the actual nonlinear flow should
be close to the predicted limit. The crossing alpha (where |S_actual| first
exceeds the limit) should be near 1.0 for well-conditioned networks.

**Design:** For each case's bottleneck line, apply worst-case perturbations
at scale factors 0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5. Solve
full nonlinear AC PF at each scale. Interpolate the crossing alpha. The
validation check requires crossing >= 0.95 for the certificate to be
considered "sound" in the nonlinear sense.

**Output:** Table 3 (crossing alpha, linearization error), Figure 3
(predicted vs actual flow curves).

### 12.4 Experiment 4: Scalability Analysis (`run_scalability.py`)

**Research question:** How does wall-clock computation time scale with
network size?

**Hypothesis:** DC computation scales nearly linearly with network size
(dominated by sparse LU). AC computation scales superlinearly due to the
larger Jacobian and per-constraint adjoint solves, but remains practical
up to several thousand buses.

**Design:** Time DC-only and AC-only computations separately across
PGLib cases of increasing size, with multiple repeats for timing
statistics.

**Output:** scalability.json with per-case DC and AC timing statistics.

---

## 13. Certificate vs Bound: Semantics of Different Results

The project produces several types of outputs with distinct semantic meanings:

### 13.1 The L2 Radius: A Deterministic Certificate

The L2 radius is a **certificate**: it guarantees that every perturbation
within the L2 ball is safe under the linear model. No distributional
assumptions are required. The guarantee is deterministic and worst-case.

Formally: if ||Delta_p||_2 <= r*, then |f_l(Delta_p)| <= c_l for all lines l
(under the linear flow model).

### 13.2 The Sigma-Radius: A Probabilistic Certificate

The sigma-radius is a **probabilistic certificate**: it gives the number of
standard deviations of margin. Under the Gaussian assumption:
- The overload probability for a single line is approximately Q(r_sigma).
- The probability is exact under the linearized model with Gaussian injections.
- It is approximate for the true nonlinear model (where linearization error
  introduces a gap).

### 13.3 The Overload Probability: A Probabilistic Bound

The overload probability P(|f| > c) is a **probabilistic bound**: it gives
the probability of overload under the assumed distribution. Under the
symmetric limit model with nonzero base flow:

    P(|f| > c) = Q((c - |f0|)/sigma) + Q((c + |f0|)/sigma)

This is a bound (not a certificate) because it relies on the distributional
assumption. If the actual injection distribution is not Gaussian, the bound
may be inaccurate (though the Cantelli bound provides a distribution-free
alternative -- see Section 11.1).

### 13.4 The Cantelli Bound: Distribution-Free but Conservative

The Cantelli (one-sided Chebyshev) bound provides a distribution-free upper
bound on overload probability that requires only knowledge of the mean and
variance:

    P(X >= headroom) <= sigma^2 / (sigma^2 + headroom^2)

This is valid for any distribution, not just Gaussian, but is generally much
looser than the Gaussian Q-function for well-behaved distributions.

---

## 14. Design Rationale: Key Implementation Decisions

### 14.1 Why Sparse LU, Not Dense Inversion?

Both DC and AC operators use sparse LU factorization (SciPy's `splu`) rather
than dense matrix inversion. This is essential for scalability:

- The B-matrix (DC) and Jacobian J (AC) are sparse with O(n) nonzeros for
  typical power networks (each bus connects to a small number of neighbors).
- Sparse LU factorization is O(n * nnz) in practice, much faster than the
  O(n^3) dense inversion.
- The factored form supports efficient forward/back-substitution for multiple
  right-hand sides (critical for the adjoint method).

### 14.2 Why OPF-Based Dispatch?

The base dispatch is typically computed via DC OPF (PyPSA + HiGHS solver)
rather than taken directly from the MATPOWER case data. This is because:

- Case data may not satisfy thermal limits, leading to negative margins and
  meaningless radii.
- OPF produces a dispatch that respects generator limits and thermal
  constraints, providing a realistic operating point.
- The OPF headroom factor (default 0.9) ensures that no line is loaded above
  90% of its limit at the base point, guaranteeing positive margins.

### 14.3 Why Deterministic Ordering?

All arrays in the project use deterministic ordering based on sorted
pandapower indices: `bus_ids = sorted(net.bus.index)`,
`line_ids = sorted(net.line.index)`. This ensures:

- Reproducibility across runs.
- Consistent array indexing between the operator, the radius computation,
  and the verification modules.
- Stable JSON keys (`line_0`, `line_1`, ...) for output files.

### 14.4 Why PV/PQ Bus Handling in AC?

The AC Jacobian correctly handles PV buses (generators with voltage control)
by:

- Including theta as a variable for all non-slack buses (PV and PQ).
- Including V as a variable only for PQ buses (PV buses have fixed V).
- Including P equations for all non-slack buses.
- Including Q equations only for PQ buses (PV buses absorb reactive power).

This produces a rectangular sub-structure in the Jacobian that is critical
for correctness. Without PV/PQ handling, the Jacobian would have the wrong
dimension and produce incorrect sensitivities for networks with generators.

**Implemented in:** `src/stability_radius/ac/ac_model.py`, functions
`_detect_pv_buses()` and `_build_reduced_pf_jacobian_mw_per_unit()`.

---

## 15. Connections to the Power Systems Literature

### 15.1 PTDF and DC Power Flow

The DC model and PTDF matrix are standard in power systems analysis. See:
- Wood, Wollenberg, Sheble, "Power Generation, Operation and Control"
  (3rd edition) for DC power flow fundamentals.
- The B-matrix construction in `dc_model.py` follows the standard formulation
  with b_ij = V^2 / X_ij.

### 15.2 AC Power Flow Jacobian

The AC Jacobian construction follows the textbook formulation using polar
coordinates (V, theta). The off-diagonal and diagonal elements are derived
from the standard S = V * conj(I), I = Ybus * V identities. The
implementation in `ac_model.py` uses the scaled (MW/MVAr units) form to
avoid per-unit confusion.

### 15.3 LODF and Contingency Analysis

The LODF (Line Outage Distribution Factor) computation follows the standard
formula: LODF_{m,k} = PTDF_{m,k} / (1 - PTDF_{k,k}). The islanding
detection (1 - PTDF_{k,k} ~ 0) and handling is consistent with standard
practice in contingency analysis tools.

### 15.4 Robust Optimization and Uncertainty Sets

The stability radius formulation is related to the robust optimization
literature on uncertainty sets. The L2 ball is a specific uncertainty set;
the ellipsoidal set (metric radius) is more general. The connection to
Cantelli bounds and chance-constrained optimization is explicit in the
probabilistic extensions.

---

## 16. Limitations and Open Questions

### 16.1 Linearization Validity Region

The certificates are valid only within the region where the linear
approximation is accurate. For large perturbations, the nonlinear effects
(voltage collapse, reactive power limits, PV/PQ switching) may cause
violations that the linear model does not predict. The worst-case
verification experiments quantify this gap but do not eliminate it.

### 16.2 Static Analysis Only

The project addresses static (steady-state) thermal limits. It does not
consider:
- Dynamic stability (rotor angle, frequency response).
- Voltage stability (PV curves, nose curves).
- Transient thermal ratings (short-term overload capability).

### 16.3 No Discrete Events

The analysis assumes continuous perturbations to injections. It does not
model:
- Generator tripping (discrete events).
- Topology changes (switching, islanding).
- Protection system interactions.

N-1 radii partially address this by considering single-line outages, but
the perturbation itself (injection changes) is still continuous.

### 16.4 Gaussian Assumption

The sigma-radius and overload probability assume Gaussian injection
distributions. Real injection uncertainty (especially from renewables)
may have heavier tails, asymmetry, or correlation structure that the
diagonal Gaussian model does not capture. The Cantelli bound provides
a partial hedge against non-Gaussian distributions.

### 16.5 Independence of Line Constraints

The per-line radius decomposition assumes that line constraints are
independent. In reality, lines share buses and coupling through the
network topology creates correlations. The global radius r* = min_l r_l
correctly accounts for this (it is the worst-case over all lines), but
the per-line radii individually do not reflect inter-line interactions.

---

## 17. Summary of Key Scientific Claims

The project makes and tests the following scientific claims:

1. **The stability radius provides a rigorous certificate** of robustness
   against injection perturbations, under the linearized power flow model.
   (Verified by deterministic certificate checks and worst-case verification.)

2. **The AC radius is generally smaller than the DC radius** because the AC
   model captures additional physics. (Tested by `run_pglib_sweep.py`.)

3. **The sigma-radius provides a more meaningful ranking** of line danger
   than simple heuristics (loading ratio, headroom), because it combines
   margin and sensitivity information with heterogeneous uncertainty.
   (Tested by Spearman rank correlation in `metrics_analysis.py`.)

4. **The linearization is tight** for moderate perturbations: the crossing
   alpha is typically near 1.0 for well-conditioned networks. (Tested by
   `run_worst_case_verify.py`.)

5. **The computation scales to practical network sizes** (thousands of
   buses) within seconds to minutes. (Tested by `run_scalability.py`.)
