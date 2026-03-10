# Experiments and Evaluation

This document describes the experimental pipeline, benchmark methodology, and evaluation framework.

> Cross-references: [execution_flow.md](execution_flow.md) for step-by-step traces, [scientific_concepts.md](scientific_concepts.md) for research motivation.

---

## 1. Overview

The project includes four main experiments designed for a research paper, plus a comparative metrics analysis pipeline. All experiment scripts are in `experiments/` and use YAML configs from `experiments/configs/`.

| Experiment | Script | Purpose | Paper Figure |
|------------|--------|---------|--------------|
| PGLib Sweep | `run_pglib_sweep.py` | DC vs AC radius comparison across network sizes | Fig. 1 |
| Sigma Radius | `run_sigma_radius.py` | Deep sigma-radius analysis with heterogeneous uncertainty | Fig. 2, Table 2 |
| Worst-Case Verification | `run_worst_case_verify.py` | Validate analytic worst-case perturbation | Verification |
| Scalability | `run_scalability.py` | Wall-clock time vs network size | Scalability curve |
| Metrics Analysis | `metrics_analysis.py` | Comparative evaluation of all metrics vs Monte Carlo | Correlation analysis |

---

## 2. Experiment 1: PGLib Sweep (`run_pglib_sweep.py`)

### Purpose

Compare DC and AC stability radii across the PGLib-OPF benchmark library to assess how well the DC approximation tracks the AC reality, and to identify cases where the two diverge.

### Benchmark Cases

From `experiments/configs/pglib_sweep.yaml`:

| Case | Buses | Lines | Source |
|------|-------|-------|--------|
| pglib_opf_case5_pjm | 5 | 6 | PJM |
| pglib_opf_case14_ieee | 14 | 20 | IEEE |
| pglib_opf_case24_ieee_rts | 24 | 38 | IEEE RTS |
| pglib_opf_case30_ieee | 30 | 41 | IEEE |
| pglib_opf_case57_ieee | 57 | 80 | IEEE |
| pglib_opf_case73_ieee_rts | 73 | 120 | IEEE RTS |
| pglib_opf_case118_ieee | 118 | 186 | IEEE |
| pglib_opf_case200_activ | 200 | 245 | ACTIVSg |
| pglib_opf_case300_ieee | 300 | 411 | IEEE |
| pglib_opf_case500_goc | 500 | 733 | GOC |
| pglib_opf_case588_sdet | 588 | 686 | SDET |
| pglib_opf_case1354_pegase | 1354 | 1991 | PEGASE |
| pglib_opf_case2000_goc | 2000 | 3206 | GOC |
| pglib_opf_case2383wp_k | 2383 | 2896 | Polish |
| pglib_opf_case2736sp_k | 2736 | 3504 | Polish |
| pglib_opf_case2869_pegase | 2869 | 4582 | PEGASE |
| pglib_opf_case10000_goc | 10000 | 13193 | GOC |

### Protocol

1. For each case, run `compute_results_for_case()` with both DC and AC
2. Extract `dc_radius_l2_min` (global minimum DC L2 radius) and `ac_radius_l2_min` (global minimum AC L2 radius)
3. Record compute time, number of buses/lines
4. Save per-case JSON results and summary.json

### Output

- `experiments/output/pglib_sweep_good_v*/` directories
- `summary.json`: Aggregated metrics per case
- `fig1_dc_vs_ac_radius.png/pdf`: Scatter plot of DC vs AC radius

### Metrics Collected

Per case:
- `n_bus`, `n_line`
- `dc_radius_l2_min`, `dc_radius_l2_mean`
- `ac_radius_l2_min`, `ac_radius_l2_mean`
- `compute_time_sec`
- `base_dispatch` used
- `ac_pf_status`, `ac_pf_attempt`

---

## 3. Experiment 2: Sigma Radius (`run_sigma_radius.py`)

### Purpose

Perform deep analysis of the sigma-radius under heterogeneous per-bus uncertainty, derived from realistic hourly demand patterns (UnitCommitment.jl data). This is the primary experiment demonstrating the sigma-radius concept.

### Protocol

1. Load experiment config (e.g., `sigma_case2000_goc.yaml`)
2. Parse UnitCommitment.jl JSON → extract per-bus sigma arrays from hourly demand standard deviation
3. Run `compute_results_for_case()` with AC, sigma, metric, and h-vector saving
4. Save results, sigma arrays, and h-vectors
5. Run AC Monte Carlo verification (sigma-radius-aware)
6. Run worst-case verification
7. Generate visualizations

### Key Configurations

From experiment YAML files:

```yaml
case_path: data/input/pglib_opf_case2000_goc.m
uc_jl_path: data/uc_jl/case118.json
sigma_q_fraction: 0.3         # Q sigma = 0.3 * P sigma
n_timesteps: 24               # hours of demand data
mc_samples: 10000
mc_seed: 42
base_dispatch: case
```

### Output Files

Per experiment directory:
- `results.json`: Full computation results
- `sigma_arrays.json`: Per-bus sigma values used
- `hvectors.npz`: Saved h-vectors (NumPy compressed)
- `validation.json`: MC validation results
- `verification_results.json`: Certificate verification
- `table2_sigma_radius.csv`: Formatted table for paper
- `mc_tightened_limit.json`: MC results under tightened limits
- Plots:
  - `fig_critical_lines.png/pdf`: Bar chart of most critical lines (lowest sigma-radius)
  - `fig_flow_vs_limit.png/pdf`: Scatter of |S0| vs thermal limit per line
  - `fig_violation_scale.png/pdf`: How violations scale with perturbation magnitude
  - `topology_sigma_radius.png/pdf`: Network topology colored by sigma-radius
  - `fig2_l2_vs_sigma.png/pdf`: L2 vs sigma radius comparison
  - `fig2b_sigma_heatmap.png/pdf`: Sigma-radius heatmap

### Cases Analyzed

- `sigma_case2000_goc`: 2000-bus GOC network
- `sigma_case2736sp_k`: 2736-bus Polish network
- `sigma_case2869_pegase`: 2869-bus PEGASE network
- `sigma_radius_hourly`: Hourly sigma analysis

---

## 4. Experiment 3: Worst-Case Verification (`run_worst_case_verify.py`)

### Purpose

Validate that the analytically computed worst-case perturbation actually causes the predicted violation when applied to the network and verified through a full nonlinear AC power flow.

### Protocol

1. Compute all radii including worst-case perturbation vectors
2. For each critical line (lines with finite, positive sigma-radius):
   a. Extract worst-case perturbation (dp_mw, dq_mvar)
   b. Apply perturbation to base dispatch in the pandapower network
   c. Run full AC PF on the perturbed network
   d. Measure actual |S| at the binding end
   e. Compare with linearized prediction: |S_predicted| = |S0| + h^T · Δu
3. Report success/failure based on whether the actual flow exceeds the thermal limit

### Success Criteria

The verification passes if:
- The perturbed AC PF converges
- The actual flow magnitude at the binding end is close to the thermal limit (within tolerance)
- The direction of violation matches the prediction

---

## 5. Experiment 4: Scalability (`run_scalability.py`)

### Purpose

Measure wall-clock time for DC and AC radius computation across network sizes to establish the computational scaling behavior.

### Protocol

1. Load pglib_sweep.yaml case list (sorted by network size)
2. For each case, repeat `repeats` times (default 3):
   a. Time `compute_results_for_case()` call (DC only)
   b. Time `compute_results_for_case()` call (AC only, if enabled)
   c. Record n_bus, n_line, wall-clock time
3. Compute mean/std across repeats
4. Save results as JSON

### Expected Scaling

- **DC**: Dominated by sparse LU factorization of B_red, approximately O(n^{1.5}) for 2D graphs
- **AC**: Dominated by sparse LU factorization of Jacobian + m adjoint solves, approximately O(n^{1.5} + m × n)

---

## 6. Metrics Analysis Pipeline (`metrics_analysis.py`)

### Purpose

Quantitatively compare stability radii against baseline robustness metrics, using Monte Carlo simulation as ground truth for per-line overload probability.

### Compared Metrics

| Metric | Type | Source | Direction |
|--------|------|--------|-----------|
| `radius_ac_l2` | Stability radius | AC L2 certificate | Lower = more dangerous |
| `radius_ac_sigma` | Stability radius | AC sigma certificate | Lower = more dangerous |
| `radius_ac_metric` | Stability radius | AC metric certificate | Lower = more dangerous |
| `loading_ratio` | Baseline | |S0|/S_limit | Higher = more loaded |
| `headroom_mva` | Baseline | S_limit - |S0| | Lower = more dangerous |
| `cheb_prob_upper` | Baseline | Cantelli bound | Higher = more dangerous |
| `overload_probability_ac` | Analytic | Gaussian Q-function | Higher = more dangerous |

### Evaluation Methodology

#### Spearman Rank Correlation

For each metric, compute Spearman's ρ between the metric values and empirical overload probability (from Monte Carlo):

- Metrics where lower = more dangerous are negated before correlation
- Positive ρ means the metric correctly identifies dangerous lines
- p-value indicates statistical significance

#### Precision-at-k

For each metric, rank all lines by "most dangerous" and measure:
- Mean empirical overload probability of the top-k lines
- Higher values indicate the metric successfully identifies actually dangerous lines
- Default k values: 3, 5, 10

### Output

- `unified_per_line_metrics.csv`: One row per line, all metrics as columns
- `spearman_correlations.csv`: ρ and p-value per metric
- `precision_at_k.csv`: Mean/max empirical probability for top-k per metric
- Scatter plots: each metric vs empirical overload probability
- Bar chart: Spearman ρ comparison
- Histograms: distribution of AC radii

---

## 7. Reproducibility

### Seeds and Randomness

- Monte Carlo RNG seed: configurable via `--mc-seed` (default: 42)
- HiGHS random seed: configurable in OPF config
- Random perturbations are generated using `numpy.random.default_rng(seed)`

### Solver Tolerances

| Solver | Parameter | Default |
|--------|-----------|---------|
| pandapower.runpp (AC PF) | tolerance_mva | 1e-8 |
| pandapower.runopp (AC FPF) | PDIPM_FEASTOL | 1e-6 |
| HiGHS (DC OPF) | time_limit | configurable |

### Deterministic Ordering

- Bus indices: always `sorted(net.bus.index)`
- Line indices: always `sorted(net.line.index)`
- Results keyed by `line_<id>` for stable ordering

### Result Schema Versioning

The `__meta__.schema_version` field (currently 3) tracks the results format version for forward compatibility.

---

## 8. Interpreting Results

### Radius Values

- **r > 0**: Base point is feasible for this line; perturbations up to norm r are safe
- **r = 0**: Line is at its thermal limit (binding constraint)
- **r < 0**: Base point already violates this line's thermal limit
- **r = +inf**: This line has zero sensitivity to perturbations (or is unconstrained)
- **r = NaN**: Computation failed or line was skipped

### Global Certificate

The global minimum r* across all lines gives the tightest certificate: any injection perturbation with ||Δp|| ≤ r* is guaranteed safe for all lines simultaneously (under the linear model).

### Sigma-Radius Interpretation

A sigma-radius of k means the operating point is k standard deviations away from the thermal limit for that line. Typical interpretations:
- k < 1: High risk (overload likely under normal fluctuations)
- 1 < k < 3: Moderate risk
- k > 3: Low risk (overload requires extreme perturbation)

### Overload Probability

The Gaussian overload probability gives P(|S| > c) assuming linearized flow sensitivity and Gaussian perturbations. It serves as a complement to the sigma-radius for probabilistic risk assessment.
