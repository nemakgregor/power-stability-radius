# TODO: AC Stability Radius — Step-by-Step Plan to Publication

This document is the master TODO list for preparing the project **stability-radius** for a Q1 journal submission. Each step is self-contained and must pass `pytest` before proceeding to the next. Steps are ordered by dependency — complete them sequentially.

***

## Phase 1 — Bugfixes and Correctness (MUST DO FIRST)

### Step 1.1 — Fix `estimate_line_limit_mva` for unconstrained lines

**File:** `src/stability_radius/radii/common.py`

**Problem:** In PGLib/MATPOWER convention, `rateA = 0` means "unconstrained" (no thermal limit), not "zero limit". The current code returns `0.0` for such lines, which produces `margin < 0` and `radius = 0` — a false-positive bottleneck that invalidates every downstream AC and DC result.[^1]

**What to change:**
- In `estimate_line_limit_mva`, when the extracted `rateA` value is `0.0` (or `NaN`, `+inf`), return a large finite surrogate value (e.g., `1e5 MVA`) instead of zero.
- Add a boolean field `is_unconstrained` to the per-line output so downstream consumers (tables, plots) can mark these lines as "no real limit" rather than displaying a misleading finite radius.

**Why:** Without this fix, every PGLib case that has even one `rateA=0` line will report `r* = 0`, making all experiments meaningless.

**Test:** `tests/test_radii_common_line_limits.py` — add a parametrized test with `rateA ∈ {0, NaN, inf, 100}` and assert that the first three return the fallback value and the last returns `100.0`.

***

### Step 1.2 — Fix `|S| ≈ 0` fallback in `ac_l2.py`

**File:** `src/stability_radius/radii/ac_l2.py`

**Problem:** When `|S_0| < ε` on a line end, the gradient weights become `(wP, wQ) = (1.0, 0.0)`, discarding the Q-channel entirely. This is mathematically incorrect: at zero apparent power, the gradient of `|S|` is undefined, and a one-sided fallback biases the sensitivity norm.

**What to change:**
- Replace the `wP=1, wQ=0` fallback with `wP = wQ = 1/√2`. This is equivalent to asking "what is the sensitivity of `|S|` in the direction of equal P and Q injection?" — a conservative, unbiased choice.
- Add a brief comment explaining the reasoning: the gradient of a norm at the origin does not exist; the equal-weight fallback is a conservative certificate choice.

**Why:** Without this, lightly loaded lines get artificially asymmetric sensitivity vectors, which distorts the radius and the worst-case direction.

**Test:** In `tests/test_ac_radius_smoke.py`, add a 2-bus case where one line has near-zero flow. Assert that `||h||2 > 0` and `radius_ac_l2 > 0` for this line.

***

### Step 1.3 — Reconcile `unconstrained_line_nom_mw` between `config.py` and YAML

**Files:** `src/stability_radius/config.py`, `conf/config_shared.yaml`

**Problem:** `OPFConfig.unconstrained_line_nom_mw` defaults to `1e6` in Python but `config_shared.yaml` sets `1.0e5`. When running from YAML the value is `1e5`; when running programmatic tests without YAML it is `1e6`. This inconsistency means CI tests and CLI experiments may silently use different limits.

**What to change:**
- Set the Python dataclass default to `1e5` (matching the YAML) or vice versa — pick one value and enforce it in both places.
- Add a determinism assertion in `workflows.py` that logs the effective `unconstrained_line_nom_mw` at compute start.

**Why:** Reproducibility: results must be identical regardless of entry point.

**Test:** `tests/test_config_extends.py` — load the full YAML chain and assert the value equals the Python default.

***

### Step 1.4 — Verify AC Jacobian units and signs against pandapower

**File:** `src/stability_radius/ac/ac_model.py`

**Problem (potential):** The Jacobian `build_reduced_pf_jacobian_mw_perunit` scales partials by `sn_mva`. The sign convention for `P_from / P_to` must match pandapower's `res_line.p_from_mw / p_to_mw` — i.e., power leaving from-bus into line is positive. If signs differ, the certificate is unsound.

**What to do (verification, not code change):**
- Create a dedicated test `tests/test_ac_jacobian_vs_pandapower.py`.
- For a 3-bus meshed net: (a) solve AC PF via pandapower; (b) build `ACOperator`; (c) apply a tiny balanced perturbation `δu` (e.g., 0.01 MW); (d) solve PF again with perturbed injections; (e) compare `Δf_actual` vs `J · δu` (finite-difference check). Require `max |error| < 0.1%` of `|Δf|`.

**Why:** This is the single most critical correctness check. If the Jacobian is wrong (sign flip, missing factor), every radius is meaningless. The finite-difference test is the gold standard.

**Formula for reference:**

\[ \Delta S_\ell^{\text{predicted}} = \nabla_{u} S_\ell \cdot \delta u, \quad \Delta S_\ell^{\text{actual}} = S_\ell(u_0 + \delta u) - S_\ell(u_0) \]

Require: \( \|\Delta S^{\text{predicted}} - \Delta S^{\text{actual}}\|_\infty / \|\Delta S^{\text{actual}}\|_\infty < 10^{-3} \)

***

### Step 1.5 — Remove dead file `res_gpt.md`

**File:** `res_gpt.md`

**Problem:** This file is a GPT conversation log, not part of the project. It appears at the repository root and may confuse reviewers or CI systems.

**What to change:** Delete it. Ensure no imports or references exist anywhere.

***

## Phase 2 — AC Sigma-Radius Module (Core New Feature)

### Step 2.1 — Create `radii/ac_sigma_radius.py`

**New file:** `src/stability_radius/radii/ac_sigma_radius.py`

**Purpose:** Compute the AC sigma-radius — the stability certificate in units of standard deviations, given per-bus injection covariance.

**Mathematical definition.** Let \( h_\ell \in \mathbb{R}^{2n} \) be the adjoint sensitivity vector for line \(\ell\) (from `ac_l2.py`), partitioned as \( h_\ell = [h_\ell^P; h_\ell^Q] \). Let \( \sigma_{P,i} \) and \( \sigma_{Q,i} \) be the per-bus standard deviations of active and reactive power injection perturbations.

Define the diagonal covariance matrix:

\[ \Sigma = \mathrm{diag}(\sigma_{P,1}^2, \ldots, \sigma_{P,n}^2, \sigma_{Q,1}^2, \ldots, \sigma_{Q,n}^2) \]

The standard deviation of flow at the binding end of line \(\ell\):

\[ \sigma_\ell = \|\Sigma^{1/2} h_\ell\|_2 = \sqrt{\sum_i (\sigma_{P,i} \cdot h_{\ell,i}^P)^2 + \sum_i (\sigma_{Q,i} \cdot h_{\ell,i}^Q)^2} \] [^2]

The sigma-radius (in units of σ):

\[ r_\ell^\sigma = \frac{c_\ell - |S_\ell^0|}{\sigma_\ell} \] [^3]

The worst-case perturbation vector (physical units, MW/MVAr):

\[ \Delta u_\ell^* = r_\ell^\sigma \cdot \frac{\Sigma \, h_\ell}{\sigma_\ell} \] [^4]

Component-wise:

\[ \Delta P_i^* = r_\ell^\sigma \cdot \frac{\sigma_{P,i}^2 \cdot h_{\ell,i}^P}{\sigma_\ell}, \quad \Delta Q_i^* = r_\ell^\sigma \cdot \frac{\sigma_{Q,i}^2 \cdot h_{\ell,i}^Q}{\sigma_\ell} \] [^5]

Overload probability (Gaussian, symmetric limits):

\[ P(|S_\ell| > c_\ell) = \Phi\!\left(-\frac{c_\ell - |S_\ell^0|}{\sigma_\ell}\right) + \Phi\!\left(-\frac{c_\ell + |S_\ell^0|}{\sigma_\ell}\right) \] [^6]

**Design decisions:**
- This module does NOT call `build_ac_operator` or `compute_ac_l2_radius` internally. Instead it accepts pre-computed `h_vectors` (a 2D array of shape `(n_lines, 2*n_bus)`) as input. This keeps the module stateless and testable.
- Sigma vectors `sigma_p_mw` and `sigma_q_mvar` are `np.ndarray` of shape `(n_bus,)`.
- Balanced-subspace projection: if `balance=True`, project h-vectors onto `1ᵀΔP=0, 1ᵀΔQ=0` before computing σ (same logic as the P/Q block projection in `ac_l2.py`).

**Outputs per line (dict keys):**
- `sigma_flow_mva` (float) — eq.[^2]
- `radius_ac_sigma` (float) — eq., dimensionless (in σ units)[^3]
- `overload_probability_ac` (float) — eq.[^6]
- `worst_case_dp_mw` (np.ndarray, `(n_bus,)`) — eq., P-block[^5]
- `worst_case_dq_mvar` (np.ndarray, `(n_bus,)`) — eq., Q-block[^5]
- `worst_case_s_predicted_mva` (float) — linearized |S| at worst-case point

**Test:** `tests/test_ac_sigma_radius.py` — synthetic 3-bus net with known σ; verify that `r_σ = margin / σ_flow` within tolerance.

***

### Step 2.2 — Modify `ac_l2.py` to export raw `h`-vectors

**File:** `src/stability_radius/radii/ac_l2.py`

**Problem:** Currently `compute_ac_l2_radius` computes `h_ℓ` inside the chunked loop and immediately reduces it to `||h||₂`, discarding the vector. The sigma-radius module (Step 2.1) needs the full vectors.

**What to change:**
- Add an optional parameter `return_h_vectors: bool = False`.
- When `True`, accumulate `h_from` and `h_to` arrays of shape `(n_lines, 2*n_red)` and return them in a separate dict key `"_h_vectors"` alongside the per-line results.
- The existing API (default `False`) is unchanged — no breaking change.

**Why:** The h-vectors are the most expensive artifact (one LU solve per line-end). Computing them twice would double the wall-clock time.

***

### Step 2.3 — Create `radii/ac_metric_radius.py`

**New file:** `src/stability_radius/radii/ac_metric_radius.py`

**Purpose:** Compute the AC metric radius — the certificate under an arbitrary SPD weight matrix \(M\) in the perturbation space.

**Mathematical definition.** Given weight matrix \(M \succ 0\) and sensitivity \(h_\ell\):

\[ r_\ell^M = \frac{c_\ell - |S_\ell^0|}{\sqrt{h_\ell^T M^{-1} h_\ell}} \] [^7]

The existing DC `metric.py` uses Cholesky decomposition of \(M\). The AC version is identical in structure but operates on the `2n`-dimensional space \([ΔP; ΔQ]\).

**Design:** Accept `h_vectors` (same as Step 2.1) and `M` matrix. Support both dense `(2n, 2n)` and diagonal `(2n,)` forms.

**Why this is a separate module (not merged with sigma):** The sigma-radius is a special case of metric radius where \(M = \Sigma^{-1}\) (inverse covariance). But the metric module supports arbitrary M (e.g., Laplacian of the network graph, or a cost-based weighting), which is useful for generalizing the paper's contribution.

**Test:** `tests/test_ac_metric_radius.py` — verify that `M = I` gives the same result as `radius_ac_l2`.

***

### Step 2.4 — Integrate AC sigma and metric radii into `workflows.py`

**File:** `src/stability_radius/workflows.py`

**What to change:**
- After the existing AC L2 block, add conditional calls to `compute_ac_sigma_radius` and/or `compute_ac_metric_radius` if `sigma_p_mw` / `sigma_q_mvar` arrays are provided.
- Add new config keys under `compute.ac`: `sigma_p_mw_source`, `sigma_q_mvar_source` (values: `"uniform"` / `"file"` / `null`). When `"uniform"`, use scalar `sigma_p_mw` and `sigma_q_mvar` from config. When `"file"`, load from a JSON/CSV path (UC.jl integration, Phase 4).
- Merge sigma/metric results into `results_lines`.
- Save h-vectors to a separate compressed file (`.npz`) alongside `results.json` if config flag `save_h_vectors: true` is set.

**Why:** Keeps the main pipeline extensible without bloating the core `ac_l2.py`.

***

## Phase 3 — Worst-Case Verification Module

### Step 3.1 — Create `verification/verify_worst_case.py`

**New file:** `src/stability_radius/verification/verify_worst_case.py`

**Purpose:** Given a worst-case perturbation vector \(\Delta u^*_\ell\) from Step 2.1, verify by running a full nonlinear AC PF (pandapower) that the predicted overload actually occurs.

**Algorithm:**
1. Deep-copy `net`; apply lossless policy (matching the certificate).
2. For each bus `i`, add `sgen` with `p_mw = ΔP_i*`, `q_mvar = ΔQ_i*`.
3. Run `pandapower.runpp`.
4. Extract `|S_from|`, `|S_to|` for the target line.
5. Compare against `c_ℓ` (thermal limit in MVA).

**Outputs per verification:**
- `predicted_s_mva` (from linear model)
- `actual_s_mva` (from nonlinear PF)
- `limit_mva`
- `violated: bool`
- `pf_converged: bool`
- `relative_error` = `|predicted - actual| / actual`

**Design:** This is a pure verification function. It does NOT modify results or recompute radii. It returns a dataclass `WorstCaseVerificationResult`.

**Key insight for the paper:** If the linearized prediction says violation occurs at `|S| = c + ε` and the nonlinear PF confirms `|S_actual| > c`, the certificate is validated. If the nonlinear PF shows no violation, the linearization error exceeds the margin — which quantifies the conservatism.

**Test:** `tests/test_verify_worst_case.py` — on a 3-bus net, construct a case with tight margin, verify that the worst-case direction achieves violation and that a 50% scaled direction does not.

***

### Step 3.2 — Create `verification/ac_monte_carlo_sigma.py`

**New file:** `src/stability_radius/verification/ac_monte_carlo_sigma.py`

**Purpose:** Run AC Monte Carlo with injection perturbations drawn from \(\mathcal{N}(0, \Sigma)\) where \(\Sigma\) is the per-bus covariance (not isotropic). This validates the sigma-radius certificate.

**How it differs from existing `monte_carlo.py`:**
- The existing MC uses `sigma_p_mw` and `sigma_q_mvar` as scalars (uniform across buses).
- The new module accepts `np.ndarray` per-bus sigma vectors.
- Sample generation: draw `z ~ N(0,I)`, then scale `ΔP_i = σ_{P,i} · z_i`, `ΔQ_i = σ_{Q,i} · z_{n+i}`.
- Balance enforcement: project each sample onto `1ᵀΔP=0, 1ᵀΔQ=0`.

**Outputs:**
- `n_samples`, `n_violations`, `n_pf_failures`
- `empirical_overload_probability` (by line)
- `soundness_inside_sigma_ball`: fraction of samples with `||Σ^{-1/2} Δu|| ≤ r_σ` that have no violations

**Test:** `tests/test_ac_mc_sigma.py` — on case14, verify that `soundness_inside_sigma_ball == 1.0` with 500 samples at `0.9 * r_σ`.

***

## Phase 4 — UnitCommitment.jl Data Integration

### Step 4.1 — Create `parsers/uc_jl.py`

**New file:** `src/stability_radius/parsers/uc_jl.py`

**Purpose:** Load a UnitCommitment.jl JSON instance and extract per-bus injection standard deviations from time-series data.

**UC.jl data format:**[^8][^9]
```json
{
  "Buses": {
    "b1": {"Load (MW)": [100, 110, 105, ...]},
    ...
  },
  "Generators": {
    "gen1": {
      "Bus": "b1",
      "Max power (MW)": [...],  // or scalar
      "Min power (MW)": [...],
      ...
    }
  },
  "Transmission lines": {
    "l1": {"Source bus": "b1", "Target bus": "b2", "Reactance (ohms)": ..., ...}
  }
}
```

**Extraction algorithm:**
1. For each bus, collect the time series of `"Load (MW)"` — this gives `σ_load` per bus.
2. For each generator, if `"Max power (MW)"` is a list (time-varying capacity), compute `std(capacity)` — this gives `σ_gen` per bus.
3. Total: `σ_{P,i} = sqrt(σ_{load,i}² + σ_{gen,i}²)`.
4. For Q: estimate `σ_{Q,i} = σ_{P,i} · tan(arccos(pf))` with a configurable power factor (default `pf=0.9`).

**Bus name mapping challenge:** UC.jl uses bus names like `"b1"`, `"b2"`, ..., while PGLib/pandapower uses integer indices. The parser must accept an explicit mapping dict `{"b1": 0, "b2": 1, ...}` or infer it from bus ordering.

**Outputs:**
- `sigma_p_mw: np.ndarray (n_bus,)`
- `sigma_q_mvar: np.ndarray (n_bus,)`
- `n_timesteps: int`
- `bus_mapping: dict[str, int]`
- `metadata: dict` (source file path, date range, etc.)

**Test:** `tests/test_uc_jl_parser.py` — create a minimal UC.jl JSON fixture (3 buses, 24 hours) and verify extracted sigma values.

***

### Step 4.2 — Create download utility for UC.jl instances

**File:** `src/stability_radius/utils/download.py` (extend existing)

**What to add:**
- Function `download_uc_jl_instance(case_name: str, dest_dir: Path) -> Path`.
- Base URL pattern: `https://axavier.org/UnitCommitment.jl/0.4/instances/matpower/{case_name}/2017-01-01.json.gz`.[^10]
- Support for `.json.gz` (gzip-compressed).
- Deterministic: hash check if file already exists, skip re-download.

**Test:** `tests/test_download.py` — mock the HTTP call and verify file is written correctly.

***

## Phase 5 — Project Structure Cleanup

### Step 5.1 — Create `experiments/` directory (separate from `tests/`)

**New directory:** `experiments/`

**Rationale:** `tests/` contains pytest unit/integration tests that assert correctness (pass/fail). Experiments are scripts that produce tables, plots, and numerical results for the paper. They should NOT be in `tests/` — they are not testing anything, they are measuring.

**Structure:**
```
experiments/
├── __init__.py
├── configs/           # YAML configs for each experiment
│   ├── pglib_sweep.yaml
│   ├── uc_jl_case118.yaml
│   └── ...
├── run_pglib_sweep.py       # Experiment 1: DC vs AC radius across PGLib
├── run_sigma_radius.py      # Experiment 2: sigma-radius with UC.jl data
├── run_worst_case_verify.py # Experiment 3: worst-case verification
├── run_scalability.py       # Experiment 4: wall-clock time vs network size
├── collect_results.py       # Aggregate JSON → CSV summary table
├── plot_radius_distribution.py
├── plot_sigma_vs_time.py
├── plot_worst_case_heatmap.py
└── README.md
```

**Each `run_*.py` script:**
- Reads a config YAML from `experiments/configs/`.
- Calls library functions from `stability_radius.*`.
- Writes results to `experiments/output/{experiment_name}/`.
- Does NOT import from `tests/`.

***

### Step 5.2 — Add `__init__.py` re-exports for new modules

**Files to update:** `src/stability_radius/radii/__init__.py`, `src/stability_radius/verification/__init__.py`, `src/stability_radius/parsers/__init__.py` (may need creation).

Add imports for:
- `ac_sigma_radius.compute_ac_sigma_radius`
- `ac_metric_radius.compute_ac_metric_radius`
- `verify_worst_case.verify_worst_case_perturbation`
- `uc_jl.load_uc_jl_sigma`

***

### Step 5.3 — Update `results.json` schema (schema_version=3)

**Files:** `src/stability_radius/workflows.py`, `UNITS_CONTRACT.md`

**New per-line fields (AC section):**
- `sigma_flow_mva` — flow std at binding end (MVA)
- `radius_ac_sigma` — dimensionless (in σ units)
- `overload_probability_ac` — Gaussian probability
- `worst_case_dp_mw` — list of floats (or null if not computed)
- `worst_case_dq_mvar` — list of floats (or null)

**New `__meta__` fields:**
- `schema_version: 3`
- `ac.sigma_source: "uniform" | "uc_jl" | null`
- `ac.sigma_p_mw: list[float] | float | null`
- `ac.sigma_q_mvar: list[float] | float | null`
- `ac.sigma_n_timesteps: int | null`

**Backward compatibility:** tests that check schema_version=2 should still pass on version=3 data (new fields are additive).

***

## Phase 6 — Experiments for Paper

### Step 6.1 — Experiment 1: PGLib DC vs AC radius sweep

**Script:** `experiments/run_pglib_sweep.py`

**Cases:** case5, case14, case30, case57, case118, case300, case1354_pegase, case2869_pegase (from PGLib-OPF).[^1]

**For each case, compute:**
- DC L2 radius (\(r^*_{\rm DC}\))
- AC L2 radius (\(r^*_{\rm AC}\))
- Ratio AC/DC
- Bottleneck line index and its margin
- Wall-clock time (total, DC-only, AC-only)

**Output Table (Table 1 in paper):**

| Case | \(n_b\) | \(n_l\) | \(r^*_{\rm DC}\) (MW) | \(r^*_{\rm AC}\) (MW) | AC/DC | Time (s) | Bottleneck |
|---|---|---|---|---|---|---|---|

**Output Plot (Fig. 1):** Bar chart comparing \(r^*_{\rm DC}\) and \(r^*_{\rm AC}\) across cases.

***

### Step 6.2 — Experiment 2: AC sigma-radius with UC.jl data

**Script:** `experiments/run_sigma_radius.py`

**Cases:** PGLib case118 + UC.jl `matpower/case118/2017-01-01.json.gz`.[^10]

**Steps:**
1. Load PGLib case118, compute AC PF base point.
2. Load UC.jl instance, extract per-bus σ via `parsers/uc_jl.py`.
3. Compute AC sigma-radius (Step 2.1).
4. Compute AC L2 radius (uniform σ) for comparison.
5. Run worst-case verification (Step 3.1) for the top-5 tightest lines.
6. Run AC MC with non-uniform σ (Step 3.2), 50k samples.

**Output Table (Table 2 in paper):**

| Line | \(r_\ell^{L2}\) (MW) | \(r_\ell^\sigma\) (σ) | σ-flow (MVA) | MC violation rate | Worst-case verified? |
|---|---|---|---|---|---|

**Output Plot (Fig. 2):** Scatter plot of \(r^{L2}\) vs \(r^\sigma\), colored by `binding_end`.

***

### Step 6.3 — Experiment 3: Worst-case direction verification

**Script:** `experiments/run_worst_case_verify.py`

**For the bottleneck line of each PGLib case:**
1. Extract worst-case direction \(\Delta u^*\).
2. Scale it to `{0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2} × r*`.
3. For each scale: run nonlinear AC PF via pandapower. Record actual `|S|`.
4. Plot predicted vs actual `|S|` as a function of scale factor.

**Output Plot (Fig. 3):** For each case, a line plot with x = scale factor, y = `|S_actual| / c`. The curve should cross `y=1` near `scale ≈ 1.0`, validating the certificate boundary.

**Output Table (Table 3 in paper):**

| Case | Scale at crossing | Linearization error at r* | PF convergence failures |
|---|---|---|---|

***

### Step 6.4 — Experiment 4: Scalability (wall-clock time)

**Script:** `experiments/run_scalability.py`

**Cases:** case14 through case2869_pegase.

**Measure (separately):**
- Time to build Ybus + Jacobian + LU factorization.
- Time for adjoint solves (per-line, chunked).
- Total time for AC L2 radius.
- Total time for AC sigma-radius (incremental over L2).

**Output Plot (Fig. 4):** Log-log plot of time vs \(n_{\rm bus}\), with separate curves for each phase. Should show near-linear scaling because LU on sparse Jacobian is \(O(n^{1.5})\) for power grids and adjoint solves are \(O(n_{\rm line})\) forward/back-substitutions.

***

### Step 6.5 — Experiment 5: Sigma-radius vs time of day (UC.jl 24h)

**Script:** `experiments/run_sigma_vs_time.py`

**For case118 with UC.jl data:**
1. For each hour \(t = 0, \ldots, 23\): extract `σ_P(t)`, `σ_Q(t)` from the load/gen profiles (rolling std over a window, or std of the detrended signal).
2. Compute `r*_σ(t)` at each hour.

**Output Plot (Fig. 5):** Line chart of \(r^*_\sigma\) vs hour, overlaid with total load curve. Shows how robustness degrades at peak load.

***

## Phase 7 — Tests (pytest)

### Step 7.1 — New unit tests

| Test file | What it tests | Phase dependency |
|---|---|---|
| `tests/test_ac_sigma_radius.py` | Eq. [^2]–[^6] on synthetic 3-bus net | 2.1 |
| `tests/test_ac_metric_radius.py` | Eq. [^7] with M=I matches L2 | 2.3 |
| `tests/test_verify_worst_case.py` | Worst-case direction causes violation | 3.1 |
| `tests/test_ac_mc_sigma.py` | MC soundness with non-uniform σ | 3.2 |
| `tests/test_uc_jl_parser.py` | Parse minimal UC.jl JSON fixture | 4.1 |
| `tests/test_ac_jacobian_vs_pandapower.py` | Finite-difference Jacobian validation | 1.4 |
| `tests/test_line_limits_pglib.py` | rateA=0 handled as unconstrained | 1.1 |

### Step 7.2 — Update existing tests

- `tests/test_ac_radius_smoke.py` — add `|S|≈0` fallback case (Step 1.2).
- `tests/test_unit_consistency_end_to_end.py` — add AC sigma fields to expected output keys.
- `tests/test_radii_common_line_limits.py` — parametrize with `rateA=0`.
- `tests/test_certificate_concept.py` — no change needed (DC-only, already green).
- `tests/test_verification_report_and_monte_carlo.py` — add AC MC sigma mode.

### Step 7.3 — CI command

After all phases, the full test suite should pass:

```bash
pytest tests/ -v --tb=short -x
```

Expected: **all green** on PGLib cases (case14, case30, case118) with `allow_download=true` or pre-downloaded data.

***

## Phase 8 — Paper Deliverables Checklist

### Tables

| # | Title | Data source | Script |
|---|---|---|---|
| 1 | DC vs AC L2 radius across PGLib cases | Experiment 1 | `collect_results.py` |
| 2 | AC sigma-radius with UC.jl data (case118) | Experiment 2 | `collect_results.py` |
| 3 | Worst-case verification: linearization accuracy | Experiment 3 | `run_worst_case_verify.py` |
| 4 | Scalability: wall-clock time breakdown | Experiment 4 | `run_scalability.py` |

### Figures

| # | Title | Type | Script |
|---|---|---|---|
| 1 | DC vs AC radius comparison (bar chart) | Bar | `plot_radius_distribution.py` |
| 2 | L2 vs sigma radius scatter (case118) | Scatter | `plot_radius_distribution.py` |
| 3 | Predicted vs actual |S| at worst-case boundary | Line | `run_worst_case_verify.py` |
| 4 | Computation time vs network size (log-log) | Line | `run_scalability.py` |
| 5 | Sigma-radius vs hour of day (case118) | Line + area | `plot_sigma_vs_time.py` |

### Artifacts to include in paper repository

- `results/*.json` — all computed results
- `results/*.npz` — h-vectors for reproducibility
- `experiments/output/*.csv` — summary tables
- `figures/*.pdf` — publication-quality plots
- `UNITS_CONTRACT.md` — full specification (appendix)

***

## Summary Dependency Graph

```
Phase 1 (Bugfixes)
  └─ 1.1 line limits
  └─ 1.2 |S|≈0 fallback
  └─ 1.3 config consistency
  └─ 1.4 Jacobian verification
  └─ 1.5 cleanup

Phase 2 (AC sigma/metric)  ← requires Phase 1
  └─ 2.1 ac_sigma_radius.py
  └─ 2.2 export h-vectors from ac_l2.py
  └─ 2.3 ac_metric_radius.py
  └─ 2.4 integrate into workflows.py

Phase 3 (Verification)  ← requires Phase 2
  └─ 3.1 verify_worst_case.py
  └─ 3.2 ac_monte_carlo_sigma.py

Phase 4 (UC.jl)  ← requires Phase 2
  └─ 4.1 parsers/uc_jl.py
  └─ 4.2 download utility

Phase 5 (Structure)  ← requires Phases 2–4
  └─ 5.1 experiments/ directory
  └─ 5.2 re-exports
  └─ 5.3 schema v3

Phase 6 (Experiments)  ← requires Phases 1–5
  └─ 6.1 PGLib sweep
  └─ 6.2 UC.jl sigma
  └─ 6.3 worst-case verify
  └─ 6.4 scalability
  └─ 6.5 time-of-day

Phase 7 (Tests)  ← in parallel with each phase
Phase 8 (Paper deliverables)  ← after Phase 6
```

---

## References

1. [The Power Grid Library for Benchmarking.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_ff8fd29a-57ee-4f08-a7dd-254bca4d6555/55b86605-15a6-4e23-9135-ab4990760725/The-Power-Grid-Library-for-Benchmarking.pdf?AWSAccessKeyId=ASIA2F3EMEYEZHQSUQJT&Signature=qjSTLfBnH7PeDrwqihbV%2BN8SVXs%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEC4aCXVzLWVhc3QtMSJIMEYCIQDC76Z88WwMoJ7DhI6Uo4%2BRZ83WSJWUhj1oIish2B2mswIhAOYPkyudIAdrLcYJrGXmZIoN1BeDXqXtDx%2FT2%2FsGwrIiKvwECPb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgzGrpoSmCs%2F5hQAbuMq0AQtr%2FdpoUiZas%2BL7avxF42tQGsHRngTOBUvdCs7vc5nj5hqju6qqgSsE87MPcjNSFKmrwaj3Vo36A4jLR3UT7N7dhc8Ip9qZswGJiZzWylFSQXLBOSqknf37E2k1BAHqBbcvPOVeiG8oE9fWiJsYVlpT5edxjQjAMfgR1oiJuNBzCpVwV87L9%2FAM7TJDPC0re8MwfOLFZqhyCmdzdd%2FbnDruWI0uPPAx%2F7SJctj8E27Y7rpaOECHzo6P%2FksnvvxoyShfft3dcWT7JbbYL%2BE6trD1ZTfQLNRPDQmOg5c9KQENn5VEVu6zb6XxZB303rzSpTNDctC3d5XYimM99%2BoCIhY9x7iWa9YS5epB0nFj8KhGiau0B0126IXsBH0wK%2B80YBxhzgbFMIIV0mXbOhHAHUBKmdcvWuenZkQC56jpbUjOL%2FdcL1SSfe%2FiKBWCuDuB4mCHEBnrFhoG%2FtdKMGmFvH6wQiTLD2fQA6zJtlSJUTeMYxqom5IOSN8vyHOAfVAn1mSx1bqiMeBD53T7sZNh0LnNqBWUqOejGGhW614JesK8uODFPOSZQbRHXs0he%2Fg4zfwInJw3SeOOHzjPNBHQFzvay6VjgHMm%2BEKPNW5q0BEWb4oBu%2BGRz9Q7O8kwGqEO%2FfAuNGyatZj%2Boxi2H%2BqMzFwVoO%2F1%2Fut4ZBANTcnEZD%2FsJGBKbbETj%2FqyQSjR%2BhNAe2RvoFwVqLtpg%2FtnMmu9qebWEtvZdOyeMaoX%2Fish8IfxdYptq3g8yHnjJwrKPiJGhB02ML%2By25h6B%2BDoOIxH0ikMPvH9swGOpcBdw4ET716dey7Wd12a2HJCWo57S1MsEDOkBYO7cgP3MlsR1cbcw0%2Bck31liLfi8tuA%2FbrvNCsK1zHpq16nIqvd5dHNITo6xwmtKhNXXfgCS3SRuW8fFO%2BIgRtZA7WYMIUZJCQX0OrrxTUWsuZpeTEkctM6gi6d2FJixpKuy1g1bRZD4SqBmvnrAB5x3i1joUXDAue%2FUBe3Q%3D%3D&Expires=1771945010)

2. [posobie-osnovy-funktsionirovaniia-rynkov-elektroenergii.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159164453/d0e485cd-7ea7-4a6d-bfc6-231088fd4c26/posobie-osnovy-funktsionirovaniia-rynkov-elektroenergii.pdf?AWSAccessKeyId=ASIA2F3EMEYEZHQSUQJT&Signature=7BMxJosXXBTS%2B%2BAcw4zH5XhtqAA%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEC4aCXVzLWVhc3QtMSJIMEYCIQDC76Z88WwMoJ7DhI6Uo4%2BRZ83WSJWUhj1oIish2B2mswIhAOYPkyudIAdrLcYJrGXmZIoN1BeDXqXtDx%2FT2%2FsGwrIiKvwECPb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgzGrpoSmCs%2F5hQAbuMq0AQtr%2FdpoUiZas%2BL7avxF42tQGsHRngTOBUvdCs7vc5nj5hqju6qqgSsE87MPcjNSFKmrwaj3Vo36A4jLR3UT7N7dhc8Ip9qZswGJiZzWylFSQXLBOSqknf37E2k1BAHqBbcvPOVeiG8oE9fWiJsYVlpT5edxjQjAMfgR1oiJuNBzCpVwV87L9%2FAM7TJDPC0re8MwfOLFZqhyCmdzdd%2FbnDruWI0uPPAx%2F7SJctj8E27Y7rpaOECHzo6P%2FksnvvxoyShfft3dcWT7JbbYL%2BE6trD1ZTfQLNRPDQmOg5c9KQENn5VEVu6zb6XxZB303rzSpTNDctC3d5XYimM99%2BoCIhY9x7iWa9YS5epB0nFj8KhGiau0B0126IXsBH0wK%2B80YBxhzgbFMIIV0mXbOhHAHUBKmdcvWuenZkQC56jpbUjOL%2FdcL1SSfe%2FiKBWCuDuB4mCHEBnrFhoG%2FtdKMGmFvH6wQiTLD2fQA6zJtlSJUTeMYxqom5IOSN8vyHOAfVAn1mSx1bqiMeBD53T7sZNh0LnNqBWUqOejGGhW614JesK8uODFPOSZQbRHXs0he%2Fg4zfwInJw3SeOOHzjPNBHQFzvay6VjgHMm%2BEKPNW5q0BEWb4oBu%2BGRz9Q7O8kwGqEO%2FfAuNGyatZj%2Boxi2H%2BqMzFwVoO%2F1%2Fut4ZBANTcnEZD%2FsJGBKbbETj%2FqyQSjR%2BhNAe2RvoFwVqLtpg%2FtnMmu9qebWEtvZdOyeMaoX%2Fish8IfxdYptq3g8yHnjJwrKPiJGhB02ML%2By25h6B%2BDoOIxH0ikMPvH9swGOpcBdw4ET716dey7Wd12a2HJCWo57S1MsEDOkBYO7cgP3MlsR1cbcw0%2Bck31liLfi8tuA%2FbrvNCsK1zHpq16nIqvd5dHNITo6xwmtKhNXXfgCS3SRuW8fFO%2BIgRtZA7WYMIUZJCQX0OrrxTUWsuZpeTEkctM6gi6d2FJixpKuy1g1bRZD4SqBmvnrAB5x3i1joUXDAue%2FUBe3Q%3D%3D&Expires=1771945010)

3. [EnergyNet_2019.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159164453/cf770c54-28c7-4204-8081-331191891195/EnergyNet_2019.pdf?AWSAccessKeyId=ASIA2F3EMEYEZHQSUQJT&Signature=aEYJ1xmzSiYYfYYr81E1U9MUcU8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEC4aCXVzLWVhc3QtMSJIMEYCIQDC76Z88WwMoJ7DhI6Uo4%2BRZ83WSJWUhj1oIish2B2mswIhAOYPkyudIAdrLcYJrGXmZIoN1BeDXqXtDx%2FT2%2FsGwrIiKvwECPb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgzGrpoSmCs%2F5hQAbuMq0AQtr%2FdpoUiZas%2BL7avxF42tQGsHRngTOBUvdCs7vc5nj5hqju6qqgSsE87MPcjNSFKmrwaj3Vo36A4jLR3UT7N7dhc8Ip9qZswGJiZzWylFSQXLBOSqknf37E2k1BAHqBbcvPOVeiG8oE9fWiJsYVlpT5edxjQjAMfgR1oiJuNBzCpVwV87L9%2FAM7TJDPC0re8MwfOLFZqhyCmdzdd%2FbnDruWI0uPPAx%2F7SJctj8E27Y7rpaOECHzo6P%2FksnvvxoyShfft3dcWT7JbbYL%2BE6trD1ZTfQLNRPDQmOg5c9KQENn5VEVu6zb6XxZB303rzSpTNDctC3d5XYimM99%2BoCIhY9x7iWa9YS5epB0nFj8KhGiau0B0126IXsBH0wK%2B80YBxhzgbFMIIV0mXbOhHAHUBKmdcvWuenZkQC56jpbUjOL%2FdcL1SSfe%2FiKBWCuDuB4mCHEBnrFhoG%2FtdKMGmFvH6wQiTLD2fQA6zJtlSJUTeMYxqom5IOSN8vyHOAfVAn1mSx1bqiMeBD53T7sZNh0LnNqBWUqOejGGhW614JesK8uODFPOSZQbRHXs0he%2Fg4zfwInJw3SeOOHzjPNBHQFzvay6VjgHMm%2BEKPNW5q0BEWb4oBu%2BGRz9Q7O8kwGqEO%2FfAuNGyatZj%2Boxi2H%2BqMzFwVoO%2F1%2Fut4ZBANTcnEZD%2FsJGBKbbETj%2FqyQSjR%2BhNAe2RvoFwVqLtpg%2FtnMmu9qebWEtvZdOyeMaoX%2Fish8IfxdYptq3g8yHnjJwrKPiJGhB02ML%2By25h6B%2BDoOIxH0ikMPvH9swGOpcBdw4ET716dey7Wd12a2HJCWo57S1MsEDOkBYO7cgP3MlsR1cbcw0%2Bck31liLfi8tuA%2FbrvNCsK1zHpq16nIqvd5dHNITo6xwmtKhNXXfgCS3SRuW8fFO%2BIgRtZA7WYMIUZJCQX0OrrxTUWsuZpeTEkctM6gi6d2FJixpKuy1g1bRZD4SqBmvnrAB5x3i1joUXDAue%2FUBe3Q%3D%3D&Expires=1771945010)

4. [SO-raschetnaia-model.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159164453/1d987d5b-e1b5-46c0-8fb5-d2733224abfe/SO-raschetnaia-model.pdf?AWSAccessKeyId=ASIA2F3EMEYEZHQSUQJT&Signature=YViFABhNuNnGOj%2F64nW63QfwUnk%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEC4aCXVzLWVhc3QtMSJIMEYCIQDC76Z88WwMoJ7DhI6Uo4%2BRZ83WSJWUhj1oIish2B2mswIhAOYPkyudIAdrLcYJrGXmZIoN1BeDXqXtDx%2FT2%2FsGwrIiKvwECPb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgzGrpoSmCs%2F5hQAbuMq0AQtr%2FdpoUiZas%2BL7avxF42tQGsHRngTOBUvdCs7vc5nj5hqju6qqgSsE87MPcjNSFKmrwaj3Vo36A4jLR3UT7N7dhc8Ip9qZswGJiZzWylFSQXLBOSqknf37E2k1BAHqBbcvPOVeiG8oE9fWiJsYVlpT5edxjQjAMfgR1oiJuNBzCpVwV87L9%2FAM7TJDPC0re8MwfOLFZqhyCmdzdd%2FbnDruWI0uPPAx%2F7SJctj8E27Y7rpaOECHzo6P%2FksnvvxoyShfft3dcWT7JbbYL%2BE6trD1ZTfQLNRPDQmOg5c9KQENn5VEVu6zb6XxZB303rzSpTNDctC3d5XYimM99%2BoCIhY9x7iWa9YS5epB0nFj8KhGiau0B0126IXsBH0wK%2B80YBxhzgbFMIIV0mXbOhHAHUBKmdcvWuenZkQC56jpbUjOL%2FdcL1SSfe%2FiKBWCuDuB4mCHEBnrFhoG%2FtdKMGmFvH6wQiTLD2fQA6zJtlSJUTeMYxqom5IOSN8vyHOAfVAn1mSx1bqiMeBD53T7sZNh0LnNqBWUqOejGGhW614JesK8uODFPOSZQbRHXs0he%2Fg4zfwInJw3SeOOHzjPNBHQFzvay6VjgHMm%2BEKPNW5q0BEWb4oBu%2BGRz9Q7O8kwGqEO%2FfAuNGyatZj%2Boxi2H%2BqMzFwVoO%2F1%2Fut4ZBANTcnEZD%2FsJGBKbbETj%2FqyQSjR%2BhNAe2RvoFwVqLtpg%2FtnMmu9qebWEtvZdOyeMaoX%2Fish8IfxdYptq3g8yHnjJwrKPiJGhB02ML%2By25h6B%2BDoOIxH0ikMPvH9swGOpcBdw4ET716dey7Wd12a2HJCWo57S1MsEDOkBYO7cgP3MlsR1cbcw0%2Bck31liLfi8tuA%2FbrvNCsK1zHpq16nIqvd5dHNITo6xwmtKhNXXfgCS3SRuW8fFO%2BIgRtZA7WYMIUZJCQX0OrrxTUWsuZpeTEkctM6gi6d2FJixpKuy1g1bRZD4SqBmvnrAB5x3i1joUXDAue%2FUBe3Q%3D%3D&Expires=1771945010)

5. [Matmodel-RSV.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159164453/612b5e56-aa71-4d46-9f23-a19ece6210a4/Matmodel-RSV.pdf?AWSAccessKeyId=ASIA2F3EMEYEZHQSUQJT&Signature=3JNaPKn98l1%2BrBjPEKRsccGXHN8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEC4aCXVzLWVhc3QtMSJIMEYCIQDC76Z88WwMoJ7DhI6Uo4%2BRZ83WSJWUhj1oIish2B2mswIhAOYPkyudIAdrLcYJrGXmZIoN1BeDXqXtDx%2FT2%2FsGwrIiKvwECPb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgzGrpoSmCs%2F5hQAbuMq0AQtr%2FdpoUiZas%2BL7avxF42tQGsHRngTOBUvdCs7vc5nj5hqju6qqgSsE87MPcjNSFKmrwaj3Vo36A4jLR3UT7N7dhc8Ip9qZswGJiZzWylFSQXLBOSqknf37E2k1BAHqBbcvPOVeiG8oE9fWiJsYVlpT5edxjQjAMfgR1oiJuNBzCpVwV87L9%2FAM7TJDPC0re8MwfOLFZqhyCmdzdd%2FbnDruWI0uPPAx%2F7SJctj8E27Y7rpaOECHzo6P%2FksnvvxoyShfft3dcWT7JbbYL%2BE6trD1ZTfQLNRPDQmOg5c9KQENn5VEVu6zb6XxZB303rzSpTNDctC3d5XYimM99%2BoCIhY9x7iWa9YS5epB0nFj8KhGiau0B0126IXsBH0wK%2B80YBxhzgbFMIIV0mXbOhHAHUBKmdcvWuenZkQC56jpbUjOL%2FdcL1SSfe%2FiKBWCuDuB4mCHEBnrFhoG%2FtdKMGmFvH6wQiTLD2fQA6zJtlSJUTeMYxqom5IOSN8vyHOAfVAn1mSx1bqiMeBD53T7sZNh0LnNqBWUqOejGGhW614JesK8uODFPOSZQbRHXs0he%2Fg4zfwInJw3SeOOHzjPNBHQFzvay6VjgHMm%2BEKPNW5q0BEWb4oBu%2BGRz9Q7O8kwGqEO%2FfAuNGyatZj%2Boxi2H%2BqMzFwVoO%2F1%2Fut4ZBANTcnEZD%2FsJGBKbbETj%2FqyQSjR%2BhNAe2RvoFwVqLtpg%2FtnMmu9qebWEtvZdOyeMaoX%2Fish8IfxdYptq3g8yHnjJwrKPiJGhB02ML%2By25h6B%2BDoOIxH0ikMPvH9swGOpcBdw4ET716dey7Wd12a2HJCWo57S1MsEDOkBYO7cgP3MlsR1cbcw0%2Bck31liLfi8tuA%2FbrvNCsK1zHpq16nIqvd5dHNITo6xwmtKhNXXfgCS3SRuW8fFO%2BIgRtZA7WYMIUZJCQX0OrrxTUWsuZpeTEkctM6gi6d2FJixpKuy1g1bRZD4SqBmvnrAB5x3i1joUXDAue%2FUBe3Q%3D%3D&Expires=1771945010)

6. [posobie1-SPB-osnovy-funktsionirovaniia-rynkov-elektroenergii.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159164453/d15890dd-3a5c-4399-b32f-108c7323f8c0/posobie1-SPB-osnovy-funktsionirovaniia-rynkov-elektroenergii.pdf?AWSAccessKeyId=ASIA2F3EMEYEZHQSUQJT&Signature=MiyPtHYpxo%2BwbcVmqb294o00t3M%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEC4aCXVzLWVhc3QtMSJIMEYCIQDC76Z88WwMoJ7DhI6Uo4%2BRZ83WSJWUhj1oIish2B2mswIhAOYPkyudIAdrLcYJrGXmZIoN1BeDXqXtDx%2FT2%2FsGwrIiKvwECPb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgzGrpoSmCs%2F5hQAbuMq0AQtr%2FdpoUiZas%2BL7avxF42tQGsHRngTOBUvdCs7vc5nj5hqju6qqgSsE87MPcjNSFKmrwaj3Vo36A4jLR3UT7N7dhc8Ip9qZswGJiZzWylFSQXLBOSqknf37E2k1BAHqBbcvPOVeiG8oE9fWiJsYVlpT5edxjQjAMfgR1oiJuNBzCpVwV87L9%2FAM7TJDPC0re8MwfOLFZqhyCmdzdd%2FbnDruWI0uPPAx%2F7SJctj8E27Y7rpaOECHzo6P%2FksnvvxoyShfft3dcWT7JbbYL%2BE6trD1ZTfQLNRPDQmOg5c9KQENn5VEVu6zb6XxZB303rzSpTNDctC3d5XYimM99%2BoCIhY9x7iWa9YS5epB0nFj8KhGiau0B0126IXsBH0wK%2B80YBxhzgbFMIIV0mXbOhHAHUBKmdcvWuenZkQC56jpbUjOL%2FdcL1SSfe%2FiKBWCuDuB4mCHEBnrFhoG%2FtdKMGmFvH6wQiTLD2fQA6zJtlSJUTeMYxqom5IOSN8vyHOAfVAn1mSx1bqiMeBD53T7sZNh0LnNqBWUqOejGGhW614JesK8uODFPOSZQbRHXs0he%2Fg4zfwInJw3SeOOHzjPNBHQFzvay6VjgHMm%2BEKPNW5q0BEWb4oBu%2BGRz9Q7O8kwGqEO%2FfAuNGyatZj%2Boxi2H%2BqMzFwVoO%2F1%2Fut4ZBANTcnEZD%2FsJGBKbbETj%2FqyQSjR%2BhNAe2RvoFwVqLtpg%2FtnMmu9qebWEtvZdOyeMaoX%2Fish8IfxdYptq3g8yHnjJwrKPiJGhB02ML%2By25h6B%2BDoOIxH0ikMPvH9swGOpcBdw4ET716dey7Wd12a2HJCWo57S1MsEDOkBYO7cgP3MlsR1cbcw0%2Bck31liLfi8tuA%2FbrvNCsK1zHpq16nIqvd5dHNITo6xwmtKhNXXfgCS3SRuW8fFO%2BIgRtZA7WYMIUZJCQX0OrrxTUWsuZpeTEkctM6gi6d2FJixpKuy1g1bRZD4SqBmvnrAB5x3i1joUXDAue%2FUBe3Q%3D%3D&Expires=1771945010)

7. [Optimal-Power-Flow-in-DC-Networks-with-Robust.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/159164453/da744248-1d0f-4f46-a215-d94a6d14f045/Optimal-Power-Flow-in-DC-Networks-with-Robust.pdf?AWSAccessKeyId=ASIA2F3EMEYEZHQSUQJT&Signature=0EbMZ%2FNQhCdzDHMGqPrJ32EKH%2FU%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEC4aCXVzLWVhc3QtMSJIMEYCIQDC76Z88WwMoJ7DhI6Uo4%2BRZ83WSJWUhj1oIish2B2mswIhAOYPkyudIAdrLcYJrGXmZIoN1BeDXqXtDx%2FT2%2FsGwrIiKvwECPb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgzGrpoSmCs%2F5hQAbuMq0AQtr%2FdpoUiZas%2BL7avxF42tQGsHRngTOBUvdCs7vc5nj5hqju6qqgSsE87MPcjNSFKmrwaj3Vo36A4jLR3UT7N7dhc8Ip9qZswGJiZzWylFSQXLBOSqknf37E2k1BAHqBbcvPOVeiG8oE9fWiJsYVlpT5edxjQjAMfgR1oiJuNBzCpVwV87L9%2FAM7TJDPC0re8MwfOLFZqhyCmdzdd%2FbnDruWI0uPPAx%2F7SJctj8E27Y7rpaOECHzo6P%2FksnvvxoyShfft3dcWT7JbbYL%2BE6trD1ZTfQLNRPDQmOg5c9KQENn5VEVu6zb6XxZB303rzSpTNDctC3d5XYimM99%2BoCIhY9x7iWa9YS5epB0nFj8KhGiau0B0126IXsBH0wK%2B80YBxhzgbFMIIV0mXbOhHAHUBKmdcvWuenZkQC56jpbUjOL%2FdcL1SSfe%2FiKBWCuDuB4mCHEBnrFhoG%2FtdKMGmFvH6wQiTLD2fQA6zJtlSJUTeMYxqom5IOSN8vyHOAfVAn1mSx1bqiMeBD53T7sZNh0LnNqBWUqOejGGhW614JesK8uODFPOSZQbRHXs0he%2Fg4zfwInJw3SeOOHzjPNBHQFzvay6VjgHMm%2BEKPNW5q0BEWb4oBu%2BGRz9Q7O8kwGqEO%2FfAuNGyatZj%2Boxi2H%2BqMzFwVoO%2F1%2Fut4ZBANTcnEZD%2FsJGBKbbETj%2FqyQSjR%2BhNAe2RvoFwVqLtpg%2FtnMmu9qebWEtvZdOyeMaoX%2Fish8IfxdYptq3g8yHnjJwrKPiJGhB02ML%2By25h6B%2BDoOIxH0ikMPvH9swGOpcBdw4ET716dey7Wd12a2HJCWo57S1MsEDOkBYO7cgP3MlsR1cbcw0%2Bck31liLfi8tuA%2FbrvNCsK1zHpq16nIqvd5dHNITo6xwmtKhNXXfgCS3SRuW8fFO%2BIgRtZA7WYMIUZJCQX0OrrxTUWsuZpeTEkctM6gi6d2FJixpKuy1g1bRZD4SqBmvnrAB5x3i1joUXDAue%2FUBe3Q%3D%3D&Expires=1771945010)

8. [ANL-CEEESA/UnitCommitment.jl: Optimization package for ... - GitHub](https://github.com/ANL-CEEESA/UnitCommitment.jl) - Data Format: The package proposes an extensible and fully-documented JSON-based data format for SCUC...

9. [Problem definition · UnitCommitment.jl - GitHub Pages](https://anl-ceeesa.github.io/UnitCommitment.jl/0.4/guides/problem/) - Some unit commitment models allow price-sensitive loads to have a piecewise-linear convex revenue cu...

10. [instances](https://axavier.org/UnitCommitment.jl/0.4/instances/) - /UnitCommitment.jl/0.4/instances/. 5 directories 1 file. Name · Size · Modified · Go up, —, —. matpo...

