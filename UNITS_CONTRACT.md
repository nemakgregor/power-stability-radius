# UNITS_CONTRACT

This file is the primary specification for the repository. It defines:

- what is computed;
- which DC and AC assumptions are in force;
- which units and sign conventions are used;
- how inputs and outputs are structured;
- which fail-fast rules are part of the contract;
- which forward-looking changes are proposed without breaking compatibility.

Tests under `tests/` should be treated as an executable part of this contract.

---

## 0. Scope and Meaning of "Stability Radius"

### Goal

For a given base operating point and a set of line limits, estimate how robust that operating point is to nodal injection perturbations.

The motivating question is:

- What is the largest perturbation size, measured in an L2 sense over buses, that is guaranteed not to overload any transmission line?

### Certificate-Style Definition

For line `ell`, write the limit in the form:

- DC: `|f_ell(Delta p)| <= c_ell`
- AC: `max(|S_from(Delta P, Delta Q)|, |S_to(Delta P, Delta Q)|) <= c_ell`

Let `Delta u` denote the perturbation vector:

- DC: `Delta u = Delta p`
- AC: `Delta u = [Delta P; Delta Q]`

The stability or robustness radius is a scalar `r >= 0` such that:

> for all `Delta u` satisfying `||Delta u||_2 <= r`, no line limit is violated within the chosen model or linearization.

The repository computes certificates, meaning lower bounds that guarantee safety inside the ball. They do not claim to be the exact nonlinear AC maximum radius.

---

## 1. Glossary

- Base point: the operating point around which a linear model and certificate are built.
- Slack bus: the reference bus used to remove one angular degree of freedom in DC and AC reductions.
- Balanced disturbances: perturbations whose net active power is zero, and in AC mode whose net reactive power is also zero.
  - DC: `1^T Delta p = 0`
  - AC: `1^T Delta P = 0` and `1^T Delta Q = 0`
- `H_full`: dense DC sensitivity matrix such that `Delta f = H_full Delta p`.
- `DCOperator`: operator-form DC model that can work without materializing `H_full`.
- `LODF`: line outage distribution factor approximation for line outages.
- `ACOperator`: sparse Jacobian plus LU factorization used for AC adjoint solves.
- Certificate soundness: if the perturbation stays inside the certified radius, the modeled constraints remain satisfied.
- Usefulness: a certificate may be logically correct but practically trivial, for example when `r* = 0`.
- AC FPF: feasible AC operating point obtained from `pandapower.runopp`, used when a standard AC PF base point is not sufficient.
- Metric radius: radius under a positive-definite metric matrix `M`, with norm `||Delta u||_M = sqrt(Delta u^T M Delta u)`.
- Sigma radius: dimensionless margin normalized by a flow standard deviation, `r_sigma = margin / sigma_flow`.
- Worst-case verification: nonlinear AC PF replay of the analytical worst-case perturbation direction.

---

## 2. Determinism and No-Hidden-Effects Rules (CURRENT)

### 2.1 No Implicit Input Downloads

If an input `.m` file is missing:

- `allow_download=false` raises an explicit error;
- `allow_download=true` triggers a deterministic download attempt derived from the case filename.

This behavior is part of reproducibility and CI.

### 2.2 Stable Ordering of Indices

The repository uses deterministic orderings throughout:

- `bus_ids = sorted(net.bus.index)`
- `line_ids = sorted(net.line.index)`

These orderings define the coordinate system for arrays written to `results.json` and for internal operators.

### 2.3 Fail-Fast Instead of Silent Heuristics

Incompatible configurations terminate with explicit errors. Examples:

- AC Monte Carlo requires `pandapower` as the per-sample PF engine.
- `ac.lossless=false` is not implemented for the supported AC certificate and AC Monte Carlo workflow.

---

## 3. Units and Sign Conventions

### 3.1 Data Sources

Inputs are MATPOWER or PGLib `.m` files converted into a `pandapower` network.

Important convention:

- MATPOWER and PGLib `rateA` values are normally in MVA.

### 3.2 DC Model

Units:

- nodal active injections: MW
- line flows: MW
- line limits: stored from MVA ratings but interpreted as MW under the repository's lossless DC `PF=1` convention

Flow sign convention:

- `flow0_mw` is a signed flow in the line orientation implied by the network representation.

### 3.3 AC Certificate Around AC PF

Units:

- `Delta P`: MW
- `Delta Q`: MVAr
- base-end flows: MW and MVAr
- AC line limits: MVA
- apparent-power checks use `|S| = sqrt(P^2 + Q^2)` in MVA

---

## 4. Base Points and Dispatch

The project distinguishes:

- the source of active-power dispatch;
- the AC base point used for the AC certificate.

### 4.1 DC Base Point (CURRENT)

#### A) `base_dispatch=case`

- Injections are read from the `pandapower` network.
- The net injection vector is balanced by adjusting the slack bus so the total becomes exactly zero.
- Base flows are reconstructed through `DCOperator.flows_from_bus_injections_mw(...)`.

#### B) `base_dispatch=dc_opf`

- A DC OPF is solved with PyPSA plus HiGHS.
- The result provides:
  - base line flows;
  - balanced bus injections;
  - generator dispatch for reproducibility.
- A consistency check verifies that OPF-reported flows match the reconstruction through `DCOperator`.

Contract:

- DC OPF is both a dispatch source and a DC base-flow source.
- When AC is enabled, the OPF active dispatch becomes the active-power input for the downstream AC PF.

### 4.2 AC Base Point (CURRENT)

- The AC certificate is always built around an AC PF solution, not around a DC OPF solution directly.
- Solvers may be `pandapower` or `pypsa` depending on configuration.
- When `base_dispatch=dc_opf`, the active dispatch is applied first and then AC PF is solved.

Contract:

- The AC certificate is computed around the resulting AC PF state and the state is stored in metadata for replay and verification.

### 4.3 AC FPF Base Point (CURRENT)

AC FPF provides an alternative AC base point using an AC-feasible optimization.

#### When It Is Used

- when a standard AC PF is not adequate for the target workflow;
- when a feasible AC operating point that respects branch constraints is explicitly required.

#### Mathematical Intent

The optimization seeks a feasible dispatch that stays as close as possible to a reference dispatch `P^0`.

#### `ACFPFConfig`

Key parameters include:

- `pg0_source`
- `vm_min_pu`, `vm_max_pu`
- `max_iteration`
- `max_loading_percent`
- `max_attempts`
- `per_attempt_timeout`

#### Post-OPP PF Validation

After `runopp`, the code runs a standard PF using the optimized dispatch so that the AC certificate linearizes around a Newton-style PF point, matching downstream verification.

#### Implementation

- `stability_radius.base_point.pandapower_opp`
- `solve_ac_fpf_base_point(...)`

Contract:

- AC FPF returns the same base-point result type used by the standard AC PF path so downstream code stays uniform.

---

## 5. DC Model: Mathematics and Implementation

### 5.1 DC Assumptions (CURRENT)

- lossless model;
- small-angle approximation;
- fixed voltage magnitudes;
- branch behavior dominated by reactance.

### 5.2 `DCOperator` (CURRENT)

The DC operator builds a sparse reduced-angle system:

- an oriented incidence matrix is assembled;
- branch coefficients are converted into MW/rad terms;
- a reduced matrix `B_red` is formed with one slack degree of freedom removed;
- flow reconstruction uses the resulting factorization.

#### Elements Included in the `B` Matrix

To avoid singular or incomplete topology handling, the operator includes:

- lines;
- transformers with DC-compatible tap and phase-shift treatment;
- impedance elements.

#### Phase-Shifting Transformers

If a transformer has a nonzero phase shift, the model stores a constant shift injection term used when reconstructing absolute base flows.

### 5.3 Operator vs Materialize Modes (CURRENT)

#### `dc.mode=operator`

- `H_full` is not materialized;
- projected row norms are computed via operator solves;
- memory use is lower;
- N-1 post-processing is unavailable.

#### `dc.mode=materialize`

- a dense `H_full` is built with configurable dtype;
- memory use is higher;
- enables N-1 effective radii and dense post-processing workflows.

---

## 6. DC Radii and Probabilistic Metrics

### 6.1 DC L2 Radius for Balanced Disturbances

The DC certificate assumes balanced perturbations `sum(Delta p)=0` and measures perturbation size in the full-bus Euclidean norm.

Because a sensitivity row `g` is defined only up to an additive multiple of the all-ones vector, the implementation uses the projected norm:

- `||Proj(g)||_2`

For line `ell`, the certificate takes the form:

- `r_ell = margin_ell / ||Proj(g_ell)||_2`

The global DC certificate is the minimum over constrained lines.

### 6.2 Sigma Radius and Overload Probability

If `Delta p` is Gaussian in the balanced subspace, the line-flow standard deviation is derived from the projected sensitivity row.

The repository then reports:

- sigma radius, a margin expressed in standard deviations;
- overload probability under the corresponding Gaussian line-flow model.

---

## 7. Effective N-1 Radii (DC)

### 7.1 Idea

The goal is to estimate robustness under a single line outage.

The implementation uses an LODF-style approximation:

- updated base flows after outage;
- updated sensitivity rows under outage.

### 7.2 Islanding

If the outage makes the approximation undefined or produces islanding:

- `islanding=skip` skips that contingency;
- `islanding=raise` fails explicitly.

---

## 8. AC Certificate: AC L2 Around AC PF

### 8.1 Assumptions (CURRENT)

- AC certificates are local linearizations around a solved AC PF base point;
- the PF Jacobian is used through adjoint solves;
- line-end constraints are handled per end and then aggregated.

### 8.2 `ACOperator`: Jacobian and Adjoint Solves

The AC operator:

- builds the reduced PF Jacobian;
- factors it sparsely;
- solves adjoint systems to obtain line-end sensitivity vectors.

### 8.3 Line Constraints and Two-Ended Handling

Each line has two ends, and the certificate treats them separately:

- a certificate is computed for each end;
- the binding end is recorded;
- the effective line certificate is the tighter one.

### 8.4 AC Balancing

When AC balancing is enabled, perturbations are projected so that both active and reactive totals remain balanced in the intended subspace.
For the reduced AC model, independent reactive perturbations are restricted
to PQ-bus coordinates. PV and slack-bus Q coordinates are zeroed/excluded in
AC sigma and metric radii unless a future active-set model explicitly adds
them.

### 8.5 AC Metric Radius (CURRENT)

The AC metric radius generalizes the AC L2 radius to a positive-definite metric.

Contract:

- if `M = I`, the metric radius matches the AC L2 radius;
- diagonal inverse-variance choices recover sigma-like weighted behavior;
- under balance, the metric dual norm uses the constrained projection
  `M^{-1} - M^{-1}C^T(CM^{-1}C^T)^+CM^{-1}`, not unweighted
  mean-subtraction.

#### Implementation

- `stability_radius.radii.ac_metric_radius`

### 8.6 AC Sigma Radius (CURRENT)

The AC sigma-radius normalizes line-end margin by the line-end flow standard deviation induced by bus-level `sigma_p` and `sigma_q`.
The Q block uses only PQ-bus coordinates from the reduced AC model; PV and
slack Q entries are not independent uncertainty coordinates for the
certificate.

#### Overload Probability

Under the Gaussian approximation, the same ingredients provide an approximate
overload probability. For AC apparent power the reported probability is
one-sided: `P(|S0| + X > c) = Q((c - |S0|) / sigma_flow)`. The two-sided
signed-flow probability remains the DC convention.

#### Sigma Sources

Sigma can come from:

- uniform scalar values;
- UC.jl-derived profiles;
- synthetic load-proportional rules in experiment pipelines.

### 8.7 Nonlinear Replay Validation (CURRENT)

`radius_ac_l2_linear` is the exact dual-norm radius for the frozen first-order
AC model. When `compute.ac.validation.nonlinear.enabled=true`, the compute
pipeline additionally replays the top-k worst-case AC L2 directions with
nonlinear pandapower PF and reports optional validation fields.

Contract:

- `radius_ac_l2_validated` is nonnegative and never exceeds the linear
  certificate radius for the replayed line.
- `nonlinear_conservatism_ratio > 1` means nonlinear replay did not violate
  before the linear boundary; `< 1` means the linear approximation was
  optimistic for that replayed direction.
- `linearization_status` records the validation state without changing the
  meaning of the linear certificate.
- `q_limit_hit` and `pv_pq_switch_detected` diagnose generator reactive-limit
  events that invalidate strict fixed-PV/PQ linearization claims.
- If the binding apparent-power base point has `|S0| <= eps`, the norm
  gradient is nondifferentiable. `radius_ac_l2` remains a signed diagnostic
  distance, while `certificate_radius_ac_l2` is set to zero and
  `constraint_status_ac_l2 = nondifferentiable_apparent_power`.
- Detailed replay trajectories are stored outside `results.json` in
  `validation_report.json`.

#### Implementation

- `stability_radius.radii.ac_sigma_radius`

### 8.7 Worst-Case Verification (CURRENT)

Worst-case verification applies the analytical worst-case perturbation direction in a nonlinear AC PF replay.

#### Expected Result

- near the certificate boundary, the selected line should approach its limit;
- below the boundary, violations should not appear systematically;
- the predicted-vs-actual mismatch quantifies linearization error.

#### Implementation

- `stability_radius.verification.verify_worst_case`

### 8.8 AC Sigma Monte Carlo (CURRENT)

This workflow samples AC perturbations, solves a PF for each sample, and compares empirical overload behavior with sigma-radius expectations.
For heterogeneous sigma, balance uses the sigma-squared weighted conditional
Gaussian projection. PF failures are reported as bad samples and as a PF
failure probability, but they are not counted as overloads on every line.

#### Implementation

- `stability_radius.verification.ac_monte_carlo_sigma`

---

## 9. Monte Carlo Verification (DC and AC)

### 9.1 DC Monte Carlo (CURRENT)

DC Monte Carlo:

- samples balanced perturbations;
- propagates them through the DC sensitivity model;
- counts overloads and soundness outcomes relative to the certificate.

### 9.2 AC Monte Carlo (CURRENT)

AC Monte Carlo:

- samples balanced active and reactive perturbations;
- applies them to the network;
- runs AC PF per sample;
- records overloads and PF failures.

Contract:

- PF non-convergence is tracked explicitly rather than hidden.

---

## 10. `results.json` Data Schema (CURRENT, `schema_version=3`)

### 10.1 Top Level

The result file is a JSON object containing:

- `__meta__`
- per-line entries keyed as `line_<idx>`

### 10.2 `__meta__` Key Fields

Metadata includes the information needed to replay or interpret the run, such as:

- input path;
- slack bus;
- selected DC and AC settings;
- base-dispatch mode;
- base-point details;
- solver and timing details;
- schema version.

### 10.3 Per-Line Fields (DC)

Typical DC fields include:

- `flow0_mw`
- `p_limit_mw_est`
- `margin_mw`
- `norm_g`
- `radius_l2`
- `constraint_status_l2`
- `certificate_radius_l2`
- `signed_distance_l2`
- `radius_sigma`
- `overload_probability`
- `radius_nminus1`
- `constraint_status_nminus1`
- `certificate_radius_nminus1`
- `signed_distance_nminus1`
- `worst_contingency_line_idx`

### 10.4 Per-Line Fields (AC)

Minimal AC fields:

- `ac_s_limit_mva`
- `ac_s0_from_mva`
- `ac_s0_to_mva`
- `margin_ac_mva`
- `||h||2`
- `binding_end`
- `radius_ac_l2`
- `radius_ac_l2_linear`
- `constraint_status_ac_l2`
- `certificate_radius_ac_l2`
- `signed_distance_ac_l2`
- `nondifferentiable_apparent_power`
- `radius_ac_l2_validated`
- `validation_scale_safe`
- `validation_scale_violation`
- `nonlinear_conservatism_ratio`
- `pf_replay_status`
- `max_replay_rel_error`
- `nonlinear_validation_n_pf_calls`
- `linearization_status`

If AC sigma is computed, additional fields may include:

- `sigma_flow_mva`
- `radius_ac_sigma`
- `constraint_status_ac_sigma`
- `certificate_radius_ac_sigma`
- `signed_distance_ac_sigma`
- `overload_probability_ac`
- `worst_case_dp_mw`
- `worst_case_dq_mvar`
- `worst_case_s_predicted_mva`

---

## 11. Failure Modes (CURRENT)

### 11.1 Zero Radius `r* = 0`

This can mean either:

- a genuinely binding base point at the limit;
- incorrect or degenerate limits;
- a base point inconsistent with the modeled limits.

Verification status handling distinguishes these situations.

### 11.1.1 Constraint-Level Certificate Status

Per-line result rows may include a `constraint_status_*` field alongside the
signed diagnostic `radius_*` field. The nonnegative certificate radius is stored
in `certificate_radius_*`; `signed_distance_*` preserves the signed
`margin / dual_norm` diagnostic when the base point is already infeasible.

Known status values:

- `ok_finite`
- `ok_infinite`
- `base_infeasible`
- `degenerate_sensitivity`
- `unconstrained_limit`
- `pf_failed`
- `jacobian_singular`
- `nonlinear_unvalidated`
- `nonlinear_optimistic`

In schema version 3, `radius_*` fields are signed diagnostic distances and can
still be negative for base-infeasible constraints. Consumers should prefer
`constraint_status_*` and `certificate_radius_*` for certificate claims.

### 11.2 AC PF Non-Convergence in Monte Carlo

- PF failure is tracked explicitly in statistics;
- a PF failure on the base point itself is treated as a configuration or operating-point problem.

### 11.3 OPF and `DCOperator` Mismatch

When `base_dispatch=dc_opf`, the repository checks:

- bus-injection balance;
- consistency between OPF-reported flows and operator-reconstructed flows.

A mismatch is treated as an error because the certificate and the verification would otherwise refer to different physical states.

---

# PROPOSED CONTRACTS

The following are forward-looking improvements that can be introduced incrementally without breaking the current schema immediately.

## P1. Explicit Units Block in `__meta__`

Add:

```json
"units": {
  "dc_flow": "MW",
  "dc_limit": "MW_assumed_from_MVA_pf1",
  "ac_flow_p": "MW",
  "ac_flow_q": "MVAr",
  "ac_limit": "MVA"
}
```

Benefit:

- less ambiguity around fields such as `p_limit_mw_est`.

## P2. Rename `p_limit_mw_est` to `dc_limit_mw_pf1`

Keep the old field as a compatibility alias for one or two schema revisions.

Benefit:

- the field name matches the actual meaning of the DC limit convention.

## P3. Explicit Distribution Defaults for Verification

Instead of relying on scattered defaults, add a dedicated metadata block such as:

- `verification_defaults: { dc_sigma_mw, ac_sigma_p_mw, ac_sigma_q_mvar }`

Benefit:

- fewer verification parameters must be supplied manually;
- report generation and Monte Carlo can derive their intended sampling settings directly from run metadata.

## P4. Stable Official Table Field Sets

Define one authoritative list of table columns and document it centrally.

Benefit:

- less drift between CLI table formatting, documentation, and downstream expectations.

## P5. Long-Term Support for `ac.lossless=false`

This requires a deeper model extension:

- PF and certificate paths must use a consistent lossy model;
- AC Monte Carlo must verify the same model assumptions;
- additional loss-driven flow redistribution effects must be handled explicitly.

Recommendation:

- implement this as a dedicated branch of functionality, not as a small refactor mixed into unrelated work.

---

## 12. How to Read the Tests as a Specification

Important executable contracts include:

- `test_certificate_concept.py`
  - boundary tightness on the worst-line direction;
  - slack invariance for balanced disturbances.
- `test_config_extends.py`
  - correct `extends:` composition with relative paths.
- `test_verification_report_and_monte_carlo.py`
  - report rendering should avoid meaningless `nan%` text.
- `test_opf_dc_consistency.py`
  - OPF and `DCOperator` must agree on flows within tolerance.
- `test_ac_metric_radius.py`
  - identity metric equivalence to L2, SPD validation, diagonal and dense handling.
- `test_ac_sigma_radius.py`
  - sigma-flow formulas, balancing, worst-case construction, overload probability.
- `test_verify_worst_case.py`
  - no violation below the expected scale, expected approach to the limit near the boundary, NaN handling on PF divergence.
- `test_ac_mc_sigma.py`
  - soundness inside the sigma ball, per-line overload fractions, input validation.
- `test_pp_helpers.py`
  - helper semantics for in-service checks, voltage lookup, and slack resolution.
- `test_verification_status.py`
  - repository status summarization semantics.
- `test_statistics_table.py`
  - table formatting, column inference, and radius summaries.
- `test_metrics_analysis.py`
  - unified dataframe construction, rank-correlation rules, precision-at-k handling.
- `test_workflows_helpers.py`
  - helper merging, sigma-array construction, and deterministic ordering logic.

---

## 13. Current-Version Summary

Current repository truth:

- DC certificates and DC probabilistic metrics are built on balanced L2 geometry and the `DCOperator`.
- The AC certificate is built around an AC PF or AC FPF base point and uses Jacobian adjoint solves.
- AC metric radius generalizes AC L2 to an SPD metric; with inverse-variance choices it aligns with sigma-style weighting.
- AC sigma-radius and worst-case verification are integral documented parts of the current workflow.
- Monte Carlo verification checks the operating point and fields actually recorded in `results.json`.
- The repository is intentionally deterministic, stable in ordering, and fail-fast on unsupported combinations.

If you change:

- units;
- index ordering;
- the contents of `__meta__`;
- base-point semantics;

then the change should be accompanied by:

1. an update to this file;
2. updates to the tests that encode the contract.
