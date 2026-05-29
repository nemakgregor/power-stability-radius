# Project Memory

Last reviewed: 2026-05-29.

## Purpose

`power-stability-radius` is a Python research toolkit for robustness certificates
of power-system operating points. It estimates how large balanced nodal
injection perturbations can be before any monitored branch thermal limit is hit.

The result is a certificate-style lower bound inside a chosen linear model:

- DC: balanced active-power perturbations, MW flows, PTDF-like sensitivities.
- AC: local linearization around a solved AC PF/FPF base point, apparent-power
  limits in MVA, adjoint Jacobian sensitivities for `[Delta P; Delta Q]`.

Use `UNITS_CONTRACT.md` as the primary specification for units, signs, schema,
and fail-fast behavior.

## Main Entry Points

- Main CLI: `entry_points/power_stability_radius.py`
  - `compute` / `demo`: write `results.json`, tables, optional `h_vectors.npz`.
  - `monte-carlo`: verify an existing `results.json`.
  - `report`: multi-case Markdown verification report.
  - `table`: format an existing result file.
- Library API: `stability_radius.workflows.compute_results_for_case(...)`.
- Experiment fronts:
  - `entry_points/run_pglib_sweep.py`: DC vs AC multi-case sweep.
  - `entry_points/run_sigma_radius.py`: AC sigma-radius workflow.
  - `entry_points/run_worst_case_verify.py`: nonlinear replay of worst-case directions.
  - `entry_points/run_scalability.py`: timing study.
  - `entry_points/metrics_analysis.py`: compare radius against heuristic metrics.
  - `entry_points/n1_stability_demo.py`: Cost OPF vs Radius OPF vs SCOPF demo.

Detailed entry-point inventory is in `docs/entry_points.md`.

## Architecture Map

- `src/stability_radius/application/`: argparse CLI orchestration and config-to-use-case translation.
- `src/stability_radius/workflows.py`: single-case compute pipeline; returns JSON-serializable results.
- `src/stability_radius/config.py`: frozen dataclass defaults plus YAML `extends` composition.
- `src/stability_radius/parsers/`: MATPOWER and UnitCommitment.jl input parsing.
- `src/stability_radius/base_point/`: DC OPF, AC PF, AC FPF, pandapower/PyPSA helpers.
- `src/stability_radius/dc/`: `DCOperator`, sparse B-matrix, PTDF-style sensitivities.
- `src/stability_radius/ac/`: `ACOperator`, Ybus/Jacobian construction, adjoint solves.
- `src/stability_radius/radii/`: L2, sigma, metric, N-1, AC feasibility computations.
- `src/stability_radius/verification/`: Monte Carlo, certificate checks, worst-case replay, reports.
- `src/stability_radius/postprocess/`: table, aggregation, and plotting helpers.
- `tests/`: executable contracts for math, units, config, docs-as-code, CI, and workflows.

## Core Contracts

- Stable ordering is part of the coordinate system: sorted bus indices and sorted
  line indices.
- Missing inputs fail unless `allow_download=true`.
- No hidden compatibility outputs: optional radii appear only when explicitly enabled.
- `base_dispatch` values: `case`, `dc_opf`, `acpf`, `ac_fpf`.
- AC certificate is built around AC PF/FPF state, not directly around DC OPF.
- DC OPF can provide active dispatch for the downstream AC PF.
- Balanced subspace projection is central; do not remove it when changing radius math.
- Shared balanced geometry lives in `src/stability_radius/geometry/balanced.py`;
  use it for AC L2, AC sigma, AC metric, and nonlinear worst-case replay so
  radius denominators and replay directions stay in the same geometry.
- Optional nonlinear AC replay is integrated into `compute` under
  `compute.ac.validation.nonlinear.*`; it writes `validation_report.json/md`
  and only compact validation diagnostics stay in `results.json`.
- For PV networks, nonlinear replay directions must restrict the Q balanced
  block to PQ buses, matching the reduced AC Jacobian geometry.
- AC sigma and AC metric radii also restrict the Q block to PQ buses; PV/slack
  Q coordinates in expanded h-vectors are diagnostic zeros, not independent
  uncertainty coordinates.
- AC metric radius under balance uses the constrained `M^{-1}` dual projection;
  with `M = diag(1/sigma^2)` it must match AC sigma for nonuniform sigma.
- `ac.lossless=false` is fail-fast for the supported AC certificate path until
  a full pi/shunt AC Jacobian and branch-flow model are implemented.
- AC PF/FPF metadata records `q_limit_hit`/`q_limit_events`; if Q-limit events
  are present, per-line AC outputs are marked
  `invalid_active_set_changed_q_limit` rather than treated as strict fixed
  PV/PQ certificates.
- Binding AC apparent-power constraints with `|S0|` near zero are marked
  `nondifferentiable_apparent_power`; legacy radii remain diagnostic, but
  nonnegative certificate radius is zero.
- Results schema is currently `__meta__.schema_version = 3`.
- H-vectors are not JSON-serializable; CLI extracts `_h_vectors` and saves `h_vectors.npz`.
- Verification should remain independent where possible: load saved results and re-derive from raw input.
- Keep unit names explicit. DC limits are MVA ratings interpreted as MW under the DC PF=1 convention; AC limits are MVA.
- AC apparent-power overload probability is one-sided:
  `P(|S0| + X > c) = Q((c - |S0|) / sigma_flow)`. The two-sided signed
  Gaussian probability remains a DC/signed-flow convention.
- Schema v3 keeps legacy signed `radius_*` fields for compatibility, but new
  consumers should use `constraint_status_*`, nonnegative
  `certificate_radius_*`, and diagnostic `signed_distance_*` fields.

## Current Config Defaults

- Dependency manager: Poetry. Python target: `^3.10`, CI uses Python 3.11.
- Default artifact root in code/config: `run_artifacts`.
- `conf/config.yaml` extends shared, compute, Monte Carlo, and report YAML files.
- `conf/config_shared.yaml` currently sets `run_tests: true`; pass
  `--run-tests 0` for fast manual compute runs.
- Default compute config has both DC and AC enabled, `base_dispatch: case`,
  AC `pf_solver: pandapower`, `pf_init: flat`, `lossless: true`.
- Advanced DC probabilistic and N-1 extensions are disabled by default.
- AC sigma/metric extensions and h-vector saving are disabled by default.
- AC nonlinear validation is disabled by default; enable with
  `--ac-validate-nonlinear 1` or `compute.ac.validation.nonlinear.enabled=true`.

Useful local checks:

```bash
poetry run python -m pytest -q
poetry run ruff format --check .
```

## Documentation Truth

- `UNITS_CONTRACT.md`: highest-level project contract.
- `docs/index.md`: public documentation hub.
- `docs/repository_structure.md`, `docs/entry_points.md`, `docs/execution_flow.md`,
  `docs/testing_and_ci.md`: operational references.
- `docs/scientific_concepts.md`, `docs/mathematical_foundations.md`,
  `docs/algorithms_and_models.md`: research/math references.

Known drift to watch:

- `ac.lossless=false` is intentionally unsupported in certificate mode. If
  lossy AC support is needed, implement the full pi/shunt Jacobian and update
  the fail-fast tests and docs together.
- `experiments/README.md` has old structure text and mojibake in box-drawing
  comments. Prefer `docs/entry_points.md` for the current runnable surface.

## Data And Artifacts

- Input cases live in `data/input/`, including PGLib cases from 5 buses up to
  10000 buses, plus API variants and UC.jl data in `data/uc_jl/case118.json`.
- New generated outputs should normally go under `run_artifacts/`.
- Existing historical results in this workspace are git-ignored and mostly under:
  - `analysis_output/*/results.json`
  - `runs/n1_stability_demo/*`
- Do not delete or rewrite generated artifacts unless the user asks.

Notable existing result summaries:

- `runs/n1_stability_demo/n1_demo_case118_limit_check_v2/`: Radius OPF raises
  min AC L2 radius from `0.8455` to `8.4564` MW versus Cost OPF/SCOPF, with
  about `5.282%` cost increase; AC N-1 screening pass rate improves from
  `2.29%` to `85.71%`.
- `runs/n1_stability_demo/n1_demo_case200_scale_probe_v2/`: all three regimes
  are identical in the summary; min AC L2 radius `38.5657` MW and N-1 pass rate
  `100%`.
- `analysis_output/n1_demo_case118_final/` is older two-regime output: Radius
  OPF improves min AC L2 radius from `-5.4127` to `47.2020` MW with `14.765%`
  cost increase. Treat as historical unless the user points to it.

## Change Style For This Repo

- Prefer small, targeted edits that preserve existing layers.
- Put reusable logic in `src/stability_radius`, not in entry-point scripts.
- Entry points should stay wrappers/orchestrators; docs-as-code tests expect
  their inventory to stay synchronized.
- For numerical behavior changes, update tests and `UNITS_CONTRACT.md`.
- For new runnable scripts, update `docs/entry_points.md`.
- For new result fields, update `UNITS_CONTRACT.md`, `docs/data_formats.md`,
  table formatting if relevant, and focused tests.
