# Session Log

Keep this log short. Each entry should capture durable context, not a transcript.

## 2026-05-30 - Design Principles Agent

- Added `.codex/docs/design-principles-agent.md` and linked it from
  `AGENTS.md` for broad DRY/KISS/YAGNI/SOLID maintainability work.
- Added `tools/design_principles_audit.py`, a heuristic whole-code Python audit
  for long/branchy functions, duplicate function bodies, unreferenced private
  definitions, broad classes/modules, and excessive parameter counts.
- Added tests for the audit tool and documented the local command in
  `docs/testing_and_ci.md`.
- Current audit output is intentionally candidate-based and shows larger
  refactor targets, especially experiment entry points and workflow functions;
  it is not wired into CI as a hard gate.

## 2026-05-30 - Radius Guardrails From Static Review

- Fixed AC metric validation to reject odd `[P; Q]` h-vector widths,
  nonfinite h-vectors, nonfinite dense metric matrices, and nonsymmetric dense
  metric matrices before Cholesky/solve.
- Centralized signed raw radius handling so nonfinite sensitivities produce
  `NaN` diagnostics consistently with `degenerate_sensitivity` status; applied
  to DC L2, operator-path DC L2, DC sigma, AC sigma, and AC metric radii.
- AC sigma now exports zero worst-case perturbation vectors for non-`ok_finite`
  rows, including base-infeasible constraints, and documents that boundary
  vectors are meaningful only for `ok_finite` rows.
- Tightened DC covariance validation and corrected the config comment to list
  the implemented `uc_jl` sigma source instead of stale `"file"` wording.
- Verified with focused radius tests, static quality tests, `ruff format
  --check .`, `git diff --check`, and full `python -m pytest -q` (350 passed).

## 2026-05-30 - Public Release Metadata And AC Unconstrained Edge Case

- Verified the pasted audit against current code: AC nondifferentiable status
  propagation, Q-limit diagnostics, thresholds documentation, logging setup,
  entry-point inventory, and generated-artifact ignore rules already had current
  coverage or docs.
- Added MIT `LICENSE`, `CITATION.cff`, pyproject license/readme metadata, and a
  small release-metadata test.
- Fixed AC L2 status aggregation so a zero-flow unconstrained line can record
  the diagnostic apparent-power subgradient without becoming a zero-radius
  limiting certificate; added regression coverage.
- Added a concise docs-index pipeline overview and README reproducibility,
  license, and citation pointers.

## 2026-05-29 - Quality Gates And Fail-Fast Cleanup

- Removed unused `src/stability_radius/opf` duplicate OPF/PF package.
- Changed AC PF and `base_dispatch=acpf/ac_fpf` paths to fail fast instead of
  switching solver/model policy after failure; DC OPF now uses the configured
  headroom exactly once.
- Removed compatibility aliases in AC Monte Carlo per-line overload output and
  cleaned old compatibility terminology from code/docs.
- Added application-wide docstring and prohibited-design-term static tests.
- Added coverage configuration with a 70% gate over domain code after omitting
  CLI/workflow/report orchestration from the denominator.
- Verified with `python -m pytest -q` (334 passed),
  `python -m pytest --cov=stability_radius --cov-report=term-missing -q`
  (71.33%, gate 70%), and `python -m ruff format --check .`.

## 2026-05-29 - Codex Project Memory Bootstrap

- Reviewed repository structure, public docs, `UNITS_CONTRACT.md`, configs,
  main workflow code, CI, data inventory, and existing result directories.
- Created `.codex/docs` as compact working memory for future Codex sessions.
- Established rule: after each meaningful conversation, update this log and
  only update `project-memory.md` for durable facts or decisions.
- Noted current docs/code drift: `ac.lossless=false` support appears present in
  current compute code/tests even though some older docs still describe it as
  unsupported or incomplete.
- Existing generated artifacts are git-ignored; important historical outputs are
  in `analysis_output/` and `runs/n1_stability_demo/`.

## 2026-05-29 - Paper Draft Bootstrap

- Drafted `paper/manuscript.tex` as a journal-agnostic Q1-oriented manuscript
  around balanced stability-radius certificates, sparse AC adjoint computation,
  metric/sigma unification, benchmark protocol, preliminary results, and
  limitations.
- Added `paper/references.bib` and `paper/reviewer_simulation.md`.
- Verified citation keys resolve and all new paper files are ASCII-only.
- PDF compilation was not run because no local LaTeX engine (`pdflatex` or
  `tectonic`) is installed.

## 2026-05-29 - MDPI Paper Project

- Replaced `paper/template.tex` with an MDPI-class manuscript using
  `Definitions/mdpi`, MDPI front-matter commands, external BibTeX references,
  figures, and table inputs.
- Added paper figures under `paper/figures/` and LaTeX table fragments under
  `paper/tables/`; updated `paper/README.md` and `paper/compile.ps1` to build
  `template.tex`.
- Verified citation keys and included figure/table paths; local PDF build is
  still untested because `latexmk`, `pdflatex`, and `tectonic` are unavailable.
- Parsed `C:\Users\egor1\Desktop\ICS_seminar.pptx` (28 slides, 39 media
  images) and realigned the manuscript results narrative toward the seminar
  flow: AC/sigma radius, dangerous-line metrics, hidden-danger analysis, and
  N-1 screening. The MDPI template now references 19 figures and 4 tables.

## 2026-05-29 - AC Probability And Certificate Status

- Changed AC sigma overload probability to the one-sided apparent-power tail
  while keeping a two-sided signed-flow helper for DC-style semantics.
- Added constraint-level status fields plus nonnegative `certificate_radius_*`
  and signed diagnostic `signed_distance_*` fields for DC L2/sigma and AC
  L2/sigma/metric outputs; signed `radius_*` fields remain schema-v3
  diagnostics.
- Updated table defaults, unit/data-format docs, and focused tests.
- Verified with `python -m pytest -q` (316 passed) and
  `python -m ruff format --check .`; `poetry run ...` was blocked by a broken
  local `.venv` (`.venv\lib64` access error).

## 2026-05-29 - Shared Balanced Geometry

- Added `src/stability_radius/geometry/balanced.py` for shared balanced
  projection, sigma-weighted dual projection, reduced-block L2 norms, and L2
  worst-case directions.
- Routed AC L2, AC sigma, AC metric, and nonlinear worst-case replay through
  the shared geometry helper to reduce radius/replay drift.
- Added `tests/test_geometry_balanced.py` for unweighted projection, weighted
  projection, row-wise projection, worst-case direction, and reduced AC block
  norm with implicit slack zero.
- Verified with `python -m pytest -q` (321 passed),
  `python -m ruff format --check .`, and `git diff --check`.

## 2026-05-29 - Compute Nonlinear AC Replay Validation

- Added optional `compute.ac.validation.nonlinear.*` config and CLI flags for
  top-k nonlinear replay of AC L2 worst-case directions.
- `compute` now writes `validation_report.json` and `validation_report.md` when
  nonlinear validation is enabled, while merging compact per-line fields such
  as `radius_ac_l2_validated`, `nonlinear_conservatism_ratio`,
  `pf_replay_status`, and `max_replay_rel_error` into `results.json`.
- Replay directions use the shared balanced geometry, restrict Q balancing to
  PQ buses when PV buses exist, and reapply the dispatch used by the AC base
  point when available.
- Verified with focused tests, `python -m pytest -q` (324 passed),
  `python -m ruff format --check .`, `git diff --check`, and a smoke compute
  on `data/input/pglib_opf_case5_pjm.m` with `--ac-validate-nonlinear 1`.

## 2026-05-29 - Model-Critical Geometry And Validation Fixes

- Fixed AC sigma/metric Q-block geometry so PV/slack Q coordinates are
  excluded and only PQ-bus Q coordinates are balanced/projected.
- Replaced AC metric's unweighted balanced projection with the constrained
  `M^{-1}` dual projection; added nonuniform sigma equivalence coverage.
- Preserved negative DC/N-1 signed distances while exposing nonnegative
  certificate radii and explicit infeasibility statuses.
- Split PF non-convergence from per-line overload probabilities in AC Monte
  Carlo; per-line probabilities are conditional on PF-converged samples.
- Made `ac.lossless=false` fail-fast in certificate mode and added Q-limit
  active-set diagnostics (`q_limit_hit`, `invalid_active_set_changed_q_limit`).
- Marked binding `|S0|≈0` AC apparent-power constraints as
  `nondifferentiable_apparent_power`; their signed radius remains diagnostic
  but the strict certificate radius is zero.
- Verified with targeted pytest subset covering sigma/metric, N-1, MC, workflow,
  and Q-limit helper changes.

## 2026-05-29 - Entry Point And Experiment Boundary Cleanup

- Replaced stale `experiments/README.md` with a config-only experiment layout
  and current runnable commands through `entry_points/` and package
  post-processing modules.
- Promoted shared h-vector, pandapower OPF, Gaussian sampling, JSON, and line
  limit helpers to public library functions; entry points no longer import
  private `stability_radius.*` symbols.
- Centralized diagonal Gaussian balanced sampling, sorted line-limit extraction,
  JSON result loading, and single-input plot CLI parsing.
- Added static tests for repeated non-trivial function bodies, private library
  imports from entry points, and the config-only `experiments/` boundary.
- Forced Matplotlib `Agg` in post-processing plot modules and covered plot main
  paths to avoid GUI/Tk-dependent failures.
- Verified with `python -m ruff format --check .`, `git diff --check`, and
  `python -m pytest --cov=stability_radius --cov-report=term-missing -q`
  (339 passed, 71.77% coverage).

## 2026-05-30 - DRY/KISS Refactor Batch

- Added shared test factories for repeated pandapower 2-bus, 3-bus dispatch,
  and triangle networks; routed duplicated AC FPF, ACPF, OPF consistency, and
  ext-grid absorption tests through them.
- Added config assertion helpers for project-default tests and parameterized the
  repeated logging output-dir checks.
- Split DC L2 and Gaussian sigma radius routines into validation, per-line row
  assembly, and summary helpers while preserving public result fields.
- Verified with `python tools/design_principles_audit.py --root .`,
  `python -m ruff format --check .`, `git diff --check`, and
  `python -m pytest -q` (352 passed).
