# Session Log

Keep this log short. Each entry should capture durable context, not a transcript.

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
  L2/sigma/metric outputs; legacy signed `radius_*` fields remain for schema-v3
  compatibility.
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
  `nondifferentiable_apparent_power`; their legacy radius remains diagnostic
  but the strict certificate radius is zero.
- Verified with targeted pytest subset covering sigma/metric, N-1, MC, workflow,
  and Q-limit helper changes.
