# Revision R1 experiments (IEEE Access resubmission)

Every table in `paper/revision_r1/` regenerates from the scripts in
`scripts/`; raw outputs are committed as JSON under `results/`.

## Root-cause fixes validated here

1. **Slack-index inconsistency** — the AC operator resolved a positional
   `slack_bus: 0` to bus position 0 while the h-vector expansion resolved it
   (ext-grid-aware) to the true slack position; every `h_P` entry below the
   slack position was shifted by one bus on any case whose ext_grid is not
   the first sorted bus (case118, case200_activ, case2000_goc, ...).
2. **Q-limit active set** — the base PF enforces reactive limits; buses whose
   every controlling generator is limit-pinned are effectively PQ but were
   linearized as PV.

Both fixed in `src/` (see commits) and regression-tested in
`tests/test_slack_extgrid_regression.py`, `tests/test_ybus_matches_pandapower.py`,
`tests/test_zero_flow_operator_norm.py`, and the tightened
`tests/test_h_vector_fd.py`.

## Scripts

| Script | Reviewer item | Output |
|---|---|---|
| `exp1_fd_convergence.py` | 2 | FD step sweep; adjoint residual |
| `exp2_sigma_calibration.py` | 5 | analytical vs empirical flow sd; tightened-limit probabilities |
| `exp3_multiscale_replay.py` | 4 | worst-direction replay at 0.25r..1.5r; crossing alphas |
| `exp4_ranking_stats.py` | 10 | scenario-bootstrap Spearman CIs, paired diffs, precision@k |
| `exp5_timing_breakdown.py` | 11 | staged repeated timings + peak RSS |
| `exp6_zero_flow_case2000.py` | 7 | operator-norm certification of zero-flow ends; threshold sweep |
| `exp7_dc_ac_paired.py` | (table regen) | paired DC/AC radii with corrected implementation |
| `exp8_participation_response.py` | 6 | participation-factor response, realizability, replay tightness |
| `make_tables_r1.py` | -- | regenerates `paper/revision_r1/tables_r1/*.tex` from the JSONs |

## Environment

Python 3.11, pandapower 2.14.10, SciPy 1.13.1, NumPy 1.26.4, pandas 2.3,
pypsa 1.3 (Linux container, 48-thread Xeon class).  PGLib cases are
downloaded via `stability_radius.utils.download`.  The UnitCommitment.jl
hourly sigma source (axavier.org) was unreachable from the experiment
environment; `exp2` uses a load-proportional sigma with the same
heterogeneous structure and documents this in its JSON.

## Known caveats

- `exp5`: the lossless `runpp` base point does not converge on
  `case2000_goc`; the zero-flow study (`exp6`) uses the AC feasibility power
  flow instead (reported, not hidden).
- `exp4`: samples whose Q-limit outer loop flip-flops indefinitely are
  skipped via a 10 s per-sample guard and counted in the artifact.
