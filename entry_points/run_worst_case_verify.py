from __future__ import annotations

"""Entry-point script for Experiment 3.

What it is:
- Runs `stability_radius.experiments.run_worst_case_verify.main`.
- Verifies analytic worst-case perturbations with nonlinear AC power flow.

Aim and scope:
- Start from sweep results, reconstruct the critical perturbation direction,
  and compare predicted versus actual limit violations across scale factors.
- Save outputs under `run_artifacts/run_worst_case_verify/...` by default.

CLI options:
- one of `--sweep-dir` or `--results`
- `--output-dir`, `--scales`, `--top-k`, `--recompute`, `--cases`

Artifacts:
- per-case `*_worst_case.json`
- aggregate `table3_summary.json`
- validation JSON and Figure 3 plots
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(run_entrypoint("stability_radius.experiments.run_worst_case_verify", "main"))

