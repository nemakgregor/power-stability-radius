from __future__ import annotations

"""Entry-point script for Experiment 2.

What it is:
- Runs `stability_radius.experiments.run_sigma_radius.main`.
- Computes AC sigma-radius using UC.jl or synthetic uncertainty data.

Aim and scope:
- Build sigma arrays, solve the AC base point, compute line-wise sigma-radius,
  run worst-case verification, and export plots and summary tables.
- Save all outputs under `run_artifacts/run_sigma_radius/...` by default.

CLI options:
- `--config`: experiment YAML describing the case, sigma source, plots, and MC settings

Artifacts:
- `results.json`, `summary.json`, `sigma_arrays.json`
- verification JSON, optional Monte Carlo JSON, `hvectors.npz`
- figure files and CSV exports
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(run_entrypoint("stability_radius.experiments.run_sigma_radius", "main"))

