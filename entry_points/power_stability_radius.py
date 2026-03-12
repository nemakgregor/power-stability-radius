from __future__ import annotations

"""Entry-point script for the main project CLI.

What it is:
- The public entry point for the compute, monte-carlo, report, and table flows.
- Runs `stability_radius.cli.main`.

Aim and scope:
- Keep the operational CLI visible at the repository root.
- Centralize all primary workflows behind one command surface.
- Default run outputs to `run_artifacts/<command>/...` unless config or flags override it.

CLI options:
- Global flags: `--config`, `--runs-dir`, `--run-dir-mode`, `--run-name`,
  `--log-level`, `--log-file-level`, `--run-tests`, `--allow-download`.
- Shared solver overrides: `--opf-*`, `--opf-dc-flow-consistency-tol-mw`,
  `--opf-bus-balance-tol-mw`.
- Subcommands: `compute` (`demo`), `monte-carlo`, `report`, `table`.
- Use `python entry_points/power_stability_radius.py <subcommand> --help` for the full option set.

Artifacts:
- `run_artifacts/compute/...`: `results.json`, tables, optional `h_vectors.npz`, `debug.log`.
- `run_artifacts/monte_carlo/...`: `monte_carlo_stats.json`, `debug.log`.
- `run_artifacts/report/...`: report copies, config snapshots, `debug.log`.
- `run_artifacts/table/...`: formatted table outputs and `debug.log`.
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(run_entrypoint("stability_radius.cli", "main"))

