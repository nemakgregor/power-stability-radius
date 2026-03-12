from __future__ import annotations

"""Entry-point script for aggregated experiment collection.

What it is:
- Runs `stability_radius.helpers.experiments.collect_results.main`.
- Scans experiment JSON outputs and builds one CSV summary table.

Aim and scope:
- Provide a single cross-experiment aggregation step.
- Default to scanning `run_artifacts/` and writing under
  `run_artifacts/collect_results/...`.

CLI options:
- `--output-dir`: root directory to scan for experiment JSON outputs
- `--csv`: output CSV path

Artifacts:
- aggregated CSV summary
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(
        run_entrypoint("stability_radius.helpers.experiments.collect_results", "main")
    )

