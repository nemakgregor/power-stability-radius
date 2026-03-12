from __future__ import annotations

"""Entry-point script for standalone table export.

What it is:
- Runs `stability_radius.helpers.reporting.table.main`.
- Formats an existing `results.json` into ASCII and optional CSV tables.

Aim and scope:
- Provide a quick reporting utility without running the full multi-command CLI.
- Always leave a reproducible table artifact under `run_artifacts/table/...`.

CLI options:
- Positional: `results_json`
- Flags: `--max-rows`, `--format`, `--radius-field`, `--columns`
- Output controls: `--table-out`, `--csv-out`

Artifacts:
- default `results_table.txt`
- optional explicit table and CSV outputs
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(
        run_entrypoint("stability_radius.helpers.reporting.table", "main")
    )

