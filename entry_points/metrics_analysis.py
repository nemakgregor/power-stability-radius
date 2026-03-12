from __future__ import annotations

"""Entry-point script for comparative metric analysis.

What it is:
- Runs `stability_radius.analysis.metrics_analysis.main`.
- Compares AC radius-based indicators against practical overload metrics.

Aim and scope:
- Run one reproducible analysis pipeline for one MATPOWER case.
- Produce ranking diagnostics, correlation tables, hidden-danger reports, and plots.
- Save all generated files under `run_artifacts/metrics_analysis/...` by default.

CLI options:
- `--input`, `--slack-bus`, `--base-dispatch`
- `--sigma-p`, `--sigma-q`, `--sigma-p-scale`, `--sigma-p-min`
- `--sigma-q-scale`, `--sigma-q-min`
- `--mc-samples`, `--mc-seed`, `--top-k`
- `--output-dir`, `--log-level`

Artifacts:
- `results.json`, `mc_verification.json`, `unified_per_line_metrics.csv`
- summary CSVs and classification CSVs
- scatter plots, histograms, heatmaps, precision-at-k charts
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(
        run_entrypoint("stability_radius.analysis.metrics_analysis", "main")
    )

