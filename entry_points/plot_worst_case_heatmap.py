from __future__ import annotations

"""Entry-point script for worst-case verification plots.

What it is:
- Runs `stability_radius.helpers.experiments.plot_worst_case_heatmap.main`.
- Converts Experiment 3 JSON outputs into heatmap and scatter visualizations.

Aim and scope:
- Make verification drift and violation status easy to inspect from one folder.
- Default input is `run_artifacts/run_worst_case_verify`.
- Default outputs go to `run_artifacts/plot_worst_case_heatmap/...`.

CLI options:
- `--input-dir`
- `--output-dir`

Artifacts:
- `worst_case_heatmap.pdf`
- `worst_case_scatter.pdf` when enough data is present
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(
        run_entrypoint(
            "stability_radius.helpers.experiments.plot_worst_case_heatmap",
            "main",
        )
    )

