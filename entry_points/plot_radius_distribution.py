from __future__ import annotations

"""Entry-point script for the DC-vs-AC radius distribution plotter.

What it is:
- Runs `stability_radius.helpers.experiments.plot_radius_distribution.main`.
- Reads sweep outputs and renders distribution plots across cases.

Aim and scope:
- Turn `run_pglib_sweep` JSON artifacts into presentation-ready figures.
- Default inputs come from `run_artifacts/run_pglib_sweep`.
- Default outputs go to `run_artifacts/plot_radius_distribution/...`.

CLI options:
- `--input-dir`
- `--output-dir`

Artifacts:
- `radius_distribution.pdf`
- `radius_distribution.png`
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(
        run_entrypoint(
            "stability_radius.helpers.experiments.plot_radius_distribution",
            "main",
        )
    )

