from __future__ import annotations

"""Entry-point script for sigma-radius and timing plots.

What it is:
- Runs `stability_radius.helpers.experiments.plot_sigma_vs_time.main`.
- Combines sigma-radius result folders with optional scalability timings.

Aim and scope:
- Visualize sorted sigma-radius values and DC-vs-AC timing bars from
  the standardized artifact tree.
- Default inputs target `run_artifacts/run_sigma_radius` and
  `run_artifacts/run_scalability/scalability.json`.

CLI options:
- `--sigma-dir`
- `--scalability`
- `--output-dir`

Artifacts:
- `sigma_radius_sorted.pdf`
- `sigma_vs_time.pdf` when scalability data is available
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(
        run_entrypoint(
            "stability_radius.helpers.experiments.plot_sigma_vs_time",
            "main",
        )
    )

