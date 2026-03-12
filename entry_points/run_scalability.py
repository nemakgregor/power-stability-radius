from __future__ import annotations

"""Entry-point script for Experiment 4.

What it is:
- Runs `stability_radius.experiments.run_scalability.main`.
- Measures DC and AC runtime versus network size across configured cases.

Aim and scope:
- Produce one timing benchmark artifact per configured benchmark set.
- Keep scalability outputs separate from sweep outputs under
  `run_artifacts/run_scalability/...`.

CLI options:
- `--config`: YAML with benchmark cases and shared solver settings
- `--repeats`: number of repeated timings per case

Artifacts:
- `scalability.json`
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(run_entrypoint("stability_radius.experiments.run_scalability", "main"))

