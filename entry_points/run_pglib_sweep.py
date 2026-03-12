from __future__ import annotations

"""Entry-point script for Experiment 1.

What it is:
- Runs `stability_radius.experiments.run_pglib_sweep.main`.
- Runs a DC-vs-AC radius sweep across configured PGLib cases.

Aim and scope:
- Compare DC and AC L2 certificates on a shared base point.
- Produce per-case JSON files, an aggregate summary, and Figure 1 outputs.
- Keep experiment artifacts under `run_artifacts/run_pglib_sweep/...`.

CLI options:
- `--config`: experiment YAML with case list and output naming
- `--reuse-dir`: reuse previously solved per-case JSON files

Artifacts:
- per-case `*.json`
- `summary.json`
- `fig1_dc_vs_ac_radius.png` and `.pdf`
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(run_entrypoint("stability_radius.experiments.run_pglib_sweep", "main"))

