from __future__ import annotations

"""Entry-point script for the three-regime N-1 demo.

What it is:
- Runs `stability_radius.demos.n1_stability_demo.main`.
- Compares Cost OPF, Radius OPF, and a screening-based SCOPF proxy on one case.

Aim and scope:
- Show how radius-driven tightening changes the operating regime.
- Measure post-contingency behavior, N-1 diagnostics, and security-cost tradeoffs.
- Keep all demo outputs under `run_artifacts/n1_stability_demo/...` by default.

CLI options:
- `--input`, `--output-dir`, `--slack-bus`
- `--r-target`, `--n-iter`, `--scopf-iter`
- `--sigma-p`, `--sigma-q`
- `--skip-n1-screening`, `--skip-dc-n1`, `--skip-ac-n1-radius`
- `--verbose`

Artifacts:
- summary text, CSV comparisons, DC and AC N-1 tables
- regime comparison plots and screening plots
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(
        run_entrypoint("stability_radius.demos.n1_stability_demo", "main")
    )

