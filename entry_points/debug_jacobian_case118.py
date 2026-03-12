from __future__ import annotations

"""Entry-point script for the Case118 Jacobian diagnostic.

What it is:
- Runs `stability_radius.debug.jacobian_case118.main`.
- Runs a focused developer diagnostic for AC Jacobian assembly on Case118.

Aim and scope:
- Keep a known debug helper discoverable under the unified entry-point surface.
- Save the diagnostic log under `run_artifacts/debug_jacobian_case118/...`.

CLI options:
- no CLI flags; edit the underlying helper for deeper investigation

Artifacts:
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(run_entrypoint("stability_radius.debug.jacobian_case118", "main"))

