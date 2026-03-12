from __future__ import annotations

"""Entry-point script for repository snapshot export.

What it is:
- Runs `stability_radius.helpers.openapi.listing.main`.
- Scans a repository tree, applies inclusion filters, and emits a markdown snapshot.

Aim and scope:
- Produce one portable repository digest for external review or prompt handoff.
- Keep default outputs under `run_artifacts/openapi_list/...` with a local `debug.log`.

CLI options:
- Positional: `root`
- Flags: `-o/--output`, `--no-tree`, `--max-bytes`

Artifacts:
- default `repository_snapshot.md`
- optional custom markdown output
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(run_entrypoint("stability_radius.helpers.openapi.listing", "main"))

