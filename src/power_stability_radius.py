from __future__ import annotations

"""
Single entrypoint script.

Design notes
------------
This module is intentionally kept thin so the repository can be used as a library
without importing CLI-only dependencies or performing sys.path side-effects at
import time.

- Library API: `stability_radius.workflows.compute_results_for_case`
- CLI: `stability_radius.cli.main`
"""

from typing import Sequence

from stability_radius.workflows import compute_results_for_case


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint used by `python src/power_stability_radius.py ...`.

    Parameters
    ----------
    argv:
        Optional argv sequence (without the program name). If None, uses sys.argv.

    Returns
    -------
    int
        Exit code.
    """
    from stability_radius.cli import main as cli_main

    return int(cli_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
