from __future__ import annotations

"""Thin CLI wrapper for the main application entry point."""

from stability_radius.application.cli import (
    _parse_report_cases_from_cfg,
    _resolve_path,
    build_parser,
    main,
    run_compute,
    run_monte_carlo,
    run_report,
    run_table,
)

__all__ = [
    "_parse_report_cases_from_cfg",
    "_resolve_path",
    "build_parser",
    "main",
    "run_compute",
    "run_monte_carlo",
    "run_report",
    "run_table",
]


if __name__ == "__main__":
    raise SystemExit(main())
