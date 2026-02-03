"""
Statistics and reporting utilities.

This subpackage provides lightweight helpers to summarize computed radii and
print/export them as a table to the terminal (no external dependencies required).
"""

from __future__ import annotations

from .table import (
    format_radius_summary,
    format_results_csv,
    format_results_table,
    print_radius_summary,
    print_results_table,
    write_results_csv,
    write_results_table,
)

__all__ = [
    "format_results_table",
    "format_results_csv",
    "write_results_table",
    "write_results_csv",
    "format_radius_summary",
    "print_results_table",
    "print_radius_summary",
]
