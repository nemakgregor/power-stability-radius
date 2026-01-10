"""
Statistics and reporting utilities.

This subpackage provides lightweight helpers to summarize computed radii and
print them as a table to the terminal (no external dependencies required).
"""

from __future__ import annotations

from .table import format_results_table, print_radius_summary, print_results_table

__all__ = ["format_results_table", "print_results_table", "print_radius_summary"]
