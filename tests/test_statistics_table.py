"""Tests for stability_radius.statistics.table — ASCII/CSV table formatting."""

from __future__ import annotations

import math

from stability_radius.statistics.table import (
    _format_float,
    _line_sort_key,
    format_radius_summary,
    format_results_csv,
    format_results_table,
    format_results_table_sections,
    infer_default_flat_columns,
    DEFAULT_AC_COLUMNS,
    DEFAULT_DC_COLUMNS,
)


# ---------------------------------------------------------------------------
# _line_sort_key
# ---------------------------------------------------------------------------


class TestLineSortKey:
    def test_numeric_sorting(self):
        keys = ["line_10", "line_2", "line_1"]
        assert sorted(keys, key=_line_sort_key) == ["line_1", "line_2", "line_10"]

    def test_fallback_for_non_numeric(self):
        key = "trafo_abc"
        result = _line_sort_key(key)
        assert result[0] == 10**18


# ---------------------------------------------------------------------------
# _format_float
# ---------------------------------------------------------------------------


class TestFormatFloat:
    def test_finite_number(self):
        assert _format_float(1.23456789) == "1.23457"

    def test_inf(self):
        assert _format_float(float("inf")) == "inf"

    def test_nan(self):
        assert _format_float(float("nan")) == "nan"

    def test_non_numeric(self):
        assert _format_float("hello") == "hello"

    def test_integer(self):
        assert _format_float(42) == "42"

    def test_zero(self):
        assert _format_float(0.0) == "0"

    def test_none_returns_str(self):
        assert _format_float(None) == "None"


# ---------------------------------------------------------------------------
# infer_default_flat_columns
# ---------------------------------------------------------------------------


class TestInferDefaultFlatColumns:
    def test_dc_only(self):
        results = {"line_0": {"radius_l2": 1.0, "norm_g": 0.5}}
        cols = infer_default_flat_columns(results)
        assert cols == tuple(DEFAULT_DC_COLUMNS)

    def test_ac_only(self):
        results = {"line_0": {"radius_ac_l2": 2.0, "margin_ac_mva": 0.1}}
        cols = infer_default_flat_columns(results)
        assert cols == tuple(DEFAULT_AC_COLUMNS)

    def test_both_dc_and_ac(self):
        results = {"line_0": {"radius_l2": 1.0, "radius_ac_l2": 2.0}}
        cols = infer_default_flat_columns(results)
        assert cols == tuple(DEFAULT_DC_COLUMNS) + tuple(DEFAULT_AC_COLUMNS)

    def test_neither_dc_nor_ac_uses_dc_fallback(self):
        results = {"line_0": {"some_field": 42}}
        cols = infer_default_flat_columns(results)
        assert cols == tuple(DEFAULT_DC_COLUMNS)

    def test_non_line_keys_ignored(self):
        results = {"__meta__": {"radius_l2": 1.0}, "line_0": {"margin_ac_mva": 1.0}}
        cols = infer_default_flat_columns(results)
        assert cols == tuple(DEFAULT_AC_COLUMNS)


# ---------------------------------------------------------------------------
# format_results_table
# ---------------------------------------------------------------------------


class TestFormatResultsTable:
    def test_basic_table_structure(self):
        results = {
            "line_0": {"p0_mw": 10.0, "radius_l2": 5.0},
            "line_1": {"p0_mw": 20.0, "radius_l2": 3.0},
        }
        table = format_results_table(results, columns=("p0_mw", "radius_l2"))
        lines = table.split("\n")
        # Header line, separator, 2 data rows
        assert len(lines) == 4
        assert "line" in lines[0]
        assert "p0_mw" in lines[0]
        assert "radius_l2" in lines[0]

    def test_max_rows_truncation(self):
        results = {f"line_{i}": {"radius_l2": float(i)} for i in range(10)}
        table = format_results_table(results, columns=("radius_l2",), max_rows=3)
        assert "7 more rows" in table

    def test_empty_results(self):
        table = format_results_table({}, columns=("radius_l2",))
        lines = table.split("\n")
        # Header + separator only, no data rows
        assert len(lines) == 2

    def test_missing_field_shows_empty(self):
        results = {"line_0": {"radius_l2": 1.0}}
        table = format_results_table(results, columns=("radius_l2", "missing_col"))
        # Should not crash; missing_col shows empty
        assert "line_0" in table


# ---------------------------------------------------------------------------
# format_results_table_sections
# ---------------------------------------------------------------------------


class TestFormatResultsTableSections:
    def test_dc_section_only(self):
        results = {"line_0": {"radius_l2": 1.0, "norm_g": 0.5}}
        table = format_results_table_sections(results)
        assert "[DC]" in table
        assert "[AC]" not in table

    def test_ac_section_only(self):
        results = {"line_0": {"radius_ac_l2": 2.0, "margin_ac_mva": 0.1}}
        table = format_results_table_sections(results)
        assert "[DC]" not in table
        assert "[AC]" in table

    def test_both_sections(self):
        results = {"line_0": {"radius_l2": 1.0, "radius_ac_l2": 2.0}}
        table = format_results_table_sections(results)
        assert "[DC]" in table
        assert "[AC]" in table

    def test_no_line_results(self):
        results = {"__meta__": {"version": 2}}
        table = format_results_table_sections(results)
        assert table == "No per-line results found."


# ---------------------------------------------------------------------------
# format_results_csv
# ---------------------------------------------------------------------------


class TestFormatResultsCsv:
    def test_csv_has_header_and_data(self):
        results = {"line_0": {"radius_l2": 1.5}}
        csv_text = format_results_csv(results, columns=("radius_l2",))
        lines = csv_text.strip().split("\n")
        assert lines[0] == "line,radius_l2"
        assert lines[1] == "line_0,1.5"

    def test_csv_numeric_sort(self):
        results = {
            "line_2": {"radius_l2": 2.0},
            "line_10": {"radius_l2": 10.0},
            "line_1": {"radius_l2": 1.0},
        }
        csv_text = format_results_csv(results, columns=("radius_l2",))
        lines = csv_text.strip().split("\n")
        # Data rows should be sorted: line_1, line_2, line_10
        assert lines[1].startswith("line_1,")
        assert lines[2].startswith("line_2,")
        assert lines[3].startswith("line_10,")


# ---------------------------------------------------------------------------
# format_radius_summary
# ---------------------------------------------------------------------------


class TestFormatRadiusSummary:
    def test_summary_with_finite_radii(self):
        results = {
            "line_0": {"radius_l2": 1.0},
            "line_1": {"radius_l2": 3.0},
            "line_2": {"radius_l2": 5.0},
        }
        summary = format_radius_summary(results, radius_field="radius_l2")
        assert "lines=3" in summary
        assert "finite_radii=3" in summary
        assert "min=1" in summary
        assert "max=5" in summary

    def test_summary_skips_nan_and_inf(self):
        results = {
            "line_0": {"radius_l2": 1.0},
            "line_1": {"radius_l2": float("nan")},
            "line_2": {"radius_l2": float("inf")},
        }
        summary = format_radius_summary(results, radius_field="radius_l2")
        assert "finite_radii=1" in summary

    def test_summary_no_finite_radii(self):
        results = {"line_0": {"radius_l2": float("inf")}}
        summary = format_radius_summary(results, radius_field="radius_l2")
        assert "finite_radii=0" in summary
