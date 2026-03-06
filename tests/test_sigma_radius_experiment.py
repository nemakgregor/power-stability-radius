from __future__ import annotations

"""
Tests for Experiment 2 (run_sigma_radius) helper functions.

Covers the audit fixes:
- Negative sigma-radius handling in aggregation
- Base-infeasible line flagging in table rows
- Scatter plot filtering for log-log axes
- Worst-case verification skipping for negative r_L2
- Monte Carlo validation with tightest-feasible-line selection
- Validation check feasibility summary
"""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_hourly_results(
    *,
    n_lines: int = 5,
    n_bus: int = 3,
    line_ids: list[int] | None = None,
    sigma_radii: list[float] | None = None,
    l2_radii: list[float] | None = None,
    s0_values: list[float] | None = None,
    limit_values: list[float] | None = None,
) -> dict:
    """Build a single-hour result dict matching _compute_hour() output."""
    if line_ids is None:
        line_ids = list(range(n_lines))
    if sigma_radii is None:
        sigma_radii = [5.0 - i for i in range(n_lines)]
    if l2_radii is None:
        l2_radii = [10.0 - i * 2 for i in range(n_lines)]
    if s0_values is None:
        s0_values = [50.0 + i * 10 for i in range(n_lines)]
    if limit_values is None:
        limit_values = [100.0] * n_lines

    sigma_results = {}
    ac_l2_results = {}
    h_bind = np.random.default_rng(42).standard_normal((n_lines, 2 * n_bus))
    s0_arr = np.array(s0_values, dtype=float)
    limit_arr = np.array(limit_values, dtype=float)

    for i, lid in enumerate(line_ids):
        k = f"line_{lid}"
        sigma_results[k] = {
            "radius_ac_sigma": sigma_radii[i],
            "sigma_flow_mva": 1.5 + i * 0.1,
            "overload_probability_ac": 0.01 * (i + 1),
            "worst_case_dp_mw": np.zeros(n_bus),
            "worst_case_dq_mvar": np.zeros(n_bus),
        }
        binding_end = "from" if i % 2 == 0 else "to"
        ac_l2_results[k] = {
            "radius_ac_l2": l2_radii[i],
            "binding_end": binding_end,
            "ac_s0_from_mva": s0_values[i],
            "ac_s0_to_mva": s0_values[i] * 0.9,
            "ac_s_limit_mva": limit_values[i],
        }

    # Mock feasibility result
    feasibility = MagicMock()
    feasibility.is_feasible = all(s < c for s, c in zip(s0_values, limit_values))
    feasibility.n_constrained_violated = sum(
        1 for s, c in zip(s0_values, limit_values) if s > c
    )
    feasibility.worst_margin_mva = min(c - s for s, c in zip(s0_values, limit_values))
    feasibility.worst_line_id = line_ids[
        int(np.argmin([c - s for s, c in zip(s0_values, limit_values)]))
    ]

    return {
        "ac_l2_results": ac_l2_results,
        "sigma_results": sigma_results,
        "h_bind": h_bind,
        "h_from_full": h_bind,
        "h_to_full": h_bind,
        "s0_mva": s0_arr,
        "s_limit_mva": limit_arr,
        "line_ids": line_ids,
        "total_load_mw": 200.0,
        "ac_feasibility": feasibility,
    }


# ---------------------------------------------------------------------------
# Tests for _aggregate_across_hours
# ---------------------------------------------------------------------------


class TestAggregateAcrossHours:
    def test_negative_sigma_radius_is_kept(self) -> None:
        """Negative r_sigma should be included in aggregation, not filtered."""
        from experiments.run_sigma_radius import _aggregate_across_hours

        hr = _make_hourly_results(
            n_lines=3,
            line_ids=[0, 1, 2],
            sigma_radii=[-2.0, 3.0, 5.0],
        )
        agg = _aggregate_across_hours({0: hr})

        assert "line_0" in agg["min_sigma_radius"]
        assert agg["min_sigma_radius"]["line_0"] == pytest.approx(-2.0)

    def test_base_infeasible_flag_set_for_negative_r_sigma(self) -> None:
        """Lines with r_sigma < 0 should be flagged as base_infeasible."""
        from experiments.run_sigma_radius import _aggregate_across_hours

        hr = _make_hourly_results(
            n_lines=3,
            line_ids=[10, 11, 12],
            sigma_radii=[-1.5, 4.0, 6.0],
        )
        agg = _aggregate_across_hours({0: hr})

        assert agg["base_infeasible"]["line_10"] is True
        assert agg["base_infeasible"]["line_11"] is False
        assert agg["base_infeasible"]["line_12"] is False

    def test_negative_r_sigma_sorts_first(self) -> None:
        """When sorted, negative r_sigma lines should appear before positive ones."""
        from experiments.run_sigma_radius import _aggregate_across_hours

        hr = _make_hourly_results(
            n_lines=4,
            line_ids=[0, 1, 2, 3],
            sigma_radii=[10.0, -3.0, 2.0, -1.0],
        )
        agg = _aggregate_across_hours({0: hr})
        sorted_lines = sorted(agg["min_sigma_radius"].items(), key=lambda kv: kv[1])
        # Most negative first
        assert sorted_lines[0][0] == "line_1"
        assert sorted_lines[0][1] == pytest.approx(-3.0)
        assert sorted_lines[1][0] == "line_3"
        assert sorted_lines[1][1] == pytest.approx(-1.0)

    def test_worst_hour_tracks_minimum_across_hours(self) -> None:
        """When the same line has different r_sigma across hours, the minimum wins."""
        from experiments.run_sigma_radius import _aggregate_across_hours

        hr0 = _make_hourly_results(
            n_lines=2,
            line_ids=[0, 1],
            sigma_radii=[5.0, 3.0],
        )
        hr1 = _make_hourly_results(
            n_lines=2,
            line_ids=[0, 1],
            sigma_radii=[2.0, 4.0],
        )
        agg = _aggregate_across_hours({0: hr0, 5: hr1})

        # line_0: min is 2.0 at hour 5
        assert agg["min_sigma_radius"]["line_0"] == pytest.approx(2.0)
        assert agg["worst_hour"]["line_0"] == 5
        # line_1: min is 3.0 at hour 0
        assert agg["min_sigma_radius"]["line_1"] == pytest.approx(3.0)
        assert agg["worst_hour"]["line_1"] == 0

    def test_nan_and_inf_sigma_radius_are_skipped(self) -> None:
        """Non-finite r_sigma values should be excluded from aggregation."""
        from experiments.run_sigma_radius import _aggregate_across_hours

        hr = _make_hourly_results(
            n_lines=3,
            line_ids=[0, 1, 2],
            sigma_radii=[float("nan"), float("inf"), 4.0],
        )
        agg = _aggregate_across_hours({0: hr})

        assert "line_0" not in agg["min_sigma_radius"]
        assert "line_1" not in agg["min_sigma_radius"]
        assert "line_2" in agg["min_sigma_radius"]


# ---------------------------------------------------------------------------
# Tests for _build_table2_rows
# ---------------------------------------------------------------------------


class TestBuildTable2Rows:
    def _make_agg(
        self,
        sigma_radii: list[float],
        l2_radii: list[float] | None = None,
    ) -> dict:
        """Build a minimal aggregated dict for _build_table2_rows."""
        from experiments.run_sigma_radius import _aggregate_across_hours

        if l2_radii is None:
            l2_radii = [10.0] * len(sigma_radii)
        n = len(sigma_radii)
        hr = _make_hourly_results(
            n_lines=n,
            line_ids=list(range(n)),
            sigma_radii=sigma_radii,
            l2_radii=l2_radii,
        )
        return _aggregate_across_hours({0: hr})

    def test_top_k_limits_output(self) -> None:
        from experiments.run_sigma_radius import _build_table2_rows

        agg = self._make_agg([1.0, 2.0, 3.0, 4.0, 5.0])
        rows = _build_table2_rows(agg, top_k=3)
        assert len(rows) == 3

    def test_rows_sorted_ascending_by_r_sigma(self) -> None:
        from experiments.run_sigma_radius import _build_table2_rows

        agg = self._make_agg([5.0, 1.0, 3.0, 2.0, 4.0])
        rows = _build_table2_rows(agg, top_k=5)
        r_values = [r["r_sigma"] for r in rows]
        assert r_values == sorted(r_values)

    def test_negative_r_sigma_lines_flagged_infeasible(self) -> None:
        from experiments.run_sigma_radius import _build_table2_rows

        agg = self._make_agg([-2.0, 1.0, 3.0])
        rows = _build_table2_rows(agg, top_k=3)
        # First row (most negative) should be infeasible
        assert rows[0]["base_infeasible"] is True
        assert rows[0]["r_sigma"] == pytest.approx(-2.0)
        # Others should be feasible
        assert rows[1]["base_infeasible"] is False
        assert rows[2]["base_infeasible"] is False

    def test_mc_and_verified_fields_are_none_initially(self) -> None:
        from experiments.run_sigma_radius import _build_table2_rows

        agg = self._make_agg([1.0, 2.0])
        rows = _build_table2_rows(agg, top_k=2)
        for row in rows:
            assert row["mc_violation_rate"] is None
            assert row["verified"] is None

    def test_margin_computed_correctly(self) -> None:
        from experiments.run_sigma_radius import _build_table2_rows

        hr = _make_hourly_results(
            n_lines=1,
            line_ids=[0],
            sigma_radii=[5.0],
            s0_values=[80.0],
            limit_values=[100.0],
        )
        from experiments.run_sigma_radius import _aggregate_across_hours

        agg = _aggregate_across_hours({0: hr})
        rows = _build_table2_rows(agg, top_k=1)
        assert rows[0]["margin_mva"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Tests for verification skipping with negative r_L2
# ---------------------------------------------------------------------------


class TestWorstCaseVerificationSkipsInfeasible:
    def test_negative_r_l2_lines_are_skipped(self) -> None:
        """Lines with r_L2 <= 0 should be skipped in verification."""
        from experiments.run_sigma_radius import (
            _aggregate_across_hours,
            _build_table2_rows,
            _run_worst_case_verification,
        )

        hr = _make_hourly_results(
            n_lines=3,
            line_ids=[0, 1, 2],
            sigma_radii=[-2.0, 3.0, 5.0],
            l2_radii=[-1.5, 8.0, 12.0],
            s0_values=[105.0, 50.0, 40.0],
            limit_values=[100.0, 100.0, 100.0],
        )
        agg = _aggregate_across_hours({0: hr})
        rows = _build_table2_rows(agg, top_k=3)

        # Mock the network-dependent parts
        mock_net = MagicMock()
        load_p = np.zeros((3, 1))
        load_q = np.zeros((3, 1))
        output_dir = Path("/tmp/test_verify")

        with (
            patch("experiments.run_sigma_radius.copy") as mock_copy,
            patch("experiments.run_sigma_radius.verify_worst_case") as mock_verify,
            patch.object(Path, "open", create=True),
            patch("experiments.run_sigma_radius.json"),
        ):
            mock_copy.deepcopy.return_value = mock_net

            # Set up mock verification result
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"mock": True}
            mock_result.pf_converged = True
            mock_result.violated = True
            mock_verify.return_value = mock_result

            output_dir.mkdir(parents=True, exist_ok=True)

            results = _run_worst_case_verification(
                net=mock_net,
                agg=agg,
                table_rows=rows,
                bus_ids=[0, 1, 2],
                load_p_mw=load_p,
                load_q_mvar=load_q,
                slack_bus=0,
                lossless=True,
                fpf_cfg=None,
                scales=[1.0],
                output_dir=output_dir,
            )

        # Line 0 has r_L2 < 0, should be skipped
        assert rows[0]["verified"] is None
        line0_result = next(r for r in results if r.get("line_key") == "line_0")
        assert line0_result["status"] == "skipped_infeasible"

        # Lines 1 and 2 have positive r_L2, should be verified
        # verify_worst_case should only be called for lines with positive r_L2
        # That's 2 lines with 1 scale each = 2 calls
        assert mock_verify.call_count == 2


# ---------------------------------------------------------------------------
# Tests for MC validation with feasible-line selection
# ---------------------------------------------------------------------------


class TestMonteCarloFeasibleLineSelection:
    def test_selects_tightest_positive_r_sigma(self) -> None:
        """MC should select the tightest r_sigma > 0 line, not the most negative."""
        from experiments.run_sigma_radius import _run_monte_carlo_validation

        table_rows = [
            {
                "line_key": "line_0",
                "r_sigma": -3.0,
                "worst_hour": 0,
                "base_infeasible": True,
            },
            {
                "line_key": "line_1",
                "r_sigma": -1.0,
                "worst_hour": 0,
                "base_infeasible": True,
            },
            {
                "line_key": "line_2",
                "r_sigma": 2.0,
                "worst_hour": 5,
                "base_infeasible": False,
            },
            {
                "line_key": "line_3",
                "r_sigma": 8.0,
                "worst_hour": 3,
                "base_infeasible": False,
            },
        ]

        mock_net = MagicMock()
        load_p = np.zeros((3, 10))
        load_q = np.zeros((3, 10))
        sigma_p = np.ones(3)
        sigma_q = np.ones(3)

        with (
            patch("experiments.run_sigma_radius.copy") as mock_copy,
            patch("experiments.run_sigma_radius.run_ac_monte_carlo_sigma") as mock_mc,
            patch("experiments.run_sigma_radius._set_loads_for_hour"),
            patch.object(Path, "open", create=True),
            patch("experiments.run_sigma_radius.json"),
        ):
            mock_copy.deepcopy.return_value = mock_net

            # Mock MC result
            mock_mc_result = MagicMock()
            mock_mc_result.n_samples = 100
            mock_mc_result.n_violations = 5
            mock_mc_result.n_pf_failures = 0
            mock_mc_result.soundness_inside_sigma_ball = 0.95
            mock_mc_result.empirical_overload_probability = {}
            mock_mc.return_value = mock_mc_result

            output_dir = Path("/tmp/test_mc")
            output_dir.mkdir(parents=True, exist_ok=True)

            result = _run_monte_carlo_validation(
                net=mock_net,
                agg={},
                table_rows=table_rows,
                bus_ids=[0, 1, 2],
                load_p_mw=load_p,
                load_q_mvar=load_q,
                sigma_p_mw=sigma_p,
                sigma_q_mvar=sigma_q,
                slack_bus=0,
                lossless=True,
                fpf_cfg=None,
                n_samples=100,
                seed=42,
                output_dir=output_dir,
            )

            # Should use line_2 (r_sigma=2.0), not line_0 (r_sigma=-3.0)
            assert result is not None
            # r_sigma_for_ball should be 2.0 (the tightest positive)
            mc_call_kwargs = mock_mc.call_args[1]
            assert mc_call_kwargs["r_sigma"] == pytest.approx(2.0)

    def test_all_infeasible_uses_inf_for_ball(self) -> None:
        """When all lines are infeasible, r_sigma_for_ball should be inf."""
        from experiments.run_sigma_radius import _run_monte_carlo_validation

        table_rows = [
            {
                "line_key": "line_0",
                "r_sigma": -3.0,
                "worst_hour": 0,
                "base_infeasible": True,
            },
            {
                "line_key": "line_1",
                "r_sigma": -1.0,
                "worst_hour": 0,
                "base_infeasible": True,
            },
        ]

        mock_net = MagicMock()
        load_p = np.zeros((3, 10))
        load_q = np.zeros((3, 10))
        sigma_p = np.ones(3)
        sigma_q = np.ones(3)

        with (
            patch("experiments.run_sigma_radius.copy") as mock_copy,
            patch("experiments.run_sigma_radius.run_ac_monte_carlo_sigma") as mock_mc,
            patch("experiments.run_sigma_radius._set_loads_for_hour"),
            patch.object(Path, "open", create=True),
            patch("experiments.run_sigma_radius.json"),
        ):
            mock_copy.deepcopy.return_value = mock_net

            mock_mc_result = MagicMock()
            mock_mc_result.n_samples = 100
            mock_mc_result.n_violations = 100
            mock_mc_result.n_pf_failures = 0
            mock_mc_result.soundness_inside_sigma_ball = float("nan")
            mock_mc_result.empirical_overload_probability = {}
            mock_mc.return_value = mock_mc_result

            output_dir = Path("/tmp/test_mc_inf")
            output_dir.mkdir(parents=True, exist_ok=True)

            result = _run_monte_carlo_validation(
                net=mock_net,
                agg={},
                table_rows=table_rows,
                bus_ids=[0, 1, 2],
                load_p_mw=load_p,
                load_q_mvar=load_q,
                sigma_p_mw=sigma_p,
                sigma_q_mvar=sigma_q,
                slack_bus=0,
                lossless=True,
                fpf_cfg=None,
                n_samples=100,
                seed=42,
                output_dir=output_dir,
            )

            assert result is not None
            # With all infeasible, r_sigma_for_ball should be inf
            mc_call_kwargs = mock_mc.call_args[1]
            assert mc_call_kwargs["r_sigma"] == float("inf")


# ---------------------------------------------------------------------------
# Tests for scatter plot filtering
# ---------------------------------------------------------------------------


class TestScatterPlotFiltering:
    def test_negative_r_sigma_excluded_from_scatter(self) -> None:
        """Lines with r_sigma <= 0 or r_L2 <= 0 must not appear on log-log scatter."""
        from experiments.run_sigma_radius import _aggregate_across_hours

        hr = _make_hourly_results(
            n_lines=5,
            line_ids=[0, 1, 2, 3, 4],
            sigma_radii=[-2.0, 3.0, 5.0, -0.5, 7.0],
            l2_radii=[-1.0, 8.0, 12.0, 5.0, 15.0],
        )
        agg = _aggregate_across_hours({0: hr})

        # Collect what the scatter plot would include
        line_keys = sorted(agg["min_sigma_radius"].keys())
        included = []
        excluded = []
        for lk in line_keys:
            r_sig = agg["min_sigma_radius"].get(lk, float("nan"))
            r_l2 = agg["worst_hour_ac_l2_radius"].get(lk, float("nan"))
            if np.isfinite(r_sig) and np.isfinite(r_l2) and r_sig > 0 and r_l2 > 0:
                included.append(lk)
            else:
                excluded.append(lk)

        # line_0: r_sig=-2, r_l2=-1 -> excluded
        # line_1: r_sig=3, r_l2=8 -> included
        # line_2: r_sig=5, r_l2=12 -> included
        # line_3: r_sig=-0.5, r_l2=5 -> excluded (negative r_sig)
        # line_4: r_sig=7, r_l2=15 -> included
        assert "line_0" in excluded
        assert "line_3" in excluded
        assert "line_1" in included
        assert "line_2" in included
        assert "line_4" in included
        assert len(included) == 3
        assert len(excluded) == 2


# ---------------------------------------------------------------------------
# Tests for validation checks
# ---------------------------------------------------------------------------


class TestValidationChecks:
    def test_feasibility_summary_counts_negative_lines(self) -> None:
        """Validation checks should report number of negative-sigma lines."""
        from experiments.run_sigma_radius import (
            _aggregate_across_hours,
            _build_table2_rows,
            _run_validation_checks,
        )

        hr = _make_hourly_results(
            n_lines=4,
            line_ids=[0, 1, 2, 3],
            sigma_radii=[-2.0, -0.5, 3.0, 5.0],
            s0_values=[110.0, 105.0, 80.0, 60.0],
            limit_values=[100.0, 100.0, 100.0, 100.0],
        )
        agg = _aggregate_across_hours({0: hr})
        rows = _build_table2_rows(agg, top_k=4)

        output_dir = Path("/tmp/test_validation")
        output_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(Path, "open", create=True),
            patch("experiments.run_sigma_radius.json"),
        ):
            checks = _run_validation_checks(
                agg=agg,
                hourly_results={0: hr},
                table_rows=rows,
                mc_results=None,
                sigma_p_mw_raw=np.ones(3),
                n_bus=3,
                output_dir=output_dir,
            )

        assert checks["feasibility"]["n_lines_negative_sigma"] == 2
        assert checks["feasibility"]["n_lines_total"] == 4
        assert checks["feasibility"]["n_top_k_infeasible"] == 2

    def test_balance_check_passes_for_zero_sum_dp(self) -> None:
        """Balance check should pass when sum(dp) < 1e-6."""
        from experiments.run_sigma_radius import (
            _aggregate_across_hours,
            _build_table2_rows,
            _run_validation_checks,
        )

        hr = _make_hourly_results(n_lines=1, line_ids=[0], sigma_radii=[5.0])
        agg = _aggregate_across_hours({0: hr})
        rows = _build_table2_rows(agg, top_k=1)

        output_dir = Path("/tmp/test_balance")
        output_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(Path, "open", create=True),
            patch("experiments.run_sigma_radius.json"),
        ):
            checks = _run_validation_checks(
                agg=agg,
                hourly_results={0: hr},
                table_rows=rows,
                mc_results=None,
                sigma_p_mw_raw=np.ones(3),
                n_bus=3,
                output_dir=output_dir,
            )

        assert checks["balance"]["all_ok"] is True


# ---------------------------------------------------------------------------
# Tests for CSV export
# ---------------------------------------------------------------------------


class TestCSVExport:
    def test_csv_includes_base_infeasible_column(self, tmp_path: Path) -> None:
        """CSV export should include base_infeasible in header."""
        from experiments.run_sigma_radius import (
            _aggregate_across_hours,
            _build_table2_rows,
            _export_table2_csv,
        )

        hr = _make_hourly_results(
            n_lines=2,
            line_ids=[0, 1],
            sigma_radii=[-1.0, 3.0],
        )
        agg = _aggregate_across_hours({0: hr})
        rows = _build_table2_rows(agg, top_k=2)
        _export_table2_csv(rows, tmp_path)

        csv_path = tmp_path / "table2_sigma_radius.csv"
        assert csv_path.exists()

        import csv

        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            assert "base_infeasible" in fieldnames
            csv_rows = list(reader)

        assert len(csv_rows) == 2
        # First row (r_sigma=-1.0) should be infeasible
        assert csv_rows[0]["base_infeasible"] == "True"
        # Second row (r_sigma=3.0) should be feasible
        assert csv_rows[1]["base_infeasible"] == "False"


# ---------------------------------------------------------------------------
# Tests for h-vector NPZ save
# ---------------------------------------------------------------------------


class TestSaveHvectorsNPZ:
    def test_hvectors_saved_and_loadable(self, tmp_path: Path) -> None:
        """Should save worst-hour h-vectors in NPZ format."""
        from experiments.run_sigma_radius import (
            _aggregate_across_hours,
            _save_hvectors_npz,
        )

        hr = _make_hourly_results(
            n_lines=3, line_ids=[0, 1, 2], sigma_radii=[5.0, 3.0, 7.0]
        )
        agg = _aggregate_across_hours({0: hr})
        _save_hvectors_npz(agg, output_dir=tmp_path)

        npz_path = tmp_path / "hvectors.npz"
        assert npz_path.exists()

        data = np.load(str(npz_path))
        assert "line_ids" in data
        assert "line_0" in data
        assert "line_1" in data
        assert "line_2" in data
        assert data["line_ids"].shape == (3,)


# ---------------------------------------------------------------------------
# Tests for AC feasibility check integration
# ---------------------------------------------------------------------------


class TestACFeasibilityCheck:
    @pytest.fixture()
    def _pp(self):
        return pytest.importorskip("pandapower")

    @pytest.fixture()
    def _scipy(self):
        return pytest.importorskip("scipy")

    @pytest.fixture()
    def _pypsa(self):
        return pytest.importorskip("pypsa")

    def test_infeasible_base_point_detected(self, _pp, _scipy, _pypsa) -> None:
        """Network with S0 > limit should be flagged infeasible."""
        pp = _pp
        from stability_radius.base_point.pypsa_pf import (
            solve_ac_pf_base_point_from_pandapower,
        )
        from stability_radius.radii.ac_feasibility import (
            check_ac_base_point_feasibility,
        )

        net = pp.create_empty_network(sn_mva=100.0)
        b0 = int(pp.create_bus(net, vn_kv=110.0))
        b1 = int(pp.create_bus(net, vn_kv=110.0))
        pp.create_ext_grid(net, b0, vm_pu=1.0)
        pp.create_load(net, b1, p_mw=50.0, q_mvar=10.0)
        pp.create_line_from_parameters(
            net,
            from_bus=b0,
            to_bus=b1,
            length_km=1.0,
            r_ohm_per_km=0.01,
            x_ohm_per_km=0.10,
            c_nf_per_km=0.0,
            max_i_ka=1.0,
        )

        base_pf = solve_ac_pf_base_point_from_pandapower(
            net=net, slack_bus=b0, solver="pandapower", init="flat", lossless=True
        )

        # Set a very tight limit (below base flow)
        net.line.loc[0, "rateA"] = 1.0  # 1 MVA << actual flow

        result = check_ac_base_point_feasibility(net=net, base_pf=base_pf)

        assert not result.is_feasible
        assert result.n_constrained_violated >= 1
        assert result.worst_margin_mva < 0

    def test_feasible_base_point_passes(self, _pp, _scipy, _pypsa) -> None:
        """Network with generous limits should be flagged feasible."""
        pp = _pp
        from stability_radius.base_point.pypsa_pf import (
            solve_ac_pf_base_point_from_pandapower,
        )
        from stability_radius.radii.ac_feasibility import (
            check_ac_base_point_feasibility,
        )

        net = pp.create_empty_network(sn_mva=100.0)
        b0 = int(pp.create_bus(net, vn_kv=110.0))
        b1 = int(pp.create_bus(net, vn_kv=110.0))
        pp.create_ext_grid(net, b0, vm_pu=1.0)
        pp.create_load(net, b1, p_mw=30.0, q_mvar=5.0)
        pp.create_line_from_parameters(
            net,
            from_bus=b0,
            to_bus=b1,
            length_km=1.0,
            r_ohm_per_km=0.01,
            x_ohm_per_km=0.10,
            c_nf_per_km=0.0,
            max_i_ka=1.0,
        )

        base_pf = solve_ac_pf_base_point_from_pandapower(
            net=net, slack_bus=b0, solver="pandapower", init="flat", lossless=True
        )

        # Set a generous limit
        net.line.loc[0, "rateA"] = 9999.0

        result = check_ac_base_point_feasibility(net=net, base_pf=base_pf)

        assert result.is_feasible
        assert result.n_constrained_violated == 0


# ---------------------------------------------------------------------------
# Test for negative sigma-radius in compute_ac_sigma_radius
# ---------------------------------------------------------------------------


class TestNegativeSigmaRadius:
    def test_s0_exceeding_limit_gives_negative_r_sigma(self) -> None:
        """When S0 > c, the sigma-radius should be negative."""
        from stability_radius.radii.ac_sigma_radius import compute_ac_sigma_radius

        n_bus = 2
        h = np.array([[1.0, -1.0, 0.5, -0.5]], dtype=float)
        sigma_p = np.array([1.0, 1.0])
        sigma_q = np.array([1.0, 1.0])
        s0 = np.array([110.0])  # exceeds limit
        c = np.array([100.0])

        res = compute_ac_sigma_radius(
            h_vectors=h,
            s_limit_mva=c,
            s0_mva=s0,
            sigma_p_mw=sigma_p,
            sigma_q_mvar=sigma_q,
            balance=True,
        )

        row = res["line_0"]
        assert row["radius_ac_sigma"] < 0, (
            f"Expected negative sigma-radius, got {row['radius_ac_sigma']}"
        )

    def test_s0_below_limit_gives_positive_r_sigma(self) -> None:
        """When S0 < c, the sigma-radius should be positive."""
        from stability_radius.radii.ac_sigma_radius import compute_ac_sigma_radius

        n_bus = 2
        h = np.array([[1.0, -1.0, 0.5, -0.5]], dtype=float)
        sigma_p = np.array([1.0, 1.0])
        sigma_q = np.array([1.0, 1.0])
        s0 = np.array([90.0])
        c = np.array([100.0])

        res = compute_ac_sigma_radius(
            h_vectors=h,
            s_limit_mva=c,
            s0_mva=s0,
            sigma_p_mw=sigma_p,
            sigma_q_mvar=sigma_q,
            balance=True,
        )

        row = res["line_0"]
        assert row["radius_ac_sigma"] > 0


# ---------------------------------------------------------------------------
# Tests for multi-scale verification structure
# ---------------------------------------------------------------------------


class TestMultiScaleVerification:
    def test_verification_produces_per_scale_results(self) -> None:
        """Multi-scale verification should return per-scale sub-results."""
        from experiments.run_sigma_radius import (
            _aggregate_across_hours,
            _build_table2_rows,
            _run_worst_case_verification,
        )

        hr = _make_hourly_results(
            n_lines=1,
            line_ids=[0],
            sigma_radii=[5.0],
            l2_radii=[10.0],
            s0_values=[80.0],
            limit_values=[100.0],
        )
        agg = _aggregate_across_hours({0: hr})
        rows = _build_table2_rows(agg, top_k=1)

        mock_net = MagicMock()
        load_p = np.zeros((3, 1))
        load_q = np.zeros((3, 1))

        with (
            patch("experiments.run_sigma_radius.copy") as mock_copy,
            patch("experiments.run_sigma_radius.verify_worst_case") as mock_verify,
            patch.object(Path, "open", create=True),
            patch("experiments.run_sigma_radius.json"),
        ):
            mock_copy.deepcopy.return_value = mock_net

            mock_result = MagicMock()
            mock_result.to_dict.return_value = {
                "line_id": 0,
                "actual_s_mva": 95.0,
                "predicted_s_mva": 100.0,
                "pf_converged": True,
                "violated": False,
                "relative_error": 0.05,
            }
            mock_result.pf_converged = True
            mock_result.violated = False
            mock_verify.return_value = mock_result

            output_dir = Path("/tmp/test_multiscale")
            output_dir.mkdir(parents=True, exist_ok=True)

            results = _run_worst_case_verification(
                net=mock_net,
                agg=agg,
                table_rows=rows,
                bus_ids=[0, 1, 2],
                load_p_mw=load_p,
                load_q_mvar=load_q,
                slack_bus=0,
                lossless=True,
                fpf_cfg=None,
                scales=[0.5, 1.0, 1.5],
                output_dir=output_dir,
            )

        # Should have 3 scale results for the one feasible line
        ok_results = [r for r in results if r.get("status") == "ok"]
        assert len(ok_results) == 1
        assert len(ok_results[0]["scale_results"]) == 3
        # verify_worst_case called 3 times (3 scales)
        assert mock_verify.call_count == 3
