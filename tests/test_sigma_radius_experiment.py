from __future__ import annotations

"""
Tests for Experiment 2 (run_sigma_radius) helper functions.

Covers:
- Injection sigma computation from hourly OPF results
- Average-point result dict building
- Table 2 row building from single-point results
- Scatter plot filtering for log-log axes
- Worst-case verification skipping for negative r_L2
- Monte Carlo validation with tightest-feasible-line selection
- Validation check feasibility summary
- CSV export and h-vector NPZ save
"""

import csv
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_avg_result(
    *,
    n_lines: int = 5,
    n_bus: int = 3,
    line_ids: list[int] | None = None,
    sigma_radii: list[float] | None = None,
    l2_radii: list[float] | None = None,
    s0_values: list[float] | None = None,
    limit_values: list[float] | None = None,
) -> dict:
    """Build a synthetic average-point result dict matching _compute_at_average_point() output."""
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


def _make_res(
    sigma_radii: list[float],
    l2_radii: list[float] | None = None,
    s0_values: list[float] | None = None,
    limit_values: list[float] | None = None,
    n_bus: int = 3,
) -> dict:
    """Build a result dict via _build_result_dict from synthetic avg_result."""
    from experiments.run_sigma_radius import _build_result_dict

    if l2_radii is None:
        l2_radii = [10.0] * len(sigma_radii)
    n = len(sigma_radii)
    avg = _make_avg_result(
        n_lines=n,
        n_bus=n_bus,
        line_ids=list(range(n)),
        sigma_radii=sigma_radii,
        l2_radii=l2_radii,
        s0_values=s0_values,
        limit_values=limit_values,
    )
    return _build_result_dict(avg)


# ---------------------------------------------------------------------------
# Tests for _compute_injection_sigma
# ---------------------------------------------------------------------------


class TestComputeInjectionSigma:
    def test_sigma_from_constant_injections_is_zero(self) -> None:
        """Constant bus injections across hours should give zero sigma."""
        from experiments.run_sigma_radius import _compute_injection_sigma

        hourly = {
            0: {"bus_p_mw": np.array([10.0, 20.0, 30.0]), "bus_q_mvar": np.array([1.0, 2.0, 3.0])},
            1: {"bus_p_mw": np.array([10.0, 20.0, 30.0]), "bus_q_mvar": np.array([1.0, 2.0, 3.0])},
            2: {"bus_p_mw": np.array([10.0, 20.0, 30.0]), "bus_q_mvar": np.array([1.0, 2.0, 3.0])},
        }
        sigma_p, sigma_q = _compute_injection_sigma(hourly, n_bus=3)
        np.testing.assert_allclose(sigma_p, 0.0, atol=1e-12)
        np.testing.assert_allclose(sigma_q, 0.0, atol=1e-12)

    def test_sigma_from_varying_injections(self) -> None:
        """Variable bus injections should give nonzero sigma."""
        from experiments.run_sigma_radius import _compute_injection_sigma

        hourly = {
            0: {"bus_p_mw": np.array([10.0, 20.0]), "bus_q_mvar": np.array([1.0, 2.0])},
            1: {"bus_p_mw": np.array([12.0, 22.0]), "bus_q_mvar": np.array([1.5, 2.5])},
            2: {"bus_p_mw": np.array([8.0, 18.0]), "bus_q_mvar": np.array([0.5, 1.5])},
        }
        sigma_p, sigma_q = _compute_injection_sigma(hourly, n_bus=2)
        assert sigma_p.shape == (2,)
        assert sigma_q.shape == (2,)
        assert np.all(sigma_p > 0)
        assert np.all(sigma_q > 0)

    def test_fallback_to_power_factor_when_no_q(self) -> None:
        """When bus_q_mvar is None, sigma_q = sigma_p * tan(arccos(pf))."""
        from experiments.run_sigma_radius import _compute_injection_sigma

        hourly = {
            0: {"bus_p_mw": np.array([10.0, 20.0]), "bus_q_mvar": None},
            1: {"bus_p_mw": np.array([14.0, 24.0]), "bus_q_mvar": None},
        }
        pf = 0.9
        sigma_p, sigma_q = _compute_injection_sigma(hourly, n_bus=2, power_factor=pf)
        tan_phi = math.tan(math.acos(pf))
        np.testing.assert_allclose(sigma_q, sigma_p * tan_phi, rtol=1e-10)

    def test_empty_results_returns_zeros(self) -> None:
        """When no bus_p_mw data is available, return zeros."""
        from experiments.run_sigma_radius import _compute_injection_sigma

        hourly = {
            0: {"bus_p_mw": None, "bus_q_mvar": None},
        }
        sigma_p, sigma_q = _compute_injection_sigma(hourly, n_bus=3)
        np.testing.assert_allclose(sigma_p, 0.0)
        np.testing.assert_allclose(sigma_q, 0.0)


# ---------------------------------------------------------------------------
# Tests for _build_result_dict
# ---------------------------------------------------------------------------


class TestBuildResultDict:
    def test_extracts_sigma_radius_per_line(self) -> None:
        from experiments.run_sigma_radius import _build_result_dict

        avg = _make_avg_result(n_lines=3, sigma_radii=[2.0, -1.0, 5.0])
        res = _build_result_dict(avg)

        assert "line_0" in res["sigma_radius"]
        assert res["sigma_radius"]["line_0"] == pytest.approx(2.0)
        assert res["sigma_radius"]["line_1"] == pytest.approx(-1.0)
        assert res["sigma_radius"]["line_2"] == pytest.approx(5.0)

    def test_base_infeasible_flag_for_negative(self) -> None:
        from experiments.run_sigma_radius import _build_result_dict

        avg = _make_avg_result(n_lines=3, sigma_radii=[-2.0, 3.0, 5.0])
        res = _build_result_dict(avg)

        assert res["base_infeasible"]["line_0"] is True
        assert res["base_infeasible"]["line_1"] is False

    def test_nan_sigma_excluded(self) -> None:
        from experiments.run_sigma_radius import _build_result_dict

        avg = _make_avg_result(n_lines=3, sigma_radii=[float("nan"), 3.0, 5.0])
        res = _build_result_dict(avg)

        assert "line_0" not in res["sigma_radius"]
        assert "line_1" in res["sigma_radius"]


# ---------------------------------------------------------------------------
# Tests for _build_table2_rows
# ---------------------------------------------------------------------------


class TestBuildTable2Rows:
    def test_top_k_limits_output(self) -> None:
        from experiments.run_sigma_radius import _build_table2_rows

        res = _make_res([1.0, 2.0, 3.0, 4.0, 5.0])
        rows = _build_table2_rows(res, top_k=3)
        assert len(rows) == 3

    def test_rows_sorted_ascending_by_r_sigma(self) -> None:
        from experiments.run_sigma_radius import _build_table2_rows

        res = _make_res([5.0, 1.0, 3.0, 2.0, 4.0])
        rows = _build_table2_rows(res, top_k=5)
        r_values = [r["r_sigma"] for r in rows]
        assert r_values == sorted(r_values)

    def test_negative_r_sigma_lines_flagged_infeasible(self) -> None:
        from experiments.run_sigma_radius import _build_table2_rows

        res = _make_res([-2.0, 1.0, 3.0])
        rows = _build_table2_rows(res, top_k=3)
        assert rows[0]["base_infeasible"] is True
        assert rows[0]["r_sigma"] == pytest.approx(-2.0)
        assert rows[1]["base_infeasible"] is False
        assert rows[2]["base_infeasible"] is False

    def test_mc_and_verified_fields_are_none_initially(self) -> None:
        from experiments.run_sigma_radius import _build_table2_rows

        res = _make_res([1.0, 2.0])
        rows = _build_table2_rows(res, top_k=2)
        for row in rows:
            assert row["mc_violation_rate"] is None
            assert row["verified"] is None

    def test_margin_computed_correctly(self) -> None:
        from experiments.run_sigma_radius import _build_table2_rows

        res = _make_res([5.0], s0_values=[80.0], limit_values=[100.0])
        rows = _build_table2_rows(res, top_k=1)
        assert rows[0]["margin_mva"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Tests for verification skipping with negative r_L2
# ---------------------------------------------------------------------------


class TestWorstCaseVerificationSkipsInfeasible:
    def test_negative_r_l2_lines_are_skipped(self) -> None:
        """Lines with r_L2 <= 0 should be skipped in verification."""
        from experiments.run_sigma_radius import (
            _build_table2_rows,
            _run_worst_case_verification,
        )

        res = _make_res(
            sigma_radii=[-2.0, 3.0, 5.0],
            l2_radii=[-1.5, 8.0, 12.0],
            s0_values=[105.0, 50.0, 40.0],
            limit_values=[100.0, 100.0, 100.0],
        )
        rows = _build_table2_rows(res, top_k=3)

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

            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"mock": True}
            mock_result.pf_converged = True
            mock_result.violated = True
            mock_verify.return_value = mock_result

            output_dir.mkdir(parents=True, exist_ok=True)

            results = _run_worst_case_verification(
                net=mock_net,
                res=res,
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
        assert mock_verify.call_count == 2


# ---------------------------------------------------------------------------
# Tests for MC validation with feasible-line selection
# ---------------------------------------------------------------------------


class TestMonteCarloFeasibleLineSelection:
    def test_selects_tightest_positive_r_sigma(self) -> None:
        """MC should select the tightest r_sigma > 0 line."""
        from experiments.run_sigma_radius import _run_monte_carlo_validation

        table_rows = [
            {"line_key": "line_0", "r_sigma": -3.0, "base_infeasible": True},
            {"line_key": "line_1", "r_sigma": -1.0, "base_infeasible": True},
            {"line_key": "line_2", "r_sigma": 2.0, "base_infeasible": False},
            {"line_key": "line_3", "r_sigma": 8.0, "base_infeasible": False},
        ]

        mock_net = MagicMock()
        load_p = np.zeros((3, 10))
        load_q = np.zeros((3, 10))
        sigma_p = np.ones(3)
        sigma_q = np.ones(3)

        with (
            patch("experiments.run_sigma_radius.copy") as mock_copy,
            patch("experiments.run_sigma_radius.run_ac_monte_carlo_sigma") as mock_mc,
            patch("experiments.run_sigma_radius._set_loads_to_average"),
            patch.object(Path, "open", create=True),
            patch("experiments.run_sigma_radius.json"),
        ):
            mock_copy.deepcopy.return_value = mock_net

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
                res={},
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
            mc_call_kwargs = mock_mc.call_args[1]
            assert mc_call_kwargs["r_sigma"] == pytest.approx(2.0)

    def test_all_infeasible_uses_inf_for_ball(self) -> None:
        """When all lines are infeasible, r_sigma_for_ball should be inf."""
        from experiments.run_sigma_radius import _run_monte_carlo_validation

        table_rows = [
            {"line_key": "line_0", "r_sigma": -3.0, "base_infeasible": True},
            {"line_key": "line_1", "r_sigma": -1.0, "base_infeasible": True},
        ]

        mock_net = MagicMock()
        load_p = np.zeros((3, 10))
        load_q = np.zeros((3, 10))
        sigma_p = np.ones(3)
        sigma_q = np.ones(3)

        with (
            patch("experiments.run_sigma_radius.copy") as mock_copy,
            patch("experiments.run_sigma_radius.run_ac_monte_carlo_sigma") as mock_mc,
            patch("experiments.run_sigma_radius._set_loads_to_average"),
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
                res={},
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
            mc_call_kwargs = mock_mc.call_args[1]
            assert mc_call_kwargs["r_sigma"] == float("inf")


# ---------------------------------------------------------------------------
# Tests for scatter plot filtering
# ---------------------------------------------------------------------------


class TestScatterPlotFiltering:
    def test_negative_r_sigma_excluded_from_scatter(self) -> None:
        """Lines with r_sigma <= 0 or r_L2 <= 0 must not appear on log-log scatter."""
        res = _make_res(
            sigma_radii=[-2.0, 3.0, 5.0, -0.5, 7.0],
            l2_radii=[-1.0, 8.0, 12.0, 5.0, 15.0],
        )

        line_keys = sorted(res["sigma_radius"].keys())
        included = []
        excluded = []
        for lk in line_keys:
            r_sig = res["sigma_radius"].get(lk, float("nan"))
            r_l2 = res["ac_l2_radius"].get(lk, float("nan"))
            if np.isfinite(r_sig) and np.isfinite(r_l2) and r_sig > 0 and r_l2 > 0:
                included.append(lk)
            else:
                excluded.append(lk)

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
        from experiments.run_sigma_radius import (
            _build_table2_rows,
            _run_validation_checks,
        )

        avg_result = _make_avg_result(
            n_lines=4,
            sigma_radii=[-2.0, -0.5, 3.0, 5.0],
            s0_values=[110.0, 105.0, 80.0, 60.0],
            limit_values=[100.0, 100.0, 100.0, 100.0],
        )
        from experiments.run_sigma_radius import _build_result_dict
        res = _build_result_dict(avg_result)
        rows = _build_table2_rows(res, top_k=4)

        output_dir = Path("/tmp/test_validation")
        output_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(Path, "open", create=True),
            patch("experiments.run_sigma_radius.json"),
        ):
            checks = _run_validation_checks(
                res=res,
                avg_result=avg_result,
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
        from experiments.run_sigma_radius import (
            _build_table2_rows,
            _run_validation_checks,
        )

        avg_result = _make_avg_result(n_lines=1, line_ids=[0], sigma_radii=[5.0])
        from experiments.run_sigma_radius import _build_result_dict
        res = _build_result_dict(avg_result)
        rows = _build_table2_rows(res, top_k=1)

        output_dir = Path("/tmp/test_balance")
        output_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(Path, "open", create=True),
            patch("experiments.run_sigma_radius.json"),
        ):
            checks = _run_validation_checks(
                res=res,
                avg_result=avg_result,
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
        from experiments.run_sigma_radius import (
            _build_table2_rows,
            _export_table2_csv,
        )

        res = _make_res([-1.0, 3.0])
        rows = _build_table2_rows(res, top_k=2)
        _export_table2_csv(rows, tmp_path)

        csv_path = tmp_path / "table2_sigma_radius.csv"
        assert csv_path.exists()

        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            assert "base_infeasible" in fieldnames
            csv_rows = list(reader)

        assert len(csv_rows) == 2
        assert csv_rows[0]["base_infeasible"] == "True"
        assert csv_rows[1]["base_infeasible"] == "False"


# ---------------------------------------------------------------------------
# Tests for h-vector NPZ save
# ---------------------------------------------------------------------------


class TestSaveHvectorsNPZ:
    def test_hvectors_saved_and_loadable(self, tmp_path: Path) -> None:
        from experiments.run_sigma_radius import _save_hvectors_npz

        res = _make_res([5.0, 3.0, 7.0])
        _save_hvectors_npz(res, output_dir=tmp_path)

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
            net, from_bus=b0, to_bus=b1, length_km=1.0,
            r_ohm_per_km=0.01, x_ohm_per_km=0.10, c_nf_per_km=0.0, max_i_ka=1.0,
        )

        base_pf = solve_ac_pf_base_point_from_pandapower(
            net=net, slack_bus=b0, solver="pandapower", init="flat", lossless=True
        )

        net.line.loc[0, "rateA"] = 1.0
        result = check_ac_base_point_feasibility(net=net, base_pf=base_pf)

        assert not result.is_feasible
        assert result.n_constrained_violated >= 1
        assert result.worst_margin_mva < 0

    def test_feasible_base_point_passes(self, _pp, _scipy, _pypsa) -> None:
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
            net, from_bus=b0, to_bus=b1, length_km=1.0,
            r_ohm_per_km=0.01, x_ohm_per_km=0.10, c_nf_per_km=0.0, max_i_ka=1.0,
        )

        base_pf = solve_ac_pf_base_point_from_pandapower(
            net=net, slack_bus=b0, solver="pandapower", init="flat", lossless=True
        )

        net.line.loc[0, "rateA"] = 9999.0
        result = check_ac_base_point_feasibility(net=net, base_pf=base_pf)

        assert result.is_feasible
        assert result.n_constrained_violated == 0


# ---------------------------------------------------------------------------
# Test for negative sigma-radius in compute_ac_sigma_radius
# ---------------------------------------------------------------------------


class TestNegativeSigmaRadius:
    def test_s0_exceeding_limit_gives_negative_r_sigma(self) -> None:
        from stability_radius.radii.ac_sigma_radius import compute_ac_sigma_radius

        h = np.array([[1.0, -1.0, 0.5, -0.5]], dtype=float)
        sigma_p = np.array([1.0, 1.0])
        sigma_q = np.array([1.0, 1.0])
        s0 = np.array([110.0])
        c = np.array([100.0])

        res = compute_ac_sigma_radius(
            h_vectors=h, s_limit_mva=c, s0_mva=s0,
            sigma_p_mw=sigma_p, sigma_q_mvar=sigma_q, balance=True,
        )
        assert res["line_0"]["radius_ac_sigma"] < 0

    def test_s0_below_limit_gives_positive_r_sigma(self) -> None:
        from stability_radius.radii.ac_sigma_radius import compute_ac_sigma_radius

        h = np.array([[1.0, -1.0, 0.5, -0.5]], dtype=float)
        sigma_p = np.array([1.0, 1.0])
        sigma_q = np.array([1.0, 1.0])
        s0 = np.array([90.0])
        c = np.array([100.0])

        res = compute_ac_sigma_radius(
            h_vectors=h, s_limit_mva=c, s0_mva=s0,
            sigma_p_mw=sigma_p, sigma_q_mvar=sigma_q, balance=True,
        )
        assert res["line_0"]["radius_ac_sigma"] > 0


# ---------------------------------------------------------------------------
# Tests for multi-scale verification structure
# ---------------------------------------------------------------------------


class TestMultiScaleVerification:
    def test_verification_produces_per_scale_results(self) -> None:
        from experiments.run_sigma_radius import (
            _build_table2_rows,
            _run_worst_case_verification,
        )

        res = _make_res(
            sigma_radii=[5.0],
            l2_radii=[10.0],
            s0_values=[80.0],
            limit_values=[100.0],
        )
        rows = _build_table2_rows(res, top_k=1)

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
                "line_id": 0, "actual_s_mva": 95.0, "predicted_s_mva": 100.0,
                "pf_converged": True, "violated": False, "relative_error": 0.05,
            }
            mock_result.pf_converged = True
            mock_result.violated = False
            mock_verify.return_value = mock_result

            output_dir = Path("/tmp/test_multiscale")
            output_dir.mkdir(parents=True, exist_ok=True)

            results = _run_worst_case_verification(
                net=mock_net,
                res=res,
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

        ok_results = [r for r in results if r.get("status") == "ok"]
        assert len(ok_results) == 1
        assert len(ok_results[0]["scale_results"]) == 3
        assert mock_verify.call_count == 3
