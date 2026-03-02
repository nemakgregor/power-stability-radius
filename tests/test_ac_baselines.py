from __future__ import annotations

import math

import pytest

from stability_radius.metrics.ac_baselines import (
    cantelli_upper_bound,
    compute_baseline_metrics,
    headroom_mva,
    loading_ratio,
)


# ------------------------------------------------------------------ #
# loading_ratio
# ------------------------------------------------------------------ #


class TestLoadingRatio:
    def test_normal_case(self) -> None:
        assert loading_ratio(s0_mva=80.0, s_limit_mva=100.0) == pytest.approx(0.8)

    def test_fully_loaded(self) -> None:
        assert loading_ratio(s0_mva=100.0, s_limit_mva=100.0) == pytest.approx(1.0)

    def test_overloaded(self) -> None:
        assert loading_ratio(s0_mva=120.0, s_limit_mva=100.0) == pytest.approx(1.2)

    def test_zero_flow(self) -> None:
        assert loading_ratio(s0_mva=0.0, s_limit_mva=100.0) == pytest.approx(0.0)

    def test_zero_limit_nonzero_flow(self) -> None:
        assert loading_ratio(s0_mva=5.0, s_limit_mva=0.0) == float("inf")

    def test_zero_limit_zero_flow(self) -> None:
        assert math.isnan(loading_ratio(s0_mva=0.0, s_limit_mva=0.0))


# ------------------------------------------------------------------ #
# headroom_mva
# ------------------------------------------------------------------ #


class TestHeadroomMva:
    def test_normal_case(self) -> None:
        assert headroom_mva(s0_mva=80.0, s_limit_mva=100.0) == pytest.approx(20.0)

    def test_binding(self) -> None:
        assert headroom_mva(s0_mva=100.0, s_limit_mva=100.0) == pytest.approx(0.0)

    def test_negative_when_overloaded(self) -> None:
        assert headroom_mva(s0_mva=110.0, s_limit_mva=100.0) == pytest.approx(-10.0)


# ------------------------------------------------------------------ #
# cantelli_upper_bound
# ------------------------------------------------------------------ #


class TestCantelliUpperBound:
    def test_equal_sigma_and_headroom(self) -> None:
        # sigma=1, headroom=1 => 1/(1+1) = 0.5
        assert cantelli_upper_bound(headroom=1.0, sigma_flow_mva=1.0) == pytest.approx(
            0.5
        )

    def test_large_headroom(self) -> None:
        # sigma=1, headroom=10 => 1/(1+100) ~ 0.0099
        result = cantelli_upper_bound(headroom=10.0, sigma_flow_mva=1.0)
        assert result == pytest.approx(1.0 / 101.0)

    def test_large_sigma(self) -> None:
        # sigma=10, headroom=1 => 100/(100+1) ~ 0.9901
        result = cantelli_upper_bound(headroom=1.0, sigma_flow_mva=10.0)
        assert result == pytest.approx(100.0 / 101.0)

    def test_zero_headroom_returns_one(self) -> None:
        assert cantelli_upper_bound(headroom=0.0, sigma_flow_mva=1.0) == 1.0

    def test_negative_headroom_returns_one(self) -> None:
        assert cantelli_upper_bound(headroom=-5.0, sigma_flow_mva=1.0) == 1.0

    def test_zero_sigma_returns_zero(self) -> None:
        assert cantelli_upper_bound(headroom=5.0, sigma_flow_mva=0.0) == 0.0

    def test_nan_sigma_returns_zero(self) -> None:
        assert cantelli_upper_bound(headroom=5.0, sigma_flow_mva=float("nan")) == 0.0


# ------------------------------------------------------------------ #
# compute_baseline_metrics (integration)
# ------------------------------------------------------------------ #


class TestComputeBaselineMetrics:
    def test_from_results_dict(self) -> None:
        results: dict = {
            "__meta__": {"schema_version": 2},
            "line_0": {
                "ac_s_limit_mva": 100.0,
                "ac_s0_from_mva": 80.0,
                "ac_s0_to_mva": 75.0,
                "binding_end": "from",
                "margin_ac_mva": 20.0,
                "sigma_flow_mva": 5.0,
            },
        }
        baselines = compute_baseline_metrics(results)
        assert "line_0" in baselines
        b = baselines["line_0"]
        assert b["loading_ratio"] == pytest.approx(0.8)
        assert b["headroom_mva"] == pytest.approx(20.0)
        # cheb = 25 / (25 + 400) = 25/425
        assert b["cheb_prob_upper"] == pytest.approx(25.0 / 425.0)

    def test_binding_end_to(self) -> None:
        results: dict = {
            "line_5": {
                "ac_s_limit_mva": 50.0,
                "ac_s0_from_mva": 30.0,
                "ac_s0_to_mva": 40.0,
                "binding_end": "to",
                "margin_ac_mva": 10.0,
                "sigma_flow_mva": 2.0,
            },
        }
        baselines = compute_baseline_metrics(results)
        assert "line_5" in baselines
        b = baselines["line_5"]
        # s0 = 40 (to end), limit = 50 => loading = 0.8
        assert b["loading_ratio"] == pytest.approx(0.8)
        assert b["headroom_mva"] == pytest.approx(10.0)

    def test_skips_non_line_keys(self) -> None:
        results: dict = {
            "__meta__": {},
            "_h_vectors": {},
            "not_a_line": {},
        }
        baselines = compute_baseline_metrics(results)
        assert len(baselines) == 0

    def test_missing_sigma_propagates_nan(self) -> None:
        results: dict = {
            "line_0": {
                "ac_s_limit_mva": 100.0,
                "ac_s0_from_mva": 80.0,
                "ac_s0_to_mva": 75.0,
                "binding_end": "from",
                "margin_ac_mva": 20.0,
                # sigma_flow_mva is missing
            },
        }
        baselines = compute_baseline_metrics(results)
        b = baselines["line_0"]
        assert b["loading_ratio"] == pytest.approx(0.8)
        # cantelli with NaN sigma returns 0.0 (safe fallback)
        assert b["cheb_prob_upper"] == 0.0
