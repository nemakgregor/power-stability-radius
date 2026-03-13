"""Tests for entry_points.metrics_analysis â€” DataFrame, correlations, precision-at-k."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from entry_points.metrics_analysis import (
    _aggregate_bus_loads_sorted,
    _resolve_metrics_analysis_slack_bus,
    build_unified_dataframe,
    compute_precision_at_k,
    compute_rank_correlations,
)


# ---------------------------------------------------------------------------
# build_unified_dataframe
# ---------------------------------------------------------------------------


class TestBuildUnifiedDataframe:
    @staticmethod
    def _sample_results() -> dict:
        return {
            "__meta__": {"schema_version": 3},
            "line_0": {
                "ac_s_limit_mva": 100.0,
                "ac_s0_from_mva": 30.0,
                "ac_s0_to_mva": 25.0,
                "margin_ac_mva": 70.0,
                "radius_ac_l2": 5.0,
                "radius_ac_sigma": 3.0,
                "radius_ac_metric": 4.0,
                "sigma_flow_mva": 1.0,
                "overload_probability_ac": 0.01,
                "binding_end": "from",
            },
            "line_1": {
                "ac_s_limit_mva": 200.0,
                "ac_s0_from_mva": 100.0,
                "ac_s0_to_mva": 90.0,
                "margin_ac_mva": 100.0,
                "radius_ac_l2": 10.0,
                "binding_end": "to",
            },
        }

    @staticmethod
    def _sample_baselines() -> dict:
        return {
            "line_0": {
                "loading_ratio": 0.3,
                "headroom_mva": 70.0,
                "cheb_prob_upper": 0.01,
            },
            "line_1": {
                "loading_ratio": 0.5,
                "headroom_mva": 100.0,
                "cheb_prob_upper": 0.005,
            },
        }

    @staticmethod
    def _sample_mc_fracs() -> dict:
        return {"line_0": 0.02, "line_1": 0.005}

    def test_returns_dataframe_with_correct_rows(self):
        df = build_unified_dataframe(
            results=self._sample_results(),
            baselines=self._sample_baselines(),
            mc_per_line_fractions=self._sample_mc_fracs(),
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_columns_include_all_metrics(self):
        df = build_unified_dataframe(
            results=self._sample_results(),
            baselines=self._sample_baselines(),
            mc_per_line_fractions=self._sample_mc_fracs(),
        )
        expected_cols = {
            "line_key",
            "ac_s_limit_mva",
            "s0_binding_mva",
            "margin_ac_mva",
            "radius_ac_l2",
            "loading_ratio",
            "headroom_mva",
            "empirical_overload_prob",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_skips_meta_key(self):
        df = build_unified_dataframe(
            results=self._sample_results(),
            baselines={},
            mc_per_line_fractions={},
        )
        assert "__meta__" not in df["line_key"].values

    def test_missing_baselines_gives_nan(self):
        df = build_unified_dataframe(
            results=self._sample_results(),
            baselines={},
            mc_per_line_fractions={},
        )
        assert all(math.isnan(x) for x in df["loading_ratio"])

    def test_binding_end_selects_correct_s0(self):
        df = build_unified_dataframe(
            results=self._sample_results(),
            baselines={},
            mc_per_line_fractions={},
        )
        # line_0 binding_end="from" â†’ s0_binding_mva=ac_s0_from_mva=30.0
        row0 = df[df["line_key"] == "line_0"].iloc[0]
        assert row0["s0_binding_mva"] == 30.0
        # line_1 binding_end="to" â†’ s0_binding_mva=ac_s0_to_mva=90.0
        row1 = df[df["line_key"] == "line_1"].iloc[0]
        assert row1["s0_binding_mva"] == 90.0


# ---------------------------------------------------------------------------
# compute_rank_correlations
# ---------------------------------------------------------------------------


class TestComputeRankCorrelations:
    def test_perfect_positive_correlation(self):
        df = pd.DataFrame(
            {
                "loading_ratio": [0.1, 0.5, 0.9],
                "empirical_overload_prob": [0.01, 0.05, 0.1],
            }
        )
        corr = compute_rank_correlations(df, metric_columns=["loading_ratio"])
        assert len(corr) == 1
        assert corr.iloc[0]["spearman_rho"] == pytest.approx(1.0)

    def test_negated_metric_gives_positive_rho(self):
        """Radii (lower=more dangerous) are negated so rho is positive
        when they correctly rank lines."""
        df = pd.DataFrame(
            {
                "radius_ac_l2": [10.0, 5.0, 1.0],  # smaller = more dangerous
                "empirical_overload_prob": [0.01, 0.05, 0.1],  # higher = more dangerous
            }
        )
        corr = compute_rank_correlations(df, metric_columns=["radius_ac_l2"])
        rho = corr.iloc[0]["spearman_rho"]
        assert rho == pytest.approx(1.0)

    def test_fewer_than_3_rows_gives_nan(self):
        df = pd.DataFrame(
            {"loading_ratio": [0.1, 0.5], "empirical_overload_prob": [0.01, 0.05]}
        )
        corr = compute_rank_correlations(df, metric_columns=["loading_ratio"])
        assert math.isnan(corr.iloc[0]["spearman_rho"])

    def test_constant_values_give_nan(self):
        df = pd.DataFrame(
            {
                "loading_ratio": [0.5, 0.5, 0.5],
                "empirical_overload_prob": [0.01, 0.05, 0.1],
            }
        )
        corr = compute_rank_correlations(df, metric_columns=["loading_ratio"])
        assert math.isnan(corr.iloc[0]["spearman_rho"])


# ---------------------------------------------------------------------------
# compute_precision_at_k
# ---------------------------------------------------------------------------


class TestComputePrecisionAtK:
    def test_basic_precision(self):
        df = pd.DataFrame(
            {
                "loading_ratio": [0.1, 0.5, 0.9, 0.2, 0.8],
                "empirical_overload_prob": [0.01, 0.05, 0.10, 0.02, 0.08],
            }
        )
        pak = compute_precision_at_k(
            df,
            metric_columns=["loading_ratio"],
            k_values=[3],
        )
        assert len(pak) == 1
        assert pak.iloc[0]["k"] == 3
        # loading_ratio is NOT in _NEGATE_FOR_CORRELATION, so higher = more dangerous
        # top-3 by loading_ratio desc: 0.9, 0.8, 0.5 â†’ probs 0.10, 0.08, 0.05
        expected_mean = (0.10 + 0.08 + 0.05) / 3
        assert pak.iloc[0]["mean_empirical_prob"] == pytest.approx(expected_mean)

    def test_empty_dataframe_gives_nan(self):
        df = pd.DataFrame(columns=["radius_ac_l2", "empirical_overload_prob"])
        pak = compute_precision_at_k(
            df,
            metric_columns=["radius_ac_l2"],
            k_values=[3],
        )
        assert math.isnan(pak.iloc[0]["mean_empirical_prob"])

    def test_k_larger_than_data(self):
        df = pd.DataFrame(
            {
                "loading_ratio": [0.1, 0.9],
                "empirical_overload_prob": [0.01, 0.10],
            }
        )
        pak = compute_precision_at_k(
            df,
            metric_columns=["loading_ratio"],
            k_values=[10],
        )
        # Only 2 rows available, should take all 2
        assert pak.iloc[0]["k"] == 10
        expected_mean = (0.01 + 0.10) / 2
        assert pak.iloc[0]["mean_empirical_prob"] == pytest.approx(expected_mean)


def test_metrics_analysis_slack_auto_detect_uses_smallest_ext_grid_bus() -> None:
    import pandapower as pp

    net = pp.create_empty_network()
    b2 = pp.create_bus(net, vn_kv=110.0, index=2)
    b0 = pp.create_bus(net, vn_kv=110.0, index=0)
    pp.create_ext_grid(net, bus=b2)
    pp.create_ext_grid(net, bus=b0)

    assert _resolve_metrics_analysis_slack_bus(net, None) == 0


def test_metrics_analysis_slack_auto_detect_falls_back_to_first_bus() -> None:
    import pandapower as pp

    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=110.0, index=5)
    pp.create_bus(net, vn_kv=110.0, index=1)

    assert _resolve_metrics_analysis_slack_bus(net, None) == 1


def test_metrics_analysis_load_aggregation_uses_sorted_bus_order() -> None:
    net = SimpleNamespace(
        bus=pd.DataFrame(index=pd.Index([5, 1, 3], dtype=int)),
        load=pd.DataFrame(
            {
                "bus": [3, 5, 3, 1],
                "p_mw": [30.0, 50.0, 5.0, 10.0],
                "q_mvar": [3.0, 5.0, 0.5, 1.0],
            }
        ),
    )

    bus_load_p, bus_load_q = _aggregate_bus_loads_sorted(net)

    assert list(bus_load_p.index) == [1, 3, 5]
    assert list(bus_load_q.index) == [1, 3, 5]
    assert bus_load_p.to_list() == pytest.approx([10.0, 35.0, 50.0])
    assert bus_load_q.to_list() == pytest.approx([1.0, 3.5, 5.0])
