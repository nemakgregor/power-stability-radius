from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stability_radius.demos import n1_stability_demo as demo
from stability_radius.demos.n1_stability_demo import (
    _align_line_limit_proxy_with_opf_model,
    _build_comparison_text,
    _opf_constraint_summary,
    _opf_line_limit_consistency_summary,
    _plot_cost_security_tradeoff,
    _plot_multi_regime_n1_overloads,
    _resolve_output_dir,
    _solve_cost_opf,
    _total_generation_dispatch_mw,
    _update_scopf_line_limits,
)


def test_resolve_output_dir_normalizes_legacy_analysis_output_under_runs(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    out_dir = _resolve_output_dir("analysis_output/n1_demo_case118")

    assert out_dir == (
        tmp_path / "run_artifacts" / "n1_stability_demo" / "n1_demo_case118"
    ).resolve()


def test_update_scopf_line_limits_tightens_only_violating_lines() -> None:
    updated, changed = _update_scopf_line_limits(
        {1: 99.0, 2: 99.0, 3: 80.0},
        {1: 105.0, 2: 98.0, 3: 130.0},
        security_target_pct=99.0,
    )

    assert changed == [1, 3]
    assert updated[1] == pytest.approx(93.34285714285714)
    assert updated[2] == 99.0
    assert updated[3] == pytest.approx(60.92307692307692)


def test_opf_constraint_summary_and_generation_dispatch() -> None:
    nn = SimpleNamespace(
        line=pd.DataFrame({"max_loading_percent": [99.0, 95.0]}, index=[10, 11]),
        res_line=pd.DataFrame({"loading_percent": [97.5, 92.0]}, index=[10, 11]),
        trafo=pd.DataFrame({"max_loading_percent": [98.0]}, index=[7]),
        res_trafo=pd.DataFrame({"loading_percent": [96.0]}, index=[7]),
        res_gen=pd.DataFrame({"p_mw": [100.0, 25.0]}),
        res_sgen=pd.DataFrame({"p_mw": [10.0]}),
        res_ext_grid=pd.DataFrame({"p_mw": [5.0]}),
    )

    summary = _opf_constraint_summary(nn, "cost_opf")

    assert summary["max_line_loading_pct"] == 97.5
    assert summary["min_line_loading_headroom_pct"] == 1.5
    assert summary["max_trafo_loading_pct"] == 96.0
    assert summary["min_trafo_loading_headroom_pct"] == 2.0
    assert _total_generation_dispatch_mw(nn) == 140.0


def test_build_comparison_text_includes_scopf_and_proxy_headroom_note() -> None:
    regime_order = [
        ("cost_opf", "Cost OPF"),
        ("radius_opf", "Radius OPF"),
        ("scopf", "SCOPF"),
    ]
    dispatch = {
        "cost_opf": {
            "total_cost_eur_h": 100.0,
            "generation_dispatch_mw": 200.0,
            "cost_increase_pct": 0.0,
        },
        "radius_opf": {
            "total_cost_eur_h": 103.0,
            "generation_dispatch_mw": 200.0,
            "cost_increase_pct": 3.0,
        },
        "scopf": {
            "total_cost_eur_h": 107.0,
            "generation_dispatch_mw": 200.0,
            "cost_increase_pct": 7.0,
        },
    }
    constraints = {
        key: {
            "max_line_loading_pct": 98.0,
            "min_line_loading_headroom_pct": 1.0,
            "max_trafo_loading_pct": 50.0,
            "min_trafo_loading_headroom_pct": 48.0,
        }
        for key, _ in regime_order
    }
    radius = {
        key: {
            "n_constrained": 10,
            "radius_min": 1.0,
            "radius_median": 2.0,
            "radius_mean": 3.0,
            "loading_ratio_mean": 0.5,
            "loading_ratio_max": 0.9,
        }
        for key, _ in regime_order
    }
    ac_n1 = {
        key: {
            "n_lines": 10,
            "n_already_n1_infeasible": 0,
            "ac_n1_radius_min": 5.0,
            "ac_n1_radius_median": 6.0,
            "ac_n1_radius_p10": 5.5,
        }
        for key, _ in regime_order
    }
    sigma = {
        "cost_opf": {
            "sigma_radius_min": 1.0,
            "sigma_radius_median": 2.0,
            "sigma_radius_p10": 1.5,
            "max_overload_prob": 0.1,
            "mean_overload_prob": 0.01,
            "n_prob_above_1pct": 2,
            "n_prob_above_5pct": 1,
            "max_cantelli_ub": 0.2,
            "pi_system": 3.0,
            "pi_max": 0.5,
            "min_headroom_mva": -2.0,
        },
        "radius_opf": {
            "sigma_radius_min": 2.0,
            "sigma_radius_median": 3.0,
            "sigma_radius_p10": 2.5,
            "max_overload_prob": 0.05,
            "mean_overload_prob": 0.005,
            "n_prob_above_1pct": 1,
            "n_prob_above_5pct": 0,
            "max_cantelli_ub": 0.1,
            "pi_system": 2.0,
            "pi_max": 0.4,
            "min_headroom_mva": 1.0,
        },
        "scopf": {
            "sigma_radius_min": 3.0,
            "sigma_radius_median": 4.0,
            "sigma_radius_p10": 3.5,
            "max_overload_prob": 0.01,
            "mean_overload_prob": 0.001,
            "n_prob_above_1pct": 0,
            "n_prob_above_5pct": 0,
            "max_cantelli_ub": 0.05,
            "pi_system": 1.5,
            "pi_max": 0.3,
            "min_headroom_mva": 2.0,
        },
    }
    dc_n1 = {
        key: {"dc_n1_radius_min": 10.0, "dc_n1_radius_median": 12.0}
        for key, _ in regime_order
    }
    screen = {
        key: {
            "n1_pass": 10,
            "n1_fail": 0,
            "n1_diverged": 0,
            "n1_pass_rate_pct": 100.0,
            "max_overloads_in_contingency": 0,
        }
        for key, _ in regime_order
    }

    text = _build_comparison_text(
        regime_order=regime_order,
        dispatch_summaries=dispatch,
        limit_consistency_summaries={
            key: {
                "n_lines_checked": 10,
                "n_limit_mismatch": 0,
                "max_abs_limit_diff_mva": 0.0,
                "max_rel_limit_diff_pct": 0.0,
            }
            for key, _ in regime_order
        },
        constraint_summaries=constraints,
        radius_summaries=radius,
        ac_n1_radius_summaries=ac_n1,
        sigma_summaries=sigma,
        dc_n1_summaries=dc_n1,
        screen_summaries=screen,
        verify={"reason": "not checked"},
        r_target=5.0,
        sigma_p_mw=5.0,
        sigma_q_mvar=2.0,
    )

    assert "SCOPF" in text
    assert "Min headroom vs MVA proxy" in text
    assert "apparent-power branch limits" in text
    assert "post-PF current-based diagnostic" in text


class _FakeNet(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_align_line_limit_proxy_with_opf_model_overwrites_explicit_rating() -> None:
    net = _FakeNet(
        bus=pd.DataFrame({"vn_kv": [110.0]}, index=[0]),
        line=pd.DataFrame(
            {
                "from_bus": [0],
                "max_i_ka": [1.0],
                "df": [1.0],
                "parallel": [1.0],
                "max_loading_percent": [100.0],
                "rateA": [50.0],
            },
            index=[7],
        ),
    )

    summary = _align_line_limit_proxy_with_opf_model(net)

    expected_nominal = np.sqrt(3.0) * 110.0 * 1.0
    assert net.line.at[7, "rateA"] == pytest.approx(expected_nominal)
    assert net.line.at[7, "rate_a_mva"] == pytest.approx(expected_nominal)
    assert summary["n_lines_checked"] == 1
    assert summary["n_lines_aligned"] == 1


def test_opf_line_limit_consistency_summary_reports_zero_mismatch_after_alignment() -> None:
    net = _FakeNet(
        bus=pd.DataFrame({"vn_kv": [110.0]}, index=[0]),
        line=pd.DataFrame(
            {
                "from_bus": [0],
                "max_i_ka": [1.0],
                "df": [0.5],
                "parallel": [2.0],
                "max_loading_percent": [99.0],
                "rateA": [25.0],
                "rate_a_mva": [25.0],
            },
            index=[5],
        ),
        _ppc_opf={"branch": np.array([[9999.0, 9999.0, 9999.0]])},
    )

    _align_line_limit_proxy_with_opf_model(net)
    summary = _opf_line_limit_consistency_summary(net, "cost_opf")

    assert summary["n_lines_checked"] == 1
    assert summary["n_limit_mismatch"] == 0
    assert summary["max_abs_limit_diff_mva"] == pytest.approx(0.0)


def test_solve_cost_opf_accepts_pf_replay_with_current_gap(monkeypatch) -> None:
    calls = {"run_cost_opf": 0, "validate": 0}

    monkeypatch.setattr(demo, "_prepare_cost_opf_network", lambda nn: None)
    monkeypatch.setattr(demo, "_apply_loading_limits", lambda nn, **kwargs: None)
    monkeypatch.setattr(demo, "_set_default_voltage_bounds", lambda nn: None)
    monkeypatch.setattr(demo, "_add_matpower_costs", lambda nn, input_path: 1)
    monkeypatch.setattr(demo, "_extract_pypsa_result_from_pp", lambda nn, line_indices: "base_pf")

    def fake_run_cost_opf(nn, label: str = "cost_opf") -> float:
        calls["run_cost_opf"] += 1
        return 123.4

    def fake_validate_opf_with_pf(nn, label: str) -> tuple[bool, float]:
        calls["validate"] += 1
        return True, 2.5

    monkeypatch.setattr(demo, "_run_cost_opf", fake_run_cost_opf)
    monkeypatch.setattr(demo, "_validate_opf_with_pf", fake_validate_opf_with_pf)

    nn, base_pf, total_cost = _solve_cost_opf(
        SimpleNamespace(),
        [1, 2],
        input_path="case118.m",
        max_loading_percent=99.0,
        label="cost_opf",
    )

    assert isinstance(nn, SimpleNamespace)
    assert base_pf == "base_pf"
    assert total_cost == pytest.approx(123.4)
    assert calls["run_cost_opf"] == 1
    assert calls["validate"] == 1


def test_plot_multi_regime_n1_overloads_writes_png(tmp_path) -> None:
    output_path = tmp_path / "n1_overloads.png"
    regime_records = {
        "cost_opf": (
            "Cost OPF",
            [
                {
                    "contingency_line": 1,
                    "pf_converged": True,
                    "n1_feasible": False,
                    "n_overloads": 4,
                    "max_loading_percent": 121.0,
                },
                {
                    "contingency_line": 2,
                    "pf_converged": True,
                    "n1_feasible": True,
                    "n_overloads": 0,
                    "max_loading_percent": 98.0,
                },
            ],
        ),
        "radius_opf": (
            "Radius OPF",
            [
                {
                    "contingency_line": 1,
                    "pf_converged": True,
                    "n1_feasible": False,
                    "n_overloads": 2,
                    "max_loading_percent": 108.0,
                },
                {
                    "contingency_line": 2,
                    "pf_converged": True,
                    "n1_feasible": True,
                    "n_overloads": 0,
                    "max_loading_percent": 94.0,
                },
            ],
        ),
        "scopf": (
            "SCOPF",
            [
                {
                    "contingency_line": 1,
                    "pf_converged": True,
                    "n1_feasible": True,
                    "n_overloads": 0,
                    "max_loading_percent": 99.0,
                },
                {
                    "contingency_line": 2,
                    "pf_converged": True,
                    "n1_feasible": True,
                    "n_overloads": 0,
                    "max_loading_percent": 91.0,
                },
            ],
        ),
    }

    _plot_multi_regime_n1_overloads(regime_records, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_cost_security_tradeoff_writes_png(tmp_path) -> None:
    output_path = tmp_path / "tradeoff.png"
    regime_order = [
        ("cost_opf", "Cost OPF"),
        ("radius_opf", "Radius OPF"),
        ("scopf", "SCOPF"),
    ]

    _plot_cost_security_tradeoff(
        regime_order=regime_order,
        dispatch_summaries={
            "cost_opf": {"cost_increase_pct": 0.0},
            "radius_opf": {"cost_increase_pct": 2.0},
            "scopf": {"cost_increase_pct": 6.0},
        },
        screen_summaries={
            "cost_opf": {"n1_pass_rate_pct": 95.0},
            "radius_opf": {"n1_pass_rate_pct": 98.0},
            "scopf": {"n1_pass_rate_pct": 100.0},
        },
        ac_n1_radius_summaries={
            "cost_opf": {"ac_n1_radius_min": 5.0},
            "radius_opf": {"ac_n1_radius_min": 12.0},
            "scopf": {"ac_n1_radius_min": 14.0},
        },
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
