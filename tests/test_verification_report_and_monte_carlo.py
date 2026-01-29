from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


def test_report_formats_na_for_nan_metrics():
    # Module import requires pandapower in this repo setup.
    pytest.importorskip("pandapower")

    from verification.generate_report import _case_section_md

    mc = {
        "status": "ok",
        "n_samples": 50000,
        "gaussian_feasible_samples": 0,
        "gaussian_feasible_percent": float("nan"),
        "gaussian_ball_mass_analytic_percent": float("nan"),
        "base_point_feasible": True,
        "base_point_violated_lines": 0,
        "base_point_max_violation_mw": 0.0,
        "certificate_status": "skipped",
        "certificate_violation_samples": 0,
        "certificate_max_violation_mw": float("nan"),
    }

    md = _case_section_md(
        case="case1354_pegase",
        status="generated_ok",
        mc=mc,
        top_risky=None,
        top_match=None,
        n1_match=None,
        time_sec=1.0,
        known_pairs=None,
    )

    # Must not leak "nan%" anywhere.
    assert "nan%" not in md.lower()
    assert "n/a" in md.lower()


def test_monte_carlo_runs_on_zero_limits_and_reports_zero_feasible(tmp_path: Path):
    pytest.importorskip("pandapower")
    pytest.importorskip("scipy")

    from stability_radius.parsers.matpower import load_network
    from verification.monte_carlo import estimate_coverage_percent

    # Minimal 2-bus MATPOWER case.
    case_text = """function mpc = case2
mpc.version = '2';
mpc.baseMVA = 100;

mpc.bus = [
1 3 0 0 0 0 1 1.00 0 110 1 1.05 0.95;
2 1 0 0 0 0 1 1.00 0 110 1 1.05 0.95;
];

mpc.gen = [
1 0 0 10 -10 1.00 100 1 50 0;
];

mpc.branch = [
1 2 0.01 0.05 0 100 100 100 0 0 1 -360 360;
];
"""
    mfile = tmp_path / "case2.m"
    mfile.write_text(case_text, encoding="utf-8")

    # Determine pandapower line index to build matching results.json.
    net = load_network(mfile)
    assert len(net.line) >= 1
    lid = int(sorted(net.line.index)[0])

    # Intentionally inconsistent radius (radius_l2=1 with limit=0) to force certificate violations,
    # but the Monte Carlo routine must still run and return deterministic keys.
    results = {
        "__meta__": {"input_path": str(mfile), "slack_bus": 0, "inj_std_mw": 1.0},
        f"line_{lid}": {
            "flow0_mw": 0.0,
            "p_limit_mw_est": 0.0,  # feasibility becomes measure-zero for continuous distributions
            "radius_l2": 1.0,
        },
    }
    rfile = tmp_path / "results.json"
    rfile.write_text(json.dumps(results), encoding="utf-8")

    stats = estimate_coverage_percent(
        results_path=rfile,
        input_case_path=mfile,
        slack_bus=0,
        n_samples=200,  # small test
        seed=0,
        chunk_size=50,
    )

    assert stats["status"] in {
        "ok",
        "certificate_violations_found",
        "base_point_infeasible",
    }
    assert int(stats["n_samples"]) == 200
    assert int(stats["gaussian_feasible_samples"]) == 0
    assert float(stats["gaussian_feasible_percent"]) == pytest.approx(0.0)

    # Certificate check should detect violations for this artificial (wrong) radius.
    assert stats["certificate_status"] in {
        "violations_found",
        "base_point_infeasible",
        "skipped",
    }
