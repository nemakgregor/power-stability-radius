from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


def test_report_formats_na_for_nan_coverage():
    # Module import requires pandapower in this repo setup.
    pytest.importorskip("pandapower")

    from verification.generate_report import _case_section_md

    mc = {
        "status": "no_feasible_samples",
        "coverage_percent": float("nan"),
        "total_feasible_in_box": 0,
        "feasible_in_ball": 0,
        "n_samples": 50000,
        "seed": 0,
        "chunk_size": 256,
        "box_lo": -10.0,
        "box_hi": 10.0,
        "min_r": 0.0,
        "max_r": 5.0,
        "feasible_rate_in_box_percent": 0.0,
        "coverage_ci95_low_percent": float("nan"),
        "coverage_ci95_high_percent": float("nan"),
    }

    md = _case_section_md(
        case="case1354_pegase",
        status="generated_no_feasible_samples",
        mc=mc,
        top_risky=None,
        top_match=None,
        n1_match=None,
        time_sec=1.0,
        known_pairs=None,
    )

    # Must not leak "nan%" into literature comparison or anywhere else.
    assert "nan%" not in md.lower()
    assert "coverage: n/a" in md.lower() or "coverage % (mc): n/a" in md.lower()


def test_monte_carlo_returns_no_feasible_samples_on_zero_limits(tmp_path: Path):
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

    results = {
        "__meta__": {"input_path": str(mfile), "slack_bus": 0},
        f"line_{lid}": {
            "flow0_mw": 0.0,
            "p_limit_mw_est": 0.0,  # makes feasibility measure-zero for continuous sampling
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

    assert stats["status"] == "no_feasible_samples"
    assert math.isnan(float(stats["coverage_percent"]))
    assert int(stats["total_feasible_in_box"]) == 0
    assert int(stats["feasible_in_ball"]) == 0

    # New scaling: box half-width is 2*max_r/sqrt(n_bus)
    n_bus = int(stats["n_bus"])
    expected_half = 2.0 * 1.0 / math.sqrt(float(n_bus))
    assert float(stats["box_lo"]) == pytest.approx(-expected_half)
    assert float(stats["box_hi"]) == pytest.approx(expected_half)
