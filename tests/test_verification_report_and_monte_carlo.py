from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_report_formats_na_for_nan_metrics() -> None:
    from stability_radius.verification.generate_report import _case_card_md
    from stability_radius.verification.types import (
        BASE_OK,
        PROB_OK,
        RADIUS_OK,
        SOUND_PASS,
        BasePointCheck,
        OverallCheck,
        ProbabilisticCheck,
        RadiusCheck,
        SoundnessCheck,
        VerificationInputs,
        VerificationResult,
    )

    vr = VerificationResult(
        schema_version=1,
        inputs=VerificationInputs(
            case_id="case1354_pegase",
            results_path="/abs/results.json",
            input_case_path="/abs/case.m",
            slack_bus=0,
            n_bus=10,
            n_line=20,
            dim_balance=9,
            n_samples=50000,
            seed=0,
            chunk_size=256,
            sigma_mw=1.0,
        ),
        base_point=BasePointCheck(
            status=BASE_OK, violated_lines=0, max_violation_mw=0.0
        ),
        radius=RadiusCheck(
            status=RADIUS_OK,
            r_star=1.0,
            argmin_line_pos=0,
            argmin_line_idx=0,
            min_margin_mw=1.0,
            argmin_margin_mw=1.0,
            argmin_norm_g=1.0,
        ),
        soundness=SoundnessCheck(
            status=SOUND_PASS,
            n_ball_samples=0,
            violation_samples=0,
            max_violation_mw=float("nan"),
            max_violation_line_idx=-1,
            tol_mw=1e-6,
        ),
        probabilistic=ProbabilisticCheck(
            status=PROB_OK,
            p_safe_gaussian_percent=float("nan"),
            p_safe_gaussian_ci95_low_percent=float("nan"),
            p_safe_gaussian_ci95_high_percent=float("nan"),
            p_ball_analytic_percent=float("nan"),
            p_ball_mc_percent=float("nan"),
            p_ball_mc_ci95_low_percent=float("nan"),
            p_ball_mc_ci95_high_percent=float("nan"),
            eta_safe_given_in_ball_percent=float("nan"),
            eta_ci95_low_percent=float("nan"),
            eta_ci95_high_percent=float("nan"),
            rho=float("nan"),
        ),
        comparisons={},
        overall=OverallCheck(status="WARN", reasons=("synthetic",)),
    )

    md = _case_card_md(
        case="case1354_pegase",
        results_status="ok",
        vr=vr,
        comparisons={},
        time_sec=1.0,
    )

    assert "nan%" not in md.lower()
    assert "n/a" in md.lower()


def test_monte_carlo_runs_on_zero_limits_and_reports_zero_feasible(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pandapower")
    pytest.importorskip("scipy")

    from stability_radius.parsers.matpower import load_network
    from stability_radius.verification.monte_carlo import run_monte_carlo_verification

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

    net = load_network(mfile)
    assert len(net.line) >= 1
    lid = int(sorted(net.line.index)[0])

    results = {
        "__meta__": {
            "input_path": str(mfile),
            "slack_bus": 0,
            "inj_std_mw": 1.0,
            "dispatch_mode": "opf_pypsa",
            "opf_solver": "highs",
            "opf_headroom_factor": 0.95,
        },
        f"line_{lid}": {
            "flow0_mw": 0.0,
            "p_limit_mw_est": 0.0,
            "radius_l2": 1.0,
            "norm_g": 1.0,
        },
    }
    rfile = tmp_path / "results.json"
    rfile.write_text(json.dumps(results), encoding="utf-8")

    vr = run_monte_carlo_verification(
        results_path=rfile,
        input_case_path=mfile,
        slack_bus=0,
        n_samples=200,
        seed=0,
        chunk_size=50,
    )

    assert vr.probabilistic.p_safe_gaussian_percent == pytest.approx(0.0)

    assert vr.soundness.status in {
        "SOUND_FAIL",
        "SOUND_SKIPPED_TRIVIAL_RADIUS",
        "SOUND_SKIPPED_BASE_INFEASIBLE",
        "SOUND_SKIPPED_INVALID_RADIUS",
        "SOUND_SKIPPED_NO_SAMPLES",
        "SOUND_PASS",
    }
