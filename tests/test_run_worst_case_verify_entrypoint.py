from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from entry_points import run_worst_case_verify as mod


def _sample_case_inputs() -> tuple[dict, np.ndarray, np.ndarray, list[int]]:
    results = {
        "line_10": {
            "radius_ac_l2": 1.0,
            "binding_end": "from",
            "ac_s0_from_mva": 60.0,
            "ac_s_limit_mva": 100.0,
        },
        "line_11": {
            "radius_ac_l2": 2.0,
            "binding_end": "to",
            "ac_s0_to_mva": 50.0,
            "ac_s_limit_mva": 90.0,
        },
        "line_12": {
            "radius_ac_l2": float("nan"),
            "binding_end": "from",
            "ac_s0_from_mva": 20.0,
            "ac_s_limit_mva": 40.0,
        },
    }
    h_from = np.zeros((3, 4), dtype=float)
    h_to = np.zeros((3, 4), dtype=float)
    line_ids = [10, 11, 12]
    return results, h_from, h_to, line_ids


def test_verify_case_uses_top_k_and_keeps_primary_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def _fake_verify_worst_case(*, line_id: int, s0_mva: float, limit_mva: float, scale: float, **_: object):
        calls.append(line_id)
        predicted = s0_mva + scale * (limit_mva - s0_mva)
        actual = s0_mva + scale * 0.9 * (limit_mva - s0_mva)
        return SimpleNamespace(
            pf_converged=True,
            actual_s_mva=actual,
            predicted_s_mva=predicted,
            violated=actual >= limit_mva,
            relative_error=0.01,
        )

    monkeypatch.setattr(mod, "verify_worst_case", _fake_verify_worst_case)

    results, h_from, h_to, line_ids = _sample_case_inputs()
    out = mod._verify_case(
        case_name="case_x",
        results=results,
        net=object(),
        h_from_full=h_from,
        h_to_full=h_to,
        line_ids=line_ids,
        scales=[1.0],
        top_k=2,
    )

    assert out["status"] == "ok"
    assert out["top_k_requested"] == 2
    assert out["n_verified_lines"] == 2
    assert [r["line_id"] for r in out["verified_lines"]] == [10, 11]

    assert out["bottleneck_line"] == 10
    assert out["radius_ac_l2"] == pytest.approx(1.0)
    assert out["scale_results"] == out["verified_lines"][0]["scale_results"]
    assert calls == [10, 11]


def test_verify_case_rejects_non_positive_top_k() -> None:
    results, h_from, h_to, line_ids = _sample_case_inputs()
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        mod._verify_case(
            case_name="case_x",
            results=results,
            net=object(),
            h_from_full=h_from,
            h_to_full=h_to,
            line_ids=line_ids,
            scales=[1.0],
            top_k=0,
        )


def test_validation_checks_cover_all_verified_lines(tmp_path) -> None:
    case_results = [
        {
            "case": "case_x",
            "status": "ok",
            "verified_lines": [
                {
                    "line_id": 10,
                    "crossing_alpha": 0.85,
                    "scale_results": [{"scale": 1.0, "pf_converged": True}],
                },
                {
                    "line_id": 11,
                    "crossing_alpha": 1.02,
                    "scale_results": [{"scale": 1.0, "pf_converged": False}],
                },
            ],
        }
    ]

    checks = mod._run_validation_checks(case_results, tmp_path)

    assert len(checks["crossing"]["details"]) == 2
    assert checks["crossing"]["any_dangerous"] is True
    assert checks["pf_divergence_at_1"]["any_diverged"] is True
