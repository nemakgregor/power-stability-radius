"""Tests for untested helpers in stability_radius.workflows."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from stability_radius.config import OPFConfig
from stability_radius.workflows import (
    ACExtensionsConfig,
    _build_headroom_schedule,
    _build_sigma_arrays,
    compute_results_for_case,
    _merge_line_results,
    _run_ac_nonlinear_validation_topk,
    _solve_dc_opf_with_adaptive_headroom,
)
from stability_radius.verification.verify_worst_case import ViolationScaleSearchResult


# ---------------------------------------------------------------------------
# _merge_line_results
# ---------------------------------------------------------------------------


class TestMergeLineResults:
    def test_merge_disjoint_fields(self):
        a = {"line_0": {"radius_l2": 1.0}, "line_1": {"radius_l2": 2.0}}
        b = {"line_0": {"margin_mw": 10.0}, "line_1": {"margin_mw": 20.0}}
        merged = _merge_line_results(a, b)
        assert merged["line_0"] == {"radius_l2": 1.0, "margin_mw": 10.0}
        assert merged["line_1"] == {"radius_l2": 2.0, "margin_mw": 20.0}

    def test_later_dict_overwrites(self):
        a = {"line_0": {"radius_l2": 1.0}}
        b = {"line_0": {"radius_l2": 99.0}}
        merged = _merge_line_results(a, b)
        assert merged["line_0"]["radius_l2"] == 99.0

    def test_disjoint_keys(self):
        a = {"line_0": {"x": 1}}
        b = {"line_1": {"y": 2}}
        merged = _merge_line_results(a, b)
        assert "line_0" in merged
        assert "line_1" in merged

    def test_empty_inputs(self):
        merged = _merge_line_results({}, {})
        assert merged == {}

    def test_single_dict(self):
        a = {"line_0": {"r": 5.0}}
        merged = _merge_line_results(a)
        assert merged == {"line_0": {"r": 5.0}}

    def test_keys_sorted_numerically(self):
        a = {"line_10": {"r": 10}, "line_2": {"r": 2}, "line_1": {"r": 1}}
        merged = _merge_line_results(a)
        keys = list(merged.keys())
        assert keys == ["line_1", "line_2", "line_10"]


# ---------------------------------------------------------------------------
# _build_sigma_arrays
# ---------------------------------------------------------------------------


class TestBuildSigmaArrays:
    def test_uniform_source(self):
        ext = ACExtensionsConfig(
            sigma_p_mw_source="uniform",
            sigma_q_mvar_source="uniform",
            sigma_p_mw_uniform=5.0,
            sigma_q_mvar_uniform=3.0,
        )
        sp, sq = _build_sigma_arrays(ac_ext=ext, n_bus=4)
        np.testing.assert_array_equal(sp, np.full(4, 5.0))
        np.testing.assert_array_equal(sq, np.full(4, 3.0))

    def test_uc_jl_source(self):
        sp_arr = np.array([1.0, 2.0, 3.0])
        sq_arr = np.array([0.5, 1.0, 1.5])
        ext = ACExtensionsConfig(
            sigma_p_mw_source="uc_jl",
            sigma_q_mvar_source="uc_jl",
            sigma_p_mw_array=sp_arr,
            sigma_q_mvar_array=sq_arr,
        )
        sp, sq = _build_sigma_arrays(ac_ext=ext, n_bus=3)
        np.testing.assert_array_equal(sp, sp_arr)
        np.testing.assert_array_equal(sq, sq_arr)

    def test_uc_jl_wrong_shape_raises(self):
        ext = ACExtensionsConfig(
            sigma_p_mw_source="uc_jl",
            sigma_q_mvar_source="uniform",
            sigma_p_mw_array=np.array([1.0, 2.0]),
        )
        with pytest.raises(ValueError, match="shape"):
            _build_sigma_arrays(ac_ext=ext, n_bus=5)

    def test_uc_jl_missing_array_raises(self):
        ext = ACExtensionsConfig(
            sigma_p_mw_source="uc_jl",
            sigma_q_mvar_source="uniform",
            sigma_p_mw_array=None,
        )
        with pytest.raises(ValueError, match="sigma_p_mw_array"):
            _build_sigma_arrays(ac_ext=ext, n_bus=3)

    def test_invalid_source_raises(self):
        ext = ACExtensionsConfig(
            sigma_p_mw_source="bad",
            sigma_q_mvar_source="uniform",
        )
        with pytest.raises(ValueError, match="sigma_p_mw_source"):
            _build_sigma_arrays(ac_ext=ext, n_bus=3)

    def test_uniform_non_positive_raises(self):
        ext = ACExtensionsConfig(
            sigma_p_mw_source="uniform",
            sigma_q_mvar_source="uniform",
            sigma_p_mw_uniform=0.0,
        )
        with pytest.raises(ValueError, match="finite and >0"):
            _build_sigma_arrays(ac_ext=ext, n_bus=3)

    def test_uniform_nan_raises(self):
        ext = ACExtensionsConfig(
            sigma_p_mw_source="uniform",
            sigma_q_mvar_source="uniform",
            sigma_p_mw_uniform=float("nan"),
        )
        with pytest.raises(ValueError, match="finite and >0"):
            _build_sigma_arrays(ac_ext=ext, n_bus=3)

    def test_mixed_sources(self):
        """P from uniform, Q from uc_jl — both must work independently."""
        sq_arr = np.array([0.1, 0.2, 0.3])
        ext = ACExtensionsConfig(
            sigma_p_mw_source="uniform",
            sigma_q_mvar_source="uc_jl",
            sigma_p_mw_uniform=10.0,
            sigma_q_mvar_array=sq_arr,
        )
        sp, sq = _build_sigma_arrays(ac_ext=ext, n_bus=3)
        np.testing.assert_array_equal(sp, np.full(3, 10.0))
        np.testing.assert_array_equal(sq, sq_arr)


class TestAdaptiveHeadroom:
    def test_build_headroom_schedule_relaxes_towards_one(self):
        assert _build_headroom_schedule(0.90) == [0.9, 0.92, 0.95, 0.98, 1.0]
        assert _build_headroom_schedule(0.98) == [0.98, 1.0]
        assert _build_headroom_schedule(1.0) == [1.0]

    def test_solve_dc_opf_with_adaptive_headroom_retries_in_order(self):
        cfg = OPFConfig(headroom_factor=0.90)
        bp_obj = object()
        base_obj = object()
        seen: list[float] = []

        def _fake_build_dc_base_point_dc_opf(*, net, slack_bus, opf_cfg, limit_factor):
            seen.append(float(opf_cfg.headroom_factor))
            if float(opf_cfg.headroom_factor) < 0.95:
                raise RuntimeError("infeasible for this headroom")
            return bp_obj, base_obj

        with patch(
            "stability_radius.workflows.build_dc_base_point_dc_opf",
            side_effect=_fake_build_dc_base_point_dc_opf,
        ):
            bp_dc, base_dc, used = _solve_dc_opf_with_adaptive_headroom(
                net=object(),
                slack_bus=0,
                opf_cfg=cfg,
                limit_factor=1.0,
                case_tag="case30",
            )

        assert bp_dc is bp_obj
        assert base_dc is base_obj
        assert used == pytest.approx(0.95)
        assert seen == [0.9, 0.92, 0.95]


def test_compute_ac_lossless_false_fails_before_certificate_build() -> None:
    with pytest.raises(NotImplementedError, match="ac.lossless=false"):
        compute_results_for_case(
            input_path="does_not_need_to_exist.m",
            slack_bus=0,
            base_dispatch="case",
            compute_dc=False,
            dc_mode="operator",
            dc_chunk_size=1,
            dc_dtype=np.float64,
            dc_inj_std_mw=1.0,
            compute_ac=True,
            ac_chunk_size=1,
            ac_balance=True,
            ac_pf_init="flat",
            ac_pf_solver="pandapower",
            ac_lossless=False,
        )


class TestACNonlinearValidationTopK:
    def test_merges_compact_fields_and_returns_report(self):
        results_lines = {
            "line_0": {
                "constraint_status_ac_l2": "ok_finite",
                "certificate_radius_ac_l2": 2.0,
                "binding_end": "from",
            },
            "line_1": {
                "constraint_status_ac_l2": "ok_finite",
                "certificate_radius_ac_l2": 1.0,
                "binding_end": "to",
            },
            "line_2": {
                "constraint_status_ac_l2": "base_infeasible",
                "certificate_radius_ac_l2": 0.0,
                "binding_end": "from",
            },
        }
        h_bind = np.array(
            [
                [1.0, 2.0, 3.0, 0.0, 4.0, 0.0],
                [2.0, 3.0, 4.0, 0.0, 5.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        captured: list[dict] = []

        def _fake_find_violation_scale(**kwargs):
            captured.append(kwargs)
            line_id = int(kwargs["line_id"])
            if line_id == 1:
                gamma = 0.8
                actual = 108.0
                trajectory = [
                    {
                        "scale": 0.0,
                        "actual_s_mva": 90.0,
                        "violated": False,
                        "pf_converged": True,
                    },
                    {
                        "scale": 0.8,
                        "actual_s_mva": actual,
                        "violated": True,
                        "pf_converged": True,
                    },
                ]
            else:
                gamma = 1.25
                actual = 125.0
                trajectory = [
                    {
                        "scale": 0.0,
                        "actual_s_mva": 80.0,
                        "violated": False,
                        "pf_converged": True,
                    },
                    {
                        "scale": 1.0,
                        "actual_s_mva": 99.0,
                        "violated": False,
                        "pf_converged": True,
                    },
                    {
                        "scale": 1.25,
                        "actual_s_mva": actual,
                        "violated": True,
                        "pf_converged": True,
                    },
                ]
            return ViolationScaleSearchResult(
                line_id=line_id,
                limit_mva=100.0,
                s0_mva=90.0,
                predicted_violation_scale=1.0,
                actual_violation_scale=gamma,
                actual_s_at_violation=actual,
                conservatism_ratio=gamma,
                n_pf_calls=len(trajectory),
                converged=True,
                scale_trajectory=trajectory,
            )

        with patch(
            "stability_radius.verification.verify_worst_case.find_violation_scale",
            side_effect=_fake_find_violation_scale,
        ):
            report = _run_ac_nonlinear_validation_topk(
                net=object(),
                case_tag="case_test",
                results_lines=results_lines,
                h_bind=h_bind,
                s0_mva=np.array([80.0, 90.0, 0.0]),
                s_limit_mva=np.array([100.0, 100.0, 100.0]),
                line_ids=[0, 1, 2],
                n_bus=3,
                pq_mask=np.array([False, True, False]),
                balance=True,
                lossless=True,
                gen_dispatch_mw_by_name={"gen_0": 10.0},
                top_k=2,
                scale_max=2.0,
                tol=0.01,
                max_iter=5,
            )

        assert [call["line_id"] for call in captured] == [1, 0]
        assert captured[0]["gen_dispatch_mw_by_name"] == {"gen_0": 10.0}
        assert captured[0]["delta_u_unit"][3] == pytest.approx(0.0)
        assert captured[0]["delta_u_unit"][5] == pytest.approx(0.0)
        assert report["top_k_replayed"] == 2
        assert report["summary"]["n_lines_gamma_lt_1"] == 1
        assert results_lines["line_1"]["radius_ac_l2_validated"] == pytest.approx(0.8)
        assert results_lines["line_1"]["linearization_status"] == "nonlinear_optimistic"
        assert results_lines["line_0"]["radius_ac_l2_validated"] == pytest.approx(2.0)
        assert results_lines["line_0"]["linearization_status"] == "validated_local"
