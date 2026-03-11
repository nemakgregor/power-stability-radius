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
    _merge_line_results,
    _solve_dc_opf_with_adaptive_headroom,
)


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
