from __future__ import annotations

import inspect

import numpy as np
import pytest


def test_per_line_helper_returns_correct_shape() -> None:
    """_ac_pf_sample_per_line_violations_mva returns bool array of correct shape."""
    pp = pytest.importorskip("pandapower")

    from stability_radius.verification.monte_carlo import (
        _ac_pf_sample_per_line_violations_mva,
    )

    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110.0)
    b2 = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=b1)
    pp.create_load(net, bus=b2, p_mw=10.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=10.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.runpp(net)

    line_ids = [int(x) for x in sorted(net.line.index)]
    limits_mva = np.array([1000.0])  # very high limit -> no overload

    ok, worst, wpos, overloaded = _ac_pf_sample_per_line_violations_mva(
        net,
        line_ids=line_ids,
        limits_mva=limits_mva,
        feas_tol_mva=0.0,
    )

    assert ok is True
    assert overloaded.shape == (len(line_ids),)
    assert overloaded.dtype == bool
    assert not overloaded[0]


def test_per_line_helper_detects_overload() -> None:
    """_ac_pf_sample_per_line_violations_mva flags overloaded lines."""
    pp = pytest.importorskip("pandapower")

    from stability_radius.verification.monte_carlo import (
        _ac_pf_sample_per_line_violations_mva,
    )

    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110.0)
    b2 = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, bus=b1)
    pp.create_load(net, bus=b2, p_mw=10.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=10.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.runpp(net)

    line_ids = [int(x) for x in sorted(net.line.index)]
    limits_mva = np.array([0.001])  # very low limit -> guaranteed overload

    ok, worst, wpos, overloaded = _ac_pf_sample_per_line_violations_mva(
        net,
        line_ids=line_ids,
        limits_mva=limits_mva,
        feas_tol_mva=0.0,
    )

    assert ok is False
    assert overloaded[0] is np.bool_(True)


def test_track_per_line_overloads_parameter_exists() -> None:
    """run_monte_carlo_verification accepts track_per_line_overloads."""
    from stability_radius.verification.monte_carlo import run_monte_carlo_verification

    sig = inspect.signature(run_monte_carlo_verification)
    assert "track_per_line_overloads" in sig.parameters
    param = sig.parameters["track_per_line_overloads"]
    assert param.default is False


def test_sample_gaussian_ac_uses_sigma_weighted_balance_projection() -> None:
    from stability_radius.verification.monte_carlo import _sample_gaussian_ac

    rng = np.random.default_rng(123)
    sigma_p = np.array([1.0, 3.0, 10.0])
    sigma_q = np.array([2.0, 5.0, 7.0])

    dp, dq = _sample_gaussian_ac(
        rng=rng,
        n=500,
        n_bus=3,
        sigma_p_mw=sigma_p,
        sigma_q_mvar=sigma_q,
    )

    assert np.max(np.abs(np.sum(dp, axis=1))) < 1e-10
    assert np.max(np.abs(np.sum(dq, axis=1))) < 1e-10
    # Heterogeneous conditional projection should not collapse to ordinary
    # unweighted sample mean subtraction before scaling.
    assert not np.allclose(np.var(dp, axis=0), np.var(dp, axis=0)[0])
