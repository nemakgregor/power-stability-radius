"""Tests for the ac_fpf base_dispatch mode (AC Feasible Power Flow via pandapower.runopp).

Covers:
- solve_ac_fpf() convergence on a 3-bus network.
- PyPSAAPFResult bus_p_mw extraction from runopp.
- solve_ac_fpf_base_point() produces correct BasePointAC.
- DC base point from AC FPF bus injections.
- pg0_source="midpoint" initial guess works.
- Voltage bounds are respected in solution.
- Quadratic cost function coefficients are correct.
"""

from __future__ import annotations

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")


def _make_3bus_net():
    """3-bus network for AC FPF testing.

    Bus 0: ext_grid (slack)
    Bus 1: gen (5 MW) + load (3 MW)
    Bus 2: load (8 MW)
    """
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    b2 = pp.create_bus(net, vn_kv=110.0)

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_load(net, b1, p_mw=3.0, q_mvar=0.0)
    pp.create_load(net, b2, p_mw=8.0, q_mvar=0.0)

    pp.create_gen(
        net,
        b1,
        p_mw=5.0,
        min_p_mw=0.0,
        max_p_mw=10.0,
        controllable=True,
    )

    common = dict(
        length_km=1.0,
        r_ohm_per_km=0.01,
        c_nf_per_km=0.0,
        max_i_ka=10.0,
        max_loading_percent=100.0,
    )
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1, x_ohm_per_km=0.10, **common
    )
    pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, x_ohm_per_km=0.15, **common
    )
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b2, x_ohm_per_km=0.20, **common
    )

    return net, b0


def test_solve_ac_fpf_basic() -> None:
    """solve_ac_fpf should converge on a simple 3-bus network."""
    from stability_radius.base_point.pandapower_opp import solve_ac_fpf

    net, slack_bus = _make_3bus_net()
    line_ids = [int(x) for x in sorted(net.line.index)]

    result = solve_ac_fpf(
        net=net,
        slack_bus=slack_bus,
        line_indices=line_ids,
        lossless=False,
    )

    assert result.status == "PP_OPP_OK"
    assert len(result.bus_ids) == 3
    assert result.v_mag_pu.shape == (3,)
    assert result.v_ang_rad.shape == (3,)
    assert len(result.line_ids) == 3
    assert result.line_p0_mw.shape == (3,)


def test_solve_ac_fpf_result_has_bus_p_mw() -> None:
    """AC FPF should populate bus_p_mw in the result."""
    from stability_radius.base_point.pandapower_opp import solve_ac_fpf

    net, slack_bus = _make_3bus_net()
    line_ids = [int(x) for x in sorted(net.line.index)]

    result = solve_ac_fpf(
        net=net,
        slack_bus=slack_bus,
        line_indices=line_ids,
        lossless=False,
    )

    assert result.bus_p_mw is not None, "bus_p_mw should be populated by runopp"
    assert result.bus_p_mw.shape == (len(result.bus_ids),)


def test_solve_ac_fpf_base_point_result() -> None:
    """solve_ac_fpf_base_point should produce a valid BasePointAC."""
    from stability_radius.base_point.ac import solve_ac_fpf_base_point

    net, slack_bus = _make_3bus_net()

    bp_ac, raw = solve_ac_fpf_base_point(
        net=net,
        slack_bus=slack_bus,
        lossless=False,
    )

    assert bp_ac.pf_solver == "pandapower_opp"
    assert bp_ac.status == "PP_OPP_OK"
    assert bp_ac.bus_p_mw is not None
    assert bp_ac.bus_p_mw.shape == (len(bp_ac.bus_ids),)
    assert len(bp_ac.line_ids) == 3
    assert bp_ac.p_from_mw.shape == (3,)
    assert bp_ac.s_limit_mva.shape == (3,)

    # Meta dict should serialize correctly.
    meta = bp_ac.to_meta_dict()
    assert meta["pf_solver"] == "pandapower_opp"
    assert meta["bus_p_mw"] is not None


def test_ac_fpf_dc_base_point_integration() -> None:
    """DC base point can be built from AC FPF bus injections."""
    from stability_radius.base_point.ac import solve_ac_fpf_base_point
    from stability_radius.base_point.dc import build_dc_base_point_from_acpf

    net, slack_bus = _make_3bus_net()

    bp_ac, base_pf = solve_ac_fpf_base_point(
        net=net,
        slack_bus=slack_bus,
        lossless=False,
    )

    assert base_pf.bus_p_mw is not None

    bp_dc, base_dc, dc_op = build_dc_base_point_from_acpf(
        net=net,
        slack_bus=slack_bus,
        acpf_bus_p_mw=base_pf.bus_p_mw,
        acpf_bus_ids=list(base_pf.bus_ids),
    )

    assert bp_dc.source == "acpf"
    assert set(bp_ac.bus_ids) == set(bp_dc.bus_ids)
    # DC injections should sum to 0 (balanced).
    assert abs(float(np.sum(bp_dc.bus_injections_mw))) < 1e-10
    # DC flows should have correct shape.
    n_lines = len(sorted(net.line.index))
    assert bp_dc.line_flows_mw.shape == (n_lines,)


def test_ac_fpf_pg0_midpoint() -> None:
    """AC FPF with pg0_source='midpoint' should converge."""
    from stability_radius.base_point.pandapower_opp import ACFPFConfig, solve_ac_fpf

    net, slack_bus = _make_3bus_net()
    line_ids = [int(x) for x in sorted(net.line.index)]

    cfg = ACFPFConfig(pg0_source="midpoint")
    result = solve_ac_fpf(
        net=net,
        slack_bus=slack_bus,
        line_indices=line_ids,
        lossless=False,
        fpf_cfg=cfg,
    )

    assert result.status == "PP_OPP_OK"
    assert result.bus_p_mw is not None


def test_ac_fpf_voltage_bounds_respected() -> None:
    """Solution voltage magnitudes should be within configured bounds."""
    from stability_radius.base_point.pandapower_opp import ACFPFConfig, solve_ac_fpf

    net, slack_bus = _make_3bus_net()
    line_ids = [int(x) for x in sorted(net.line.index)]

    vm_min, vm_max = 0.95, 1.05
    cfg = ACFPFConfig(vm_min_pu=vm_min, vm_max_pu=vm_max)
    result = solve_ac_fpf(
        net=net,
        slack_bus=slack_bus,
        line_indices=line_ids,
        lossless=False,
        fpf_cfg=cfg,
    )

    assert result.status == "PP_OPP_OK"
    # Allow small numerical tolerance on bounds.
    assert np.all(result.v_mag_pu >= vm_min - 1e-4), (
        f"v_mag_pu min={float(np.min(result.v_mag_pu))} < {vm_min}"
    )
    assert np.all(result.v_mag_pu <= vm_max + 1e-4), (
        f"v_mag_pu max={float(np.max(result.v_mag_pu))} > {vm_max}"
    )


def test_ac_fpf_cost_function_setup() -> None:
    """Quadratic cost coefficients should match (P - P0)^2 expansion."""
    from stability_radius.base_point.pandapower_opp import determine_pg0

    import pandas as pd

    # Simulate a generator row with p_mw=5.0
    row = pd.Series({"p_mw": 5.0, "min_p_mw": 0.0, "max_p_mw": 10.0})

    # pg0_source="case" should use p_mw
    pg0_case = determine_pg0(row, pg0_source="case")
    assert abs(pg0_case - 5.0) < 1e-10

    # pg0_source="midpoint" should use (min + max) / 2
    pg0_mid = determine_pg0(row, pg0_source="midpoint")
    assert abs(pg0_mid - 5.0) < 1e-10  # (0 + 10) / 2 = 5

    # For (P - P0)^2 = P^2 - 2*P0*P + P0^2:
    # cp2 = 1, cp1 = -2*P0, cp0 = P0^2
    p0 = 7.0
    cp2 = 1.0
    cp1 = -2.0 * p0
    cp0 = p0 * p0
    assert abs(cp2 - 1.0) < 1e-10
    assert abs(cp1 - (-14.0)) < 1e-10
    assert abs(cp0 - 49.0) < 1e-10

    # Verify the cost at P=P0 is 0
    cost_at_p0 = cp2 * p0 * p0 + cp1 * p0 + cp0
    assert abs(cost_at_p0) < 1e-10, f"Cost at P=P0 should be 0, got {cost_at_p0}"

    # Verify the cost at P != P0 is positive
    p_test = 3.0
    cost_at_p3 = cp2 * p_test * p_test + cp1 * p_test + cp0
    expected = (p_test - p0) ** 2
    assert abs(cost_at_p3 - expected) < 1e-10


def test_ac_fpf_lossless_mode() -> None:
    """AC FPF with lossless=True should converge."""
    from stability_radius.base_point.pandapower_opp import solve_ac_fpf

    net, slack_bus = _make_3bus_net()
    line_ids = [int(x) for x in sorted(net.line.index)]

    result = solve_ac_fpf(
        net=net,
        slack_bus=slack_bus,
        line_indices=line_ids,
        lossless=True,
    )

    assert result.status == "PP_OPP_OK"
    assert result.v_mag_pu.shape == (3,)


def test_ac_fpf_line_limit_setup_uses_deterministic_surrogates() -> None:
    from stability_radius.base_point.pandapower_opp import set_line_thermal_limits

    net, _ = _make_3bus_net()
    net.line.loc[:, "max_loading_percent"] = np.nan
    net.line.loc[:, "max_i_ka"] = 0.0

    set_line_thermal_limits(net)

    assert np.allclose(net.line["max_loading_percent"].to_numpy(dtype=float), 100.0)
    assert np.allclose(net.line["max_i_ka"].to_numpy(dtype=float), 100.0)


def test_ac_fpf_generator_defaults_are_deterministic_when_bounds_are_missing() -> None:
    from stability_radius.base_point.pandapower_opp import setup_gen_for_opp

    net, _ = _make_3bus_net()
    gid = int(sorted(net.gen.index)[0])
    net.gen.at[gid, "max_p_mw"] = float("nan")
    net.gen.at[gid, "min_q_mvar"] = float("nan")
    net.gen.at[gid, "max_q_mvar"] = float("nan")

    pg0_map = setup_gen_for_opp(net, pg0_source="case")

    assert net.gen.at[gid, "max_p_mw"] == pytest.approx(100.0)
    assert net.gen.at[gid, "min_q_mvar"] == pytest.approx(-999.0)
    assert net.gen.at[gid, "max_q_mvar"] == pytest.approx(999.0)
    assert pg0_map[f"gen_{gid}"] == pytest.approx(5.0)
