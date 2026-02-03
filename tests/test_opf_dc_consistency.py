from __future__ import annotations

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")
pytest.importorskip("pypsa")
pytest.importorskip("pandas")
pytest.importorskip("highspy")


def _make_triangle_net():
    """
    Small meshed (non-radial) network where line flows depend on branch reactances.

    This is important: a single-line 2-bus system would trivially match regardless of x.
    """
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    b2 = pp.create_bus(net, vn_kv=110.0)

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_load(net, b1, p_mw=10.0, q_mvar=0.0)
    pp.create_load(net, b2, p_mw=5.0, q_mvar=0.0)

    common = dict(
        length_km=1.0,
        r_ohm_per_km=0.01,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )

    # Triangle with different reactances to enforce non-trivial split.
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


def _make_triangle_net_with_tapped_trafo():
    """
    Meshed network where a transformer branch (with non-unit tap) influences flow split.

    Topology:
      b0 --line-- b1
      b0 --line-- b2
      b1 --trafo(tap!=1)-- b2

    This exercises the OPF<->DCOperator consistency with tap handling.
    """
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    b2 = pp.create_bus(net, vn_kv=110.0)

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_load(net, b1, p_mw=10.0, q_mvar=0.0)
    pp.create_load(net, b2, p_mw=5.0, q_mvar=0.0)

    common_line = dict(
        length_km=1.0,
        r_ohm_per_km=0.01,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )

    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1, x_ohm_per_km=0.10, **common_line
    )
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b2, x_ohm_per_km=0.20, **common_line
    )

    # "Same voltage" transformer is valid for a DC abstraction test; we only care about tap handling.
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        sn_mva=40.0,
        vn_hv_kv=110.0,
        vn_lv_kv=110.0,
        vk_percent=10.0,
        vkr_percent=0.5,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=0.0,
        tap_side="hv",
        tap_neutral=0,
        tap_min=-10,
        tap_max=10,
        tap_step_percent=1.0,
        tap_pos=5,  # => tap ratio 1.05
    )

    return net, b0


def _make_multivoltage_cycle_net():
    """
    Regression helper: a meshed network with multiple voltage levels.

    Why this exists
    ---------------
    A historical bug was caused by passing per-unit reactance into PyPSA `Line.x` which
    actually expects Ohm. In single-voltage networks this can go unnoticed (uniform
    scaling cancels out in DC flows). In multi-voltage networks it breaks the physics
    and causes OPF->DCOperator mismatches.

    Topology (cycle that crosses voltage levels):
      110kV: b0 --line-- b1
      trafo: b1 (110kV) -> b2 (10kV)
      10kV:  b2 --line-- b3
      trafo: b0 (110kV) -> b3 (10kV)
    """
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    b2 = pp.create_bus(net, vn_kv=10.0)
    b3 = pp.create_bus(net, vn_kv=10.0)

    pp.create_ext_grid(net, b0, vm_pu=1.0)

    # Put loads on the LV side to ensure power flows through transformers and both voltage levels.
    pp.create_load(net, b2, p_mw=5.0, q_mvar=0.0)
    pp.create_load(net, b3, p_mw=5.0, q_mvar=0.0)

    common_line = dict(
        length_km=1.0,
        r_ohm_per_km=0.01,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )

    # HV line
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1, x_ohm_per_km=0.10, **common_line
    )
    # LV line
    pp.create_line_from_parameters(
        net, from_bus=b2, to_bus=b3, x_ohm_per_km=0.02, **common_line
    )

    trafo_common = dict(
        sn_mva=40.0,
        vk_percent=10.0,
        vkr_percent=0.5,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=0.0,
        tap_side="hv",
        tap_neutral=0,
        tap_min=0,
        tap_max=0,
        tap_step_percent=1.0,
        tap_pos=0,
    )

    pp.create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        vn_hv_kv=110.0,
        vn_lv_kv=10.0,
        **trafo_common,
    )
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b0,
        lv_bus=b3,
        vn_hv_kv=110.0,
        vn_lv_kv=10.0,
        **trafo_common,
    )

    return net, b0


def test_pypsa_opf_flows_match_dc_operator_reconstruction() -> None:
    """
    Regression test for OPF->DCOperator consistency.

    If PyPSA uses any resistance-dependent coefficients in its linear model and we pass r!=0,
    reconstructed flows from DCOperator (x-only) drift. Project policy requires lossless DC
    consistency, therefore max|Δf| must be below the configured tolerance (~1e-3 MW).
    """
    from stability_radius.dc.dc_model import build_dc_operator
    from stability_radius.opf.pypsa_opf import solve_dc_opf_base_flows_from_pandapower

    net, slack_bus = _make_triangle_net()

    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    inj_sum = float(np.sum(opf_res.bus_injections_mw))
    assert abs(inj_sum) < 1e-6

    dc_op = build_dc_operator(net, slack_bus=int(slack_bus))

    f_opf = np.asarray(opf_res.line_flows_mw, dtype=float).reshape(-1)
    f_dc = np.asarray(
        dc_op.flows_from_delta_injections(opf_res.bus_injections_mw),
        dtype=float,
    ).reshape(-1)

    assert f_opf.shape == f_dc.shape
    assert float(np.max(np.abs(f_opf - f_dc))) < 1e-3


def test_pypsa_opf_flows_match_dc_operator_reconstruction_with_tapped_trafo() -> None:
    """
    Regression: consistency must hold on networks with tap-changing transformers.

    This test would typically fail if:
    - OPF includes tap effects but DCOperator doesn't, or
    - both include taps but use different conventions.
    """
    from stability_radius.dc.dc_model import build_dc_operator
    from stability_radius.opf.pypsa_opf import solve_dc_opf_base_flows_from_pandapower

    net, slack_bus = _make_triangle_net_with_tapped_trafo()

    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    inj_sum = float(np.sum(opf_res.bus_injections_mw))
    assert abs(inj_sum) < 1e-6

    dc_op = build_dc_operator(net, slack_bus=int(slack_bus))

    f_opf = np.asarray(opf_res.line_flows_mw, dtype=float).reshape(-1)
    f_dc = np.asarray(
        dc_op.flows_from_delta_injections(opf_res.bus_injections_mw),
        dtype=float,
    ).reshape(-1)

    assert f_opf.shape == f_dc.shape
    assert float(np.max(np.abs(f_opf - f_dc))) < 1e-3


def test_pypsa_opf_flows_match_dc_operator_reconstruction_multivoltage() -> None:
    """
    Regression: OPF->DCOperator consistency must hold on multi-voltage networks.

    This specifically detects unit mismatches between:
      - pandapower data (x in Ohm)
      - PyPSA Line API (expects x in Ohm, then computes per-unit internally)
    """
    from stability_radius.dc.dc_model import build_dc_operator
    from stability_radius.opf.pypsa_opf import solve_dc_opf_base_flows_from_pandapower

    net, slack_bus = _make_multivoltage_cycle_net()

    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    inj_sum = float(np.sum(opf_res.bus_injections_mw))
    assert abs(inj_sum) < 1e-6

    dc_op = build_dc_operator(net, slack_bus=int(slack_bus))

    f_opf = np.asarray(opf_res.line_flows_mw, dtype=float).reshape(-1)
    f_dc = np.asarray(
        dc_op.flows_from_delta_injections(opf_res.bus_injections_mw),
        dtype=float,
    ).reshape(-1)

    assert f_opf.shape == f_dc.shape
    assert float(np.max(np.abs(f_opf - f_dc))) < 1e-3
