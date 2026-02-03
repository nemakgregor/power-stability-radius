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
    """
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    b2 = pp.create_bus(net, vn_kv=10.0)
    b3 = pp.create_bus(net, vn_kv=10.0)

    pp.create_ext_grid(net, b0, vm_pu=1.0)

    pp.create_load(net, b2, p_mw=5.0, q_mvar=0.0)
    pp.create_load(net, b3, p_mw=5.0, q_mvar=0.0)

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
