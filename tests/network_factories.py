from __future__ import annotations

from typing import Any


def make_three_bus_dispatch_net(pp: Any):
    """Build the shared three-bus dispatch network used by AC base-point tests."""
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


def make_triangle_net(pp: Any):
    """Build a small meshed network where branch reactances set flow split."""
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


def make_two_bus_opf_net(
    pp: Any, *, load_p_mw: float, gen_min_p_mw: float, gen_max_p_mw: float = 20.0
):
    """Build a two-bus OPF network for ext-grid generation or absorption tests."""
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_load(net, b1, p_mw=float(load_p_mw), q_mvar=0.0)

    pp.create_gen(
        net,
        b1,
        p_mw=10.0,
        min_p_mw=float(gen_min_p_mw),
        max_p_mw=float(gen_max_p_mw),
        controllable=True,
    )

    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=0.01,
        x_ohm_per_km=0.10,
        c_nf_per_km=0.0,
        max_i_ka=10.0,
        max_loading_percent=100.0,
    )

    return net, b0
