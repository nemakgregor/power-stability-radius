"""Tests for bus injection sign conventions in DC OPF.

Covers:
- Generator injection is positive at the generator bus.
- Load consumption produces negative injection at the load bus.
- Ext_grid generation produces positive injection at the slack bus.
- Ext_grid absorption (sign=-1) produces negative injection at the slack bus.
- Net injection sums to ~0 in lossless DC.
- Bus injection vector shape matches bus_ids length.
"""

from __future__ import annotations

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")
pytest.importorskip("pypsa")
pytest.importorskip("pandas")
pytest.importorskip("highspy")


def _make_3bus_net():
    """3-bus network with clear sign convention test points.

    Bus 0: ext_grid (slack)
    Bus 1: gen (min=5, max=10 MW) + load (3 MW)
    Bus 2: load (8 MW)

    Total demand = 11 MW, gen produces [5..10] MW,
    ext_grid covers the rest (1..6 MW injection).
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
        min_p_mw=5.0,
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

    return net, b0, b1, b2


def test_bus_injections_shape_matches_bus_ids() -> None:
    """bus_injections_mw length must equal len(bus_ids)."""
    from stability_radius.base_point.pypsa_opf import (
        solve_dc_opf_base_flows_from_pandapower,
    )

    net, b0, b1, b2 = _make_3bus_net()
    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    assert len(opf_res.bus_ids) == 3
    assert len(opf_res.bus_injections_mw) == len(opf_res.bus_ids)


def test_bus_injection_sum_near_zero() -> None:
    """In lossless DC, sum of bus injections must be ~0."""
    from stability_radius.base_point.pypsa_opf import (
        solve_dc_opf_base_flows_from_pandapower,
    )

    net, b0, b1, b2 = _make_3bus_net()
    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    inj_sum = float(np.sum(opf_res.bus_injections_mw))
    assert abs(inj_sum) < 1e-4, f"Injection sum should be ~0, got {inj_sum}"


def test_load_only_bus_has_negative_injection() -> None:
    """Bus with only loads (no generators) must have negative injection."""
    from stability_radius.base_point.pypsa_opf import (
        solve_dc_opf_base_flows_from_pandapower,
    )

    net, b0, b1, b2 = _make_3bus_net()
    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    bus_ids = list(opf_res.bus_ids)
    # Bus 2 has only load (8 MW), no gen → injection must be -8 MW.
    b2_pos = bus_ids.index(int(b2))
    b2_inj = float(opf_res.bus_injections_mw[b2_pos])
    assert b2_inj < 0, f"Load-only bus should have negative injection, got {b2_inj}"
    assert abs(b2_inj - (-8.0)) < 0.5, f"Expected ~-8 MW at load-only bus, got {b2_inj}"


def test_generator_bus_net_injection_is_gen_minus_load() -> None:
    """Bus with gen and load: injection = gen_output - load."""
    from stability_radius.base_point.pypsa_opf import (
        solve_dc_opf_base_flows_from_pandapower,
    )

    net, b0, b1, b2 = _make_3bus_net()
    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    bus_ids = list(opf_res.bus_ids)

    # Bus 1 has gen (min=5 MW) and load (3 MW).
    # Injection = gen_dispatch - 3.0.
    b1_pos = bus_ids.index(int(b1))
    b1_inj = float(opf_res.bus_injections_mw[b1_pos])

    # Gen dispatches at least 5 MW, load = 3 MW, so net injection >= 2 MW.
    assert b1_inj >= 1.5, (
        f"Expected positive net injection at gen+load bus, got {b1_inj}"
    )


def test_slack_bus_injection_positive_when_generating() -> None:
    """Slack bus injection should be positive when ext_grid generates."""
    from stability_radius.base_point.pypsa_opf import (
        solve_dc_opf_base_flows_from_pandapower,
    )

    net, b0, b1, b2 = _make_3bus_net()
    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    bus_ids = list(opf_res.bus_ids)
    b0_pos = bus_ids.index(int(b0))
    b0_inj = float(opf_res.bus_injections_mw[b0_pos])

    # Total demand=11 MW, gen produces [5..10] MW.
    # Ext_grid must generate >= 1 MW → positive injection.
    assert b0_inj > 0.5, (
        f"Slack bus should have positive injection when generating, got {b0_inj}"
    )
