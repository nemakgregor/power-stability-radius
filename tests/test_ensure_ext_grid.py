"""Tests for ensure_ext_grid_at_slack auto-creation behaviour.

Covers:
- No-op when in-service ext_grid already exists at slack bus.
- Auto-create when network has no ext_grid at all.
- Auto-create when ext_grid exists but not at the requested slack bus.
- Auto-create when all ext_grid entries are out of service.
"""

from __future__ import annotations

import pytest

pp = pytest.importorskip("pandapower")

from stability_radius.base_point.pandapower_tools import ensure_ext_grid_at_slack


def _make_net_with_ext_grid() -> tuple:
    """3-bus network with ext_grid at bus 0."""
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    b2 = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_gen(net, b1, p_mw=5.0, min_p_mw=0.0, max_p_mw=10.0)
    pp.create_load(net, b2, p_mw=8.0, q_mvar=0.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=0.01,
        x_ohm_per_km=0.10,
        c_nf_per_km=0.0,
        max_i_ka=10.0,
    )
    pp.create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=1.0,
        r_ohm_per_km=0.01,
        x_ohm_per_km=0.15,
        c_nf_per_km=0.0,
        max_i_ka=10.0,
    )
    return net, b0, b1, b2


def _make_net_no_ext_grid() -> tuple:
    """3-bus network without any ext_grid."""
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    b2 = pp.create_bus(net, vn_kv=110.0)
    pp.create_gen(net, b0, p_mw=10.0, min_p_mw=0.0, max_p_mw=20.0)
    pp.create_gen(net, b1, p_mw=5.0, min_p_mw=0.0, max_p_mw=10.0)
    pp.create_load(net, b2, p_mw=8.0, q_mvar=0.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=0.01,
        x_ohm_per_km=0.10,
        c_nf_per_km=0.0,
        max_i_ka=10.0,
    )
    pp.create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=1.0,
        r_ohm_per_km=0.01,
        x_ohm_per_km=0.15,
        c_nf_per_km=0.0,
        max_i_ka=10.0,
    )
    return net, b0, b1, b2


def test_noop_when_ext_grid_present() -> None:
    """ensure_ext_grid_at_slack is a no-op when ext_grid already exists at slack bus."""
    net, b0, _, _ = _make_net_with_ext_grid()
    n_before = len(net.ext_grid)

    ensure_ext_grid_at_slack(net, b0)

    assert len(net.ext_grid) == n_before


def test_creates_ext_grid_when_none_exists() -> None:
    """Auto-creates ext_grid when network has none."""
    net, b0, _, _ = _make_net_no_ext_grid()
    assert len(net.ext_grid) == 0

    ensure_ext_grid_at_slack(net, b0)

    assert len(net.ext_grid) == 1
    row = net.ext_grid.iloc[0]
    assert int(row["bus"]) == b0
    assert float(row["vm_pu"]) == pytest.approx(1.0)
    assert bool(row.get("in_service", True)) is True


def test_creates_ext_grid_at_different_bus() -> None:
    """Auto-creates ext_grid when existing one is at a different bus."""
    net, b0, b1, _ = _make_net_with_ext_grid()
    # ext_grid exists at b0 but we request b1
    assert len(net.ext_grid) == 1

    ensure_ext_grid_at_slack(net, b1)

    assert len(net.ext_grid) == 2
    # New ext_grid should be at b1
    new_row = net.ext_grid.iloc[-1]
    assert int(new_row["bus"]) == b1


def test_creates_ext_grid_when_all_out_of_service() -> None:
    """Auto-creates ext_grid when all existing entries are out of service."""
    net, b0, _, _ = _make_net_with_ext_grid()
    # Set existing ext_grid out of service
    net.ext_grid.loc[:, "in_service"] = False
    n_before = len(net.ext_grid)

    ensure_ext_grid_at_slack(net, b0)

    assert len(net.ext_grid) == n_before + 1
    # New entry should be in service at b0
    new_row = net.ext_grid.iloc[-1]
    assert int(new_row["bus"]) == b0
    assert bool(new_row["in_service"]) is True


def test_created_ext_grid_allows_runpp() -> None:
    """Network with auto-created ext_grid should be solvable by pandapower."""
    net, b0, _, _ = _make_net_no_ext_grid()

    ensure_ext_grid_at_slack(net, b0)
    pp.runpp(net)

    assert net.res_bus.vm_pu.notna().all()
