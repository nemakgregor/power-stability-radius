from __future__ import annotations

import pytest

pp = pytest.importorskip("pandapower")

from stability_radius.base_point.pandapower_tools import resolve_slack_bus_id


def test_resolve_slack_bus_id_auto_detects_smallest_ext_grid_bus() -> None:
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = int(pp.create_bus(net, vn_kv=110.0))
    _b1 = int(pp.create_bus(net, vn_kv=110.0))
    b2 = int(pp.create_bus(net, vn_kv=110.0))

    # Create ext_grids in reverse bus order; auto-detection should still pick the
    # smallest bus id, not the first row in net.ext_grid.
    pp.create_ext_grid(net, b2, vm_pu=1.0)
    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_ext_grid(net, b0, vm_pu=1.0)

    assert resolve_slack_bus_id(net, -1) == min(b0, b2)
