from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from stability_radius.base_point.pandapower_tools import (
    detect_q_limit_events,
    resolve_slack_bus_id,
)

pp = pytest.importorskip("pandapower")


def test_resolve_slack_bus_id_auto_detects_smallest_ext_grid_bus() -> None:
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = int(pp.create_bus(net, vn_kv=110.0))
    _b1 = int(pp.create_bus(net, vn_kv=110.0))
    b2 = int(pp.create_bus(net, vn_kv=110.0))

    # Create ext_grids in reverse bus order; auto-detection should still pick
    # the smallest bus id, not the first row in net.ext_grid.
    pp.create_ext_grid(net, b2, vm_pu=1.0)
    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_ext_grid(net, b0, vm_pu=1.0)

    assert resolve_slack_bus_id(net, -1) == min(b0, b2)


def test_detect_q_limit_events_reports_generator_at_limit() -> None:
    net = SimpleNamespace()
    net.gen = pd.DataFrame(
        {
            "bus": [3],
            "in_service": [True],
            "min_q_mvar": [-5.0],
            "max_q_mvar": [10.0],
        },
        index=[7],
    )
    net.res_gen = pd.DataFrame({"q_mvar": [10.0]}, index=[7])
    net.ext_grid = pd.DataFrame()
    net.res_ext_grid = pd.DataFrame()

    events = detect_q_limit_events(net, tol_mvar=1e-8)

    assert events == [
        {
            "element": "gen",
            "element_index": 7,
            "bus": 3,
            "q_mvar": 10.0,
            "q_min_mvar": -5.0,
            "q_max_mvar": 10.0,
            "at_min": False,
            "at_max": True,
        }
    ]
