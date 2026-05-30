"""Tests for ext_grid absorption in DC OPF.

Covers:
- Ext_grid absorption generator with sign=-1 is created and used when needed.
- Bus injections correctly account for sign attribute.
- Absorption is tracked in PyPSAOPFResult.ext_grid_absorption_mw.
- Absorption generators are excluded from gen_dispatch_mw_by_name.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.network_factories import make_two_bus_opf_net

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")
pytest.importorskip("pypsa")
pytest.importorskip("pandas")
pytest.importorskip("highspy")


def _make_net_requiring_absorption():
    """Build a network where generator min-output forces ext-grid absorption."""
    return make_two_bus_opf_net(pp, load_p_mw=5.0, gen_min_p_mw=10.0)


def _make_net_no_absorption():
    """Normal network where ext_grid only generates (no absorption needed).

    Load = 15 MW, gen max = 20 MW, gen min = 0 MW.
    OPF can satisfy load without ext_grid absorption.
    """
    return make_two_bus_opf_net(pp, load_p_mw=15.0, gen_min_p_mw=0.0)


def test_ext_grid_absorption_used_when_gen_min_exceeds_demand() -> None:
    """When generator min output exceeds demand, ext_grid must absorb excess."""
    from stability_radius.base_point.pypsa_opf import (
        solve_dc_opf_base_flows_from_pandapower,
    )

    net, slack_bus = _make_net_requiring_absorption()
    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    assert opf_res.status in {"ok", "optimal"}

    # Absorption must be > 0 (gen produces >=10 MW but demand is only 5 MW).
    assert opf_res.ext_grid_absorption_mw > 1.0, (
        f"Expected ext_grid absorption >1 MW, got {opf_res.ext_grid_absorption_mw}"
    )

    # Bus injections must balance (sum ~= 0 in lossless DC).
    inj_sum = float(np.sum(opf_res.bus_injections_mw))
    assert abs(inj_sum) < 1e-4, f"Bus injections not balanced: sum={inj_sum}"


def test_ext_grid_absorption_zero_when_not_needed() -> None:
    """When load exceeds gen min, no absorption should be used."""
    from stability_radius.base_point.pypsa_opf import (
        solve_dc_opf_base_flows_from_pandapower,
    )

    net, slack_bus = _make_net_no_absorption()
    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    assert opf_res.status in {"ok", "optimal"}

    # No absorption needed.
    assert opf_res.ext_grid_absorption_mw < 1e-3, (
        f"Expected no absorption, got {opf_res.ext_grid_absorption_mw} MW"
    )

    # Bus injections must balance.
    inj_sum = float(np.sum(opf_res.bus_injections_mw))
    assert abs(inj_sum) < 1e-4


def test_absorption_generators_excluded_from_dispatch() -> None:
    """Absorption generators (ext_*_absorb) must NOT appear in gen_dispatch."""
    from stability_radius.base_point.pypsa_opf import (
        solve_dc_opf_base_flows_from_pandapower,
    )

    net, slack_bus = _make_net_requiring_absorption()
    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    gen_names = [name for name, _ in opf_res.gen_dispatch_mw_by_name]
    absorb_names = [n for n in gen_names if n.endswith("_absorb")]
    assert len(absorb_names) == 0, (
        f"Absorption generators should be excluded from dispatch: {absorb_names}"
    )


def test_bus_injections_sign_correct_with_absorption() -> None:
    """Bus injection at slack bus must be negative when ext_grid absorbs."""
    from stability_radius.base_point.pypsa_opf import (
        solve_dc_opf_base_flows_from_pandapower,
    )

    net, slack_bus = _make_net_requiring_absorption()
    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    bus_ids = list(opf_res.bus_ids)
    slack_pos = bus_ids.index(int(slack_bus))
    slack_inj = float(opf_res.bus_injections_mw[slack_pos])

    # Slack bus absorbs excess power: injection should be negative.
    assert slack_inj < -1.0, (
        f"Slack bus injection should be negative (absorbing), got {slack_inj} MW"
    )


def test_opf_flows_consistent_with_dc_operator_when_absorbing() -> None:
    """OPF flows with absorption must match DC operator reconstruction."""
    from stability_radius.base_point.pypsa_opf import (
        solve_dc_opf_base_flows_from_pandapower,
    )
    from stability_radius.dc.dc_model import build_dc_operator

    net, slack_bus = _make_net_requiring_absorption()
    line_indices = [int(x) for x in sorted(net.line.index)]
    line_limits = np.full(len(line_indices), 1.0e6, dtype=float)

    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=line_indices,
        line_limits_mw=line_limits,
    )

    dc_op = build_dc_operator(net, slack_bus=int(slack_bus))

    f_opf = np.asarray(opf_res.line_flows_mw, dtype=float).reshape(-1)
    f_dc = np.asarray(
        dc_op.flows_from_bus_injections_mw(opf_res.bus_injections_mw),
        dtype=float,
    ).reshape(-1)

    assert f_opf.shape == f_dc.shape
    max_diff = float(np.max(np.abs(f_opf - f_dc)))
    assert max_diff < 1e-3, (
        f"OPF flows don't match DC operator reconstruction: max|Δf|={max_diff} MW"
    )
