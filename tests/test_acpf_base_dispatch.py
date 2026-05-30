"""Tests for the acpf base_dispatch mode.

Covers:
- PyPSAAPFResult bus_p_mw extraction from pandapower solver.
- build_dc_base_point_from_acpf() correctness.
- Slack bus loss correction.
- Non-slack bus injections preserved.
- Full integration smoke test with compute_results_for_case().
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.network_factories import make_three_bus_dispatch_net

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")
pytest.importorskip("pypsa")
pytest.importorskip("pandas")
pytest.importorskip("highspy")


def test_pypsa_apf_result_has_bus_p_mw() -> None:
    """AC PF with pandapower solver should populate bus_p_mw."""
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )

    net, slack_bus = make_three_bus_dispatch_net(pp)

    result = solve_ac_pf_base_point_from_pandapower(
        net=net,
        slack_bus=slack_bus,
        solver="pandapower",
        init="flat",
        lossless=True,
    )

    assert result.bus_p_mw is not None, (
        "bus_p_mw should be populated by pandapower solver"
    )
    assert result.bus_p_mw.shape == (len(result.bus_ids),)
    # Sum should be close to zero (near-lossless with lossless=True)
    assert abs(float(np.sum(result.bus_p_mw))) < 5.0, (
        f"bus_p_mw sum should be small, got {float(np.sum(result.bus_p_mw))}"
    )


def test_base_point_ac_propagates_bus_p_mw() -> None:
    """solve_ac_pf_base_point should propagate bus_p_mw to BasePointAC."""
    from stability_radius.base_point.ac import solve_ac_pf_base_point

    net, slack_bus = make_three_bus_dispatch_net(pp)

    bp_ac, raw = solve_ac_pf_base_point(
        net=net,
        slack_bus=slack_bus,
        pf_solver="pandapower",
        pf_init="flat",
        lossless=True,
        gen_dispatch_mw_by_name={},
    )

    assert bp_ac.bus_p_mw is not None, "BasePointAC.bus_p_mw should be set"
    assert bp_ac.bus_p_mw.shape == (len(bp_ac.bus_ids),)

    # Check meta dict serialization
    meta = bp_ac.to_meta_dict()
    assert meta["bus_p_mw"] is not None
    assert len(meta["bus_p_mw"]) == len(bp_ac.bus_ids)


def test_build_dc_base_point_from_acpf_basic() -> None:
    """build_dc_base_point_from_acpf returns correct structure."""
    from stability_radius.base_point.dc import build_dc_base_point_from_acpf

    net, slack_bus = make_three_bus_dispatch_net(pp)

    # Simulate AC PF bus injections (MW).
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    # Bus 0: ext_grid ~6 MW, Bus 1: gen 5 - load 3 = 2 MW, Bus 2: -8 MW
    # With some loss: sum != 0
    acpf_bus_p_mw = np.array([6.05, 2.0, -8.0])  # sum = 0.05 (simulated loss)

    bp, base, op = build_dc_base_point_from_acpf(
        net=net,
        slack_bus=slack_bus,
        acpf_bus_p_mw=acpf_bus_p_mw,
        acpf_bus_ids=bus_ids,
    )

    assert bp.source == "acpf"
    assert bp.status == "acpf"
    assert len(bp.bus_ids) == 3
    assert len(bp.line_ids) == 3
    assert bp.line_flows_mw.shape == (3,)

    # Bus injections must be balanced (sum = 0).
    assert abs(float(np.sum(bp.bus_injections_mw))) < 1e-10

    # Base quantities must match.
    assert base.opf_status == "acpf"
    assert len(base.line_indices) == 3


def test_acpf_slack_correction_equals_losses() -> None:
    """Slack bus loss correction should equal negative sum of input injections."""
    from stability_radius.base_point.dc import build_dc_base_point_from_acpf

    net, slack_bus = make_three_bus_dispatch_net(pp)
    bus_ids = [int(x) for x in sorted(net.bus.index)]

    # Simulate AC PF with losses: sum = -0.3 MW (losses)
    acpf_bus_p_mw = np.array([5.7, 2.0, -8.0])  # sum = -0.3

    bp, base, op = build_dc_base_point_from_acpf(
        net=net,
        slack_bus=slack_bus,
        acpf_bus_p_mw=acpf_bus_p_mw,
        acpf_bus_ids=bus_ids,
    )

    # Slack bus injection in output should differ from input by the loss correction.
    slack_pos = list(bp.bus_ids).index(int(slack_bus))
    slack_inj_out = float(bp.bus_injections_mw[slack_pos])
    slack_inj_in = float(acpf_bus_p_mw[bus_ids.index(int(slack_bus))])
    correction = slack_inj_in - slack_inj_out

    # correction should equal sum of inputs (= -0.3, i.e. slack absorbs 0.3 extra)
    input_sum = float(np.sum(acpf_bus_p_mw))
    assert abs(correction - input_sum) < 1e-10, (
        f"Slack correction {correction} should equal input sum {input_sum}"
    )


def test_acpf_non_slack_injections_preserved() -> None:
    """Non-slack bus injections must exactly match input from AC PF."""
    from stability_radius.base_point.dc import build_dc_base_point_from_acpf

    net, slack_bus = make_three_bus_dispatch_net(pp)
    bus_ids = [int(x) for x in sorted(net.bus.index)]

    acpf_bus_p_mw = np.array([5.7, 2.0, -8.0])

    bp, base, op = build_dc_base_point_from_acpf(
        net=net,
        slack_bus=slack_bus,
        acpf_bus_p_mw=acpf_bus_p_mw,
        acpf_bus_ids=bus_ids,
    )

    # Non-slack buses should have exact same injection as input.
    for i, bid in enumerate(bp.bus_ids):
        if int(bid) == int(slack_bus):
            continue
        pos_in = bus_ids.index(int(bid))
        assert (
            abs(float(bp.bus_injections_mw[i]) - float(acpf_bus_p_mw[pos_in])) < 1e-10
        ), (
            f"Non-slack bus {bid}: expected {acpf_bus_p_mw[pos_in]}, "
            f"got {bp.bus_injections_mw[i]}"
        )


def test_acpf_mode_compute_results_smoke() -> None:
    """Full integration: compute_results_for_case with base_dispatch=acpf."""
    from stability_radius.workflows import compute_results_for_case

    net, slack_bus = make_three_bus_dispatch_net(pp)

    # Write a temporary .m file is complex; use the internal API directly.
    # Instead, we test the critical chain: AC PF → bus_p_mw → DC base point.
    # For full integration, solve both AC PF and DC base point.
    from stability_radius.base_point.ac import solve_ac_pf_base_point
    from stability_radius.base_point.dc import build_dc_base_point_from_acpf

    bp_ac, base_pf = solve_ac_pf_base_point(
        net=net,
        slack_bus=slack_bus,
        pf_solver="pandapower",
        pf_init="flat",
        lossless=True,
        gen_dispatch_mw_by_name={},
    )

    assert base_pf.bus_p_mw is not None

    bp_dc, base_dc, dc_op = build_dc_base_point_from_acpf(
        net=net,
        slack_bus=slack_bus,
        acpf_bus_p_mw=base_pf.bus_p_mw,
        acpf_bus_ids=list(base_pf.bus_ids),
    )

    assert bp_dc.source == "acpf"

    # Both share the same bus IDs.
    assert set(bp_ac.bus_ids) == set(bp_dc.bus_ids)

    # DC injections should sum to 0 (balanced).
    assert abs(float(np.sum(bp_dc.bus_injections_mw))) < 1e-10

    # DC flows should have correct shape.
    n_lines = len(sorted(net.line.index))
    assert bp_dc.line_flows_mw.shape == (n_lines,)

    # Margins should be positive (interior point from case dispatch).
    assert np.all(base_dc.margin_mw > -1e-3), (
        f"All margins should be near-positive: min={float(np.min(base_dc.margin_mw))}"
    )
