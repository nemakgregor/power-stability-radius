"""Tests for AC PF fail-fast behavior and metadata propagation.

Covers:
- Primary solve sets pf_attempt="primary", pf_repairs=[].
- A runpp failure raises immediately without changing model policy.
- Metadata propagates through BasePointAC.to_meta_dict().
- Metadata propagates through solve_ac_pf_base_point().
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")
pytest.importorskip("pypsa")
pytest.importorskip("pandas")


def _make_simple_net():
    """Create a minimal 2-bus network for AC PF testing."""
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = int(pp.create_bus(net, vn_kv=110.0))
    b1 = int(pp.create_bus(net, vn_kv=110.0))

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_load(net, b1, p_mw=10.0, q_mvar=2.0)

    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=0.01,
        x_ohm_per_km=0.10,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )

    return net, b0


def _make_chain_net(n_buses: int):
    net = pp.create_empty_network(sn_mva=100.0)
    buses = [int(pp.create_bus(net, vn_kv=110.0)) for _ in range(n_buses)]
    pp.create_ext_grid(net, buses[0], vm_pu=1.0)

    for idx in range(1, n_buses):
        pp.create_load(net, buses[idx], p_mw=0.1, q_mvar=0.0)
        pp.create_line_from_parameters(
            net,
            from_bus=buses[idx - 1],
            to_bus=buses[idx],
            length_km=1.0,
            r_ohm_per_km=0.01,
            x_ohm_per_km=0.10,
            c_nf_per_km=0.0,
            max_i_ka=1.0,
            max_loading_percent=100.0,
        )

    return net, buses[0]


def _populate_fake_pp_results(nn) -> None:
    nn.converged = True
    nn.res_bus = pd.DataFrame(
        {
            "vm_pu": 1.0,
            "va_degree": 0.0,
            "p_mw": 0.0,
            "q_mvar": 0.0,
        },
        index=nn.bus.index,
    )
    nn.res_line = pd.DataFrame(
        {
            "p_from_mw": 0.0,
            "q_from_mvar": 0.0,
            "p_to_mw": 0.0,
            "q_to_mvar": 0.0,
        },
        index=nn.line.index,
    )


# ---------- Tests for primary solve ----------


def test_primary_solve_metadata() -> None:
    """When PF converges on the first try, metadata reflects primary attempt."""
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )

    net, slack_bus = _make_simple_net()

    result = solve_ac_pf_base_point_from_pandapower(
        net=net,
        slack_bus=slack_bus,
        solver="pandapower",
        init="flat",
        lossless=True,
    )

    assert result.pf_attempt == "primary"
    assert result.pf_repairs is None or len(result.pf_repairs) == 0


def test_runpp_failure_is_fail_fast_without_retry() -> None:
    """A failed AC PF solve must not retry with altered model settings."""
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )

    net, slack_bus = _make_simple_net()

    with patch("pandapower.runpp", side_effect=RuntimeError("boom")) as runpp:
        with pytest.raises(RuntimeError, match="primary solve"):
            solve_ac_pf_base_point_from_pandapower(
                net=net,
                slack_bus=slack_bus,
                solver="pandapower",
                init="flat",
                lossless=True,
            )

    assert runpp.call_count == 1


# ---------- Tests for BasePointAC.to_meta_dict propagation ----------


def test_base_point_ac_meta_dict_propagates_repair_fields() -> None:
    """BasePointAC.to_meta_dict must include pf_attempt and pf_repairs."""
    from stability_radius.base_point.types import BasePointAC

    bp = BasePointAC(
        pf_solver="pandapower",
        pf_init="flat",
        lossless=True,
        slack_bus=0,
        bus_ids=(0, 1),
        vm_pu=np.array([1.0, 0.99]),
        va_rad=np.array([0.0, -0.01]),
        line_ids=(0,),
        p_from_mw=np.array([10.0]),
        q_from_mvar=np.array([2.0]),
        p_to_mw=np.array([-9.9]),
        q_to_mvar=np.array([-1.9]),
        s_limit_mva=np.array([100.0]),
        status="PP_PF_OK",
        pf_attempt="primary",
        pf_repairs=("distributed_slack_auto_disabled_large_network",),
    )

    meta = bp.to_meta_dict()

    assert meta["pf_attempt"] == "primary"
    assert meta["pf_repairs"] == ["distributed_slack_auto_disabled_large_network"]
    assert meta["status"] == "PP_PF_OK"
    assert meta["pf_solver"] == "pandapower"


def test_base_point_ac_meta_dict_default_repair() -> None:
    """Default BasePointAC has pf_attempt='primary' and empty pf_repairs."""
    from stability_radius.base_point.types import BasePointAC

    bp = BasePointAC(
        pf_solver="pandapower",
        pf_init="flat",
        lossless=True,
        slack_bus=0,
        bus_ids=(0,),
        vm_pu=np.array([1.0]),
        va_rad=np.array([0.0]),
        line_ids=(0,),
        p_from_mw=np.array([0.0]),
        q_from_mvar=np.array([0.0]),
        p_to_mw=np.array([0.0]),
        q_to_mvar=np.array([0.0]),
        s_limit_mva=np.array([100.0]),
        status="ok",
    )

    meta = bp.to_meta_dict()
    assert meta["pf_attempt"] == "primary"
    assert meta["pf_repairs"] == []


# ---------- Tests for solve_ac_pf_base_point propagation ----------


def test_solve_ac_pf_base_point_propagates_metadata() -> None:
    """solve_ac_pf_base_point should propagate pf_attempt/pf_repairs to BasePointAC."""
    from stability_radius.base_point.ac import solve_ac_pf_base_point

    net, slack_bus = _make_simple_net()

    bp, raw = solve_ac_pf_base_point(
        net=net,
        slack_bus=slack_bus,
        pf_solver="pandapower",
        pf_init="flat",
        lossless=True,
        gen_dispatch_mw_by_name={},
    )

    # Both the raw result and BasePointAC should have consistent metadata.
    assert bp.pf_attempt == raw.pf_attempt
    # pf_repairs may be None in raw or () in BasePointAC.
    raw_repairs = list(raw.pf_repairs or [])
    bp_repairs = list(bp.pf_repairs)
    assert raw_repairs == bp_repairs


# ---------- Test for PyPSAAPFResult fields ----------


def test_pypsa_apf_result_default_fields() -> None:
    """PyPSAAPFResult default pf_attempt and pf_repairs."""
    from stability_radius.base_point.pypsa_pf import PyPSAAPFResult

    result = PyPSAAPFResult(
        bus_ids=(0,),
        v_mag_pu=np.array([1.0]),
        v_ang_rad=np.array([0.0]),
        line_ids=(0,),
        line_p0_mw=np.array([0.0]),
        line_q0_mvar=np.array([0.0]),
        line_p1_mw=np.array([0.0]),
        line_q1_mvar=np.array([0.0]),
        status="ok",
    )

    assert result.pf_attempt == "primary"
    assert result.pf_repairs is None


def test_pypsa_apf_result_explicit_repair_metadata() -> None:
    """PyPSAAPFResult can record non-solver repair metadata."""
    from stability_radius.base_point.pypsa_pf import PyPSAAPFResult

    result = PyPSAAPFResult(
        bus_ids=(0, 1),
        v_mag_pu=np.array([1.0, 0.99]),
        v_ang_rad=np.array([0.0, -0.01]),
        line_ids=(0,),
        line_p0_mw=np.array([5.0]),
        line_q0_mvar=np.array([1.0]),
        line_p1_mw=np.array([-4.9]),
        line_q1_mvar=np.array([-0.9]),
        status="PP_PF_OK",
        pf_attempt="primary",
        pf_repairs=["distributed_slack_auto_disabled_large_network"],
    )

    assert result.pf_attempt == "primary"
    assert result.pf_repairs == ["distributed_slack_auto_disabled_large_network"]


def test_distributed_slack_metadata_reports_requested_and_used() -> None:
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )

    net, slack_bus = _make_simple_net()

    def _fake_runpp(nn, **kwargs):
        _populate_fake_pp_results(nn)

    with patch("pandapower.runpp", side_effect=_fake_runpp):
        result = solve_ac_pf_base_point_from_pandapower(
            net=net,
            slack_bus=slack_bus,
            solver="pandapower",
            init="flat",
            lossless=True,
            distributed_slack=True,
        )

    assert result.distributed_slack_requested is True
    assert result.distributed_slack_used is True
    assert "distributed_slack_auto_disabled_large_network" not in (
        result.pf_repairs or []
    )


def test_large_network_auto_disables_distributed_slack_and_propagates_meta() -> None:
    from stability_radius.base_point.ac import solve_ac_pf_base_point

    net, slack_bus = _make_chain_net(301)

    def _fake_runpp(nn, **kwargs):
        _populate_fake_pp_results(nn)

    with patch("pandapower.runpp", side_effect=_fake_runpp):
        bp, raw = solve_ac_pf_base_point(
            net=net,
            slack_bus=slack_bus,
            pf_solver="pandapower",
            pf_init="flat",
            lossless=True,
            gen_dispatch_mw_by_name={},
            distributed_slack=True,
        )

    assert raw.distributed_slack_requested is True
    assert raw.distributed_slack_used is False
    assert "distributed_slack_auto_disabled_large_network" in (raw.pf_repairs or [])

    assert bp.distributed_slack_requested is True
    assert bp.distributed_slack_used is False
    meta = bp.to_meta_dict()
    assert meta["distributed_slack_requested"] is True
    assert meta["distributed_slack_used"] is False
    assert "distributed_slack_auto_disabled_large_network" in meta["pf_repairs"]
