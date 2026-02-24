from __future__ import annotations

import math

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")
pytest.importorskip("pypsa")
pytest.importorskip("pandas")


def test_ac_l2_radius_smoke_small_net() -> None:
    """
    Smoke test:
    - build a tiny 2-bus system
    - solve AC PF via PyPSA
    - compute AC L2 radius (operator-based)

    Goal: ensure the new AC pipeline is importable and runs on a minimal case.
    """
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius

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
    lid = int(sorted(net.line.index)[0])
    net.line.loc[lid, "rateA"] = 100.0  # MVA

    base_pf = solve_ac_pf_base_point_from_pandapower(net=net, slack_bus=b0)

    res = compute_ac_l2_radius(net, base_pf=base_pf, slack_bus=b0, chunk_size=32)

    key = f"line_{lid}"
    assert key in res
    row = res[key]

    assert "radius_ac_l2" in row
    r = float(row["radius_ac_l2"])
    assert math.isfinite(r) or math.isinf(r)

    assert float(row["ac_s_limit_mva"]) > 0.0
    assert float(row["ac_s0_from_mva"]) >= 0.0
    assert float(row["ac_s0_to_mva"]) >= 0.0


def test_ac_l2_radius_near_zero_flow_keeps_nonzero_sensitivity() -> None:
    """
    Regression test for the |S|≈0 fallback (certificate soundness / symmetry).

    We build a 2-bus system with two parallel lines:
    - line_fast: normal impedance, carries essentially all (tiny) load
    - line_slow: extremely high reactance, carries ~0 flow => |S0| < eps at both ends

    Contract:
    - The computed sensitivity norm must remain strictly positive (||h||2 > 0).
    - The resulting radius must be > 0 (finite or +inf).
    """
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius

    net = pp.create_empty_network(sn_mva=100.0)
    b0 = int(pp.create_bus(net, vn_kv=110.0))
    b1 = int(pp.create_bus(net, vn_kv=110.0))

    pp.create_ext_grid(net, b0, vm_pu=1.0)

    # Tiny load to make one branch's flow truly ~0 (below the internal |S0| epsilon).
    pp.create_load(net, b1, p_mw=1.0e-6, q_mvar=0.0)

    common = dict(
        length_km=1.0,
        r_ohm_per_km=0.01,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )

    # Normal line.
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1, x_ohm_per_km=0.10, **common
    )
    # Almost-open line (huge reactance) -> near-zero flow.
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1, x_ohm_per_km=1.0e8, **common
    )

    line_ids = [int(x) for x in sorted(net.line.index)]
    assert len(line_ids) == 2
    lid_slow = int(line_ids[1])

    # Explicit MVA limits (avoid dependence on current-based fallback).
    net.line.loc[line_ids, "rateA"] = 100.0

    # Use pandapower PF backend to keep the base point robust for extreme impedances.
    base_pf = solve_ac_pf_base_point_from_pandapower(
        net=net,
        slack_bus=b0,
        solver="pandapower",
        init="flat",
        lossless=True,
    )

    res = compute_ac_l2_radius(net, base_pf=base_pf, slack_bus=b0, chunk_size=32)

    row = res[f"line_{lid_slow}"]

    # Ensure we really are in the near-zero-flow regime for that line.
    assert float(row["ac_s0_from_mva"]) < 1.0e-9
    assert float(row["ac_s0_to_mva"]) < 1.0e-9

    assert float(row["||h||2"]) > 0.0
    assert float(row["radius_ac_l2"]) > 0.0
