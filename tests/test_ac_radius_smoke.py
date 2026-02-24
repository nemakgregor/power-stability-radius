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
