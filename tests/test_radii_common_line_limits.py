from __future__ import annotations

import math

import pytest

pp = pytest.importorskip("pandapower")


def test_estimate_line_limit_mva_from_max_i_ka_and_vn_kv():
    from stability_radius.radii.common import estimate_line_limit_mva

    net = pp.create_empty_network(sn_mva=100.0)
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)

    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=0.01,
        x_ohm_per_km=0.10,
        c_nf_per_km=0.0,
        max_i_ka=0.5,
        max_loading_percent=80.0,
    )

    for col in ("rateA", "rate_a_mva", "sn_mva", "max_mva"):
        if col in net.line.columns:
            net.line.drop(columns=[col], inplace=True)

    lid = int(sorted(net.line.index)[0])
    row = net.line.loc[lid]

    limit = float(estimate_line_limit_mva(net, row))
    expected = math.sqrt(3.0) * 110.0 * 0.5 * 0.8
    assert limit == pytest.approx(expected)
