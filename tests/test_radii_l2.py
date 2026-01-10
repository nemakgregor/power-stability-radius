from __future__ import annotations

import math
import pytest

pp = pytest.importorskip("pandapower")


def _make_small_net():
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_load(net, b1, p_mw=5.0, q_mvar=0.0)

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


def test_compute_l2_radius_returns_expected_keys():
    from stability_radius.dc.dc_model import build_dc_matrices
    from stability_radius.radii.l2 import compute_l2_radius

    net, slack_bus = _make_small_net()
    H, _ = build_dc_matrices(net, slack_bus=slack_bus)
    res = compute_l2_radius(net, H, margin_factor=1.0)

    assert len(res) == len(net.line)
    key = f"line_{net.line.index[0]}"
    assert key in res

    row = res[key]
    for required in ("p0_mw", "p_limit_mw_est", "margin_mw", "norm_g", "radius_l2"):
        assert required in row

    assert row["norm_g"] >= 0.0
    assert row["margin_mw"] >= 0.0
    assert math.isfinite(row["radius_l2"]) or math.isinf(row["radius_l2"])


def test_compute_l2_radius_validates_margin_factor():
    from stability_radius.radii.l2 import compute_l2_radius
    import numpy as np

    net, _ = _make_small_net()
    H = np.zeros((len(net.line), len(net.bus)))

    with pytest.raises(ValueError):
        compute_l2_radius(net, H, margin_factor=0.0)
