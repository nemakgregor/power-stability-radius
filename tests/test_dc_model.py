from __future__ import annotations

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")


def _make_small_net():
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    b2 = pp.create_bus(net, vn_kv=110.0)

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_load(net, b1, p_mw=10.0, q_mvar=0.0)
    pp.create_load(net, b2, p_mw=5.0, q_mvar=0.0)

    # Use parameters that produce a valid runpp and valid x_total for DC.
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
    pp.create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=1.0,
        r_ohm_per_km=0.01,
        x_ohm_per_km=0.10,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )
    return net, b0


def test_build_dc_matrices_shape_and_slack_column():
    from stability_radius.dc.dc_model import build_dc_matrices

    net, slack_bus = _make_small_net()
    H, _ = build_dc_matrices(net, slack_bus=slack_bus)

    assert H.shape == (len(net.line), len(net.bus))
    # Slack column should be all zeros
    slack_pos = list(net.bus.index).index(slack_bus)
    assert np.allclose(H[:, slack_pos], 0.0)
