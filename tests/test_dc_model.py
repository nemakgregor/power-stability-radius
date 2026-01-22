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


def test_build_dc_operator_accepts_negative_reactance():
    pytest.importorskip("scipy")

    from stability_radius.dc.dc_model import build_dc_operator

    net = pp.create_empty_network(sn_mva=100.0)
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_load(net, b1, p_mw=10.0, q_mvar=0.0)

    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=0.01,
        x_ohm_per_km=-0.10,  # negative reactance should not be forced to b=0
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )

    op = build_dc_operator(net, slack_bus=b0)
    assert op.n_bus == 2
    assert op.n_line == 1
    assert np.isfinite(op.b[0])
    assert op.b[0] < 0.0


def test_build_dc_operator_uses_trafo_in_b_matrix_to_avoid_singularity():
    """
    Regression: pandapower can convert MATPOWER/PGLib branches into transformers.
    If DC B matrix is built from net.line only, the reduced B can become singular.

    We create a network where the slack is connected to the monitored line component
    ONLY via a transformer. Operator construction must succeed.
    """
    pytest.importorskip("scipy")

    from stability_radius.dc.dc_model import build_dc_operator

    net = pp.create_empty_network(sn_mva=100.0)

    b_hv = pp.create_bus(net, vn_kv=110.0)
    b_lv = pp.create_bus(net, vn_kv=10.0)
    b2 = pp.create_bus(net, vn_kv=10.0)

    pp.create_ext_grid(net, b_hv, vm_pu=1.0)
    pp.create_load(net, b2, p_mw=5.0, q_mvar=0.0)

    # Transformer connects slack component to the line component.
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b_hv,
        lv_bus=b_lv,
        sn_mva=40.0,
        vn_hv_kv=110.0,
        vn_lv_kv=10.0,
        vk_percent=10.0,
        vkr_percent=0.5,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=0.0,
        tap_side="hv",
        tap_neutral=0,
        tap_min=0,
        tap_max=0,
        tap_step_percent=1.0,
        tap_pos=0,
    )

    # Monitored line is only on LV side; without trafo in B it would be disconnected from slack.
    pp.create_line_from_parameters(
        net,
        from_bus=b_lv,
        to_bus=b2,
        length_km=1.0,
        r_ohm_per_km=0.01,
        x_ohm_per_km=0.10,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )

    op = build_dc_operator(net, slack_bus=b_hv)
    assert op.n_bus == 3
    assert op.n_line == 1
