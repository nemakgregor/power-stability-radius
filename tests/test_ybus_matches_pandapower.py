from __future__ import annotations

"""
Regression test: the certificate's series-only Ybus must equal pandapower's
internal Ybus after the lossless policy is applied.

Historical bug: ``apply_lossless_policy_to_pandapower_net`` did not zero the
transformer magnetizing / iron-loss shunt (``i0_percent``, ``pfe_kw``), so
pandapower's PF included shunt admittances the certificate Jacobian never
modeled (e.g. pglib case118: i0_percent=-3.156 on two transformers produced
diagonal Ybus mismatches of ~0.08 pu).
"""

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")


def _make_net_with_trafo_shunts() -> tuple[object, int]:
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = int(pp.create_bus(net, vn_kv=110.0))
    b1 = int(pp.create_bus(net, vn_kv=110.0))
    b2 = int(pp.create_bus(net, vn_kv=20.0))
    b3 = int(pp.create_bus(net, vn_kv=20.0))

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_load(net, b1, p_mw=20.0, q_mvar=5.0)
    pp.create_load(net, b3, p_mw=10.0, q_mvar=2.0)

    common = dict(length_km=1.0, max_i_ka=1.0, max_loading_percent=100.0)
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        r_ohm_per_km=0.05,
        x_ohm_per_km=0.20,
        c_nf_per_km=200.0,
        **common,
    )
    pp.create_line_from_parameters(
        net,
        from_bus=b2,
        to_bus=b3,
        r_ohm_per_km=0.04,
        x_ohm_per_km=0.15,
        c_nf_per_km=150.0,
        **common,
    )
    # Transformer WITH series resistance, magnetizing current, and iron losses.
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        sn_mva=40.0,
        vn_hv_kv=110.0,
        vn_lv_kv=20.0,
        vk_percent=10.0,
        vkr_percent=0.5,
        pfe_kw=30.0,
        i0_percent=0.4,
    )
    # Bus shunt (must be disabled by the policy).
    pp.create_shunt(net, b1, q_mvar=-10.0, p_mw=0.0)
    net.line.loc[:, "rateA"] = 1000.0
    return net, b0


def test_policy_zeroes_trafo_shunt_parameters() -> None:
    from stability_radius.base_point.pandapower_tools import (
        apply_lossless_policy_to_pandapower_net,
    )

    net, _ = _make_net_with_trafo_shunts()
    nn = apply_lossless_policy_to_pandapower_net(net)

    assert float(nn.trafo["vkr_percent"].abs().max()) == 0.0
    assert float(nn.trafo["i0_percent"].abs().max()) == 0.0
    assert float(nn.trafo["pfe_kw"].abs().max()) == 0.0
    assert not bool(nn.shunt["in_service"].any())
    assert float(nn.line["r_ohm_per_km"].abs().max()) == 0.0
    assert float(nn.line["c_nf_per_km"].abs().max()) == 0.0


def test_policy_fails_fast_on_trafo3w() -> None:
    from stability_radius.base_point.pandapower_tools import (
        apply_lossless_policy_to_pandapower_net,
    )

    net = pp.create_empty_network(sn_mva=100.0)
    b0 = int(pp.create_bus(net, vn_kv=110.0))
    b1 = int(pp.create_bus(net, vn_kv=20.0))
    b2 = int(pp.create_bus(net, vn_kv=10.0))
    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_transformer3w_from_parameters(
        net,
        hv_bus=b0,
        mv_bus=b1,
        lv_bus=b2,
        vn_hv_kv=110.0,
        vn_mv_kv=20.0,
        vn_lv_kv=10.0,
        sn_hv_mva=40.0,
        sn_mv_mva=20.0,
        sn_lv_mva=20.0,
        vk_hv_percent=10.0,
        vk_mv_percent=10.0,
        vk_lv_percent=10.0,
        vkr_hv_percent=0.3,
        vkr_mv_percent=0.3,
        vkr_lv_percent=0.3,
        pfe_kw=0.0,
        i0_percent=0.0,
    )
    with pytest.raises(ValueError, match="trafo3w"):
        apply_lossless_policy_to_pandapower_net(net)


def test_ybus_matches_pandapower_internal_after_policy() -> None:
    from stability_radius.ac.ac_model import build_ac_operator
    from stability_radius.base_point.pandapower_tools import (
        apply_lossless_policy_to_pandapower_net,
    )

    net, slack = _make_net_with_trafo_shunts()
    nn = apply_lossless_policy_to_pandapower_net(net)

    pp.runpp(
        nn,
        calculate_voltage_angles=True,
        enforce_q_lims=False,
        init="flat",
        trafo_model="pi",
        tolerance_mva=1e-10,
        max_iteration=100,
    )

    bus_ids = [int(x) for x in sorted(nn.bus.index)]
    lookup = nn._pd2ppc_lookups["bus"]
    idx = np.array([lookup[b] for b in bus_ids])
    y_pp = nn._ppc["internal"]["Ybus"].toarray()[np.ix_(idx, idx)]

    vm = np.array([float(nn.res_bus.vm_pu.loc[b]) for b in bus_ids])
    va = np.deg2rad(np.array([float(nn.res_bus.va_degree.loc[b]) for b in bus_ids]))
    op = build_ac_operator(net=nn, slack_bus=slack, vm_pu=vm, va_rad=va, lossless=True)
    y_op = np.asarray(op.Ybus.toarray())

    diff = np.abs(y_op - y_pp)
    assert float(diff.max()) < 1e-8, (
        f"certificate Ybus deviates from pandapower internal Ybus by "
        f"{float(diff.max()):.3e} (policy incomplete?)"
    )


def test_ybus_matches_pandapower_from_raw_net_production_path() -> None:
    """Production path: operator built from the RAW net (lossless=True) must
    still match the pandapower Ybus of the POLICIED net used for PF/replay.

    Catches the transformer |z|-vs-|x| convention mismatch: pandapower zeroes
    vkr BEFORE x = sqrt(z^2 - r^2), so the PF reactance is vk, not
    sqrt(vk^2 - vkr^2).
    """
    from stability_radius.ac.ac_model import build_ac_operator
    from stability_radius.base_point.pandapower_tools import (
        apply_lossless_policy_to_pandapower_net,
    )

    net_raw, slack = _make_net_with_trafo_shunts()
    nn = apply_lossless_policy_to_pandapower_net(net_raw)

    pp.runpp(
        nn,
        calculate_voltage_angles=True,
        enforce_q_lims=False,
        init="flat",
        trafo_model="pi",
        tolerance_mva=1e-10,
        max_iteration=100,
    )

    bus_ids = [int(x) for x in sorted(nn.bus.index)]
    lookup = nn._pd2ppc_lookups["bus"]
    idx = np.array([lookup[b] for b in bus_ids])
    y_pp = nn._ppc["internal"]["Ybus"].toarray()[np.ix_(idx, idx)]

    vm = np.array([float(nn.res_bus.vm_pu.loc[b]) for b in bus_ids])
    va = np.deg2rad(np.array([float(nn.res_bus.va_degree.loc[b]) for b in bus_ids]))
    # NOTE: raw net here, as in the production workflow.
    op = build_ac_operator(
        net=net_raw, slack_bus=slack, vm_pu=vm, va_rad=va, lossless=True
    )
    y_op = np.asarray(op.Ybus.toarray())

    diff = np.abs(y_op - y_pp)
    assert float(diff.max()) < 1e-8, (
        f"raw-net operator Ybus deviates from policied-PF Ybus by "
        f"{float(diff.max()):.3e} (trafo vkr convention mismatch?)"
    )
