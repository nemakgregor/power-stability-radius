from __future__ import annotations

import copy

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")
pytest.importorskip("pypsa")
pytest.importorskip("pandas")
pytest.importorskip("highspy")

from stability_radius.dc.dc_model import build_dc_matrices
from stability_radius.radii.common import get_line_base_quantities
from stability_radius.radii.l2 import compute_l2_radius


def _make_multivoltage_cycle_net():
    """
    Meshed multi-voltage network (small) for unit-consistency tests.

    Topology (cycle across voltage levels):
      110kV: b0 --line-- b1
      trafo: b1 (110kV) -> b2 (10kV)
      10kV:  b2 --line-- b3
      trafo: b0 (110kV) -> b3 (10kV)

    Important:
    - we set explicit `rateA` so line limits do NOT depend on vn_kv (avoid mixing concerns).
    """
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = int(pp.create_bus(net, vn_kv=110.0))
    b1 = int(pp.create_bus(net, vn_kv=110.0))
    b2 = int(pp.create_bus(net, vn_kv=10.0))
    b3 = int(pp.create_bus(net, vn_kv=10.0))

    pp.create_ext_grid(net, b0, vm_pu=1.0)

    pp.create_load(net, b2, p_mw=5.0, q_mvar=0.0)
    pp.create_load(net, b3, p_mw=5.0, q_mvar=0.0)

    common_line = dict(
        length_km=1.0,
        r_ohm_per_km=0.01,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )

    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1, x_ohm_per_km=0.10, **common_line
    )
    pp.create_line_from_parameters(
        net, from_bus=b2, to_bus=b3, x_ohm_per_km=0.02, **common_line
    )

    trafo_common = dict(
        sn_mva=40.0,
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

    pp.create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        vn_hv_kv=110.0,
        vn_lv_kv=10.0,
        **trafo_common,
    )
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b0,
        lv_bus=b3,
        vn_hv_kv=110.0,
        vn_lv_kv=10.0,
        **trafo_common,
    )

    net.line.loc[:, "rateA"] = 100.0

    return net, b0


def _scale_vn_and_x(net: object, *, k: float) -> object:
    """
    Unit-consistency scaling transform:

    - vn_kv := k * vn_kv
    - x_ohm := k^2 * x_ohm  (implemented via x_ohm_per_km scaling)

    This keeps b = V^2 / X invariant.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    net2 = copy.deepcopy(net)

    net2.bus.loc[:, "vn_kv"] = net2.bus["vn_kv"].astype(float) * float(k)

    if hasattr(net2, "line") and net2.line is not None and len(net2.line):
        net2.line.loc[:, "x_ohm_per_km"] = net2.line["x_ohm_per_km"].astype(
            float
        ) * float(k * k)
        if "r_ohm_per_km" in net2.line.columns:
            net2.line.loc[:, "r_ohm_per_km"] = net2.line["r_ohm_per_km"].astype(
                float
            ) * float(k * k)

    if hasattr(net2, "trafo") and net2.trafo is not None and len(net2.trafo):
        if "vn_hv_kv" in net2.trafo.columns:
            net2.trafo.loc[:, "vn_hv_kv"] = net2.trafo["vn_hv_kv"].astype(
                float
            ) * float(k)
        if "vn_lv_kv" in net2.trafo.columns:
            net2.trafo.loc[:, "vn_lv_kv"] = net2.trafo["vn_lv_kv"].astype(
                float
            ) * float(k)

    return net2


def _r_star_from_l2_results(results: dict, *, line_indices: list[int]) -> float:
    radii = np.asarray(
        [float(results[f"line_{lid}"]["radius_l2"]) for lid in line_indices],
        dtype=float,
    )
    finite = np.isfinite(radii)
    assert bool(np.any(finite))
    return float(np.min(radii[finite]))


def test_unit_consistency_vn_kv_and_x_ohm_scaling_invariance_end_to_end() -> None:
    """
    End-to-end unit-consistency regression test.

    If we scale:
      - all vn_kv by k
      - all x_ohm by k^2  (so b=V^2/X does not change)

    then the DCOperator/PTDF matrix H, OPF base flows f0 and the L2 certificate
    must be invariant (up to tiny numerical noise).
    """
    net, slack_bus = _make_multivoltage_cycle_net()

    base1 = get_line_base_quantities(net, limit_factor=1.0)
    H1, _ = build_dc_matrices(
        net, slack_bus=slack_bus, dtype=np.float64, chunk_size=256
    )
    l2_1 = compute_l2_radius(net, H1, limit_factor=1.0, base=base1)
    r_star_1 = _r_star_from_l2_results(l2_1, line_indices=base1.line_indices)

    k = 3.0
    net2 = _scale_vn_and_x(net, k=k)

    base2 = get_line_base_quantities(net2, limit_factor=1.0)
    H2, _ = build_dc_matrices(
        net2, slack_bus=slack_bus, dtype=np.float64, chunk_size=256
    )
    l2_2 = compute_l2_radius(net2, H2, limit_factor=1.0, base=base2)
    r_star_2 = _r_star_from_l2_results(l2_2, line_indices=base2.line_indices)

    assert np.allclose(H1, H2, atol=1e-9, rtol=0.0)
    assert np.allclose(base1.flow0_mw, base2.flow0_mw, atol=1e-6, rtol=0.0)
    assert r_star_1 == pytest.approx(r_star_2, abs=1e-6)
