from __future__ import annotations

"""
Finite-difference validation of adjoint h-vectors (sensitivity of |S| to injections).

The AC L2 certificate computes h = J^{-T} · b where:
  - J is the reduced PF Jacobian  (rows = [P,Q] equations, cols = [θ,V] variables)
  - b = ∂|S|/∂x is the state-space gradient of line-end apparent power

The result h is the injection-space gradient: h = ∂|S|/∂u, where u = [ΔP; ΔQ].
This means:
    Δ|S| ≈ h^T · δu   for small δu

This test verifies that identity by:
1) Building an ACOperator on a small network.
2) Computing h-vectors via compute_ac_l2_radius(..., return_h_vectors=True).
3) Applying a tiny balanced perturbation δu and re-running pandapower AC PF.
4) Comparing the linear prediction h^T δu against the actual Δ|S| from PF.
"""

import logging
import math

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")
pytest.importorskip("pypsa")

logger = logging.getLogger(__name__)


def _run_pp_pf(net: object, *, init: str) -> None:
    """Solve AC PF via pandapower with tight tolerance."""
    pp.runpp(
        net,
        algorithm="nr",
        calculate_voltage_angles=True,
        enforce_q_lims=False,
        init=str(init),
        max_iteration=50,
        tolerance_mva=1e-10,
    )
    assert bool(getattr(net, "converged", True))


def _make_3bus_meshed_net() -> tuple[object, int]:
    """Create a small deterministic 3-bus triangle network (lossless)."""
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = int(pp.create_bus(net, vn_kv=110.0))
    b1 = int(pp.create_bus(net, vn_kv=110.0))
    b2 = int(pp.create_bus(net, vn_kv=110.0))

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_load(net, b1, p_mw=20.0, q_mvar=5.0)
    pp.create_load(net, b2, p_mw=15.0, q_mvar=3.0)

    common = dict(
        length_km=1.0,
        r_ohm_per_km=0.0,  # lossless
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )

    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1, x_ohm_per_km=0.10, **common
    )
    pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, x_ohm_per_km=0.13, **common
    )
    pp.create_line_from_parameters(
        net, from_bus=b2, to_bus=b0, x_ohm_per_km=0.11, **common
    )

    net.line.loc[:, "rateA"] = 1000.0

    return net, b0


def test_h_vector_predicts_delta_s_via_finite_difference() -> None:
    """
    Gold-standard check: h^T δu ≈ Δ|S| for each line end.

    This validates that the h-vectors from compute_ac_l2_radius are
    injection-space sensitivities (not state-space).
    """
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius
    from stability_radius.workflows import _expand_h_reduced_to_full

    net, slack_bus = _make_3bus_meshed_net()

    # Solve base-point PF
    _run_pp_pf(net, init="flat")

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    line_ids = [int(x) for x in sorted(net.line.index)]
    n_bus = len(bus_ids)
    n_lines = len(line_ids)

    # Get base-point PF result via the library solver
    base_pf = solve_ac_pf_base_point_from_pandapower(
        net=net, slack_bus=slack_bus, solver="pandapower", init="flat", lossless=True
    )

    # Compute h-vectors
    ac_results = compute_ac_l2_radius(
        net,
        base_pf=base_pf,
        slack_bus=slack_bus,
        lossless=True,
        return_h_vectors=True,
    )

    h_vecs_raw = ac_results.pop("_h_vectors")
    h_from_raw = h_vecs_raw["h_from"]  # (n_lines, 2*n_red)
    h_to_raw = h_vecs_raw["h_to"]  # (n_lines, 2*n_red)

    # Expand to full dimension
    slack_pos = bus_ids.index(slack_bus)
    h_from_full = _expand_h_reduced_to_full(
        h_from_raw, n_bus=n_bus, slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )
    h_to_full = _expand_h_reduced_to_full(
        h_to_raw, n_bus=n_bus, slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )

    assert h_from_full.shape == (n_lines, 2 * n_bus)
    assert h_to_full.shape == (n_lines, 2 * n_bus)

    # Base |S| from PF
    s0_from = np.sqrt(
        np.array(base_pf.line_p0_mw) ** 2 + np.array(base_pf.line_q0_mvar) ** 2
    )
    s0_to = np.sqrt(
        np.array(base_pf.line_p1_mw) ** 2 + np.array(base_pf.line_q1_mvar) ** 2
    )

    # ---- Finite-difference test with balanced perturbation ----
    eps = 0.01  # MW — small enough for linear regime

    # Balanced P perturbation (sum = 0): inject at b1, withdraw at b2
    delta_p = np.zeros(n_bus, dtype=float)
    delta_p[bus_ids.index(1)] = +eps
    delta_p[bus_ids.index(2)] = -eps
    delta_q = np.zeros(n_bus, dtype=float)
    delta_u = np.concatenate([delta_p, delta_q])

    # Apply perturbation via sgen
    import copy

    net_pert = copy.deepcopy(net)
    for pos, bid in enumerate(bus_ids):
        pp.create_sgen(
            net_pert,
            bus=int(bid),
            p_mw=float(delta_p[pos]),
            q_mvar=float(delta_q[pos]),
            in_service=True,
        )

    _run_pp_pf(net_pert, init="flat")

    # Perturbed |S|
    s1_from = np.array(
        [
            math.sqrt(
                float(net_pert.res_line.loc[lid, "p_from_mw"]) ** 2
                + float(net_pert.res_line.loc[lid, "q_from_mvar"]) ** 2
            )
            for lid in line_ids
        ]
    )
    s1_to = np.array(
        [
            math.sqrt(
                float(net_pert.res_line.loc[lid, "p_to_mw"]) ** 2
                + float(net_pert.res_line.loc[lid, "q_to_mvar"]) ** 2
            )
            for lid in line_ids
        ]
    )

    # Actual ΔS
    ds_from_actual = s1_from - s0_from
    ds_to_actual = s1_to - s0_to

    # Linear predictions
    ds_from_pred = h_from_full @ delta_u
    ds_to_pred = h_to_full @ delta_u

    # Compare
    for pos in range(n_lines):
        if abs(ds_from_actual[pos]) > 1e-10:
            err_from = abs(ds_from_pred[pos] - ds_from_actual[pos]) / abs(
                ds_from_actual[pos]
            )
            logger.info(
                "Line %d from-end: dS_pred=%.6e, dS_actual=%.6e, rel_err=%.4f",
                line_ids[pos],
                ds_from_pred[pos],
                ds_from_actual[pos],
                err_from,
            )
            assert err_from < 0.05, (
                f"Line {line_ids[pos]} from-end: h-vector FD mismatch "
                f"(pred={ds_from_pred[pos]:.6e}, actual={ds_from_actual[pos]:.6e}, "
                f"rel_err={err_from:.4f})"
            )

        if abs(ds_to_actual[pos]) > 1e-10:
            err_to = abs(ds_to_pred[pos] - ds_to_actual[pos]) / abs(ds_to_actual[pos])
            logger.info(
                "Line %d to-end: dS_pred=%.6e, dS_actual=%.6e, rel_err=%.4f",
                line_ids[pos],
                ds_to_pred[pos],
                ds_to_actual[pos],
                err_to,
            )
            assert err_to < 0.05, (
                f"Line {line_ids[pos]} to-end: h-vector FD mismatch "
                f"(pred={ds_to_pred[pos]:.6e}, actual={ds_to_actual[pos]:.6e}, "
                f"rel_err={err_to:.4f})"
            )

    # At least one line should have a non-trivial ΔS
    assert np.any(np.abs(ds_from_actual) > 1e-10) or np.any(
        np.abs(ds_to_actual) > 1e-10
    ), "Degenerate: no detectable flow change"


def test_h_vector_with_q_perturbation() -> None:
    """
    Additional FD check with a reactive power perturbation.

    This catches bugs where P/Q blocks are swapped or misaligned.
    """
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius
    from stability_radius.workflows import _expand_h_reduced_to_full

    net, slack_bus = _make_3bus_meshed_net()
    _run_pp_pf(net, init="flat")

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    line_ids = [int(x) for x in sorted(net.line.index)]
    n_bus = len(bus_ids)

    base_pf = solve_ac_pf_base_point_from_pandapower(
        net=net, slack_bus=slack_bus, solver="pandapower", init="flat", lossless=True
    )

    ac_results = compute_ac_l2_radius(
        net,
        base_pf=base_pf,
        slack_bus=slack_bus,
        lossless=True,
        return_h_vectors=True,
    )

    h_vecs_raw = ac_results.pop("_h_vectors")
    slack_pos = bus_ids.index(slack_bus)
    h_from_full = _expand_h_reduced_to_full(
        h_vecs_raw["h_from"], n_bus=n_bus, slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )
    h_to_full = _expand_h_reduced_to_full(
        h_vecs_raw["h_to"], n_bus=n_bus, slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )

    s0_from = np.sqrt(
        np.array(base_pf.line_p0_mw) ** 2 + np.array(base_pf.line_q0_mvar) ** 2
    )
    s0_to = np.sqrt(
        np.array(base_pf.line_p1_mw) ** 2 + np.array(base_pf.line_q1_mvar) ** 2
    )

    # Balanced Q perturbation
    eps = 0.01
    delta_p = np.zeros(n_bus, dtype=float)
    delta_q = np.zeros(n_bus, dtype=float)
    delta_q[bus_ids.index(1)] = +eps
    delta_q[bus_ids.index(2)] = -eps
    delta_u = np.concatenate([delta_p, delta_q])

    import copy

    net_pert = copy.deepcopy(net)
    for pos, bid in enumerate(bus_ids):
        pp.create_sgen(
            net_pert,
            bus=int(bid),
            p_mw=float(delta_p[pos]),
            q_mvar=float(delta_q[pos]),
            in_service=True,
        )

    _run_pp_pf(net_pert, init="flat")

    s1_from = np.array(
        [
            math.sqrt(
                float(net_pert.res_line.loc[lid, "p_from_mw"]) ** 2
                + float(net_pert.res_line.loc[lid, "q_from_mvar"]) ** 2
            )
            for lid in line_ids
        ]
    )
    s1_to = np.array(
        [
            math.sqrt(
                float(net_pert.res_line.loc[lid, "p_to_mw"]) ** 2
                + float(net_pert.res_line.loc[lid, "q_to_mvar"]) ** 2
            )
            for lid in line_ids
        ]
    )

    ds_from_actual = s1_from - s0_from
    ds_to_actual = s1_to - s0_to
    ds_from_pred = h_from_full @ delta_u
    ds_to_pred = h_to_full @ delta_u

    for pos in range(len(line_ids)):
        if abs(ds_from_actual[pos]) > 1e-10:
            err = abs(ds_from_pred[pos] - ds_from_actual[pos]) / abs(
                ds_from_actual[pos]
            )
            assert err < 0.05, (
                f"Line {line_ids[pos]} from-end Q-pert: rel_err={err:.4f}"
            )

        if abs(ds_to_actual[pos]) > 1e-10:
            err = abs(ds_to_pred[pos] - ds_to_actual[pos]) / abs(ds_to_actual[pos])
            assert err < 0.05, f"Line {line_ids[pos]} to-end Q-pert: rel_err={err:.4f}"


def _make_5bus_with_shunts_and_charging() -> tuple[object, int]:
    """Create a 5-bus network with bus shunts and line charging.

    This exercises the lossless policy: the series-only Jacobian does NOT
    model shunts or line charging, so the PP network must have these
    elements disabled via apply_lossless_policy before FD comparison.
    """
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = int(pp.create_bus(net, vn_kv=110.0))
    b1 = int(pp.create_bus(net, vn_kv=110.0))
    b2 = int(pp.create_bus(net, vn_kv=110.0))
    b3 = int(pp.create_bus(net, vn_kv=110.0))
    b4 = int(pp.create_bus(net, vn_kv=110.0))

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_load(net, b1, p_mw=30.0, q_mvar=10.0)
    pp.create_load(net, b2, p_mw=20.0, q_mvar=5.0)
    pp.create_load(net, b3, p_mw=25.0, q_mvar=8.0)
    pp.create_load(net, b4, p_mw=10.0, q_mvar=2.0)

    # Lines with non-zero r and c_nf (will be zeroed by lossless policy)
    common = dict(length_km=10.0, max_i_ka=1.0, max_loading_percent=100.0)
    pp.create_line_from_parameters(
        net, b0, b1, r_ohm_per_km=0.05, x_ohm_per_km=0.20, c_nf_per_km=200.0, **common
    )
    pp.create_line_from_parameters(
        net, b1, b2, r_ohm_per_km=0.04, x_ohm_per_km=0.25, c_nf_per_km=150.0, **common
    )
    pp.create_line_from_parameters(
        net, b2, b3, r_ohm_per_km=0.06, x_ohm_per_km=0.18, c_nf_per_km=180.0, **common
    )
    pp.create_line_from_parameters(
        net, b3, b4, r_ohm_per_km=0.03, x_ohm_per_km=0.22, c_nf_per_km=120.0, **common
    )
    pp.create_line_from_parameters(
        net, b4, b0, r_ohm_per_km=0.05, x_ohm_per_km=0.15, c_nf_per_km=250.0, **common
    )
    pp.create_line_from_parameters(
        net, b1, b3, r_ohm_per_km=0.04, x_ohm_per_km=0.30, c_nf_per_km=100.0, **common
    )

    net.line.loc[:, "rateA"] = 1000.0

    # Bus shunts (will be disabled by lossless policy)
    pp.create_shunt(net, b1, q_mvar=-15.0, p_mw=0.0)  # capacitor
    pp.create_shunt(net, b3, q_mvar=10.0, p_mw=0.0)  # reactor

    return net, b0


def test_h_vector_fd_with_shunts_and_charging() -> None:
    """FD validation on a network that has bus shunts and line charging.

    The lossless policy must disable shunts and zero c_nf_per_km/r_ohm_per_km
    to align pandapower PF with the series-only Jacobian.
    """
    from stability_radius.base_point.pandapower_tools import (
        apply_lossless_policy_to_pandapower_net,
    )
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius
    from stability_radius.workflows import _expand_h_reduced_to_full

    net_raw, slack_bus = _make_5bus_with_shunts_and_charging()

    # Apply lossless policy (zeros r, c_nf, g_us; disables shunts)
    net = apply_lossless_policy_to_pandapower_net(net_raw)

    # Verify shunts are disabled
    assert not net.shunt["in_service"].any(), "Shunts should be disabled by lossless policy"

    _run_pp_pf(net, init="flat")

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    line_ids = [int(x) for x in sorted(net.line.index)]
    n_bus = len(bus_ids)
    n_lines = len(line_ids)

    base_pf = solve_ac_pf_base_point_from_pandapower(
        net=net, slack_bus=slack_bus, solver="pandapower", init="flat", lossless=True
    )

    ac_results = compute_ac_l2_radius(
        net,
        base_pf=base_pf,
        slack_bus=slack_bus,
        lossless=True,
        return_h_vectors=True,
    )

    h_vecs_raw = ac_results.pop("_h_vectors")
    slack_pos = bus_ids.index(slack_bus)
    h_from_full = _expand_h_reduced_to_full(
        h_vecs_raw["h_from"], n_bus=n_bus, slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )
    h_to_full = _expand_h_reduced_to_full(
        h_vecs_raw["h_to"], n_bus=n_bus, slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )

    s0_from = np.sqrt(
        np.array(base_pf.line_p0_mw) ** 2 + np.array(base_pf.line_q0_mvar) ** 2
    )
    s0_to = np.sqrt(
        np.array(base_pf.line_p1_mw) ** 2 + np.array(base_pf.line_q1_mvar) ** 2
    )

    # Balanced P perturbation
    eps = 0.01
    delta_p = np.zeros(n_bus, dtype=float)
    delta_p[bus_ids.index(bus_ids[1])] = +eps
    delta_p[bus_ids.index(bus_ids[2])] = -eps
    delta_q = np.zeros(n_bus, dtype=float)
    delta_u = np.concatenate([delta_p, delta_q])

    import copy

    net_pert = copy.deepcopy(net)
    for pos, bid in enumerate(bus_ids):
        pp.create_sgen(
            net_pert,
            bus=int(bid),
            p_mw=float(delta_p[pos]),
            q_mvar=float(delta_q[pos]),
            in_service=True,
        )

    _run_pp_pf(net_pert, init="flat")

    s1_from = np.array(
        [
            math.sqrt(
                float(net_pert.res_line.loc[lid, "p_from_mw"]) ** 2
                + float(net_pert.res_line.loc[lid, "q_from_mvar"]) ** 2
            )
            for lid in line_ids
        ]
    )
    s1_to = np.array(
        [
            math.sqrt(
                float(net_pert.res_line.loc[lid, "p_to_mw"]) ** 2
                + float(net_pert.res_line.loc[lid, "q_to_mvar"]) ** 2
            )
            for lid in line_ids
        ]
    )

    ds_from_actual = s1_from - s0_from
    ds_to_actual = s1_to - s0_to
    ds_from_pred = h_from_full @ delta_u
    ds_to_pred = h_to_full @ delta_u

    checked = 0
    for pos in range(n_lines):
        if abs(ds_from_actual[pos]) > 1e-10:
            err = abs(ds_from_pred[pos] - ds_from_actual[pos]) / abs(
                ds_from_actual[pos]
            )
            logger.info(
                "Line %d from-end (shunt net): dS_pred=%.6e, dS_actual=%.6e, rel_err=%.4f",
                line_ids[pos],
                ds_from_pred[pos],
                ds_from_actual[pos],
                err,
            )
            assert err < 0.05, (
                f"Line {line_ids[pos]} from-end (shunt net): rel_err={err:.4f}"
            )
            checked += 1

        if abs(ds_to_actual[pos]) > 1e-10:
            err = abs(ds_to_pred[pos] - ds_to_actual[pos]) / abs(ds_to_actual[pos])
            logger.info(
                "Line %d to-end (shunt net): dS_pred=%.6e, dS_actual=%.6e, rel_err=%.4f",
                line_ids[pos],
                ds_to_pred[pos],
                ds_to_actual[pos],
                err,
            )
            assert err < 0.05, (
                f"Line {line_ids[pos]} to-end (shunt net): rel_err={err:.4f}"
            )
            checked += 1

    assert checked > 0, "No line ends had measurable ΔS"


def _make_4bus_with_pv() -> tuple[object, int]:
    """Create a 4-bus network with 1 PV (generator) bus.

    Bus 0: ext_grid (slack)
    Bus 1: gen (PV bus, voltage-controlled)
    Bus 2: load (PQ bus)
    Bus 3: load (PQ bus)

    Meshed topology: 0-1, 1-2, 2-3, 3-0, 0-2 (5 lines).
    """
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = int(pp.create_bus(net, vn_kv=110.0))
    b1 = int(pp.create_bus(net, vn_kv=110.0))
    b2 = int(pp.create_bus(net, vn_kv=110.0))
    b3 = int(pp.create_bus(net, vn_kv=110.0))

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_gen(net, b1, p_mw=30.0, vm_pu=1.02, in_service=True)
    pp.create_load(net, b2, p_mw=40.0, q_mvar=10.0)
    pp.create_load(net, b3, p_mw=25.0, q_mvar=5.0)

    common = dict(
        length_km=1.0,
        r_ohm_per_km=0.0,  # lossless
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )

    pp.create_line_from_parameters(net, b0, b1, x_ohm_per_km=0.10, **common)
    pp.create_line_from_parameters(net, b1, b2, x_ohm_per_km=0.15, **common)
    pp.create_line_from_parameters(net, b2, b3, x_ohm_per_km=0.12, **common)
    pp.create_line_from_parameters(net, b3, b0, x_ohm_per_km=0.11, **common)
    pp.create_line_from_parameters(net, b0, b2, x_ohm_per_km=0.20, **common)

    net.line.loc[:, "rateA"] = 1000.0

    return net, b0


def test_h_vector_fd_with_pv_buses() -> None:
    """FD validation on a network with a PV (generator) bus.

    This tests that the PV-aware Jacobian correctly handles buses where
    voltage magnitude is fixed by generator control.
    """
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius
    from stability_radius.workflows import _expand_h_reduced_to_full

    net, slack_bus = _make_4bus_with_pv()

    _run_pp_pf(net, init="flat")

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    line_ids = [int(x) for x in sorted(net.line.index)]
    n_bus = len(bus_ids)
    n_lines = len(line_ids)

    base_pf = solve_ac_pf_base_point_from_pandapower(
        net=net, slack_bus=slack_bus, solver="pandapower", init="flat", lossless=True
    )

    ac_results = compute_ac_l2_radius(
        net,
        base_pf=base_pf,
        slack_bus=slack_bus,
        lossless=True,
        return_h_vectors=True,
    )

    h_vecs_raw = ac_results.pop("_h_vectors")
    slack_pos = bus_ids.index(slack_bus)
    h_from_full = _expand_h_reduced_to_full(
        h_vecs_raw["h_from"], n_bus=n_bus, slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )
    h_to_full = _expand_h_reduced_to_full(
        h_vecs_raw["h_to"], n_bus=n_bus, slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )

    assert h_from_full.shape == (n_lines, 2 * n_bus)
    assert h_to_full.shape == (n_lines, 2 * n_bus)

    s0_from = np.sqrt(
        np.array(base_pf.line_p0_mw) ** 2 + np.array(base_pf.line_q0_mvar) ** 2
    )
    s0_to = np.sqrt(
        np.array(base_pf.line_p1_mw) ** 2 + np.array(base_pf.line_q1_mvar) ** 2
    )

    # Balanced P perturbation at PQ buses only (bus 2 and bus 3)
    eps = 0.01
    delta_p = np.zeros(n_bus, dtype=float)
    delta_p[bus_ids.index(2)] = +eps
    delta_p[bus_ids.index(3)] = -eps
    delta_q = np.zeros(n_bus, dtype=float)
    delta_u = np.concatenate([delta_p, delta_q])

    import copy

    net_pert = copy.deepcopy(net)
    for pos, bid in enumerate(bus_ids):
        pp.create_sgen(
            net_pert,
            bus=int(bid),
            p_mw=float(delta_p[pos]),
            q_mvar=float(delta_q[pos]),
            in_service=True,
        )

    _run_pp_pf(net_pert, init="flat")

    s1_from = np.array(
        [
            math.sqrt(
                float(net_pert.res_line.loc[lid, "p_from_mw"]) ** 2
                + float(net_pert.res_line.loc[lid, "q_from_mvar"]) ** 2
            )
            for lid in line_ids
        ]
    )
    s1_to = np.array(
        [
            math.sqrt(
                float(net_pert.res_line.loc[lid, "p_to_mw"]) ** 2
                + float(net_pert.res_line.loc[lid, "q_to_mvar"]) ** 2
            )
            for lid in line_ids
        ]
    )

    ds_from_actual = s1_from - s0_from
    ds_to_actual = s1_to - s0_to
    ds_from_pred = h_from_full @ delta_u
    ds_to_pred = h_to_full @ delta_u

    checked = 0
    for pos in range(n_lines):
        if abs(ds_from_actual[pos]) > 1e-10:
            err = abs(ds_from_pred[pos] - ds_from_actual[pos]) / abs(
                ds_from_actual[pos]
            )
            logger.info(
                "Line %d from-end (PV net): dS_pred=%.6e, dS_actual=%.6e, rel_err=%.4f",
                line_ids[pos],
                ds_from_pred[pos],
                ds_from_actual[pos],
                err,
            )
            assert err < 0.05, (
                f"Line {line_ids[pos]} from-end (PV net): rel_err={err:.4f}"
            )
            checked += 1

        if abs(ds_to_actual[pos]) > 1e-10:
            err = abs(ds_to_pred[pos] - ds_to_actual[pos]) / abs(ds_to_actual[pos])
            logger.info(
                "Line %d to-end (PV net): dS_pred=%.6e, dS_actual=%.6e, rel_err=%.4f",
                line_ids[pos],
                ds_to_pred[pos],
                ds_to_actual[pos],
                err,
            )
            assert err < 0.05, (
                f"Line {line_ids[pos]} to-end (PV net): rel_err={err:.4f}"
            )
            checked += 1

    assert checked > 0, "No line ends had measurable ΔS (PV net)"
