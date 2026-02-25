from __future__ import annotations

"""
Finite-difference verification of the AC PF Jacobian sign/units against pandapower.

Why this test is critical
-------------------------
The AC certificate ultimately relies on the PF Jacobian scaling and sign conventions.
A sign flip or missing sn_mva factor makes all computed radii meaningless (unsound).

This test validates, on a tiny deterministic 3-bus *meshed* network, that:

1) The *base* line-end active power flows computed from the same series-only model
   match pandapower's res_line.{p_from_mw,p_to_mw} sign convention:
       - positive means power leaving the bus into the line.

2) For a tiny balanced injection perturbation δu (MW), the *predicted* change in
   line-end flows from the linearization matches the *actual* change observed by
   re-solving AC PF in pandapower.

We use the infinity-norm relative error criterion:
    ||Δf_pred - Δf_actual||_inf / ||Δf_actual||_inf < 1e-3
"""

import logging
import math

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")

logger = logging.getLogger(__name__)


def _run_pp_pf(net: object, *, init: str) -> None:
    """
    Solve AC PF via pandapower with tight tolerance for stable finite differences.

    Notes
    -----
    - calculate_voltage_angles=True is required, because ACOperator expects angles.
    - We use a tight tolerance to minimize numerical noise in Δf_actual.
    """
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


def _make_3bus_meshed_net() -> tuple[object, int, tuple[int, int, int]]:
    """
    Create a small deterministic 3-bus triangle network.

    Design choices
    --------------
    - lossless lines: r_ohm_per_km = 0 to match certificate "lossless" policy
    - no line charging: c_nf_per_km = 0 to match ACOperator's series-only Ybus
    - slack at bus0 (ext_grid)
    """
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = int(pp.create_bus(net, vn_kv=110.0))
    b1 = int(pp.create_bus(net, vn_kv=110.0))
    b2 = int(pp.create_bus(net, vn_kv=110.0))

    pp.create_ext_grid(net, b0, vm_pu=1.0)

    # Make base flows non-trivial (so Δf_actual has a meaningful scale).
    pp.create_load(net, b1, p_mw=20.0, q_mvar=5.0)
    pp.create_load(net, b2, p_mw=15.0, q_mvar=3.0)

    common = dict(
        length_km=1.0,
        r_ohm_per_km=0.0,  # lossless
        c_nf_per_km=0.0,  # no charging
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

    # Explicit rating is not required for this test, but keeps the net "well-formed"
    # w.r.t. the project-wide limits contract.
    net.line.loc[:, "rateA"] = 1000.0

    return net, b0, (b0, b1, b2)


def _line_end_pq_mw_from_series_only_model(
    *,
    sn_mva: float,
    Vi: float,
    Vk: float,
    theta_i: float,
    theta_k: float,
    y_series_pu: complex,
) -> tuple[float, float]:
    """
    Compute (P,Q) at the i-end of a *series-only* branch i--k.

    Convention matches pandapower:
      P > 0 means power leaving bus i into the branch.
    """
    g = float(np.real(y_series_pu))
    b = float(np.imag(y_series_pu))

    theta = float(theta_i - theta_k)
    s = math.sin(theta)
    c = math.cos(theta)

    # Helper terms (match stability_radius.radii.ac_l2 derivations)
    A = g * c + b * s
    Btmp = g * s - b * c

    # Series-only S_i->k (per-unit), then scale by sn_mva
    P_pu = g * Vi * Vi - Vi * Vk * A
    Q_pu = -b * Vi * Vi - Vi * Vk * Btmp

    return float(sn_mva) * float(P_pu), float(sn_mva) * float(Q_pu)


def _predict_delta_p_line_ends_mw(
    *,
    op: object,
    dx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict Δp_from_mw and Δp_to_mw for all monitored lines using:
        ΔP_end ≈ (∂P_end/∂x) · dx
    where dx solves:
        J dx = δu
    """
    # Import locally to keep test import graph minimal/deterministic.
    from stability_radius.ac.ac_model import ACOperator

    if not isinstance(op, ACOperator):
        raise TypeError("op must be ACOperator")

    dxv = np.asarray(dx, dtype=float).reshape(-1)
    n_red = int(op.n_red)
    if dxv.shape != (2 * n_red,):
        raise ValueError(f"dx must have shape ({2 * n_red},), got {dxv.shape}")

    dtheta = dxv[0:n_red]
    dV = dxv[n_red : 2 * n_red]

    m = int(op.n_line)
    dp_from = np.zeros(m, dtype=float)
    dp_to = np.zeros(m, dtype=float)

    for pos in range(m):
        fb = int(op.from_bus_pos[pos])
        tb = int(op.to_bus_pos[pos])
        y = complex(op.y_series_pu[pos])

        Vi = float(op.vm_pu[fb])
        Vk = float(op.vm_pu[tb])
        ti = float(op.va_rad[fb])
        tk = float(op.va_rad[tb])

        # from-end: i=from, k=to
        dp_from[pos] = _predict_delta_p_end_mw_single(
            sn_mva=float(op.sn_mva),
            Vi=Vi,
            Vk=Vk,
            ti=ti,
            tk=tk,
            y=y,
            red_i=int(op.red_pos_of_bus_pos[fb]),
            red_k=int(op.red_pos_of_bus_pos[tb]),
            dtheta=dtheta,
            dV=dV,
        )

        # to-end: i=to, k=from
        dp_to[pos] = _predict_delta_p_end_mw_single(
            sn_mva=float(op.sn_mva),
            Vi=Vk,
            Vk=Vi,
            ti=tk,
            tk=ti,
            y=y,
            red_i=int(op.red_pos_of_bus_pos[tb]),
            red_k=int(op.red_pos_of_bus_pos[fb]),
            dtheta=dtheta,
            dV=dV,
        )

    return dp_from, dp_to


def _predict_delta_p_end_mw_single(
    *,
    sn_mva: float,
    Vi: float,
    Vk: float,
    ti: float,
    tk: float,
    y: complex,
    red_i: int,
    red_k: int,
    dtheta: np.ndarray,
    dV: np.ndarray,
) -> float:
    """Predict ΔP (MW) at i-end for a series-only branch i--k."""
    g = float(np.real(y))
    b = float(np.imag(y))

    theta = float(ti - tk)
    s = math.sin(theta)
    c = math.cos(theta)

    A = g * c + b * s
    Btmp = g * s - b * c

    # Per-unit partials for P_end (leaving bus i towards k).
    dP_dti_pu = Vi * Vk * Btmp
    dP_dtk_pu = -dP_dti_pu
    dP_dVi_pu = 2.0 * g * Vi - Vk * A
    dP_dVk_pu = -Vi * A

    scale = float(sn_mva)

    dP_dti = scale * dP_dti_pu
    dP_dtk = scale * dP_dtk_pu
    dP_dVi = scale * dP_dVi_pu
    dP_dVk = scale * dP_dVk_pu

    # Slack variables are not part of the reduced state => their deltas are 0.
    dti = float(dtheta[red_i]) if red_i >= 0 else 0.0
    dtk = float(dtheta[red_k]) if red_k >= 0 else 0.0
    dVi = float(dV[red_i]) if red_i >= 0 else 0.0
    dVk = float(dV[red_k]) if red_k >= 0 else 0.0

    return float(dP_dti * dti + dP_dtk * dtk + dP_dVi * dVi + dP_dVk * dVk)


def test_ac_jacobian_and_line_end_flow_signs_match_pandapower_finite_difference() -> (
    None
):
    """
    Gold-standard check: finite differences vs linear prediction.

    Steps (as required by the task)
    -------------------------------
    (a) solve AC PF via pandapower
    (b) build ACOperator
    (c) apply tiny balanced perturbation δu (0.01 MW)
    (d) solve PF again with perturbed injections
    (e) compare Δf_actual vs predicted from the linearization using J
    """
    from stability_radius.ac.ac_model import build_ac_operator

    net, slack_bus, (b0, b1, b2) = _make_3bus_meshed_net()

    # Deterministic per-bus "delta injection" devices (as in AC MC implementation).
    sgen_by_bus: dict[int, int] = {}
    for bid in sorted(net.bus.index):
        sid = int(
            pp.create_sgen(
                net,
                bus=int(bid),
                p_mw=0.0,
                q_mvar=0.0,
                in_service=True,
                name=f"fd_delta_bus_{int(bid)}",
            )
        )
        sgen_by_bus[int(bid)] = sid

    # (a) Base PF
    _run_pp_pf(net, init="flat")

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    line_ids = [int(x) for x in sorted(net.line.index)]
    sn_mva = float(getattr(net, "sn_mva", np.nan))
    assert math.isfinite(sn_mva) and sn_mva > 0.0

    vm = np.asarray(
        [float(net.res_bus.loc[bid, "vm_pu"]) for bid in bus_ids], dtype=float
    )
    va = (
        np.asarray(
            [float(net.res_bus.loc[bid, "va_degree"]) for bid in bus_ids], dtype=float
        )
        * math.pi
        / 180.0
    )

    p_from_0 = np.asarray(
        [float(net.res_line.loc[lid, "p_from_mw"]) for lid in line_ids], dtype=float
    )
    p_to_0 = np.asarray(
        [float(net.res_line.loc[lid, "p_to_mw"]) for lid in line_ids], dtype=float
    )

    # (b) Build ACOperator at the base point
    op = build_ac_operator(
        net=net,
        slack_bus=int(slack_bus),
        vm_pu=vm,
        va_rad=va,
        line_indices=line_ids,
        lossless=True,
    )

    # (b.1) Sanity: series-only model base line-end P should match pandapower
    p_from_model = np.zeros(len(line_ids), dtype=float)
    p_to_model = np.zeros(len(line_ids), dtype=float)

    for pos, lid in enumerate(line_ids):
        fb_pos = int(op.from_bus_pos[pos])
        tb_pos = int(op.to_bus_pos[pos])
        y = complex(op.y_series_pu[pos])

        pf, _qf = _line_end_pq_mw_from_series_only_model(
            sn_mva=float(op.sn_mva),
            Vi=float(op.vm_pu[fb_pos]),
            Vk=float(op.vm_pu[tb_pos]),
            theta_i=float(op.va_rad[fb_pos]),
            theta_k=float(op.va_rad[tb_pos]),
            y_series_pu=y,
        )
        pt, _qt = _line_end_pq_mw_from_series_only_model(
            sn_mva=float(op.sn_mva),
            Vi=float(op.vm_pu[tb_pos]),
            Vk=float(op.vm_pu[fb_pos]),
            theta_i=float(op.va_rad[tb_pos]),
            theta_k=float(op.va_rad[fb_pos]),
            y_series_pu=y,
        )
        p_from_model[pos] = float(pf)
        p_to_model[pos] = float(pt)

    # Tight absolute tolerance: the networks are constructed to match models (r=0, no charging).
    assert np.allclose(p_from_model, p_from_0, atol=1e-4, rtol=0.0)
    assert np.allclose(p_to_model, p_to_0, atol=1e-4, rtol=0.0)

    # (c) Tiny balanced perturbation δu (MW) for active injections only
    eps = 0.01  # MW
    delta_p = np.zeros(len(bus_ids), dtype=float)
    delta_p[bus_ids.index(b1)] = +eps
    delta_p[bus_ids.index(b2)] = -eps
    assert abs(float(np.sum(delta_p))) <= 1e-12

    delta_q = np.zeros(len(bus_ids), dtype=float)  # keep Q unchanged for a clean check
    assert abs(float(np.sum(delta_q))) <= 1e-12

    # Apply perturbation via sgen (generation injection convention: +p means injecting into the bus)
    for bid, dp in zip(bus_ids, delta_p.tolist()):
        net.sgen.at[sgen_by_bus[int(bid)], "p_mw"] = float(dp)
    for bid, dq in zip(bus_ids, delta_q.tolist()):
        net.sgen.at[sgen_by_bus[int(bid)], "q_mvar"] = float(dq)

    # (d) Perturbed PF (warm-start from base)
    _run_pp_pf(net, init="results")

    p_from_1 = np.asarray(
        [float(net.res_line.loc[lid, "p_from_mw"]) for lid in line_ids], dtype=float
    )
    p_to_1 = np.asarray(
        [float(net.res_line.loc[lid, "p_to_mw"]) for lid in line_ids], dtype=float
    )

    # (e) Finite-difference actual deltas
    dp_from_actual = p_from_1 - p_from_0
    dp_to_actual = p_to_1 - p_to_0
    d_actual = np.concatenate([dp_from_actual, dp_to_actual])

    max_abs_actual = float(np.max(np.abs(d_actual)))
    assert max_abs_actual > 1e-8, (
        "Degenerate Δf_actual; increase perturbation if this triggers."
    )

    # Linear prediction using the PF Jacobian J (through dx = J^{-1} δu)
    rhs = np.concatenate(
        [
            delta_p[op.mask_non_slack],
            delta_q[op.mask_non_slack],
        ]
    )
    dx = op.solve_J(rhs)

    dp_from_pred, dp_to_pred = _predict_delta_p_line_ends_mw(op=op, dx=dx)
    d_pred = np.concatenate([dp_from_pred, dp_to_pred])

    max_abs_err = float(np.max(np.abs(d_pred - d_actual)))
    rel_inf = max_abs_err / max_abs_actual

    logger.info(
        "AC Jacobian FD check: max|Δactual|=%.6g MW, max|err|=%.6g MW, rel_inf=%.6g",
        max_abs_actual,
        max_abs_err,
        rel_inf,
    )

    assert rel_inf < 1.0e-3
