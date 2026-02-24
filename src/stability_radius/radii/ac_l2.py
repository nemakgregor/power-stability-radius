from __future__ import annotations

import logging
import math
from typing import Any, Dict

import numpy as np

from stability_radius.ac.ac_model import build_ac_operator
from stability_radius.base_point.pypsa_pf import PyPSAAPFResult
from stability_radius.radii.common import (
    estimate_line_limit_mva_with_flag,
    line_key,
)

logger = logging.getLogger(__name__)

_EPS_NORM = 1e-12
_EPS_S0_MVA = 1e-9
_FALLBACK_WP_WQ = 1.0 / math.sqrt(2.0)


def _balanced_two_block_norm_from_red(
    *, a_p_red: np.ndarray, a_q_red: np.ndarray, n_bus_total: int
) -> float:
    """
    Balanced dual norm for AC injections with two constraints:
      1^T ΔP = 0,  1^T ΔQ = 0

    Combined:
      sqrt( ||Proj(aP)||^2 + ||Proj(aQ)||^2 )
    """
    ap = np.asarray(a_p_red, dtype=float).reshape(-1)
    aq = np.asarray(a_q_red, dtype=float).reshape(-1)

    t_p = float(np.dot(ap, ap))
    s_p = float(np.sum(ap))
    proj2_p = t_p - (s_p * s_p) / float(n_bus_total)

    t_q = float(np.dot(aq, aq))
    s_q = float(np.sum(aq))
    proj2_q = t_q - (s_q * s_q) / float(n_bus_total)

    return math.sqrt(max(proj2_p, 0.0) + max(proj2_q, 0.0))


def compute_ac_l2_radius(
    net: Any,
    *,
    base_pf: PyPSAAPFResult,
    slack_bus: int,
    chunk_size: int = 256,
    balance: bool = True,
    lossless: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute a fast AC L2 "stability radius" certificate around an AC PF base point.

    Output fields (key additions for unified tables)
    ------------------------------------------------
    In addition to detailed per-end fields, each line includes explicit unified keys:
      - margin_ac_mva : margin at the binding end (MVA)
      - "||h||2"      : dual sensitivity norm at the binding end (dimensionless)
      - binding_end   : "from" | "to"
      - is_unconstrained : True iff the thermal limit is a surrogate (rateA==0/NaN/+inf).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    line_ids = [int(x) for x in sorted(net.line.index)]
    if tuple(line_ids) != tuple(base_pf.line_ids):
        raise ValueError(
            "AC PF base point line ordering mismatch: expected base_pf.line_ids == sorted(net.line.index)."
        )

    op = build_ac_operator(
        net=net,
        slack_bus=int(slack_bus),
        vm_pu=np.asarray(base_pf.v_mag_pu, dtype=float),
        va_rad=np.asarray(base_pf.v_ang_rad, dtype=float),
        line_indices=line_ids,
        lossless=bool(lossless),
    )

    m = int(len(line_ids))
    n_bus = int(len(op.bus_ids))
    n_red = int(op.n_red)

    # ---------- limits + unconstrained flags ----------
    limits_mva = np.empty(m, dtype=float)
    is_unconstrained = np.zeros(m, dtype=bool)
    for pos, lid in enumerate(line_ids):
        lim, is_uc = estimate_line_limit_mva_with_flag(net, net.line.loc[int(lid)])
        limits_mva[pos] = float(lim)
        is_unconstrained[pos] = bool(is_uc)

    # ---------- base flows ----------
    p0 = np.asarray(base_pf.line_p0_mw, dtype=float).reshape(-1)
    q0 = np.asarray(base_pf.line_q0_mvar, dtype=float).reshape(-1)
    p1 = np.asarray(base_pf.line_p1_mw, dtype=float).reshape(-1)
    q1 = np.asarray(base_pf.line_q1_mvar, dtype=float).reshape(-1)

    if p0.shape != (m,) or q0.shape != (m,) or p1.shape != (m,) or q1.shape != (m,):
        raise ValueError("Base PF line flow arrays shape mismatch.")

    s0_from = np.sqrt(p0 * p0 + q0 * q0)
    s0_to = np.sqrt(p1 * p1 + q1 * q1)

    margin_from = limits_mva - s0_from
    margin_to = limits_mva - s0_to

    # ---------- constraints: 2 per line (from-end and to-end) ----------
    n_con = 2 * m
    norms = np.zeros(n_con, dtype=float)
    radii = np.full(n_con, float("inf"), dtype=float)

    p_end = np.empty(n_con, dtype=float)
    q_end = np.empty(n_con, dtype=float)
    s_end = np.empty(n_con, dtype=float)
    margin_end = np.empty(n_con, dtype=float)

    for i in range(m):
        p_end[2 * i] = float(p0[i])
        q_end[2 * i] = float(q0[i])
        s_end[2 * i] = float(s0_from[i])
        margin_end[2 * i] = float(margin_from[i])

        p_end[2 * i + 1] = float(p1[i])
        q_end[2 * i + 1] = float(q1[i])
        s_end[2 * i + 1] = float(s0_to[i])
        margin_end[2 * i + 1] = float(margin_to[i])

    # ---------- chunked adjoint solves ----------
    fallback_used = 0

    start = 0
    while start < n_con:
        end = min(n_con, start + int(chunk_size))
        k = int(end - start)

        B = np.zeros((2 * n_red, k), dtype=float)

        for j in range(k):
            con_idx = int(start + j)
            line_pos = int(con_idx // 2)
            is_from_end = (con_idx % 2) == 0

            fb_pos = int(op.from_bus_pos[line_pos])
            tb_pos = int(op.to_bus_pos[line_pos])

            i_pos = fb_pos if is_from_end else tb_pos
            k_pos = tb_pos if is_from_end else fb_pos

            y = complex(op.y_series_pu[line_pos])
            if abs(y) <= 0.0:
                continue

            g = float(np.real(y))
            b = float(np.imag(y))

            Vi = float(op.vm_pu[i_pos])
            Vk = float(op.vm_pu[k_pos])
            if Vi <= 0.0 or Vk <= 0.0:
                raise ValueError(
                    "Non-positive Vm in AC base point; cannot compute sensitivities."
                )

            theta = float(op.va_rad[i_pos] - op.va_rad[k_pos])
            s = math.sin(theta)
            c = math.cos(theta)

            A = g * c + b * s
            Btmp = g * s - b * c

            # per-unit derivatives for flow leaving bus i towards bus k
            dP_dti_pu = Vi * Vk * Btmp
            dP_dtk_pu = -dP_dti_pu
            dQ_dti_pu = -Vi * Vk * A
            dQ_dtk_pu = -dQ_dti_pu

            dP_dVi_pu = 2.0 * g * Vi - Vk * A
            dP_dVk_pu = -Vi * A
            dQ_dVi_pu = -2.0 * b * Vi - Vk * Btmp
            dQ_dVk_pu = -Vi * Btmp

            scale = float(op.sn_mva)
            dP_dti = scale * dP_dti_pu
            dP_dtk = scale * dP_dtk_pu
            dQ_dti = scale * dQ_dti_pu
            dQ_dtk = scale * dQ_dtk_pu

            dP_dVi = scale * dP_dVi_pu
            dP_dVk = scale * dP_dVk_pu
            dQ_dVi = scale * dQ_dVi_pu
            dQ_dVk = scale * dQ_dVk_pu

            s0 = float(s_end[con_idx])
            if s0 > _EPS_S0_MVA:
                wP = float(p_end[con_idx]) / s0
                wQ = float(q_end[con_idx]) / s0
            else:
                # At |S|=0 the gradient of a norm is undefined.
                # For a conservative, unbiased certificate we use an equal P/Q direction:
                # (wP, wQ) = (1/sqrt(2), 1/sqrt(2)), instead of a one-sided (1,0) fallback.
                wP = _FALLBACK_WP_WQ
                wQ = _FALLBACK_WP_WQ
                fallback_used += 1

            b_ti = wP * dP_dti + wQ * dQ_dti
            b_tk = wP * dP_dtk + wQ * dQ_dtk
            b_Vi = wP * dP_dVi + wQ * dQ_dVi
            b_Vk = wP * dP_dVk + wQ * dQ_dVk

            ri = int(op.red_pos_of_bus_pos[i_pos])
            rk = int(op.red_pos_of_bus_pos[k_pos])

            if ri >= 0:
                B[ri, j] += float(b_ti)
                B[n_red + ri, j] += float(b_Vi)
            if rk >= 0:
                B[rk, j] += float(b_tk)
                B[n_red + rk, j] += float(b_Vk)

        Y = op.solve_J_transpose(B)

        for j in range(k):
            con_idx = int(start + j)
            a_p = Y[0:n_red, j]
            a_q = Y[n_red : 2 * n_red, j]

            if bool(balance):
                denom = _balanced_two_block_norm_from_red(
                    a_p_red=a_p, a_q_red=a_q, n_bus_total=n_bus
                )
            else:
                denom = float(np.linalg.norm(Y[:, j], ord=2))

            norms[con_idx] = float(denom)

            margin = float(margin_end[con_idx])
            if denom > _EPS_NORM:
                radii[con_idx] = margin / denom
            else:
                radii[con_idx] = float("inf") if margin >= 0.0 else float("-inf")

        start = end

    if fallback_used > 0:
        logger.debug(
            "AC |S| gradient fallback used: %d/%d constraint-ends with |S0|<=%.3g MVA "
            "(used equal P/Q weights).",
            int(fallback_used),
            int(n_con),
            float(_EPS_S0_MVA),
        )

    # ---------- aggregate per line (min of from/to end) ----------
    results: Dict[str, Dict[str, Any]] = {}
    finite_vals: list[float] = []

    for pos, lid in enumerate(line_ids):
        r_from = float(radii[2 * pos])
        r_to = float(radii[2 * pos + 1])
        r_line = min(r_from, r_to)

        binding_end = "from" if r_from <= r_to else "to"
        margin_bind = (
            float(margin_from[pos]) if binding_end == "from" else float(margin_to[pos])
        )
        norm_bind = (
            float(norms[2 * pos])
            if binding_end == "from"
            else float(norms[2 * pos + 1])
        )

        k = line_key(int(lid))
        results[k] = {
            # detailed end-specific
            "ac_s_limit_mva": float(limits_mva[pos]),
            "is_unconstrained": bool(is_unconstrained[pos]),
            "ac_p0_from_mw": float(p0[pos]),
            "ac_q0_from_mvar": float(q0[pos]),
            "ac_s0_from_mva": float(s0_from[pos]),
            "ac_margin_from_mva": float(margin_from[pos]),
            "ac_norm_a_from": float(norms[2 * pos]),
            "radius_ac_l2_from": float(r_from),
            "ac_p0_to_mw": float(p1[pos]),
            "ac_q0_to_mvar": float(q1[pos]),
            "ac_s0_to_mva": float(s0_to[pos]),
            "ac_margin_to_mva": float(margin_to[pos]),
            "ac_norm_a_to": float(norms[2 * pos + 1]),
            "radius_ac_l2_to": float(r_to),
            # per-line aggregate
            "radius_ac_l2": float(r_line),
            "ac_limiting_end": str(binding_end),
            # unified table fields requested
            "binding_end": str(binding_end),
            "margin_ac_mva": float(margin_bind),
            "||h||2": float(norm_bind),
        }

        if math.isfinite(r_line):
            finite_vals.append(float(r_line))

    if finite_vals:
        logger.info(
            "AC L2 radius computed: lines=%d, mean=%.6g, min=%.6g, max=%.6g (balance=%s)",
            int(len(line_ids)),
            float(np.mean(finite_vals)),
            float(np.min(finite_vals)),
            float(np.max(finite_vals)),
            bool(balance),
        )
    else:
        logger.info(
            "AC L2 radius computed: lines=%d, finite_radii=0 (balance=%s)",
            int(len(line_ids)),
            bool(balance),
        )

    return results
