from __future__ import annotations

import logging
import math
from typing import Any, Dict

import numpy as np

from stability_radius.ac.ac_model import build_ac_operator
from stability_radius.base_point.pandapower_tools import resolve_slack_bus_id
from stability_radius.base_point.pypsa_pf import PyPSAAPFResult
from stability_radius.geometry.balanced import dual_norm_l2_balanced_from_block_vectors
from stability_radius.radii.common import (
    ConstraintStatus,
    classify_constraint_certificate,
    estimate_line_limit_mva_with_flag,
    line_key,
)

logger = logging.getLogger(__name__)

_EPS_NORM = 1e-12
_EPS_S0_MVA = 1e-9
_EPS_S0_REL_LIMIT = 1e-9
_DIAGNOSTIC_SUBGRADIENT_WP_WQ = 1.0 / math.sqrt(2.0)


def zero_flow_threshold_mva(limit_mva: float) -> float:
    """
    Scale-aware nondifferentiability threshold for |S0|.

    A fixed absolute threshold behaves differently across systems with
    different base MVA and line ratings, so the threshold is tied to the
    line rating with an absolute floor:

        eps(limit) = max(_EPS_S0_MVA, _EPS_S0_REL_LIMIT * limit)
    """
    lim = float(limit_mva)
    if not math.isfinite(lim) or lim <= 0.0:
        return float(_EPS_S0_MVA)
    return float(max(_EPS_S0_MVA, _EPS_S0_REL_LIMIT * lim))


def _balanced_two_block_norm_from_red(
    *,
    a_p_red: np.ndarray,
    a_q_red: np.ndarray,
    n_bus_total: int,
    n_pq_total: int | None = None,
) -> float:
    """
    Balanced dual norm for AC injections with two constraints:
      1^T ΔP = 0,  1^T ΔQ = 0

    Combined:
      sqrt( ||Proj(aP)||^2 + ||Proj(aQ)||^2 )

    Parameters
    ----------
    n_bus_total : int
        Total bus count for the P-block projection.
    n_pq_total : int or None
        PQ bus count for the Q-block projection.  If None, uses n_bus_total.
    """
    return dual_norm_l2_balanced_from_block_vectors(
        (a_p_red, a_q_red),
        total_sizes=(
            int(n_bus_total),
            int(n_pq_total) if n_pq_total is not None else int(n_bus_total),
        ),
        balance=True,
    )


def _balanced_cross_inner(
    u_p: np.ndarray,
    u_q: np.ndarray,
    v_p: np.ndarray,
    v_q: np.ndarray,
    *,
    n_bus_total: int,
    n_pq_total: int,
    balance: bool,
) -> float:
    """
    Inner product <Proj(u), Proj(v)> under the balanced two-block projection.

    Uses the same implicit-zero convention as
    ``dual_norm_l2_balanced_from_block_vectors``: reduced vectors carry
    ``n_bus_total`` (P block) and ``n_pq_total`` (Q block) implicit
    coordinates, and the balanced projector removes the block mean over the
    full block size.  For an orthogonal projector P:
      <Pu, Pv> = u.v - (sum u)(sum v)/n.
    """
    t = float(np.dot(u_p, v_p)) + float(np.dot(u_q, v_q))
    if bool(balance):
        n_p = int(n_bus_total)
        if n_p > 0:
            t -= (float(np.sum(u_p)) * float(np.sum(v_p))) / float(n_p)
        n_q = int(n_pq_total)
        if n_q > 0:
            t -= (float(np.sum(u_q)) * float(np.sum(v_q))) / float(n_q)
    return t


def _end_gradient_rows(
    op: Any, line_pos: int, is_from_end: bool
) -> list[tuple[int, float, float]] | None:
    """
    Reduced-state gradient entries of the line-end (P, Q) flows.

    Returns a list of ``(reduced_row, dP, dQ)`` triples in MW/MVAr per
    (rad, pu) for the flow leaving the selected end, or None for an
    out-of-service branch.
    """
    fb_pos = int(op.from_bus_pos[line_pos])
    tb_pos = int(op.to_bus_pos[line_pos])
    i_pos = fb_pos if is_from_end else tb_pos
    k_pos = tb_pos if is_from_end else fb_pos

    y = complex(op.y_series_pu[line_pos])
    if abs(y) <= 0.0:
        return None

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

    scale = float(op.sn_mva)

    # per-unit derivatives for flow leaving bus i towards bus k, scaled to MW/MVAr
    dP_dti = scale * (Vi * Vk * Btmp)
    dP_dtk = -dP_dti
    dQ_dti = scale * (-Vi * Vk * A)
    dQ_dtk = -dQ_dti

    dP_dVi = scale * (2.0 * g * Vi - Vk * A)
    dP_dVk = scale * (-Vi * A)
    dQ_dVi = scale * (-2.0 * b * Vi - Vk * Btmp)
    dQ_dVk = scale * (-Vi * Btmp)

    n_theta = int(op.n_red)
    entries: list[tuple[int, float, float]] = []

    ri_theta = int(op.theta_red_pos[i_pos])
    rk_theta = int(op.theta_red_pos[k_pos])
    if ri_theta >= 0:
        entries.append((ri_theta, float(dP_dti), float(dQ_dti)))
    if rk_theta >= 0:
        entries.append((rk_theta, float(dP_dtk), float(dQ_dtk)))

    ri_v = int(op.v_red_pos[i_pos])
    rk_v = int(op.v_red_pos[k_pos])
    if ri_v >= 0:
        entries.append((n_theta + ri_v, float(dP_dVi), float(dQ_dVi)))
    if rk_v >= 0:
        entries.append((n_theta + rk_v, float(dP_dVk), float(dQ_dVk)))

    return entries


def _q_saturated_pv_buses(net: Any, q_limit_events: Any) -> list[int]:
    """
    Bus ids whose every in-service gen/ext_grid is Q-limit-saturated.

    A bus keeps PV status if at least one in-service voltage-controlling
    element at that bus still has reactive headroom.  The slack (ext_grid)
    bus is never converted.
    """
    events = list(q_limit_events or [])
    if not events:
        return []

    saturated: dict[int, set[tuple[str, int]]] = {}
    for ev in events:
        try:
            bid = int(ev.get("bus", -1))
            key = (str(ev.get("element", "")), int(ev.get("element_index", -1)))
        except (AttributeError, TypeError, ValueError):
            continue
        if bid >= 0 and key[0] == "gen":
            saturated.setdefault(bid, set()).add(key)
    if not saturated:
        return []

    ext_buses: set[int] = set()
    eg = getattr(net, "ext_grid", None)
    if eg is not None and len(eg):
        for _, row in eg.iterrows():
            if bool(row.get("in_service", True)):
                ext_buses.add(int(row.get("bus", -1)))

    out: list[int] = []
    gen = getattr(net, "gen", None)
    for bid, keys in saturated.items():
        if bid in ext_buses:
            continue
        controllers: set[tuple[str, int]] = set()
        if gen is not None and len(gen):
            for gid in gen.index:
                row = gen.loc[gid]
                if bool(row.get("in_service", True)) and int(row.get("bus", -1)) == bid:
                    controllers.add(("gen", int(gid)))
        if controllers and controllers.issubset(keys):
            out.append(int(bid))
    return sorted(out)


def compute_ac_l2_radius(
    net: Any,
    *,
    base_pf: PyPSAAPFResult,
    slack_bus: int,
    chunk_size: int = 256,
    balance: bool = True,
    lossless: bool = True,
    return_h_vectors: bool = False,
    zero_flow_operator_norm: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute a fast AC L2 "stability radius" certificate around an AC PF base point.

    Slack consistency
    -----------------
    The slack bus is resolved with ``resolve_slack_bus_id`` (ext_grid-aware)
    BEFORE building the AC operator, so the Jacobian eliminates exactly the
    bus that the power-flow solver used as slack.  The resolved slack position
    is returned in ``_h_vectors["slack_pos"]`` and MUST be used by callers of
    ``expand_h_reduced_to_full`` instead of re-deriving it independently.

    Zero-flow (nondifferentiable) line ends
    ---------------------------------------
    At ``|S0| <= eps(limit)`` (scale-aware threshold, see
    ``zero_flow_threshold_mva``) the apparent-power magnitude has no scalar
    gradient.  When ``zero_flow_operator_norm=True`` (default), such ends get
    a first-order operator-norm radius instead of being excluded: with
    ``H = [∇P; ∇Q]`` mapped to injection space by two adjoint solves and
    projected onto the balanced subspace, the exact first-order bound is
    ``|S(Δu)| = ||H Δu||_2 <= sigma_max(H_proj) ||Δu||_2``, so
    ``r = margin / sigma_max``.  These ends are labeled with status
    ``ok_finite_operator_norm`` and remain flagged as nondifferentiable for
    accounting.

    Sensitivity norms below ``1e-12`` are treated as degenerate/infinite
    according to ``classify_constraint_certificate``.

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

    # Resolve the slack bus AGAINST net.ext_grid so the reduced Jacobian
    # eliminates the same bus the PF solver used.  (Regression: a positional
    # slack like 0 with an ext_grid elsewhere used to shift every h_P entry
    # by one bus position downstream.)
    slack_bus_id = int(resolve_slack_bus_id(net, int(slack_bus)))

    # Active-set consistency: buses whose EVERY in-service voltage-controlling
    # element saturated at a Q limit in the base PF are effectively PQ in the
    # converged pandapower solution; linearize them as PQ.
    pv_to_pq_bus_ids = _q_saturated_pv_buses(
        net, getattr(base_pf, "q_limit_events", ()) or ()
    )

    op = build_ac_operator(
        net=net,
        slack_bus=slack_bus_id,
        vm_pu=np.asarray(base_pf.v_mag_pu, dtype=float),
        va_rad=np.asarray(base_pf.v_ang_rad, dtype=float),
        line_indices=line_ids,
        lossless=bool(lossless),
        pv_to_pq_bus_ids=pv_to_pq_bus_ids,
    )

    m = int(len(line_ids))
    n_bus = int(len(op.bus_ids))
    slack_pos = int(op.slack_pos)
    if list(op.bus_ids).index(slack_bus_id) != slack_pos:
        raise AssertionError(
            "Internal slack inconsistency between resolve_slack_bus_id and ACOperator."
        )

    # ---------- limits + unconstrained flags ----------
    limits_mva = np.empty(m, dtype=float)
    is_unconstrained = np.zeros(m, dtype=bool)
    for pos, lid in enumerate(line_ids):
        lim, is_uc = estimate_line_limit_mva_with_flag(net, net.line.loc[int(lid)])
        limits_mva[pos] = float(lim)
        is_unconstrained[pos] = bool(is_uc)

    nd_threshold_mva = np.array(
        [zero_flow_threshold_mva(limits_mva[pos]) for pos in range(m)], dtype=float
    )

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

    # ---------- optional h-vector storage ----------
    n_vars = int(op.n_vars)
    n_theta = int(op.n_red)
    n_pq = int(op.n_pq)

    if return_h_vectors:
        h_from = np.zeros((m, n_vars), dtype=float)
        h_to = np.zeros((m, n_vars), dtype=float)

    # ---------- chunked adjoint solves ----------
    diagnostic_subgradient_used = 0
    nondifferentiable_end = np.zeros(n_con, dtype=bool)
    adjoint_residual_max = 0.0

    start = 0
    while start < n_con:
        end = min(n_con, start + int(chunk_size))
        k = int(end - start)

        B = np.zeros((n_vars, k), dtype=float)

        for j in range(k):
            con_idx = int(start + j)
            line_pos = int(con_idx // 2)
            is_from_end = (con_idx % 2) == 0

            entries = _end_gradient_rows(op, line_pos, is_from_end)
            if entries is None:
                continue

            s0 = float(s_end[con_idx])
            if s0 > float(nd_threshold_mva[line_pos]):
                wP = float(p_end[con_idx]) / s0
                wQ = float(q_end[con_idx]) / s0
            else:
                # At |S|=0 the gradient of a norm is undefined.
                # Use an equal P/Q diagnostic subgradient here; the certified
                # value for this end comes from the operator-norm pass below.
                wP = _DIAGNOSTIC_SUBGRADIENT_WP_WQ
                wQ = _DIAGNOSTIC_SUBGRADIENT_WP_WQ
                diagnostic_subgradient_used += 1
                nondifferentiable_end[con_idx] = True

            for row, dP, dQ in entries:
                B[row, j] += wP * dP + wQ * dQ

        Y = op.solve_J_transpose(B)

        # Adjoint residual diagnostic: || J^T Y - B ||_inf / max(1, ||B||_inf)
        resid = float(np.max(np.abs(op.J.T @ Y - B))) if k > 0 else 0.0
        denom_resid = max(1.0, float(np.max(np.abs(B))) if k > 0 else 0.0)
        adjoint_residual_max = max(adjoint_residual_max, resid / denom_resid)

        for j in range(k):
            con_idx = int(start + j)
            a_p = Y[0:n_theta, j]
            a_q = Y[n_theta:n_vars, j]

            if return_h_vectors:
                line_pos = con_idx // 2
                if (con_idx % 2) == 0:
                    h_from[line_pos, :] = Y[:, j]
                else:
                    h_to[line_pos, :] = Y[:, j]

            if bool(balance):
                denom = _balanced_two_block_norm_from_red(
                    a_p_red=a_p,
                    a_q_red=a_q,
                    n_bus_total=n_bus,
                    n_pq_total=n_pq,
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

    # ---------- operator-norm pass for zero-flow (nondifferentiable) ends ----------
    operator_norm_end = np.zeros(n_con, dtype=bool)
    nd_indices = np.flatnonzero(nondifferentiable_end)

    if bool(zero_flow_operator_norm) and nd_indices.size:
        ends_per_chunk = max(1, int(chunk_size) // 2)
        for cs in range(0, int(nd_indices.size), ends_per_chunk):
            batch = nd_indices[cs : cs + ends_per_chunk]
            kb = int(batch.size)
            B2 = np.zeros((n_vars, 2 * kb), dtype=float)

            valid = np.zeros(kb, dtype=bool)
            for j, con_idx in enumerate(batch):
                line_pos = int(con_idx // 2)
                is_from_end = (int(con_idx) % 2) == 0
                entries = _end_gradient_rows(op, line_pos, is_from_end)
                if entries is None:
                    continue
                valid[j] = True
                for row, dP, dQ in entries:
                    B2[row, 2 * j] += dP
                    B2[row, 2 * j + 1] += dQ

            Y2 = op.solve_J_transpose(B2)

            resid = float(np.max(np.abs(op.J.T @ Y2 - B2)))
            denom_resid = max(1.0, float(np.max(np.abs(B2))))
            adjoint_residual_max = max(adjoint_residual_max, resid / denom_resid)

            for j, con_idx in enumerate(batch):
                if not valid[j]:
                    continue
                hP = Y2[:, 2 * j]
                hQ = Y2[:, 2 * j + 1]

                g11 = _balanced_cross_inner(
                    hP[0:n_theta],
                    hP[n_theta:n_vars],
                    hP[0:n_theta],
                    hP[n_theta:n_vars],
                    n_bus_total=n_bus,
                    n_pq_total=n_pq,
                    balance=bool(balance),
                )
                g22 = _balanced_cross_inner(
                    hQ[0:n_theta],
                    hQ[n_theta:n_vars],
                    hQ[0:n_theta],
                    hQ[n_theta:n_vars],
                    n_bus_total=n_bus,
                    n_pq_total=n_pq,
                    balance=bool(balance),
                )
                g12 = _balanced_cross_inner(
                    hP[0:n_theta],
                    hP[n_theta:n_vars],
                    hQ[0:n_theta],
                    hQ[n_theta:n_vars],
                    n_bus_total=n_bus,
                    n_pq_total=n_pq,
                    balance=bool(balance),
                )

                tr = g11 + g22
                disc = math.sqrt(max((g11 - g22) ** 2 + 4.0 * g12 * g12, 0.0))
                lam_max = 0.5 * (tr + disc)
                sigma = math.sqrt(max(lam_max, 0.0))

                operator_norm_end[int(con_idx)] = True
                norms[int(con_idx)] = float(sigma)
                margin = float(margin_end[int(con_idx)])
                if sigma > _EPS_NORM:
                    radii[int(con_idx)] = margin / sigma
                else:
                    radii[int(con_idx)] = (
                        float("inf") if margin >= 0.0 else float("-inf")
                    )

        logger.info(
            "AC zero-flow operator-norm radii computed for %d/%d nondifferentiable "
            "line end(s) (2D (P,Q) directional derivative, balanced sigma_max).",
            int(np.sum(operator_norm_end)),
            int(nd_indices.size),
        )

    if diagnostic_subgradient_used > 0:
        logger.debug(
            "AC |S| nondifferentiable ends: %d/%d constraint-ends below the "
            "scale-aware threshold max(%.3g, %.3g*limit) MVA.",
            int(diagnostic_subgradient_used),
            int(n_con),
            float(_EPS_S0_MVA),
            float(_EPS_S0_REL_LIMIT),
        )

    logger.info(
        "AC adjoint residual: max ||J^T h - b||_inf / max(1, ||b||_inf) = %.3e",
        float(adjoint_residual_max),
    )

    # ---------- aggregate per line (min of from/to end) ----------
    results: Dict[str, Dict[str, Any]] = {}
    finite_vals: list[float] = []

    for pos, lid in enumerate(line_ids):
        r_from = float(radii[2 * pos])
        r_to = float(radii[2 * pos + 1])
        r_line = min(r_from, r_to)

        binding_end = "from" if r_from <= r_to else "to"
        bind_idx = 2 * pos if binding_end == "from" else 2 * pos + 1
        margin_bind = (
            float(margin_from[pos]) if binding_end == "from" else float(margin_to[pos])
        )
        norm_bind = float(norms[bind_idx])
        nondiff_from = bool(nondifferentiable_end[2 * pos])
        nondiff_to = bool(nondifferentiable_end[2 * pos + 1])
        nondiff_bind = nondiff_from if binding_end == "from" else nondiff_to
        opnorm_bind = bool(operator_norm_end[bind_idx])
        status_bind, cert_radius_bind, signed_distance_bind = (
            classify_constraint_certificate(
                margin=float(margin_bind),
                dual_norm=float(norm_bind),
                eps=_EPS_NORM,
                is_unconstrained=bool(is_unconstrained[pos]),
            )
        )
        linearization_status = "nonlinear_unvalidated"
        if bool(nondiff_bind) and not bool(is_unconstrained[pos]):
            if opnorm_bind:
                if status_bind == ConstraintStatus.OK_FINITE.value:
                    status_bind = ConstraintStatus.OK_FINITE_OPERATOR_NORM.value
                linearization_status = "operator_norm_first_order"
            else:
                status_bind = ConstraintStatus.NONDIFFERENTIABLE_APPARENT_POWER.value
                cert_radius_bind = 0.0
                linearization_status = (
                    ConstraintStatus.NONDIFFERENTIABLE_APPARENT_POWER.value
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
            "ac_nondifferentiable_from": bool(nondiff_from),
            "ac_operator_norm_from": bool(operator_norm_end[2 * pos]),
            "radius_ac_l2_from": float(r_from),
            "ac_p0_to_mw": float(p1[pos]),
            "ac_q0_to_mvar": float(q1[pos]),
            "ac_s0_to_mva": float(s0_to[pos]),
            "ac_margin_to_mva": float(margin_to[pos]),
            "ac_norm_a_to": float(norms[2 * pos + 1]),
            "ac_nondifferentiable_to": bool(nondiff_to),
            "ac_operator_norm_to": bool(operator_norm_end[2 * pos + 1]),
            "radius_ac_l2_to": float(r_to),
            "ac_nd_threshold_mva": float(nd_threshold_mva[pos]),
            # per-line aggregate
            "radius_ac_l2": float(r_line),
            "ac_limiting_end": str(binding_end),
            # unified table fields requested
            "binding_end": str(binding_end),
            "margin_ac_mva": float(margin_bind),
            "||h||2": float(norm_bind),
            "radius_ac_l2_linear": float(r_line),
            "certificate_radius_ac_l2": float(cert_radius_bind),
            "signed_distance_ac_l2": float(signed_distance_bind),
            "constraint_status_ac_l2": str(status_bind),
            "nondifferentiable_apparent_power": bool(nondiff_bind),
            "linearization_status": str(linearization_status),
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

    if return_h_vectors:
        results["_h_vectors"] = {
            "h_from": h_from,
            "h_to": h_to,
            "pq_mask": op.pq_mask,
            "slack_pos": int(slack_pos),
            "slack_bus_id": int(slack_bus_id),
            "pv_to_pq_bus_ids": list(pv_to_pq_bus_ids),
            "adjoint_residual_max": float(adjoint_residual_max),
        }

    return results
