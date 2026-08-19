from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict

import numpy as np

from stability_radius.ac.ac_model import build_ac_operator
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
_DEFAULT_ZERO_FLOW_REL_TOL = 1e-10


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


def _balanced_two_block_projection_from_red(
    *,
    a_p_red: np.ndarray,
    a_q_red: np.ndarray,
    n_bus_total: int,
    n_pq_total: int,
) -> np.ndarray:
    """Return the explicit Euclidean projection used by the balanced dual norm."""

    def _project(block: np.ndarray, total_size: int) -> np.ndarray:
        """Restore a reduced block and remove its all-ones component."""
        values = np.asarray(block, dtype=float).reshape(-1)
        if total_size == 0:
            if values.size:
                raise ValueError("A zero-size balance block must be empty.")
            return values
        if values.size == total_size:
            full = values.copy()
        elif values.size == total_size - 1:
            full = np.concatenate([values, np.zeros(1, dtype=float)])
        else:
            raise ValueError(
                f"Reduced block has size {values.size}, expected {total_size} "
                f"or {total_size - 1}."
            )
        return full - float(np.mean(full))

    return np.concatenate(
        [
            _project(np.asarray(a_p_red, dtype=float), int(n_bus_total)),
            _project(np.asarray(a_q_red, dtype=float), int(n_pq_total)),
        ]
    )


def compute_ac_l2_radius(
    net: Any,
    *,
    base_pf: PyPSAAPFResult,
    slack_bus: int,
    chunk_size: int = 256,
    balance: bool = True,
    lossless: bool = True,
    return_h_vectors: bool = False,
    return_timings: bool = False,
    zero_flow_rel_tol: float = _DEFAULT_ZERO_FLOW_REL_TOL,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute a fast AC L2 "stability radius" certificate around an AC PF base point.

    The certificate differentiates the apparent-power magnitude ``|S|`` at each
    monitored line end. Near zero flow, where that scalar derivative is not
    unique, it instead retains the two-dimensional ``(P,Q)`` response and uses
    its induced operator norm over the balanced injection ball.

    Sensitivity norms below ``1e-12`` are treated as degenerate/infinite
    according to ``classify_constraint_certificate``. These thresholds are
    documented in ``docs/algorithms_and_models.md`` and are intentionally kept
    local to the numerical kernels rather than hidden in the CLI config.

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
    if not math.isfinite(float(zero_flow_rel_tol)) or float(zero_flow_rel_tol) <= 0.0:
        raise ValueError("zero_flow_rel_tol must be finite and positive.")

    line_ids = [int(x) for x in sorted(net.line.index)]
    if tuple(line_ids) != tuple(base_pf.line_ids):
        raise ValueError(
            "AC PF base point line ordering mismatch: expected base_pf.line_ids == sorted(net.line.index)."
        )

    t_total0 = time.perf_counter()
    timing_build0 = time.perf_counter()
    forced_pq_bus_ids = {
        int(event["bus"])
        for event in (getattr(base_pf, "q_limit_events", ()) or ())
        if int(event.get("bus", -1)) >= 0
        and str(event.get("element", "")) != "ext_grid"
    }
    op = build_ac_operator(
        net=net,
        slack_bus=int(slack_bus),
        vm_pu=np.asarray(base_pf.v_mag_pu, dtype=float),
        va_rad=np.asarray(base_pf.v_ang_rad, dtype=float),
        line_indices=line_ids,
        lossless=bool(lossless),
        forced_pq_bus_ids=forced_pq_bus_ids,
    )
    timing_build_sec = time.perf_counter() - timing_build0

    m = int(len(line_ids))
    n_bus = int(len(op.bus_ids))

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

    # ---------- optional h-vector storage ----------
    n_vars = int(op.n_vars)
    n_theta = int(op.n_red)
    n_pq = int(op.n_pq)

    if return_h_vectors:
        h_from = np.zeros((m, n_vars), dtype=float)
        h_to = np.zeros((m, n_vars), dtype=float)

    # ---------- chunked adjoint solves ----------
    scale_end = np.maximum(
        np.repeat(np.maximum(limits_mva, float(op.sn_mva)), 2),
        1.0,
    )
    zero_flow_threshold_mva = float(zero_flow_rel_tol) * scale_end
    nondifferentiable_end = s_end <= zero_flow_threshold_mva
    zero_flow_operator_norm_end = np.zeros(n_con, dtype=bool)
    timing_rhs_sec = 0.0
    timing_adjoint_sec = 0.0
    timing_support_sec = 0.0
    adjoint_max_relative_residual = 0.0

    start = 0
    while start < n_con:
        end = min(n_con, start + int(chunk_size))
        k = int(end - start)

        timing_rhs0 = time.perf_counter()
        B = np.zeros((n_vars, k), dtype=float)

        for j in range(k):
            con_idx = int(start + j)
            line_pos = int(con_idx // 2)
            is_from_end = (con_idx % 2) == 0

            s0 = float(s_end[con_idx])
            if not bool(nondifferentiable_end[con_idx]):
                wP = float(p_end[con_idx]) / s0
                wQ = float(q_end[con_idx]) / s0
            else:
                # Filled by the two-output operator-norm pass below.
                continue

            dS_dx = op.dS_from_dx if is_from_end else op.dS_to_dx
            complex_gradient = np.asarray(
                dS_dx.getrow(line_pos).toarray(), dtype=np.complex128
            ).reshape(-1)
            B[:, j] = wP * complex_gradient.real + wQ * complex_gradient.imag
        timing_rhs_sec += time.perf_counter() - timing_rhs0

        timing_adjoint0 = time.perf_counter()
        Y = op.solve_J_transpose(B)
        timing_adjoint_sec += time.perf_counter() - timing_adjoint0

        if return_timings:
            residual = op.J.T @ Y - B
            residual_norm = np.linalg.norm(residual, axis=0)
            rhs_norm = np.linalg.norm(B, axis=0)
            rel_residual = residual_norm / np.maximum(1.0, rhs_norm)
            if rel_residual.size:
                adjoint_max_relative_residual = max(
                    float(adjoint_max_relative_residual),
                    float(np.max(rel_residual)),
                )

        timing_support0 = time.perf_counter()
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
        timing_support_sec += time.perf_counter() - timing_support0

        start = end

    # At a zero-flow end, max ||H du||_2 over the admissible unit ball is
    # sqrt(lambda_max(H K H^T)). The matrix is only 2x2, so this completes
    # the first-order treatment without choosing an arbitrary |S| subgradient.
    zero_indices = np.where(nondifferentiable_end)[0]
    for zero_start in range(0, int(zero_indices.size), int(chunk_size)):
        zero_chunk = zero_indices[zero_start : zero_start + int(chunk_size)]
        if zero_chunk.size == 0:
            continue
        Bp = np.zeros((n_vars, int(zero_chunk.size)), dtype=float)
        Bq = np.zeros((n_vars, int(zero_chunk.size)), dtype=float)
        for local_pos, con_idx_raw in enumerate(zero_chunk):
            con_idx = int(con_idx_raw)
            line_pos = int(con_idx // 2)
            dS_dx = op.dS_from_dx if con_idx % 2 == 0 else op.dS_to_dx
            complex_gradient = np.asarray(
                dS_dx.getrow(line_pos).toarray(), dtype=np.complex128
            ).reshape(-1)
            Bp[:, local_pos] = complex_gradient.real
            Bq[:, local_pos] = complex_gradient.imag
        Hp = op.solve_J_transpose(Bp)
        Hq = op.solve_J_transpose(Bq)
        for local_pos, con_idx_raw in enumerate(zero_chunk):
            con_idx = int(con_idx_raw)
            if bool(balance):
                projected_p = _balanced_two_block_projection_from_red(
                    a_p_red=Hp[:n_theta, local_pos],
                    a_q_red=Hp[n_theta:n_vars, local_pos],
                    n_bus_total=n_bus,
                    n_pq_total=n_pq,
                )
                projected_q = _balanced_two_block_projection_from_red(
                    a_p_red=Hq[:n_theta, local_pos],
                    a_q_red=Hq[n_theta:n_vars, local_pos],
                    n_bus_total=n_bus,
                    n_pq_total=n_pq,
                )
            else:
                projected_p = np.asarray(Hp[:, local_pos], dtype=float)
                projected_q = np.asarray(Hq[:, local_pos], dtype=float)
            gram = np.asarray(
                [
                    [np.dot(projected_p, projected_p), np.dot(projected_p, projected_q)],
                    [np.dot(projected_q, projected_p), np.dot(projected_q, projected_q)],
                ],
                dtype=float,
            )
            eigvals, eigvecs = np.linalg.eigh(gram)
            max_index = int(np.argmax(eigvals))
            max_eigenvalue = max(float(eigvals[max_index]), 0.0)
            denom = math.sqrt(max_eigenvalue)
            norms[con_idx] = denom
            margin = float(margin_end[con_idx])
            radii[con_idx] = (
                margin / denom
                if denom > _EPS_NORM
                else (float("inf") if margin >= 0.0 else float("-inf"))
            )
            zero_flow_operator_norm_end[con_idx] = True
            if return_h_vectors:
                output_direction = eigvecs[:, max_index]
                effective_h = (
                    float(output_direction[0]) * Hp[:, local_pos]
                    + float(output_direction[1]) * Hq[:, local_pos]
                )
                line_pos = int(con_idx // 2)
                if con_idx % 2 == 0:
                    h_from[line_pos, :] = effective_h
                else:
                    h_to[line_pos, :] = effective_h

    if zero_indices.size:
        logger.info(
            "AC zero-flow operator-norm treatment used for %d/%d line ends "
            "(relative threshold %.3g).",
            int(zero_indices.size),
            int(n_con),
            float(zero_flow_rel_tol),
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
        nondiff_from = bool(nondifferentiable_end[2 * pos])
        nondiff_to = bool(nondifferentiable_end[2 * pos + 1])
        zero_op_from = bool(zero_flow_operator_norm_end[2 * pos])
        zero_op_to = bool(zero_flow_operator_norm_end[2 * pos + 1])
        nondiff_bind = nondiff_from if binding_end == "from" else nondiff_to
        zero_op_bind = zero_op_from if binding_end == "from" else zero_op_to
        status_bind, cert_radius_bind, signed_distance_bind = (
            classify_constraint_certificate(
                margin=float(margin_bind),
                dual_norm=float(norm_bind),
                eps=_EPS_NORM,
                is_unconstrained=bool(is_unconstrained[pos]),
            )
        )
        linearization_status = "nonlinear_unvalidated"
        if bool(nondiff_bind) and not bool(zero_op_bind):
            status_bind = ConstraintStatus.NONDIFFERENTIABLE_APPARENT_POWER.value
            cert_radius_bind = 0.0
            linearization_status = ConstraintStatus.NONDIFFERENTIABLE_APPARENT_POWER.value

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
            "ac_zero_flow_operator_norm_from": bool(zero_op_from),
            "radius_ac_l2_from": float(r_from),
            "ac_p0_to_mw": float(p1[pos]),
            "ac_q0_to_mvar": float(q1[pos]),
            "ac_s0_to_mva": float(s0_to[pos]),
            "ac_margin_to_mva": float(margin_to[pos]),
            "ac_norm_a_to": float(norms[2 * pos + 1]),
            "ac_nondifferentiable_to": bool(nondiff_to),
            "ac_zero_flow_operator_norm_to": bool(zero_op_to),
            "radius_ac_l2_to": float(r_to),
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
            "zero_flow_operator_norm_certified": bool(zero_op_bind),
            "zero_flow_threshold_mva": float(zero_flow_threshold_mva[2 * pos if binding_end == "from" else 2 * pos + 1]),
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
        }

    if return_timings:
        h_vectors_mb = 0.0
        if return_h_vectors:
            h_vectors_mb = float((h_from.nbytes + h_to.nbytes) / 1.0e6)
        results["_timings"] = {
            "total_sec": float(time.perf_counter() - t_total0),
            "operator_build_lu_sec": float(timing_build_sec),
            "line_gradient_rhs_sec": float(timing_rhs_sec),
            "adjoint_solve_sec": float(timing_adjoint_sec),
            "adjoint_max_relative_residual": float(adjoint_max_relative_residual),
            "support_eval_sec": float(timing_support_sec),
            "n_bus": int(n_bus),
            "n_line": int(m),
            "n_line_ends": int(n_con),
            "n_vars": int(n_vars),
            "n_theta": int(n_theta),
            "n_pq": int(n_pq),
            "J_nnz": int(op.J.nnz),
            "chunk_size": int(chunk_size),
            "balance": bool(balance),
            "lossless": bool(lossless),
            "h_vectors_mb": float(h_vectors_mb),
            "zero_flow_operator_norm_ends": int(zero_indices.size),
            "zero_flow_rel_tol": float(zero_flow_rel_tol),
            "forced_pq_bus_count": int(len(forced_pq_bus_ids)),
        }

    return results
