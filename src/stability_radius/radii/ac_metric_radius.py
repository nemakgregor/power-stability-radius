from __future__ import annotations

"""
AC metric radius — certificate under an arbitrary SPD weight matrix M.

This module is stateless and consumes precomputed adjoint h-vectors
(from the AC L2 module with ``return_h_vectors=True``).

Mathematical definition
-----------------------
Given SPD weight matrix M ∈ R^{2n×2n} (or diagonal vector ∈ R^{2n})
and per-line adjoint sensitivity h_ℓ ∈ R^{2n}:

    r_ℓ^M = (c_ℓ − |S_ℓ⁰|) / √(h_ℓᵀ M⁻¹ h_ℓ)

Special cases:
  - M = I        →  r^M = r^{L2}  (standard AC L2 radius)
  - M = Σ⁻¹      →  r^M = r^σ     (sigma-radius)

Diagonal optimisation
---------------------
When M is supplied as a 1-D vector of length 2n, the module avoids
the O(n³) Cholesky and computes:

    h^T M^{-1} h = Σ_i h_i² / m_i

in O(n) time.
"""

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from stability_radius.geometry.balanced import (
    BlockSpec,
    project_dual_balanced_rows,
)
from stability_radius.radii.common import classify_constraint_certificate, line_key

logger = logging.getLogger(__name__)

_EPS_DENOM = 1e-12


def _validate_inputs(
    *,
    h_vectors: np.ndarray,
    s_limit_mva: np.ndarray,
    s0_mva: np.ndarray,
    M: np.ndarray,
    line_ids: Sequence[int] | None,
    pq_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Validate and normalise inputs, return (H, s_lim, s0, M, ids)."""
    H = np.asarray(h_vectors, dtype=float)
    if H.ndim != 2:
        raise ValueError(f"h_vectors must be 2D, got shape={H.shape}")
    n_lines, d = int(H.shape[0]), int(H.shape[1])

    s_lim = np.asarray(s_limit_mva, dtype=float).reshape(-1)
    s0 = np.asarray(s0_mva, dtype=float).reshape(-1)
    if s_lim.shape != (n_lines,):
        raise ValueError(f"s_limit_mva must have shape ({n_lines},), got {s_lim.shape}")
    if s0.shape != (n_lines,):
        raise ValueError(f"s0_mva must have shape ({n_lines},), got {s0.shape}")

    if np.any(~np.isfinite(s_lim)) or np.any(s_lim < 0.0):
        raise ValueError("s_limit_mva must be finite and non-negative per line.")
    if np.any(~np.isfinite(s0)) or np.any(s0 < 0.0):
        raise ValueError("s0_mva must be finite and non-negative per line.")

    M_arr = np.asarray(M, dtype=float)
    if M_arr.ndim == 1:
        if M_arr.shape != (d,):
            raise ValueError(f"Diagonal M must have shape ({d},), got {M_arr.shape}")
        if np.any(M_arr <= 0.0) or np.any(~np.isfinite(M_arr)):
            raise ValueError("Diagonal M entries must be finite and strictly positive.")
    elif M_arr.ndim == 2:
        if M_arr.shape != (d, d):
            raise ValueError(f"Dense M must have shape ({d},{d}), got {M_arr.shape}")
    else:
        raise ValueError(
            f"M must be 1D (diagonal) or 2D (dense), got ndim={M_arr.ndim}"
        )

    if line_ids is None:
        ids = tuple(range(n_lines))
    else:
        ids = tuple(int(x) for x in line_ids)
        if len(ids) != n_lines:
            raise ValueError(
                f"line_ids length must match n_lines={n_lines}, got {len(ids)}"
            )

    if pq_mask is not None:
        n_bus = d // 2
        pq = np.asarray(pq_mask, dtype=bool).reshape(-1)
        if d != 2 * n_bus or pq.shape != (n_bus,):
            raise ValueError(f"pq_mask must have shape ({n_bus},), got {pq.shape}")

    return H, s_lim, s0, M_arr, ids


def _metric_active_coordinates(*, n_bus: int, pq_mask: np.ndarray | None) -> np.ndarray:
    """Return active metric coordinates, optionally excluding PV/slack Q entries."""
    n = int(n_bus)
    if pq_mask is None:
        return np.arange(2 * n, dtype=int)
    pq = np.asarray(pq_mask, dtype=bool).reshape(-1)
    q_idx = np.where(pq)[0]
    return np.concatenate([np.arange(n, dtype=int), n + q_idx])


def _metric_balance_blocks(
    *, n_bus: int, n_q_active: int, balance: bool
) -> tuple[BlockSpec, BlockSpec]:
    """Block specs in reduced active coordinates [P_all; Q_active]."""
    n = int(n_bus)
    nq = int(n_q_active)
    return (
        BlockSpec(
            name="P",
            indices=np.arange(0, n, dtype=int),
            balance=bool(balance),
        ),
        BlockSpec(
            name="Q",
            indices=np.arange(n, n + nq, dtype=int),
            balance=bool(balance),
        ),
    )


def _constraint_matrix_from_blocks(
    *, d: int, blocks: tuple[BlockSpec, ...]
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for block in blocks:
        idx = np.asarray(block.indices, dtype=int).reshape(-1)
        if idx.size == 0 or not bool(block.balance):
            continue
        row = np.zeros(int(d), dtype=float)
        row[idx] = 1.0
        rows.append(row)
    if not rows:
        return np.zeros((0, int(d)), dtype=float)
    return np.vstack(rows)


def compute_ac_metric_radius(
    *,
    h_vectors: np.ndarray,
    s_limit_mva: np.ndarray,
    s0_mva: np.ndarray,
    M: np.ndarray,
    line_ids: Sequence[int] | None = None,
    balance: bool = True,
    pq_mask: np.ndarray | None = None,
    eps_denom: float = _EPS_DENOM,
) -> dict[str, dict[str, Any]]:
    """
    Compute AC metric radius per line using precomputed adjoint h-vectors.

    Parameters
    ----------
    h_vectors :
        Array of shape ``(n_lines, 2*n_bus)``.  Each row is
        ``h_ℓ = [hP; hQ]`` for the binding end of that line.
    s_limit_mva :
        Symmetric thermal limits ``c_ℓ`` in MVA, shape ``(n_lines,)``.
    s0_mva :
        Base apparent power magnitudes ``|S⁰_ℓ|`` at the binding end,
        in MVA, shape ``(n_lines,)``.
    M :
        SPD weight matrix.  Either:
        - 1-D array of shape ``(2*n_bus,)`` for diagonal M, or
        - 2-D array of shape ``(2*n_bus, 2*n_bus)`` for dense M.
    line_ids :
        Optional line indices for stable keys ``line_<id>``.
        If None, uses ``0..n_lines-1``.
    balance :
        If True, computes the constrained metric dual norm on the balanced
        subspace using the M^{-1}-weighted projection.
    pq_mask:
        Optional full-bus boolean mask for PQ buses.  When provided, the metric
        is restricted to active coordinates `[P_all; Q_PQ]`; PV and slack Q
        coordinates are excluded from the independent perturbation space.
    eps_denom :
        Numerical threshold below which the denominator is treated
        as zero (degenerate sensitivity).

    Returns
    -------
    dict
        Mapping ``"line_<id>"`` -> dict with keys:
          - ``metric_denom``       : ``sqrt(h^T M^{-1} h)``
          - ``margin_mva``         : ``c - |S⁰|``
          - ``radius_ac_metric``   : ``margin / metric_denom``
    """
    H, c, s0, M_arr, ids = _validate_inputs(
        h_vectors=h_vectors,
        s_limit_mva=s_limit_mva,
        s0_mva=s0_mva,
        M=M,
        line_ids=line_ids,
        pq_mask=pq_mask,
    )

    eps = float(eps_denom)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps_denom must be finite and >0.")

    n_lines = int(H.shape[0])
    d_full = int(H.shape[1])
    n_bus = d_full // 2
    active = _metric_active_coordinates(n_bus=n_bus, pq_mask=pq_mask)
    H_work = H[:, active]
    if M_arr.ndim == 1:
        M_work = M_arr[active]
    else:
        M_work = M_arr[np.ix_(active, active)]
    d = int(H_work.shape[1])
    n_q_active = int(d - n_bus)
    blocks = _metric_balance_blocks(
        n_bus=n_bus, n_q_active=n_q_active, balance=bool(balance)
    )

    # Compute denominator: sqrt(h^T M^{-1} h) for each line.
    denom = np.empty(n_lines, dtype=float)

    is_diagonal = M_work.ndim == 1

    if is_diagonal:
        # Diagonal case: balanced projection uses M^{-1} weights.  This makes
        # M=diag(1/sigma^2) exactly match AC sigma-radius.
        M_inv = 1.0 / M_work  # shape (d,)
        if bool(balance):
            H_proj = project_dual_balanced_rows(
                H_work,
                (
                    BlockSpec(
                        name="P",
                        indices=np.arange(0, n_bus, dtype=int),
                        balance=True,
                        weights=M_inv[:n_bus],
                    ),
                    BlockSpec(
                        name="Q",
                        indices=np.arange(n_bus, d, dtype=int),
                        balance=True,
                        weights=M_inv[n_bus:],
                    ),
                ),
            )
        else:
            H_proj = H_work
        denom = np.sqrt(np.sum(H_proj * H_proj * M_inv[None, :], axis=1))
    else:
        # Dense case: use the exact constrained dual norm
        # h^T[M^{-1} - M^{-1}C^T(CM^{-1}C^T)^+CM^{-1}]h.
        try:
            np.linalg.cholesky(M_work)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "M must be symmetric positive definite (Cholesky failed)."
            ) from e

        M_inv_Ht = np.linalg.solve(M_work, H_work.T)  # shape (d, n_lines)
        quad = np.sum(H_work.T * M_inv_Ht, axis=0)
        if bool(balance):
            C = _constraint_matrix_from_blocks(d=d, blocks=blocks)
            if C.shape[0] > 0:
                M_inv_Ct = np.linalg.solve(M_work, C.T)
                A = C @ M_inv_Ct
                rhs = C @ M_inv_Ht
                A_pinv = np.linalg.pinv(A)
                corr = np.sum(rhs * (A_pinv @ rhs), axis=0)
                quad = quad - corr
        denom = np.sqrt(np.maximum(quad, 0.0))

    margin = c - s0

    radius = np.empty(n_lines, dtype=float)
    valid = denom > eps
    radius[valid] = margin[valid] / denom[valid]
    radius[~valid] = np.where(margin[~valid] >= 0.0, float("inf"), float("-inf"))

    results: dict[str, dict[str, Any]] = {}
    for pos, lid in enumerate(ids):
        status, cert_radius, signed_distance = classify_constraint_certificate(
            margin=float(margin[pos]),
            dual_norm=float(denom[pos]),
            eps=eps,
        )
        k = line_key(int(lid))
        results[k] = {
            "metric_denom": float(denom[pos]),
            "margin_mva": float(margin[pos]),
            "radius_ac_metric": float(radius[pos]),
            "certificate_radius_ac_metric": float(cert_radius),
            "signed_distance_ac_metric": float(signed_distance),
            "constraint_status_ac_metric": str(status),
        }

    # Logging summary.
    finite_r = radius[np.isfinite(radius)]
    if finite_r.size > 0:
        logger.info(
            "AC metric-radius computed: lines=%d finite=%d mean=%.6g min=%.6g max=%.6g (balance=%s)",
            int(n_lines),
            int(finite_r.size),
            float(np.mean(finite_r)),
            float(np.min(finite_r)),
            float(np.max(finite_r)),
            bool(balance),
        )
    else:
        logger.info(
            "AC metric-radius computed: lines=%d finite=0 (balance=%s)",
            int(n_lines),
            bool(balance),
        )

    n_degenerate = int(np.sum(~valid))
    if n_degenerate > 0:
        logger.debug(
            "AC metric-radius: degenerate denom<=eps count=%d/%d (eps=%.3g).",
            n_degenerate,
            int(n_lines),
            float(eps),
        )

    return results
