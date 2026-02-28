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

from stability_radius.radii.common import line_key

logger = logging.getLogger(__name__)

_EPS_DENOM = 1e-12


def _validate_inputs(
    *,
    h_vectors: np.ndarray,
    s_limit_mva: np.ndarray,
    s0_mva: np.ndarray,
    M: np.ndarray,
    line_ids: Sequence[int] | None,
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

    return H, s_lim, s0, M_arr, ids


def compute_ac_metric_radius(
    *,
    h_vectors: np.ndarray,
    s_limit_mva: np.ndarray,
    s0_mva: np.ndarray,
    M: np.ndarray,
    line_ids: Sequence[int] | None = None,
    balance: bool = True,
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
        If True, projects each h-vector's P/Q blocks onto the
        sum-zero subspace via mean-subtraction before computing
        the metric norm.
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
    )

    eps = float(eps_denom)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps_denom must be finite and >0.")

    n_lines = int(H.shape[0])
    d = int(H.shape[1])
    n_bus = d // 2

    # Optional balance projection (mean-subtract P/Q blocks independently).
    if bool(balance):
        hP = H[:, :n_bus].copy()
        hQ = H[:, n_bus:].copy()
        hP -= np.mean(hP, axis=1, keepdims=True)
        hQ -= np.mean(hQ, axis=1, keepdims=True)
        H_proj = np.hstack([hP, hQ])
        logger.debug("Applied balanced projection to h-vectors (P/Q blocks).")
    else:
        H_proj = H

    # Compute denominator: sqrt(h^T M^{-1} h) for each line.
    denom = np.empty(n_lines, dtype=float)

    is_diagonal = M_arr.ndim == 1

    if is_diagonal:
        # Diagonal case: h^T M^{-1} h = sum(h_i^2 / m_i)
        M_inv = 1.0 / M_arr  # shape (d,)
        denom = np.sqrt(np.sum(H_proj * H_proj * M_inv[None, :], axis=1))
    else:
        # Dense case: Cholesky factorisation L L^T = M, then ||L^{-1} h||_2.
        try:
            L = np.linalg.cholesky(M_arr)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "M must be symmetric positive definite (Cholesky failed)."
            ) from e

        # Solve L z = h^T for each line (each row of H_proj).
        # scipy.linalg.solve_triangular would be faster, but numpy suffices
        # and avoids the extra dependency at the radii level.
        Z = np.linalg.solve(L, H_proj.T)  # shape (d, n_lines)
        denom = np.sqrt(np.sum(Z * Z, axis=0))  # shape (n_lines,)

    margin = c - s0

    radius = np.empty(n_lines, dtype=float)
    valid = denom > eps
    radius[valid] = margin[valid] / denom[valid]
    radius[~valid] = np.where(margin[~valid] >= 0.0, float("inf"), float("-inf"))

    results: dict[str, dict[str, Any]] = {}
    for pos, lid in enumerate(ids):
        k = line_key(int(lid))
        results[k] = {
            "metric_denom": float(denom[pos]),
            "margin_mva": float(margin[pos]),
            "radius_ac_metric": float(radius[pos]),
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
