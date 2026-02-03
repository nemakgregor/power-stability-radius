from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .common import (
    LineBaseQuantities,
    as_2d_square_matrix,
    get_line_base_quantities,
    line_key,
)

logger = logging.getLogger(__name__)


def metric_denominator_l2_weighted(g: np.ndarray, M: np.ndarray) -> float:
    """
    Compute sqrt(g^T M^{-1} g) for SPD weight matrix M.

    Implementation uses Cholesky: M = L L^T, so g^T M^{-1} g = ||L^{-1} g||_2^2.

    Raises
    ------
    ValueError
        If M is not SPD.
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    M = np.asarray(M, dtype=float)
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "M must be symmetric positive definite (Cholesky failed)."
        ) from e

    z = np.linalg.solve(L, g)
    return float(np.linalg.norm(z, ord=2))


def metric_radius(margin: float, g: np.ndarray, M: np.ndarray) -> float:
    """
    Metric radius: r = margin / sqrt(g^T M^{-1} g).

    Returns +inf when denominator is ~0 and margin>0.
    """
    denom = metric_denominator_l2_weighted(g, M)
    if denom <= 1e-12:
        return float("inf") if float(margin) > 0 else 0.0
    return float(margin) / denom


def compute_metric_radius(
    net,
    H_full: np.ndarray,
    M: np.ndarray,
    *,
    limit_factor: float = 1.0,
    base: LineBaseQuantities | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-line metric radii under weighted nodal perturbations.

    For each line l:
        r_l^(M) = margin_l / sqrt(g_l^T M^{-1} g_l)

    Parameters
    ----------
    net:
        pandapower network.
    H_full:
        Sensitivity matrix (m_lines x n_buses).
    M:
        SPD weight matrix defining ||dp||_M = sqrt(dp^T M dp).
    limit_factor:
        Applied to extracted limits when base is not provided.
    base:
        Optional precomputed per-line base quantities (to avoid repeated OPF).

    Returns
    -------
    dict
        Mapping "line_{line_index}" -> metrics dict (includes 'radius_metric').
    """
    base_q = (
        base
        if base is not None
        else get_line_base_quantities(net, limit_factor=float(limit_factor))
    )
    n_bus = int(H_full.shape[1])
    M = as_2d_square_matrix(np.asarray(M, dtype=float), n_bus, name="M")

    if len(base_q.line_indices) != H_full.shape[0]:
        raise ValueError(
            f"H_full row count ({H_full.shape[0]}) does not match net.line count ({len(base_q.line_indices)})."
        )

    # Keep CLI output compact: detailed progress goes to DEBUG.
    logger.debug("Computing metric radii (n_bus=%d)...", n_bus)

    # Factor once for all lines
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "M must be symmetric positive definite (Cholesky failed)."
        ) from e

    results: Dict[str, Dict[str, Any]] = {}
    for pos, lid in enumerate(base_q.line_indices):
        g_l = np.asarray(H_full[pos, :], dtype=float)
        z = np.linalg.solve(L, g_l)
        denom = float(np.linalg.norm(z, ord=2))

        margin = float(base_q.margin_mw[pos])
        r = (
            float(margin / denom)
            if denom > 1e-12
            else (float("inf") if margin > 0 else 0.0)
        )

        k = line_key(lid)
        results[k] = {
            "flow0_mw": float(base_q.flow0_mw[pos]),
            "p0_mw": float(base_q.p0_abs_mw[pos]),
            "p_limit_mw_est": float(base_q.limit_mva_assumed_mw[pos]),
            "margin_mw": margin,
            "metric_denom": denom,
            "radius_metric": r,
        }

    return results
