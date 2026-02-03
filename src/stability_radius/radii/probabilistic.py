from __future__ import annotations

import logging
import math
from typing import Any, Dict

import numpy as np

from .common import (
    LineBaseQuantities,
    as_1d_vector,
    as_2d_square_matrix,
    get_line_base_quantities,
    line_key,
)

logger = logging.getLogger(__name__)


def _qfunc(x: float) -> float:
    """
    Gaussian Q-function: Q(x) = P(Z > x) for Z~N(0,1).

    Uses erfc for numerical stability:
        Q(x) = 0.5 * erfc(x / sqrt(2))
    """
    return 0.5 * math.erfc(float(x) / math.sqrt(2.0))


def flow_stddev(g: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Compute sigma = sqrt(g^T Sigma g).

    Supports:
    - Sigma shape (n,n): covariance matrix
    - Sigma shape (n,): diagonal variances

    Raises
    ------
    ValueError
        On shape mismatch or negative numerical variance.
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    Sigma_arr = np.asarray(Sigma, dtype=float)

    if Sigma_arr.ndim == 1:
        v = as_1d_vector(Sigma_arr, g.size, name="Sigma(diag)")
        var = float(np.dot(g * g, v))
    else:
        S = as_2d_square_matrix(Sigma_arr, g.size, name="Sigma")
        var = float(g @ S @ g)

    if var < -1e-10:
        raise ValueError(
            f"Computed negative variance g^T Sigma g = {var}. Check Sigma PSD."
        )
    return math.sqrt(max(var, 0.0))


def sigma_radius(margin: float, sigma: float) -> float:
    """
    Sigma-radius: r = margin / sigma.

    Returns
    -------
    float
        +inf if sigma==0 and margin>0; 0 if both are 0.
    """
    margin = float(margin)
    sigma = float(sigma)
    if sigma <= 0.0:
        return float("inf") if margin > 0 else 0.0
    return margin / sigma


def overload_probability_symmetric_limit(
    *,
    flow0: float,
    limit: float,
    sigma: float,
) -> float:
    """
    Overload probability for a line under symmetric limit ±c with nonzero base flow.

    Model:
        F = f0 + X,   X ~ N(0, sigma^2)
    Then:
        P(|F| > c) = Q((c - |f0|)/sigma) + Q((c + |f0|)/sigma)

    Units
    -----
    - flow0, limit, sigma: MW

    Edge cases:
    - sigma==0: returns 1.0 if |f0|>c else 0.0
    """
    f0 = float(flow0)
    c = float(limit)
    s = float(sigma)

    if not math.isfinite(c) or c < 0:
        raise ValueError(f"limit must be finite and non-negative; got {limit!r}")

    if s <= 0.0:
        return 1.0 if abs(f0) > c else 0.0

    a = (c - abs(f0)) / s
    b = (c + abs(f0)) / s
    return _qfunc(a) + _qfunc(b)


def compute_sigma_radius(
    net,
    H_full: np.ndarray,
    Sigma: np.ndarray,
    *,
    limit_factor: float = 1.0,
    base: LineBaseQuantities | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-line sigma-radii and overload probabilities under Gaussian injections.

    For each line l:
        sigma_l^2 = g_l^T Sigma g_l
        r_sigma = margin_l / sigma_l
        P(|f| > c) computed using nonzero base flow f0 (see docstring above)

    Parameters
    ----------
    net:
        pandapower network.
    H_full:
        Sensitivity matrix (m_lines x n_buses).
    Sigma:
        Covariance matrix of nodal perturbations (n,n) or diagonal variances (n,).
    limit_factor:
        Applied to extracted limits when base is not provided.
    base:
        Optional precomputed per-line base quantities (to avoid repeated OPF).

    Returns
    -------
    dict
        Mapping "line_{line_index}" -> metrics dict including 'sigma_flow', 'radius_sigma',
        and 'overload_probability'.
    """
    base_q = (
        base
        if base is not None
        else get_line_base_quantities(net, limit_factor=float(limit_factor))
    )
    n_bus = int(H_full.shape[1])

    Sigma_arr = np.asarray(Sigma, dtype=float)
    if Sigma_arr.ndim == 1:
        as_1d_vector(Sigma_arr, n_bus, name="Sigma(diag)")
    else:
        as_2d_square_matrix(Sigma_arr, n_bus, name="Sigma")

    if len(base_q.line_indices) != H_full.shape[0]:
        raise ValueError(
            f"H_full row count ({H_full.shape[0]}) does not match net.line count ({len(base_q.line_indices)})."
        )

    # Keep CLI output compact: detailed progress goes to DEBUG.
    logger.debug("Computing sigma radii (Sigma ndim=%d)...", int(Sigma_arr.ndim))

    results: Dict[str, Dict[str, Any]] = {}
    for pos, lid in enumerate(base_q.line_indices):
        g_l = np.asarray(H_full[pos, :], dtype=float)
        sig = flow_stddev(g_l, Sigma_arr)

        margin = float(base_q.margin_mw[pos])
        r = sigma_radius(margin, sig)

        c = float(base_q.limit_mva_assumed_mw[pos])
        f0 = float(base_q.flow0_mw[pos])
        prob = overload_probability_symmetric_limit(flow0=f0, limit=c, sigma=sig)

        k = line_key(lid)
        results[k] = {
            "flow0_mw": f0,
            "p0_mw": float(base_q.p0_abs_mw[pos]),
            "p_limit_mw_est": c,
            "margin_mw": margin,
            "sigma_flow": float(sig),
            "radius_sigma": float(r),
            "overload_probability": float(prob),
        }

    return results
