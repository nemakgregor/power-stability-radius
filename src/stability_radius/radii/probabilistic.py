from __future__ import annotations

import logging
import math
from typing import Any, Dict

import numpy as np

from .common import (
    LineBaseQuantities,
    as_1d_vector,
    as_2d_square_matrix,
    classify_constraint_certificate,
    get_line_base_quantities,
    line_key,
    signed_radius_from_margin_norm,
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
    if np.any(~np.isfinite(g)):
        raise ValueError("g must contain only finite values.")
    Sigma_arr = np.asarray(Sigma, dtype=float)

    if Sigma_arr.ndim == 1:
        v = as_1d_vector(Sigma_arr, g.size, name="Sigma(diag)")
        if np.any(~np.isfinite(v)) or np.any(v < 0.0):
            raise ValueError("Sigma(diag) entries must be finite and non-negative.")
        var = float(np.dot(g * g, v))
    else:
        S = as_2d_square_matrix(Sigma_arr, g.size, name="Sigma")
        if np.any(~np.isfinite(S)):
            raise ValueError("Sigma entries must be finite.")
        if not np.allclose(S, S.T, rtol=1e-10, atol=1e-12):
            raise ValueError("Sigma must be symmetric.")
        eig_min = float(np.min(np.linalg.eigvalsh(S))) if S.size else 0.0
        if eig_min < -1e-10:
            raise ValueError("Sigma must be positive semidefinite.")
        var = float(g @ S @ g)

    if not math.isfinite(var):
        raise ValueError("Computed variance must be finite.")
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
        Signed diagnostic radius, with +/-inf for zero sensitivity and NaN for
        non-finite inputs.
    """
    return signed_radius_from_margin_norm(margin=margin, dual_norm=sigma, eps=0.0)


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


def _base_and_sigma_inputs(
    *,
    net: Any,
    H_full: np.ndarray,
    Sigma: np.ndarray,
    limit_factor: float,
    base: LineBaseQuantities | None,
) -> tuple[LineBaseQuantities, np.ndarray, np.ndarray]:
    """Return validated base quantities, sensitivity matrix, and covariance."""
    base_q = (
        base
        if base is not None
        else get_line_base_quantities(net, limit_factor=float(limit_factor))
    )
    H = np.asarray(H_full, dtype=float)
    if H.ndim != 2:
        raise ValueError("H_full must be a 2D sensitivity matrix.")
    if len(base_q.line_indices) != H.shape[0]:
        raise ValueError(
            f"H_full row count ({H.shape[0]}) does not match net.line count ({len(base_q.line_indices)})."
        )

    n_bus = int(H.shape[1])
    Sigma_arr = np.asarray(Sigma, dtype=float)
    if Sigma_arr.ndim == 1:
        as_1d_vector(Sigma_arr, n_bus, name="Sigma(diag)")
    else:
        as_2d_square_matrix(Sigma_arr, n_bus, name="Sigma")
    return base_q, H, Sigma_arr


def _line_is_unconstrained(base_q: LineBaseQuantities, pos: int) -> bool:
    """Return whether a line represents an unconstrained thermal certificate."""
    if base_q.is_unconstrained is None:
        return False
    flags = np.asarray(base_q.is_unconstrained, dtype=bool).reshape(-1)
    if flags.shape != (len(base_q.line_indices),):
        raise ValueError("base.is_unconstrained shape mismatch.")
    return bool(flags[pos])


def _sigma_result_row(
    *,
    base_q: LineBaseQuantities,
    pos: int,
    sigma_flow: float,
    is_unconstrained: bool,
) -> dict[str, Any]:
    """Build one Gaussian DC radius result row."""
    margin = float(base_q.margin_mw[pos])
    radius = sigma_radius(margin, sigma_flow)
    status, cert_radius, signed_distance = classify_constraint_certificate(
        margin=margin,
        dual_norm=sigma_flow,
        eps=1e-12,
        is_unconstrained=bool(is_unconstrained),
    )

    limit = float(base_q.limit_mva_assumed_mw[pos])
    flow0 = float(base_q.flow0_mw[pos])
    prob = overload_probability_symmetric_limit(
        flow0=flow0,
        limit=limit,
        sigma=sigma_flow,
    )
    return {
        "flow0_mw": flow0,
        "p0_mw": float(base_q.p0_abs_mw[pos]),
        "p_limit_mw_est": limit,
        "margin_mw": margin,
        "sigma_flow": float(sigma_flow),
        "radius_sigma": float(radius),
        "certificate_radius_sigma": float(cert_radius),
        "signed_distance_sigma": float(signed_distance),
        "constraint_status_sigma": str(status),
        "overload_probability": float(prob),
    }


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
    base_q, H, Sigma_arr = _base_and_sigma_inputs(
        net=net,
        H_full=H_full,
        Sigma=Sigma,
        limit_factor=float(limit_factor),
        base=base,
    )

    # Keep CLI output compact: detailed progress goes to DEBUG.
    logger.debug("Computing sigma radii (Sigma ndim=%d)...", int(Sigma_arr.ndim))

    results: Dict[str, Dict[str, Any]] = {}
    for pos, lid in enumerate(base_q.line_indices):
        sigma_flow = flow_stddev(np.asarray(H[pos, :], dtype=float), Sigma_arr)
        results[line_key(lid)] = _sigma_result_row(
            base_q=base_q,
            pos=pos,
            sigma_flow=sigma_flow,
            is_unconstrained=_line_is_unconstrained(base_q, pos),
        )

    return results
