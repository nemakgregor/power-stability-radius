from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .common import LineBaseQuantities, get_line_base_quantities, line_key

logger = logging.getLogger(__name__)


def compute_l2_radius(
    net,
    H_full: np.ndarray,
    margin_factor: float = 1.0,
    *,
    base: LineBaseQuantities | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute a simple per-line L2 robustness radius using a PTDF-like sensitivity.

    For each line l:
        margin_l = P_limit_l - |P0_l|
        r_l2 = margin_l / ||g_l||_2
    where g_l is the l-th row of H_full.

    Parameters
    ----------
    net:
        pandapower network.
    H_full:
        Sensitivity matrix (m_lines x n_buses).
    margin_factor:
        Multiplier applied to estimated limits (e.g., 0.9 for conservative, 1.1 for relaxed).
    base:
        Optional precomputed per-line base quantities (to avoid repeated runpp()).

    Returns
    -------
    dict
        Mapping "line_{line_index}" -> metrics dict.
    """
    base_q = (
        base
        if base is not None
        else get_line_base_quantities(net, margin_factor=margin_factor)
    )

    if len(base_q.line_indices) != H_full.shape[0]:
        raise ValueError(
            f"H_full row count ({H_full.shape[0]}) does not match net.line count ({len(base_q.line_indices)})."
        )

    results: Dict[str, Dict[str, Any]] = {}
    finite_radii: list[float] = []

    for pos, lid in enumerate(base_q.line_indices):
        margin = float(base_q.margin_mw[pos])
        g_l = np.asarray(H_full[pos, :], dtype=float)
        norm_g = float(np.linalg.norm(g_l, ord=2))
        r_l2 = float(margin / norm_g) if norm_g > 1e-12 else float("inf")

        k = line_key(lid)
        results[k] = {
            "p0_mw": float(base_q.p0_abs_mw[pos]),
            "p_limit_mw_est": float(base_q.limit_mw_est[pos]),
            "margin_mw": margin,
            "norm_g": norm_g,
            "radius_l2": r_l2,
        }
        if np.isfinite(r_l2):
            finite_radii.append(r_l2)

    if finite_radii:
        logger.info("Mean L2 radius: %.6g", float(np.mean(finite_radii)))
    else:
        logger.info("Mean L2 radius: n/a (no finite radii)")

    return results
