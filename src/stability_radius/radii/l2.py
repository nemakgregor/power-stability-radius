from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .common import LineBaseQuantities, get_line_base_quantities, line_key
from .core_l2 import row_l2_norms_projected_ones_complement

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

    Certificate (balanced disturbances)
    -----------------------------------
    This project certifies robustness against **balanced** injection perturbations:
        sum(Δp) = 0

    and measures disturbance size in the **full-bus** Euclidean norm:
        ||Δp||_2  over R^{n_bus}.

    In this setting, a line sensitivity row g is only defined up to adding a constant
    all-ones vector. The correct dual norm is:
        ||Proj(g)||_2,  where Proj(g) = g - mean(g)*1.

    For each line l:
        margin_l = P_limit_l - |P0_l|
        r_l2 = margin_l / ||Proj(g_l)||_2

    Units contract
    --------------
    - flow0_mw, p0_mw, margin_mw, radius_l2: MW
    - p_limit_mw_est: MW (derived from MVA rating under PF=1 assumption in DC)

    Parameters
    ----------
    net:
        pandapower network.
    H_full:
        Sensitivity matrix (m_lines x n_buses). (Slack choice may change H_full columns,
        but the projected norms are slack-invariant.)
    margin_factor:
        Multiplier applied to estimated limits (e.g., 0.9 for conservative, 1.1 for relaxed).
    base:
        Optional precomputed per-line base quantities (to avoid repeated OPF).

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

    H = np.asarray(H_full, dtype=float)
    if len(base_q.line_indices) != H.shape[0]:
        raise ValueError(
            f"H_full row count ({H.shape[0]}) does not match net.line count ({len(base_q.line_indices)})."
        )

    norms = row_l2_norms_projected_ones_complement(H)
    if norms.shape != (H.shape[0],):
        raise ValueError("Internal error: projected row norms shape mismatch.")

    results: Dict[str, Dict[str, Any]] = {}
    finite_radii: list[float] = []

    for pos, lid in enumerate(base_q.line_indices):
        margin = float(base_q.margin_mw[pos])
        norm_g = float(norms[pos])

        # If a line is insensitive on the balanced subspace (norm_g==0), it cannot restrict Δp.
        r_l2 = float(margin / norm_g) if norm_g > 1e-12 else float("inf")

        k = line_key(lid)
        results[k] = {
            "flow0_mw": float(base_q.flow0_mw[pos]),
            "p0_mw": float(base_q.p0_abs_mw[pos]),
            "p_limit_mw_est": float(base_q.limit_mva_assumed_mw[pos]),
            "margin_mw": margin,
            "norm_g": norm_g,
            "radius_l2": r_l2,
        }
        if np.isfinite(r_l2):
            finite_radii.append(r_l2)

    # Keep CLI output compact: summary goes to DEBUG (still written to file log by default).
    if finite_radii:
        logger.debug("Mean L2 radius: %.6g", float(np.mean(finite_radii)))
    else:
        logger.debug("Mean L2 radius: n/a (no finite radii)")

    return results
