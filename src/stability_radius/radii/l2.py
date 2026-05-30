from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .common import (
    LineBaseQuantities,
    classify_constraint_certificate,
    estimate_line_limit_mva_with_flag,
    get_line_base_quantities,
    line_key,
    signed_radius_from_margin_norm,
)
from .core_l2 import row_l2_norms_projected_ones_complement

logger = logging.getLogger(__name__)


def _base_and_sensitivity_matrix(
    *,
    net: Any,
    H_full: np.ndarray,
    limit_factor: float,
    base: LineBaseQuantities | None,
) -> tuple[LineBaseQuantities, np.ndarray]:
    """Return validated base quantities and sensitivity matrix."""
    base_q = (
        base
        if base is not None
        else get_line_base_quantities(net, limit_factor=float(limit_factor))
    )
    H = np.asarray(H_full, dtype=float)
    if len(base_q.line_indices) != H.shape[0]:
        raise ValueError(
            f"H_full row count ({H.shape[0]}) does not match net.line count ({len(base_q.line_indices)})."
        )
    return base_q, H


def _projected_l2_norms(H: np.ndarray) -> np.ndarray:
    """Return balanced-subspace row norms for a sensitivity matrix."""
    norms = row_l2_norms_projected_ones_complement(H)
    if norms.shape != (H.shape[0],):
        raise ValueError("Internal error: projected row norms shape mismatch.")
    return norms


def _unconstrained_flags(net: Any, base_q: LineBaseQuantities) -> np.ndarray:
    """Return per-line flags for MATPOWER/PGLib unconstrained thermal ratings."""
    if base_q.is_unconstrained is not None:
        flags = np.asarray(base_q.is_unconstrained, dtype=bool).reshape(-1)
        if flags.shape != (len(base_q.line_indices),):
            raise ValueError("base.is_unconstrained shape mismatch.")
        return flags

    flags = np.zeros(len(base_q.line_indices), dtype=bool)
    for pos, lid in enumerate(base_q.line_indices):
        _, is_uc = estimate_line_limit_mva_with_flag(net, net.line.loc[int(lid)])
        flags[pos] = bool(is_uc)
    return flags


def _l2_result_row(
    *,
    base_q: LineBaseQuantities,
    pos: int,
    norm_g: float,
    is_unconstrained: bool,
) -> dict[str, Any]:
    """Build one DC L2 result row."""
    margin = float(base_q.margin_mw[pos])
    r_l2 = signed_radius_from_margin_norm(margin=margin, dual_norm=norm_g, eps=1e-12)
    status, cert_radius, signed_distance = classify_constraint_certificate(
        margin=margin,
        dual_norm=norm_g,
        eps=1e-12,
        is_unconstrained=bool(is_unconstrained),
    )
    return {
        "flow0_mw": float(base_q.flow0_mw[pos]),
        "p0_mw": float(base_q.p0_abs_mw[pos]),
        "p_limit_mw_est": float(base_q.limit_mva_assumed_mw[pos]),
        "is_unconstrained": bool(is_unconstrained),
        "margin_mw": margin,
        "norm_g": float(norm_g),
        "radius_l2": r_l2,
        "certificate_radius_l2": float(cert_radius),
        "signed_distance_l2": float(signed_distance),
        "constraint_status_l2": str(status),
    }


def _log_radius_summary(finite_radii: list[float]) -> None:
    """Log a compact L2 radius summary."""
    if finite_radii:
        logger.debug("Mean L2 radius: %.6g", float(np.mean(finite_radii)))
    else:
        logger.debug("Mean L2 radius: n/a (no finite radii)")


def compute_l2_radius(
    net: Any,
    H_full: np.ndarray,
    limit_factor: float = 1.0,
    *,
    base: LineBaseQuantities | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-line DC L2 robustness radii using PTDF-like sensitivities.

    The certificate is evaluated on balanced active-power disturbances. Each
    sensitivity row is projected onto the ones-complement before taking the
    dual L2 norm, so the result is invariant to the slack-reference component
    of a PTDF row.
    """
    base_q, H = _base_and_sensitivity_matrix(
        net=net,
        H_full=H_full,
        limit_factor=float(limit_factor),
        base=base,
    )
    norms = _projected_l2_norms(H)
    is_unconstrained = _unconstrained_flags(net, base_q)

    results: Dict[str, Dict[str, Any]] = {}
    finite_radii: list[float] = []

    for pos, lid in enumerate(base_q.line_indices):
        row = _l2_result_row(
            base_q=base_q,
            pos=pos,
            norm_g=float(norms[pos]),
            is_unconstrained=bool(is_unconstrained[pos]),
        )
        results[line_key(lid)] = row
        r_l2 = float(row["radius_l2"])
        if np.isfinite(r_l2):
            finite_radii.append(r_l2)

    _log_radius_summary(finite_radii)
    return results
