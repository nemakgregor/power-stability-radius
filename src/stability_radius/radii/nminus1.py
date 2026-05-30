from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

import numpy as np

from .common import (
    ConstraintStatus,
    LineBaseQuantities,
    get_line_base_quantities,
    line_key,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LODFResult:
    """LODF computation output."""

    ptdf: np.ndarray  # (m,m)
    lodf: np.ndarray  # (m,m), with diagonal forced to -1
    islanded_contingencies: list[int]


@dataclass(frozen=True)
class _ProjectedSensitivityCache:
    """Cached row quantities for balanced L2 sensitivity norms."""

    matrix: np.ndarray
    raw_norm2: np.ndarray
    row_sum: np.ndarray
    projected_norm2: np.ndarray
    n_bus: int


def ptdf_for_line_transfers(H_full: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Compute PTDF matrix for line endpoint transfers.

    PTDF_{m,k} = h_m^T (e_from(k) - e_to(k))

    If E is the oriented incidence (m x n) with +1 at from bus and -1 at to bus,
    then:
        PTDF = H_full @ E^T   (m x n) @ (n x m) = (m x m)
    """
    H = np.asarray(H_full, dtype=float)
    Ei = np.asarray(E, dtype=float)
    if Ei.ndim != 2 or H.ndim != 2:
        raise ValueError("H_full and E must be 2D arrays.")
    if H.shape[1] != Ei.shape[1]:
        raise ValueError(
            f"Dimension mismatch: H_full is {H.shape}, E is {Ei.shape} (bus dimension must match)."
        )
    return H @ Ei.T


def lodf_from_ptdf(
    ptdf: np.ndarray,
    *,
    tol: float = 1e-10,
    islanding: Literal["skip", "raise"] = "skip",
) -> LODFResult:
    """
    Compute LODF from PTDF using:
        LODF_{m,k} = PTDF_{m,k} / (1 - PTDF_{k,k}),  for m != k
    and force:
        LODF_{k,k} = -1.

    Handling of (1 - PTDF_{k,k}) ~ 0 (islanding / radial cut):
    - islanding="skip": the entire contingency column k is set to NaN (except diagonal -1),
      and k is returned in `islanded_contingencies`.
    - islanding="raise": raises ValueError.

    Returns
    -------
    LODFResult
    """
    P = np.asarray(ptdf, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"ptdf must be square (m,m); got {P.shape}.")

    m = P.shape[0]
    denom = 1.0 - np.diag(P)
    lodf = np.empty_like(P, dtype=float)
    islanded: list[int] = []

    for k in range(m):
        if abs(float(denom[k])) <= tol:
            if islanding == "raise":
                raise ValueError(
                    f"Contingency k={k}: 1 - PTDF[k,k] is ~0 (={denom[k]}). "
                    "This suggests islanding / radial cut; LODF undefined."
                )
            islanded.append(k)
            lodf[:, k] = np.nan
            lodf[k, k] = -1.0
            continue

        lodf[:, k] = P[:, k] / denom[k]
        lodf[k, k] = -1.0

    if islanded:
        # This is important information for interpretation of results.
        logger.warning(
            "LODF: skipped %d islanded/undefined contingencies (islanding=%s). First: %s",
            len(islanded),
            islanding,
            islanded[:20],
        )

    return LODFResult(ptdf=P, lodf=lodf, islanded_contingencies=islanded)


def incidence_from_pandapower_net(
    net, *, line_indices: list[int] | None = None
) -> np.ndarray:
    """
    Build oriented incidence E (m x n) for pandapower net lines.

    For each line (from_bus -> to_bus): row has +1 at from_bus position and -1 at to_bus position.
    Out-of-service lines are represented as all-zero rows (consistent with DC matrix builder behavior).
    """
    bus_index = sorted(net.bus.index)
    bus_pos = {int(bid): pos for pos, bid in enumerate(bus_index)}

    idx = sorted(net.line.index) if line_indices is None else list(line_indices)
    m = len(idx)
    n = len(bus_index)
    E = np.zeros((m, n), dtype=float)

    for row_pos, (_, row) in enumerate(net.line.loc[idx].iterrows()):
        if "in_service" in row and not bool(row["in_service"]):
            continue

        fb = int(row["from_bus"])
        tb = int(row["to_bus"])
        if fb not in bus_pos or tb not in bus_pos:
            continue

        E[row_pos, bus_pos[fb]] = 1.0
        E[row_pos, bus_pos[tb]] = -1.0

    return E


def _validate_effective_inputs(
    *,
    base_flows: np.ndarray,
    limits: np.ndarray,
    G: np.ndarray,
    lodf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return validated arrays for effective N-1 radius computation."""
    f = np.asarray(base_flows, dtype=float).reshape(-1)
    c = np.asarray(limits, dtype=float).reshape(-1)
    Gm = np.asarray(G, dtype=float)
    L = np.asarray(lodf, dtype=float)

    m = f.size
    if c.shape != (m,):
        raise ValueError(f"limits must have shape ({m},); got {c.shape}.")
    if Gm.shape[0] != m:
        raise ValueError(f"G must have shape (m,n) with m={m}; got {Gm.shape}.")
    if L.shape != (m, m):
        raise ValueError(f"lodf must have shape ({m},{m}); got {L.shape}.")
    if Gm.shape[1] <= 0:
        raise ValueError("G must have a positive bus dimension.")
    return f, c, Gm, L


def _projected_sensitivity_cache(Gm: np.ndarray) -> _ProjectedSensitivityCache:
    """Precompute quantities for projected balanced L2 norms."""
    n_bus = int(Gm.shape[1])
    raw_norm2 = np.sum(Gm * Gm, axis=1)
    row_sum = np.sum(Gm, axis=1)
    projected_norm2 = raw_norm2 - (row_sum * row_sum) / float(n_bus)
    projected_norm2 = np.maximum(projected_norm2, 0.0)
    return _ProjectedSensitivityCache(
        matrix=Gm,
        raw_norm2=raw_norm2,
        row_sum=row_sum,
        projected_norm2=projected_norm2,
        n_bus=n_bus,
    )


def _post_contingency_denominator(
    cache: _ProjectedSensitivityCache,
    *,
    alpha: np.ndarray,
    contingency_pos: int,
    update_sensitivities: bool,
) -> np.ndarray:
    """Return projected sensitivity denominators after one contingency."""
    if not update_sensitivities:
        return np.sqrt(cache.projected_norm2)

    gk = cache.matrix[contingency_pos, :]
    dots = cache.matrix @ gk
    alpha2 = alpha * alpha
    norm2_post = (
        cache.raw_norm2 + 2.0 * alpha * dots + alpha2 * cache.raw_norm2[contingency_pos]
    )
    norm2_post = np.maximum(norm2_post, 0.0)

    sum_post = cache.row_sum + alpha * cache.row_sum[contingency_pos]
    proj_norm2_post = norm2_post - (sum_post * sum_post) / float(cache.n_bus)
    return np.sqrt(np.maximum(proj_norm2_post, 0.0))


def _contingency_radii(
    *,
    base_flows: np.ndarray,
    limits: np.ndarray,
    alpha: np.ndarray,
    denominator: np.ndarray,
    contingency_pos: int,
    eps: float,
) -> np.ndarray:
    """Compute line radii under one outage column."""
    f_post = base_flows + alpha * float(base_flows[contingency_pos])
    f_post[contingency_pos] = 0.0

    margin_post = limits - np.abs(f_post)
    radii = np.where(margin_post >= 0.0, float("inf"), float("-inf"))
    np.divide(margin_post, denominator, out=radii, where=denominator > eps)
    radii[contingency_pos] = float("inf")
    return radii


def effective_nminus1_l2_radii(
    *,
    base_flows: np.ndarray,
    limits: np.ndarray,
    G: np.ndarray,
    lodf: np.ndarray,
    update_sensitivities: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute effective N-1 L2 radii:
        r_m^(N-1) = min_{k != m}  margin_m^(k) / ||g_m^(k)||_2

    with fast LODF approximations:
        f^(k) = f + LODF[:,k] * f_k, and f_k^(k)=0
        g_m^(k) = g_m + LODF[m,k] * g_k

    Balanced-norm consistency
    ------------------------
    This project measures disturbances in the **balanced** subspace sum(Δp)=0 with the
    full Euclidean norm over all buses. Therefore the effective sensitivity norm used here is:

        ||Proj(g)||_2  where Proj(g) = g - mean(g)*1

    which can be computed as:
        ||Proj(g)||_2^2 = ||g||_2^2 - (sum(g))^2 / n_bus

    Parameters
    ----------
    base_flows:
        Signed base flows f (m,).
    limits:
        Symmetric thermal limits c (m,) in MW (MVA assumed MW under PF=1).
    G:
        Sensitivity matrix (m,n) mapping injection perturbations to line flow perturbations.
    lodf:
        LODF matrix (m,m), with diag expected to be -1. Columns may contain NaN for islanded contingencies.
    update_sensitivities:
        If True, use g_m^(k) update. If False, reuse g_m (faster, less accurate).
    eps:
        Threshold for "zero" sensitivity norm.

    Returns
    -------
    (best_radii, worst_contingency)
        best_radii: (m,) radii per monitored line.
        worst_contingency: (m,) integer contingency index that attains the min, or -1 if none.
    """
    f, c, Gm, L = _validate_effective_inputs(
        base_flows=base_flows,
        limits=limits,
        G=G,
        lodf=lodf,
    )

    m = f.size
    best = np.full(m, float("inf"), dtype=float)
    argmin = np.full(m, -1, dtype=int)
    cache = _projected_sensitivity_cache(Gm)

    for k in range(m):
        alpha = L[:, k]
        if np.isnan(alpha).any():
            logger.debug(
                "Skipping contingency k=%d due to NaN LODF column (islanding).", k
            )
            continue

        denom = _post_contingency_denominator(
            cache,
            alpha=alpha,
            contingency_pos=k,
            update_sensitivities=bool(update_sensitivities),
        )
        radii_k = _contingency_radii(
            base_flows=f,
            limits=c,
            alpha=alpha,
            denominator=denom,
            contingency_pos=k,
            eps=float(eps),
        )
        improved = radii_k < best
        best[improved] = radii_k[improved]
        argmin[improved] = k

    return best, argmin


def compute_nminus1_l2_radius(
    net,
    H_full: np.ndarray,
    *,
    limit_factor: float = 1.0,
    update_sensitivities: bool = True,
    islanding: Literal["skip", "raise"] = "skip",
    base: LineBaseQuantities | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute effective N-1 L2 radii on a pandapower network.

    Notes
    -----
    The returned `worst_contingency` is the contingency *position* (0..m-1) in the
    internal line ordering (base_q.line_indices). For convenience, we also return
    the mapped pandapower line index in `worst_contingency_line_idx`.
    """
    base_q = (
        base
        if base is not None
        else get_line_base_quantities(net, limit_factor=float(limit_factor))
    )
    if H_full.shape[0] != len(base_q.line_indices):
        raise ValueError(
            f"H_full row count ({H_full.shape[0]}) does not match net.line count ({len(base_q.line_indices)})."
        )

    logger.debug(
        "Computing N-1 effective L2 radii (update_sensitivities=%s, islanding=%s)...",
        update_sensitivities,
        islanding,
    )

    E = incidence_from_pandapower_net(net, line_indices=base_q.line_indices)
    ptdf = ptdf_for_line_transfers(H_full, E)
    lodf_res = lodf_from_ptdf(ptdf, islanding=islanding)

    best_r, argmin = effective_nminus1_l2_radii(
        base_flows=base_q.flow0_mw,
        limits=base_q.limit_mva_assumed_mw,
        G=H_full,
        lodf=lodf_res.lodf,
        update_sensitivities=update_sensitivities,
    )

    results: Dict[str, Dict[str, Any]] = {}
    for pos, lid in enumerate(base_q.line_indices):
        worst_pos = int(argmin[pos])
        worst_line_idx = (
            int(base_q.line_indices[worst_pos])
            if 0 <= worst_pos < len(base_q.line_indices)
            else -1
        )

        k = line_key(lid)
        r_n1 = float(best_r[pos])
        if np.isfinite(r_n1) and r_n1 < 0.0:
            status_n1 = ConstraintStatus.POST_CONTINGENCY_INFEASIBLE.value
            cert_r_n1 = 0.0
        elif np.isneginf(r_n1):
            status_n1 = ConstraintStatus.POST_CONTINGENCY_INFEASIBLE.value
            cert_r_n1 = 0.0
        elif np.isposinf(r_n1):
            status_n1 = ConstraintStatus.OK_INFINITE.value
            cert_r_n1 = float("inf")
        elif np.isfinite(r_n1):
            status_n1 = ConstraintStatus.OK_FINITE.value
            cert_r_n1 = max(float(r_n1), 0.0)
        else:
            status_n1 = ConstraintStatus.DEGENERATE_SENSITIVITY.value
            cert_r_n1 = float("nan")
        results[k] = {
            "flow0_mw": float(base_q.flow0_mw[pos]),
            "p0_mw": float(base_q.p0_abs_mw[pos]),
            "p_limit_mw_est": float(base_q.limit_mva_assumed_mw[pos]),
            "margin_mw": float(base_q.margin_mw[pos]),
            "radius_nminus1": float(r_n1),
            "certificate_radius_nminus1": float(cert_r_n1),
            "signed_distance_nminus1": float(r_n1),
            "constraint_status_nminus1": str(status_n1),
            "worst_contingency": worst_pos,  # position in base_q.line_indices
            "worst_contingency_line_idx": worst_line_idx,
        }

    return results
