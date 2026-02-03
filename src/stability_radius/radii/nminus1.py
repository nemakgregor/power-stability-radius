from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

import numpy as np

from .common import LineBaseQuantities, get_line_base_quantities, line_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LODFResult:
    """LODF computation output."""

    ptdf: np.ndarray  # (m,m)
    lodf: np.ndarray  # (m,m), with diagonal forced to -1
    islanded_contingencies: list[int]


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

    n_bus = int(Gm.shape[1])
    if n_bus <= 0:
        raise ValueError("G must have a positive bus dimension.")

    best = np.full(m, float("inf"), dtype=float)
    argmin = np.full(m, -1, dtype=int)

    # Precompute base norms and sums for projected (balanced) norms.
    g_norm2 = np.sum(Gm * Gm, axis=1)  # (m,)
    g_sum = np.sum(Gm, axis=1)  # (m,)
    g_proj_norm2 = g_norm2 - (g_sum * g_sum) / float(n_bus)
    g_proj_norm2 = np.maximum(g_proj_norm2, 0.0)

    for k in range(m):
        alpha = L[:, k]
        if np.isnan(alpha).any():
            logger.debug(
                "Skipping contingency k=%d due to NaN LODF column (islanding).", k
            )
            continue

        fk = float(f[k])
        f_post = f + alpha * fk
        f_post[k] = 0.0

        margin_post = np.maximum(c - np.abs(f_post), 0.0)

        if update_sensitivities:
            gk = Gm[k, :]  # (n,)
            dots = Gm @ gk  # (m,)

            # Raw ||g_m + alpha_m*gk||^2
            norm2_post = g_norm2 + 2.0 * alpha * dots + (alpha * alpha) * g_norm2[k]
            norm2_post = np.maximum(norm2_post, 0.0)

            # Projected norm: ||Proj(v)||^2 = ||v||^2 - sum(v)^2 / n
            sum_post = g_sum + alpha * g_sum[k]
            proj_norm2_post = norm2_post - (sum_post * sum_post) / float(n_bus)
            proj_norm2_post = np.maximum(proj_norm2_post, 0.0)

            denom = np.sqrt(proj_norm2_post)
        else:
            denom = np.sqrt(g_proj_norm2)

        radii_k = np.full(m, float("inf"), dtype=float)
        np.divide(margin_post, denom, out=radii_k, where=denom > eps)

        radii_k[k] = float("inf")  # skip the outaged line itself

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
    High-level wrapper: compute effective N-1 L2 radii on a pandapower network.

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
        results[k] = {
            "flow0_mw": float(base_q.flow0_mw[pos]),
            "p0_mw": float(base_q.p0_abs_mw[pos]),
            "p_limit_mw_est": float(base_q.limit_mva_assumed_mw[pos]),
            "margin_mw": float(base_q.margin_mw[pos]),
            "radius_nminus1": float(best_r[pos]),
            "worst_contingency": worst_pos,  # position in base_q.line_indices
            "worst_contingency_line_idx": worst_line_idx,
        }

    return results
