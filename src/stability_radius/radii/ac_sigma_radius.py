from __future__ import annotations

"""
AC sigma-radius (probabilistic certificate in "number of sigmas" units).

This module is intentionally *stateless* and does not build AC operators internally.
It consumes:
- precomputed adjoint sensitivity vectors h_l (one per line), and
- per-bus injection standard deviations (sigma_p, sigma_q),
- base flow magnitudes |S0| at the binding end,
- apparent-power thermal limits c (MVA).

Mathematical contract (as provided in the task)
-----------------------------------------------
Let h_l ∈ R^{2n} be the adjoint sensitivity of |S| (binding end for line l) to
injection perturbations [ΔP; ΔQ], with the partition:
    h_l = [h_l^P; h_l^Q]

Let per-bus injection std devs be:
    sigma_p_mw[i], sigma_q_mvar[i]

Define diagonal covariance:
    Σ = diag( sigma_p^2 , sigma_q^2 )

Then:
    sigma_flow_l = || Σ^{1/2} h_l ||_2
    r_sigma_l = (c_l - |S0_l|) / sigma_flow_l

Worst-case perturbation (physical units, MW/MVAr):
    Δu_l* = r_sigma_l * (Σ h_l) / sigma_flow_l

Gaussian overload probability (one-sided apparent-power limit):
    P(|S0| + X > c) = Q((c - |S0|)/sigma)

Balanced disturbances (optional)
--------------------------------
If balance=True, we enforce the physical power balance constraint:
    1^T ΔP = 0 and 1^T ΔQ = 0

Because the worst-case perturbation is dp_i = r * σ_i² * hP_i / σ_flow,
the constraint sum(dp_i) = 0 requires sum(σ_i² * hP_i) = 0.  We achieve
this via a σ²-weighted mean subtraction:
    hP <- hP - sum(σ_P² · hP) / sum(σ_P²)
    hQ <- hQ - sum(σ_Q² · hQ) / sum(σ_Q²)

This differs from the L2 certificate (which uses unweighted mean-subtraction)
because here the perturbation ellipsoid is anisotropic (Σ-weighted).
"""

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from stability_radius.geometry.balanced import (
    make_ac_block_specs,
    project_dual_balanced_rows,
)
from stability_radius.radii.common import (
    ConstraintStatus,
    classify_constraint_certificate,
    line_key,
    signed_radius_from_margin_norm,
)

logger = logging.getLogger(__name__)

_EPS_SIGMA_FLOW = 1e-15


def _qfunc(x: float) -> float:
    """
    Gaussian Q-function: Q(x) = P(Z > x), Z~N(0,1).

    Implemented via erfc for numerical stability:
        Q(x) = 0.5 * erfc(x / sqrt(2)).
    """
    return 0.5 * math.erfc(float(x) / math.sqrt(2.0))


def overload_probability_one_sided_limit(
    *, y0: float, limit: float, sigma: float
) -> float:
    """
    Overload probability for a one-sided limit y0 + X <= limit.

    Model: Y = y0 + X, X ~ N(0, sigma^2).
    Then: P(Y > limit) = Q((limit - y0) / sigma).

    If sigma <= 0, the random variable is degenerate.
    """
    y0_f = float(y0)
    c = float(limit)
    s = float(sigma)

    if not math.isfinite(c) or c < 0.0:
        raise ValueError(f"limit must be finite and >=0, got {limit!r}")

    if not math.isfinite(y0_f):
        raise ValueError(f"y0 must be finite, got {y0!r}")

    if s <= 0.0:
        return 1.0 if y0_f > c else 0.0

    return _qfunc((c - y0_f) / s)


def overload_probability_two_sided_signed(
    *, flow0: float, limit: float, sigma: float
) -> float:
    """
    Overload probability for a signed flow with symmetric limits |F| <= limit.

    This is appropriate for signed flow models, not for AC apparent-power
    magnitude constraints.
    """
    f0 = float(flow0)
    c = float(limit)
    s = float(sigma)

    if not math.isfinite(c) or c < 0.0:
        raise ValueError(f"limit must be finite and >=0, got {limit!r}")
    if not math.isfinite(f0):
        raise ValueError(f"flow0 must be finite, got {flow0!r}")

    if s <= 0.0:
        return 1.0 if abs(f0) > c else 0.0

    f_abs = abs(f0)
    return _qfunc((c - f_abs) / s) + _qfunc((c + f_abs) / s)


def _validate_inputs(
    *,
    h_vectors: np.ndarray,
    s_limit_mva: np.ndarray,
    s0_mva: np.ndarray,
    sigma_p_mw: np.ndarray,
    sigma_q_mvar: np.ndarray,
    line_ids: Sequence[int] | None,
    pq_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Internal helper for module-local processing."""
    H = np.asarray(h_vectors, dtype=float)
    if H.ndim != 2:
        raise ValueError(f"h_vectors must be 2D, got shape={H.shape}")
    n_lines, d = int(H.shape[0]), int(H.shape[1])
    if np.any(~np.isfinite(H)):
        raise ValueError("h_vectors must be finite.")

    sig_p = np.asarray(sigma_p_mw, dtype=float).reshape(-1)
    sig_q = np.asarray(sigma_q_mvar, dtype=float).reshape(-1)
    n_bus = int(sig_p.size)

    if sig_q.shape != (n_bus,):
        raise ValueError(f"sigma_q_mvar must have shape ({n_bus},), got {sig_q.shape}")
    if d != 2 * n_bus:
        raise ValueError(
            f"h_vectors second dim must be 2*n_bus={2 * n_bus}, got {d} "
            f"(n_bus inferred from sigma vectors)."
        )

    s_lim = np.asarray(s_limit_mva, dtype=float).reshape(-1)
    s0 = np.asarray(s0_mva, dtype=float).reshape(-1)
    if s_lim.shape != (n_lines,):
        raise ValueError(f"s_limit_mva must have shape ({n_lines},), got {s_lim.shape}")
    if s0.shape != (n_lines,):
        raise ValueError(f"s0_mva must have shape ({n_lines},), got {s0.shape}")

    if np.any(~np.isfinite(sig_p)) or np.any(sig_p < 0.0):
        raise ValueError("sigma_p_mw must be finite and non-negative per bus.")
    if np.any(~np.isfinite(sig_q)) or np.any(sig_q < 0.0):
        raise ValueError("sigma_q_mvar must be finite and non-negative per bus.")
    if np.any(~np.isfinite(s_lim)) or np.any(s_lim < 0.0):
        raise ValueError("s_limit_mva must be finite and non-negative per line.")
    if np.any(~np.isfinite(s0)) or np.any(s0 < 0.0):
        raise ValueError("s0_mva must be finite and non-negative per line.")

    if line_ids is None:
        ids = tuple(range(n_lines))
    else:
        ids = tuple(int(x) for x in line_ids)
        if len(ids) != n_lines:
            raise ValueError(
                f"line_ids length must match n_lines={n_lines}, got {len(ids)}"
            )

    if pq_mask is not None:
        pq = np.asarray(pq_mask, dtype=bool).reshape(-1)
        if pq.shape != (n_bus,):
            raise ValueError(f"pq_mask must have shape ({n_bus},), got {pq.shape}")

    return H, s_lim, s0, sig_p, sig_q, ids


def compute_ac_sigma_radius(
    *,
    h_vectors: np.ndarray,
    s_limit_mva: np.ndarray,
    s0_mva: np.ndarray,
    sigma_p_mw: np.ndarray,
    sigma_q_mvar: np.ndarray,
    line_ids: Sequence[int] | None = None,
    balance: bool = True,
    pq_mask: np.ndarray | None = None,
    eps_sigma_flow: float = _EPS_SIGMA_FLOW,
) -> dict[str, dict[str, Any]]:
    """
    Compute AC sigma-radius per line using precomputed adjoint h-vectors.

    Parameters
    ----------
    h_vectors:
        Array of shape (n_lines, 2*n_bus). Each row is h_l = [hP; hQ] for the *binding end*
        of that line, consistent with the AC L2 certificate's h-vector definition.
    s_limit_mva:
        Symmetric thermal limits c_l in MVA, shape (n_lines,).
    s0_mva:
        Base apparent power magnitudes |S0_l| at the binding end, in MVA, shape (n_lines,).
    sigma_p_mw, sigma_q_mvar:
        Per-bus std deviations for ΔP (MW) and ΔQ (MVAr), both shape (n_bus,).
    line_ids:
        Optional line indices used to form stable keys `line_<id>`. If None, uses 0..n_lines-1.
    balance:
        If True, projects each h-vector P/Q block onto sum-zero subspace by
        sigma^2-weighted mean-subtraction.
    pq_mask:
        Optional full-bus boolean mask for PQ buses.  When provided, the Q block
        is restricted to these coordinates; PV and slack reactive injections are
        not treated as independent perturbation coordinates.
    eps_sigma_flow:
        Numerical threshold for treating sigma_flow as zero (degenerate sensitivity).

    Returns
    -------
    dict
        Mapping "line_<id>" -> dict with keys (per task contract):
          - sigma_flow_mva (float)
          - certificate_radius_ac_sigma (float)
          - signed_distance_ac_sigma (float)
          - constraint_status_ac_sigma (str)
          - radius_ac_sigma (float)          # dimensionless (in σ units)
          - overload_probability_ac (float)
          - worst_case_dp_mw (np.ndarray, (n_bus,))
          - worst_case_dq_mvar (np.ndarray, (n_bus,))
          - worst_case_s_predicted_mva (float)
    """
    H, c, s0, sig_p, sig_q, ids = _validate_inputs(
        h_vectors=h_vectors,
        s_limit_mva=s_limit_mva,
        s0_mva=s0_mva,
        sigma_p_mw=sigma_p_mw,
        sigma_q_mvar=sigma_q_mvar,
        line_ids=line_ids,
        pq_mask=pq_mask,
    )

    eps = float(eps_sigma_flow)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps_sigma_flow must be finite and >0.")

    n_lines = int(H.shape[0])
    n_bus = int(sig_p.size)
    q_bus_indices: np.ndarray | None = None
    if pq_mask is not None:
        pq = np.asarray(pq_mask, dtype=bool).reshape(-1)
        q_bus_indices = np.where(pq)[0]
        inactive_q = np.where(~pq)[0]
        if inactive_q.size:
            H = H.copy()
            H[:, n_bus + inactive_q] = 0.0

    # Split P/Q blocks.
    hP = np.asarray(H[:, 0:n_bus], dtype=float, order="C")
    hQ = np.asarray(H[:, n_bus : 2 * n_bus], dtype=float, order="C")

    if bool(balance):
        # Physical balance constraint: 1^T ΔP = 0 and 1^T ΔQ = 0.
        #
        # Worst-case perturbation: dp_i = σ_i² · hP_adj_i · r / σ_flow.
        # For sum(dp) = 0 we need sum(σ_i² · hP_adj_i) = 0, so we subtract
        # the σ²-weighted mean:
        #   hP_adj = hP - sum(σ_P² · hP) / sum(σ_P²)
        #
        # This is the Lagrangian solution for max h^T Δu s.t. ||Σ^{-1/2} Δu|| ≤ r
        # and 1^T ΔP = 0, 1^T ΔQ = 0.
        H_proj = project_dual_balanced_rows(
            H,
            make_ac_block_specs(
                n_bus,
                balance=True,
                p_weights=sig_p * sig_p,
                q_weights=sig_q * sig_q,
                q_bus_indices=q_bus_indices,
            ),
        )
        hP = H_proj[:, 0:n_bus]
        hQ = H_proj[:, n_bus : 2 * n_bus]
        logger.debug("Applied sigma²-weighted balanced projection to h-vectors.")

    # sigma_flow_l = || [sigma_p*hP, sigma_q*hQ] ||_2
    scaledP = hP * sig_p[None, :]
    scaledQ = hQ * sig_q[None, :]
    sigma_flow = np.sqrt(
        np.sum(scaledP * scaledP, axis=1) + np.sum(scaledQ * scaledQ, axis=1)
    )

    margin = c - s0
    valid = np.isfinite(sigma_flow) & (sigma_flow > eps)

    radius = np.empty(n_lines, dtype=float)
    for i in range(n_lines):
        radius[i] = signed_radius_from_margin_norm(
            margin=float(margin[i]), dual_norm=float(sigma_flow[i]), eps=eps
        )

    status: list[str] = []
    certificate_radius = np.empty(n_lines, dtype=float)
    signed_distance = np.empty(n_lines, dtype=float)
    for i in range(n_lines):
        st, cert_r, signed_d = classify_constraint_certificate(
            margin=float(margin[i]),
            dual_norm=float(sigma_flow[i]),
            eps=eps,
        )
        status.append(st)
        certificate_radius[i] = float(cert_r)
        signed_distance[i] = float(signed_d)

    ok_finite = np.asarray(
        [st == ConstraintStatus.OK_FINITE.value for st in status], dtype=bool
    )

    # Worst-case perturbations (MW/MVAr). For non-ok rows, zeros avoid exporting
    # a boundary direction that is not a valid robustness certificate.
    dp = np.zeros((n_lines, n_bus), dtype=float)
    dq = np.zeros((n_lines, n_bus), dtype=float)

    if bool(np.any(ok_finite)):
        # Vectorized formula:
        #   dp = r * (sigma_p^2 * hP) / sigma_flow
        #   dq = r * (sigma_q^2 * hQ) / sigma_flow
        sigp2 = (sig_p * sig_p)[None, :]
        sigq2 = (sig_q * sig_q)[None, :]
        denom = sigma_flow[ok_finite, None]
        dp[ok_finite, :] = (
            radius[ok_finite, None] * (sigp2 * hP[ok_finite, :])
        ) / denom
        dq[ok_finite, :] = (
            radius[ok_finite, None] * (sigq2 * hQ[ok_finite, :])
        ) / denom

    # Predicted worst-case |S| (linearized): s0 + h·Δu*
    # For ok finite lines, this equals c (up to numerical noise).
    s_pred = np.asarray(s0, dtype=float).copy()
    if bool(np.any(ok_finite)):
        # Use explicit dot product to reflect the linear model definition.
        s_pred[ok_finite] = (
            s0[ok_finite]
            + np.sum(hP[ok_finite, :] * dp[ok_finite, :], axis=1)
            + np.sum(hQ[ok_finite, :] * dq[ok_finite, :], axis=1)
        )

    # Overload probabilities
    prob = np.empty(n_lines, dtype=float)
    for i in range(n_lines):
        prob[i] = overload_probability_one_sided_limit(
            y0=float(s0[i]),
            limit=float(c[i]),
            sigma=float(sigma_flow[i]),
        )

    results: dict[str, dict[str, Any]] = {}
    for pos, lid in enumerate(ids):
        k = line_key(int(lid))
        results[k] = {
            "sigma_flow_mva": float(sigma_flow[pos]),
            "radius_ac_sigma": float(radius[pos]),
            "certificate_radius_ac_sigma": float(certificate_radius[pos]),
            "signed_distance_ac_sigma": float(signed_distance[pos]),
            "constraint_status_ac_sigma": str(status[pos]),
            "overload_probability_ac": float(prob[pos]),
            "worst_case_dp_mw": np.asarray(dp[pos, :], dtype=float),
            "worst_case_dq_mvar": np.asarray(dq[pos, :], dtype=float),
            "worst_case_s_predicted_mva": float(s_pred[pos]),
        }

    # Logging summary (keep INFO concise).
    finite_r = radius[np.isfinite(radius)]
    if finite_r.size > 0:
        logger.info(
            "AC sigma-radius computed: lines=%d finite=%d mean=%.6g min=%.6g max=%.6g (balance=%s)",
            int(n_lines),
            int(finite_r.size),
            float(np.mean(finite_r)),
            float(np.min(finite_r)),
            float(np.max(finite_r)),
            bool(balance),
        )
    else:
        logger.info(
            "AC sigma-radius computed: lines=%d finite=0 (balance=%s)",
            int(n_lines),
            bool(balance),
        )

    # Extra debug: report degeneracy count.
    n_degenerate = int(np.sum(~valid))
    if n_degenerate > 0:
        logger.debug(
            "AC sigma-radius: degenerate sigma_flow<=eps count=%d/%d (eps=%.3g).",
            n_degenerate,
            int(n_lines),
            float(eps),
        )

    return results
