from __future__ import annotations

"""
Pure L2-certificate helpers (no pandapower / no DCOperator dependencies).

This module exists to make the "radius concept" testable in isolation:
- deterministic unit tests
- clear mathematical contract

Contract (per-line L2 certificate)
----------------------------------
Given:
  - linear flow map: Δf = H Δp
  - base flows: f0
  - symmetric line limits: |f| <= c

Define per-line margins:
  m_i = c_i - |f0_i|

Define per-line row vectors:
  g_i = H[i, :]

Then the per-line L2 radius:
  r_i = m_i / ||g_i||_2

Global certificate:
  r* = min_i r_i

If the base point is feasible (m_i >= 0), then for any Δp with ||Δp||_2 <= r*,
all line constraints are satisfied in this linear model (by Cauchy–Schwarz).

Balanced (sum-zero) disturbance norm helper
-------------------------------------------
In power-grid robustness, injections are often constrained to the balanced subspace:
  1^T Δp = 0.

For a row sensitivity vector g (defined up to adding a constant 1-vector), the correct
dual norm on that subspace (with the standard Euclidean norm in R^n) is:

  ||Proj(g)||_2, where Proj(g) = g - mean(g) * 1

Equivalently:
  ||Proj(g)||_2^2 = ||g||_2^2 - (sum(g))^2 / n

We expose helpers for this projection norm to support:
- slack-invariant certificates (angle slack is only a reference)
- deterministic unit tests of slack invariance
"""

import math
from dataclasses import dataclass

import numpy as np


def l2_norm_projected_ones_complement(v: np.ndarray) -> float:
    """
    Compute ||v - mean(v) * 1||_2.

    This is the L2 norm of the projection of v onto the hyperplane:
        {x : sum(x) = 0}.

    Notes
    -----
    The closed-form is:
        ||Proj(v)||_2^2 = ||v||_2^2 - (sum(v))^2 / n

    Returns
    -------
    float
        Non-negative projected norm.
    """
    vv = np.asarray(v, dtype=float).reshape(-1)
    n = int(vv.size)
    if n <= 0:
        return 0.0

    s = float(np.sum(vv))
    t = float(np.dot(vv, vv))
    val = t - (s * s) / float(n)
    return math.sqrt(max(val, 0.0))


def row_l2_norms_projected_ones_complement(H: np.ndarray) -> np.ndarray:
    """
    Compute per-row projected norms ||row - mean(row)*1||_2 for a matrix H.

    Parameters
    ----------
    H:
        Matrix (m, n).

    Returns
    -------
    np.ndarray
        Vector (m,) of non-negative projected norms.
    """
    Hm = np.asarray(H, dtype=float)
    if Hm.ndim != 2:
        raise ValueError(f"H must be 2D, got shape={Hm.shape}")

    m, n = int(Hm.shape[0]), int(Hm.shape[1])
    if n <= 0:
        return np.zeros(m, dtype=float)

    s = np.sum(Hm, axis=1)
    t = np.sum(Hm * Hm, axis=1)
    val = t - (s * s) / float(n)
    return np.sqrt(np.maximum(val, 0.0))


@dataclass(frozen=True)
class L2RadiusCertificate:
    """Computed certificate quantities for debugging and reporting."""

    margins: np.ndarray  # (m,)
    row_norms: np.ndarray  # (m,)
    radii: np.ndarray  # (m,)
    r_star: float
    argmin_pos: int  # 0..m-1, or -1 if undefined


def compute_l2_certificate_from_H(
    *,
    H: np.ndarray,
    f0: np.ndarray,
    c: np.ndarray,
    eps_norm: float = 1e-12,
) -> L2RadiusCertificate:
    """
    Compute per-line radii and the global L2 certificate r*.

    Parameters
    ----------
    H:
        Sensitivity matrix (m_lines, d_dim).
    f0:
        Base line flows (m_lines,).
    c:
        Symmetric line limits (m_lines,).
    eps_norm:
        Threshold for treating ||g|| as zero.

    Returns
    -------
    L2RadiusCertificate
        Includes per-line margins, norms, radii, and (r_star, argmin_pos).
    """
    Hm = np.asarray(H, dtype=float)
    if Hm.ndim != 2:
        raise ValueError(f"H must be 2D, got shape={Hm.shape}")

    m = int(Hm.shape[0])

    f0v = np.asarray(f0, dtype=float).reshape(-1)
    cv = np.asarray(c, dtype=float).reshape(-1)

    if f0v.shape != (m,):
        raise ValueError(f"f0 must have shape ({m},), got {f0v.shape}")
    if cv.shape != (m,):
        raise ValueError(f"c must have shape ({m},), got {cv.shape}")

    if not math.isfinite(float(eps_norm)) or float(eps_norm) <= 0.0:
        raise ValueError("eps_norm must be finite and positive.")

    margins = cv - np.abs(
        f0v
    )  # do NOT clip here (base feasibility is a separate check)
    row_norms = np.linalg.norm(Hm, ord=2, axis=1)

    radii = np.full(m, float("inf"), dtype=float)

    nonzero = row_norms > float(eps_norm)
    radii[nonzero] = margins[nonzero] / row_norms[nonzero]

    # For zero-norm rows: treat as non-restrictive (radius=+inf) if margin>=0.
    zero = ~nonzero
    if bool(np.any(zero)):
        radii[zero] = np.where(
            margins[zero] >= 0.0,
            float("inf"),
            float("-inf"),
        )

    finite = np.isfinite(radii)
    if not bool(np.any(finite)):
        return L2RadiusCertificate(
            margins=margins,
            row_norms=row_norms,
            radii=radii,
            r_star=float("nan"),
            argmin_pos=-1,
        )

    argmin_pos = int(np.argmin(np.where(finite, radii, float("inf"))))
    r_star = float(radii[argmin_pos])

    return L2RadiusCertificate(
        margins=margins,
        row_norms=row_norms,
        radii=radii,
        r_star=r_star,
        argmin_pos=argmin_pos,
    )
