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

Notes
-----
- If some m_i < 0, the base point violates limits; the "radius around a feasible point"
  interpretation breaks. We intentionally do NOT clip margins here: the caller must
  handle base feasibility explicitly.
- If ||g_i||_2 == 0: that line is insensitive to Δp in this model; the corresponding
  r_i is +inf if m_i > 0, and 0 if m_i == 0.
"""

import math
from dataclasses import dataclass

import numpy as np


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

    # For zero-norm rows: r=+inf if margin>0, r=0 if margin==0, r=-inf if margin<0
    zero = ~nonzero
    if bool(np.any(zero)):
        radii[zero] = np.where(
            margins[zero] > 0.0,
            float("inf"),
            np.where(margins[zero] < 0.0, float("-inf"), 0.0),
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
