from __future__ import annotations

import numpy as np
import pytest

from stability_radius.radii.core_l2 import compute_l2_certificate_from_H


def _random_points_in_l2_ball(*, dim: int, r: float, n: int, seed: int) -> np.ndarray:
    """Deterministic uniform sampling in L2 ball in R^dim."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    if n <= 0:
        raise ValueError("n must be positive")

    rng = np.random.default_rng(int(seed))

    if r == 0.0:
        return np.zeros((n, dim), dtype=float)

    z = rng.standard_normal(size=(n, dim)).astype(float, copy=False)
    norms = np.linalg.norm(z, axis=1)

    bad = norms <= 1e-12
    while bool(np.any(bad)):
        k_bad = int(np.sum(bad))
        z_bad = rng.standard_normal(size=(k_bad, dim)).astype(float, copy=False)
        z[bad, :] = z_bad
        norms = np.linalg.norm(z, axis=1)
        bad = norms <= 1e-12

    dirs = z / norms[:, None]

    u = rng.random(size=n).astype(float, copy=False)
    rad = float(r) * np.power(u, 1.0 / float(dim))
    return dirs * rad[:, None]


def test_radius_soundness_synthetic_inside_outside() -> None:
    H = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float)
    f0 = np.array([0.0, 0.0], dtype=float)
    c = np.array([2.0, 2.0], dtype=float)

    cert = compute_l2_certificate_from_H(H=H, f0=f0, c=c)
    assert cert.r_star == pytest.approx(1.0)
    assert cert.argmin_pos == 0

    # inside
    for dp in _random_points_in_l2_ball(dim=2, r=cert.r_star, n=5000, seed=0):
        f = f0 + H @ dp
        assert np.all(np.abs(f) <= c + 1e-12)

    # outside along worst direction (line 1)
    g = H[0, :]
    u = g / np.linalg.norm(g)
    dp = (cert.r_star + 1e-3) * u
    f = f0 + H @ dp
    assert np.abs(f[0]) > c[0]


def test_radius_scales_with_limits() -> None:
    H = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float)
    f0 = np.array([0.0, 0.0], dtype=float)
    c = np.array([2.0, 2.0], dtype=float)

    cert1 = compute_l2_certificate_from_H(H=H, f0=f0, c=c)
    alpha = 3.0
    cert2 = compute_l2_certificate_from_H(H=H, f0=f0, c=alpha * c)

    assert cert2.r_star == pytest.approx(alpha * cert1.r_star)
