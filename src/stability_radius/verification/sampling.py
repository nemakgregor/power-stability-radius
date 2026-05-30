"""Sampling utilities shared by verification and experiment fronts."""

from __future__ import annotations

import numpy as np


def condition_diagonal_gaussian_balance_inplace(
    dp: np.ndarray,
    dq: np.ndarray,
    sigma_p: np.ndarray,
    sigma_q: np.ndarray,
) -> None:
    """Condition diagonal Gaussian P/Q samples on zero active and reactive sums.

    The input arrays are modified in place.  For each block the projection uses
    the covariance-weighted conditional Gaussian formula, so heterogeneous
    standard deviations produce the same balanced distribution used by the AC
    sigma-radius denominator.
    """
    if dp.ndim != 2 or dq.ndim != 2:
        raise ValueError("dp and dq must be 2D")
    if dp.shape != dq.shape:
        raise ValueError("dp and dq shape mismatch")

    sigp2 = np.asarray(sigma_p, dtype=float).reshape(-1) ** 2
    sigq2 = np.asarray(sigma_q, dtype=float).reshape(-1) ** 2
    if sigp2.shape != (dp.shape[1],) or sigq2.shape != (dq.shape[1],):
        raise ValueError("sigma arrays must match dp/dq bus dimension")

    sum_sigp2 = float(np.sum(sigp2))
    if sum_sigp2 > 0.0:
        dp_sum = np.sum(dp, axis=1, keepdims=True)
        dp -= sigp2[None, :] * dp_sum / sum_sigp2

    sum_sigq2 = float(np.sum(sigq2))
    if sum_sigq2 > 0.0:
        dq_sum = np.sum(dq, axis=1, keepdims=True)
        dq -= sigq2[None, :] * dq_sum / sum_sigq2


def sample_balanced_gaussian_sigma(
    *,
    rng: np.random.Generator,
    n: int,
    sigma_p: np.ndarray,
    sigma_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw balanced Gaussian samples with per-bus P/Q standard deviations."""
    sig_p = np.asarray(sigma_p, dtype=float).reshape(-1)
    sig_q = np.asarray(sigma_q, dtype=float).reshape(-1)
    if sig_p.shape != sig_q.shape:
        raise ValueError("sigma_p and sigma_q must have the same shape")
    if np.any(~np.isfinite(sig_p)) or np.any(sig_p <= 0.0):
        raise ValueError("sigma_p must be finite and positive")
    if np.any(~np.isfinite(sig_q)) or np.any(sig_q <= 0.0):
        raise ValueError("sigma_q must be finite and positive")

    n_bus = int(sig_p.shape[0])
    z_p = rng.standard_normal(size=(int(n), n_bus))
    z_q = rng.standard_normal(size=(int(n), n_bus))
    dp = (sig_p[None, :] * z_p).astype(float, copy=False)
    dq = (sig_q[None, :] * z_q).astype(float, copy=False)
    condition_diagonal_gaussian_balance_inplace(dp, dq, sig_p, sig_q)
    return dp, dq


def sigma_inverse_norm(
    dp: np.ndarray,
    dq: np.ndarray,
    inv_sigma_p: np.ndarray,
    inv_sigma_q: np.ndarray,
) -> np.ndarray:
    """Compute row-wise Euclidean norms after diagonal sigma inverse scaling."""
    if dp.ndim != 2 or dq.ndim != 2:
        raise ValueError("dp and dq must be 2D")
    if dp.shape != dq.shape:
        raise ValueError("dp and dq shape mismatch")
    inv_p = np.asarray(inv_sigma_p, dtype=float).reshape(-1)
    inv_q = np.asarray(inv_sigma_q, dtype=float).reshape(-1)
    if inv_p.shape != (dp.shape[1],) or inv_q.shape != (dq.shape[1],):
        raise ValueError("inverse sigma arrays must match dp/dq bus dimension")

    scaled_p = dp * inv_p[None, :]
    scaled_q = dq * inv_q[None, :]
    return np.sqrt(
        np.sum(scaled_p * scaled_p, axis=1) + np.sum(scaled_q * scaled_q, axis=1)
    )
