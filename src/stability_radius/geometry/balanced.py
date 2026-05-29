from __future__ import annotations

"""Balanced-subspace projection and dual-norm helpers."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockSpec:
    """
    A coordinate block whose perturbations may be constrained to sum to zero.

    ``weights`` are dual-space weights for anisotropic covariance geometry.  For
    AC sigma-radius this is ``sigma^2`` and gives the weighted mean subtraction
    required by the Lagrangian first-order condition.
    """

    name: str
    indices: np.ndarray
    balance: bool = True
    weights: np.ndarray | None = None


def _as_indices(indices: np.ndarray, *, name: str) -> np.ndarray:
    """Internal helper for module-local processing."""
    idx = np.asarray(indices, dtype=int).reshape(-1)
    if np.any(idx < 0):
        raise ValueError(f"{name}.indices must be non-negative.")
    return idx


def _as_weights(weights: np.ndarray | None, n: int, *, name: str) -> np.ndarray | None:
    """Internal helper for module-local processing."""
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape != (n,):
        raise ValueError(f"{name}.weights must have shape ({n},), got {w.shape}.")
    if np.any(~np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError(f"{name}.weights must be finite and non-negative.")
    return w


def project_dual_balanced(h: np.ndarray, blocks: Sequence[BlockSpec]) -> np.ndarray:
    """
    Project a dual vector onto balanced blocks by mean subtraction.

    For weighted blocks, subtract the weighted mean
    ``sum(weights * h) / sum(weights)``. Blocks with zero total weight are left
    unchanged, matching the existing sigma-radius degenerate behavior.
    """
    out = np.asarray(h, dtype=float).reshape(-1).copy()

    for block in blocks:
        idx = _as_indices(block.indices, name=block.name)
        if idx.size == 0 or not bool(block.balance):
            continue
        if int(np.max(idx)) >= out.size:
            raise ValueError(
                f"{block.name}.indices exceed vector dimension {out.size}."
            )

        w = _as_weights(block.weights, int(idx.size), name=block.name)
        x = out[idx]
        if w is None:
            mu = float(np.mean(x))
        else:
            w_sum = float(np.sum(w))
            if w_sum <= 0.0:
                continue
            mu = float(np.sum(w * x) / w_sum)
        out[idx] = x - mu

    return out


def project_dual_balanced_rows(
    H: np.ndarray, blocks: Sequence[BlockSpec]
) -> np.ndarray:
    """Row-wise version of :func:`project_dual_balanced` for dense matrices."""
    out = np.asarray(H, dtype=float).copy()
    if out.ndim != 2:
        raise ValueError(f"H must be 2D, got shape={out.shape}.")

    for block in blocks:
        idx = _as_indices(block.indices, name=block.name)
        if idx.size == 0 or not bool(block.balance):
            continue
        if int(np.max(idx)) >= out.shape[1]:
            raise ValueError(
                f"{block.name}.indices exceed matrix width {out.shape[1]}."
            )

        w = _as_weights(block.weights, int(idx.size), name=block.name)
        if w is None:
            mu = np.mean(out[:, idx], axis=1, keepdims=True)
        else:
            w_sum = float(np.sum(w))
            if w_sum <= 0.0:
                continue
            mu = np.sum(out[:, idx] * w[None, :], axis=1, keepdims=True) / w_sum
        out[:, idx] = out[:, idx] - mu

    return out


def dual_norm_l2_balanced(h: np.ndarray, blocks: Sequence[BlockSpec]) -> float:
    """L2 dual norm after applying the configured balanced projection."""
    hp = project_dual_balanced(h, blocks)
    return float(np.linalg.norm(hp, ord=2))


def dual_norm_l2_balanced_rows(
    H: np.ndarray, blocks: Sequence[BlockSpec]
) -> np.ndarray:
    """Row-wise L2 dual norms after balanced projection."""
    Hp = project_dual_balanced_rows(H, blocks)
    return np.sqrt(np.sum(Hp * Hp, axis=1))


def worst_case_l2_direction(h: np.ndarray, blocks: Sequence[BlockSpec]) -> np.ndarray:
    """Unit L2 worst-case direction in the balanced subspace."""
    hp = project_dual_balanced(h, blocks)
    n = float(np.linalg.norm(hp, ord=2))
    if n <= 0.0:
        return np.zeros_like(hp)
    return hp / n


def _balanced_block_projected_norm2(
    values: np.ndarray, *, total_size: int | None
) -> float:
    """Internal helper for module-local processing."""
    v = np.asarray(values, dtype=float).reshape(-1)
    n = int(v.size if total_size is None else total_size)
    t = float(np.dot(v, v))
    if n <= 0:
        return t
    s = float(np.sum(v))
    return max(t - (s * s) / float(n), 0.0)


def dual_norm_l2_balanced_from_block_vectors(
    block_vectors: Sequence[np.ndarray],
    *,
    total_sizes: Sequence[int | None] | None = None,
    balance: bool = True,
) -> float:
    """
    L2 balanced dual norm from separate block vectors.

    ``total_sizes`` supports reduced coordinates with implicit zero entries,
    e.g. the AC P block has ``n_bus - 1`` explicit non-slack entries but is
    projected in the full ``n_bus`` active-power balanced subspace.
    """
    if total_sizes is None:
        sizes: tuple[int | None, ...] = tuple(None for _ in block_vectors)
    else:
        sizes = tuple(total_sizes)
        if len(sizes) != len(block_vectors):
            raise ValueError("total_sizes length must match block_vectors.")

    total = 0.0
    for values, n_total in zip(block_vectors, sizes):
        v = np.asarray(values, dtype=float).reshape(-1)
        if bool(balance):
            total += _balanced_block_projected_norm2(v, total_size=n_total)
        else:
            total += float(np.dot(v, v))

    return math.sqrt(max(total, 0.0))


def make_ac_block_specs(
    n_bus: int,
    *,
    balance: bool = True,
    p_weights: np.ndarray | None = None,
    q_weights: np.ndarray | None = None,
    q_bus_indices: np.ndarray | None = None,
) -> tuple[BlockSpec, BlockSpec]:
    """Build full-dimensional AC ``[P; Q]`` block specs."""
    n = int(n_bus)
    if n < 0:
        raise ValueError("n_bus must be non-negative.")

    if q_bus_indices is None:
        q_indices = np.arange(n, 2 * n, dtype=int)
        q_weights_eff = q_weights
    else:
        q_bus = np.asarray(q_bus_indices, dtype=int).reshape(-1)
        if np.any(q_bus < 0) or np.any(q_bus >= n):
            raise ValueError("q_bus_indices must be valid bus positions.")
        q_indices = n + q_bus
        q_weights_eff = (
            None if q_weights is None else np.asarray(q_weights, dtype=float)[q_bus]
        )

    return (
        BlockSpec(
            name="P",
            indices=np.arange(0, n, dtype=int),
            balance=bool(balance),
            weights=p_weights,
        ),
        BlockSpec(
            name="Q",
            indices=q_indices,
            balance=bool(balance),
            weights=q_weights_eff,
        ),
    )
