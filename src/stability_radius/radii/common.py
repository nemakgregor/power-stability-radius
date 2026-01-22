from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class LineBaseQuantities:
    """
    Container for per-line base quantities used by radius calculations.

    Notes
    -----
    - `flow0_mw` is signed (pandapower's `p_from_mw`).
    - `p0_abs_mw` is absolute value of `flow0_mw` (used for margins vs symmetric limits).
    - `limit_mw_est` is an estimated thermal limit (treated as MW proxy).
    - `margin_mw` is `max(limit - abs(flow0), 0)`.
    """

    line_indices: list[int]
    flow0_mw: np.ndarray  # shape (m,)
    p0_abs_mw: np.ndarray  # shape (m,)
    limit_mw_est: np.ndarray  # shape (m,)
    margin_mw: np.ndarray  # shape (m,)


def estimate_line_limit_mva(net, line_row, line_res_row) -> float:
    """
    Estimate a line thermal limit in MVA using available pandapower data.

    Priority (robust, version-tolerant)
    -----------------------------------
    0) If a MATPOWER-like MVA rating column exists on the line table:
       - rateA / rate_a_mva / sn_mva / max_mva
       then treat it as the limit in MVA (apply max_loading_percent if present).

    1) If `max_i_ka` and from-bus `vn_kv` are available:
         Smax ≈ sqrt(3) * V_kV * I_kA
       then apply `max_loading_percent` if present.

    2) Fallback: infer from `loading_percent` if available:
         Smax ≈ S0 * 100 / loading_percent
       then apply `max_loading_percent`.

    Returns
    -------
    float
        Estimated limit in MVA, or +inf if it cannot be estimated.
    """
    max_loading_percent = float(line_row.get("max_loading_percent", 100.0))
    if not np.isfinite(max_loading_percent) or max_loading_percent <= 0:
        max_loading_percent = 100.0

    # 0) Direct MVA rating if present (PGLib/MATPOWER converters often keep something like this).
    for k in ("rateA", "rate_a_mva", "sn_mva", "max_mva"):
        if k in line_row:
            try:
                v = float(line_row[k])
            except Exception:
                continue
            if np.isfinite(v) and v > 0:
                return v * (max_loading_percent / 100.0)

    # 1) Use max_i_ka and voltage level
    if "max_i_ka" in line_row:
        max_i_ka = float(line_row["max_i_ka"])
        if np.isfinite(max_i_ka) and max_i_ka > 0:
            from_bus = line_row["from_bus"]
            vn_kv = (
                float(net.bus.loc[from_bus, "vn_kv"])
                if from_bus in net.bus.index
                else np.nan
            )
            if np.isfinite(vn_kv) and vn_kv > 0:
                smax = np.sqrt(3.0) * vn_kv * max_i_ka  # kV * kA = MVA
                return smax * (max_loading_percent / 100.0)

    # 2) Infer from loading_percent (if present)
    loading_percent = float(line_res_row.get("loading_percent", np.nan))
    p_mw = float(line_res_row.get("p_from_mw", np.nan))
    q_mvar = float(line_res_row.get("q_from_mvar", 0.0))
    if np.isfinite(loading_percent) and loading_percent > 1e-9 and np.isfinite(p_mw):
        s0 = float(np.hypot(p_mw, q_mvar))
        smax = s0 * 100.0 / loading_percent
        return smax * (max_loading_percent / 100.0)

    return float("inf")


def get_line_base_quantities(
    net,
    *,
    margin_factor: float = 1.0,
    pf_mode: Literal["ac", "dc"] = "ac",
    line_indices: Sequence[int] | None = None,
) -> LineBaseQuantities:
    """
    Run power flow and extract per-line base flows, limits, and margins.

    Parameters
    ----------
    net:
        pandapower network.
    margin_factor:
        Multiplier applied to estimated limits (e.g., 0.9 for conservative).
    pf_mode:
        "ac" -> pandapower.runpp (AC PF)
        "dc" -> pandapower.rundcpp (DC PF, faster on large cases)
    line_indices:
        Optional explicit ordering of line indices. Defaults to sorted(net.line.index).

    Returns
    -------
    LineBaseQuantities

    Raises
    ------
    ValueError
        If margin_factor is non-positive or pf_mode is invalid.
    ImportError
        If pandapower is not installed.
    """
    if margin_factor <= 0:
        raise ValueError("margin_factor must be positive.")
    if pf_mode not in ("ac", "dc"):
        raise ValueError("pf_mode must be 'ac' or 'dc'.")

    try:
        import pandapower as pp  # local import to keep math-only modules importable
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "pandapower is required for net-based radii computations."
        ) from e

    if pf_mode == "ac":
        pp.runpp(net)
    else:
        pp.rundcpp(net)

    idx = sorted(net.line.index) if line_indices is None else list(line_indices)
    res_line = net.res_line.loc[idx]

    flow0 = res_line["p_from_mw"].to_numpy(dtype=float)
    p0_abs = np.abs(flow0)

    limits = np.empty(len(idx), dtype=float)
    for pos, (_, line_row) in enumerate(net.line.loc[idx].iterrows()):
        line_res_row = res_line.iloc[pos]
        s_limit = estimate_line_limit_mva(net, line_row, line_res_row)
        limits[pos] = float(s_limit) * float(margin_factor)

    margins = np.maximum(limits - p0_abs, 0.0)

    return LineBaseQuantities(
        line_indices=idx,
        flow0_mw=flow0,
        p0_abs_mw=p0_abs,
        limit_mw_est=limits,
        margin_mw=margins,
    )


def line_key(line_idx: int) -> str:
    """Stable external key format for per-line result dictionaries."""
    return f"line_{int(line_idx)}"


def as_2d_square_matrix(x: np.ndarray, n: int, *, name: str) -> np.ndarray:
    """
    Validate and return x as a (n,n) float matrix.

    Raises ValueError on shape mismatch.
    """
    X = np.asarray(x, dtype=float)
    if X.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n},{n}); got {X.shape}.")
    return X


def as_1d_vector(x: np.ndarray, n: int, *, name: str) -> np.ndarray:
    """
    Validate and return x as a (n,) float vector.

    Raises ValueError on shape mismatch.
    """
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},); got {v.shape}.")
    return v
