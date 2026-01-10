from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandapower as pp

logger = logging.getLogger(__name__)


def _estimate_line_limit_mva(net, line_row, line_res_row) -> float:
    """
    Estimate a line thermal limit in MVA using available pandapower data.

    Priority:
      1) If max_i_ka and from-bus vn_kv are available: Smax ≈ sqrt(3) * V_kV * I_kA
         Then apply max_loading_percent if present.
      2) Fallback: infer from loading_percent if available.

    Returns
    -------
    float
        Estimated limit in MVA, or +inf if it cannot be estimated.
    """
    max_loading_percent = float(line_row.get("max_loading_percent", 100.0))
    if not np.isfinite(max_loading_percent) or max_loading_percent <= 0:
        max_loading_percent = 100.0

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


def compute_l2_radius(
    net, H_full: np.ndarray, margin_factor: float = 1.0
) -> Dict[str, Dict[str, Any]]:
    """
    Compute a simple per-line L2 robustness radius using a PTDF-like sensitivity.

    For each line l:
        margin_l = P_limit_l - |P0_l|
        r_l2 = margin_l / ||g_l||_2
    where g_l is the l-th row of H_full.

    Parameters
    ----------
    net:
        pandapower network.
    H_full:
        Sensitivity matrix (m_lines x n_buses).
    margin_factor:
        Multiplier applied to estimated limits (e.g., 0.9 for conservative, 1.1 for relaxed).

    Returns
    -------
    dict
        Mapping "line_{line_index}" -> metrics dict.
    """
    if margin_factor <= 0:
        raise ValueError("margin_factor must be positive.")

    results: Dict[str, Dict[str, Any]] = {}

    pp.runpp(net)

    line_indices = list(net.line.index)
    if len(line_indices) != H_full.shape[0]:
        raise ValueError(
            f"H_full row count ({H_full.shape[0]}) does not match net.line count ({len(line_indices)})."
        )

    res_line = net.res_line.loc[line_indices]
    p_from = res_line["p_from_mw"].to_numpy(dtype=float)

    finite_radii = []

    for pos, (line_idx, line_row) in enumerate(net.line.loc[line_indices].iterrows()):
        p0 = abs(float(p_from[pos]))

        line_res_row = res_line.iloc[pos]
        s_limit = _estimate_line_limit_mva(net, line_row, line_res_row)

        # For compatibility with H_full (active power sensitivity), treat s_limit as an MW proxy.
        p_limit = float(s_limit) * float(margin_factor)
        margin = max(p_limit - p0, 0.0)

        g_l = np.asarray(H_full[pos, :], dtype=float)
        norm_g = float(np.linalg.norm(g_l, ord=2))

        r_l2 = float(margin / norm_g) if norm_g > 1e-12 else float("inf")

        key = f"line_{line_idx}"
        results[key] = {
            "p0_mw": p0,
            "p_limit_mw_est": p_limit,
            "margin_mw": margin,
            "norm_g": norm_g,
            "radius_l2": r_l2,
        }

        if np.isfinite(r_l2):
            finite_radii.append(r_l2)

    if finite_radii:
        logger.info("Mean L2 radius: %.6g", float(np.mean(finite_radii)))
    else:
        logger.info("Mean L2 radius: n/a (no finite radii)")

    return results
