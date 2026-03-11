from __future__ import annotations

"""
Baseline and practical robustness metrics for AC lines.

This module contains both simple heuristic metrics (loading ratio, headroom,
Cantelli bound) and industry-standard practical metrics used in static security
assessment, contingency ranking, transfer capability studies, and risk-based
security assessment.  Together they serve as baselines against which the
stability-radius-based certificates can be compared.

Practical metrics implemented
-----------------------------
- **Performance Index (PI_MVA)** — classical EMS contingency-ranking metric
  (per-line contribution).
- **Transfer margin / ATC** (linearised) — directional security margin using
  AC Jacobian sensitivities.  In our simplified physical setting (no commercial
  commitments) the transfer margin equals the simplified ATC.
- **Thermal overload risk index** — probability × severity weight, following
  risk-based security assessment practice.

All scalar functions are stateless.  Array-level helpers consume data from
the per-line results dict produced by ``compute_results_for_case()``.
"""

import math
from typing import Any

import numpy as np


def loading_ratio(*, s0_mva: float, s_limit_mva: float) -> float:
    """Loading ratio: |S0| / c.

    Higher values indicate lines closer to their thermal limit.

    Returns
    -------
    float
        In [0, inf).  Returns inf if *s_limit_mva* == 0 and flow > 0.
    """
    c = float(s_limit_mva)
    s = float(abs(s0_mva))
    if c <= 0.0:
        return float("inf") if s > 0.0 else float("nan")
    return s / c


def headroom_mva(*, s0_mva: float, s_limit_mva: float) -> float:
    """Headroom: c - |S0|.

    Lower values indicate less margin before overload.

    Returns
    -------
    float
        Can be negative if already overloaded.
    """
    return float(s_limit_mva) - float(abs(s0_mva))


def cantelli_upper_bound(*, headroom: float, sigma_flow_mva: float) -> float:
    """One-sided Cantelli (Chebyshev) upper bound on overload probability.

    P(X >= headroom) <= sigma^2 / (sigma^2 + headroom^2)

    Only valid when *headroom* > 0 (line not already overloaded at base).

    Parameters
    ----------
    headroom : margin c - |S0| in MVA.
    sigma_flow_mva : standard deviation of linearised flow magnitude in MVA.

    Returns
    -------
    float
        Upper bound probability in [0, 1].
        Returns 1.0 if headroom <= 0 (already overloaded / binding).
        Returns 0.0 if sigma_flow_mva <= 0 and headroom > 0.
    """
    h = float(headroom)
    s = float(sigma_flow_mva)

    if h <= 0.0:
        return 1.0
    if not math.isfinite(s) or s <= 0.0:
        return 0.0

    s2 = s * s
    h2 = h * h
    return s2 / (s2 + h2)


# ---------------------------------------------------------------------------
# Performance Index (PIMVA) — contingency-ranking metric
# ---------------------------------------------------------------------------


def performance_index_line(
    *,
    s0_mva: float,
    s_limit_mva: float,
    w: float = 1.0,
    n: int = 1,
) -> float:
    """Per-line contribution to the Performance Index PI_MVA.

    PI_l = (w / 2n) * (|S_l| / S_l^max)^{2n}

    This is the standard form used in EMS contingency ranking (Heliyon 2023
    review, PIMVA).  Higher values indicate greater severity of loading.

    Parameters
    ----------
    s0_mva : Apparent power flow magnitude |S_l| (MVA).
    s_limit_mva : Thermal limit S_l^max (MVA).
    w : Line weight (default 1.0).
    n : Exponent order (default 1 → quadratic penalty ``2n = 2``).

    Returns
    -------
    float
        Non-negative.  Returns inf if *s_limit_mva* <= 0 and flow > 0.
    """
    c = float(s_limit_mva)
    s = float(abs(s0_mva))
    nn = int(n)
    if nn < 1:
        nn = 1
    ww = float(w)

    if c <= 0.0:
        return float("inf") if s > 0.0 else 0.0

    ratio = s / c
    return (ww / (2.0 * nn)) * (ratio ** (2 * nn))


def performance_index_system(
    line_pi_values: list[float] | np.ndarray,
) -> float:
    """System-wide Performance Index: sum of per-line contributions.

    Parameters
    ----------
    line_pi_values : Per-line PI values from ``performance_index_line()``.

    Returns
    -------
    float
        Sum of all per-line PI contributions.
    """
    return float(np.nansum(np.asarray(line_pi_values, dtype=float)))


# ---------------------------------------------------------------------------
# Transfer margin / ATC (linearised)
# ---------------------------------------------------------------------------


def transfer_margin_linearized(
    *,
    margins_mva: np.ndarray,
    h_vectors: np.ndarray,
    direction: np.ndarray,
) -> tuple[float, int]:
    """Linearised transfer margin for a given injection-change direction.

    TM(d) = min_l { margin_l / |h_l^T d| }

    where margin_l = S_l^max - |S_l^0| and h_l is the AC sensitivity vector
    (row of J^{-T} applied to the flow gradient).

    In a simplified physical setting (no commercial commitments), TM equals
    the simplified ATC: ATC*(d) = TTC(d) - current_loading ≈ TM(d).

    Parameters
    ----------
    margins_mva : Per-line margins (MVA), shape ``(n_lines,)``.
    h_vectors : Sensitivity matrix, shape ``(n_lines, n_vars)``.
    direction : Transfer direction vector, shape ``(n_vars,)``.

    Returns
    -------
    (tm_value, limiting_line_idx)
        Transfer margin (MVA) and index of the limiting line.
        Returns ``(inf, -1)`` if no line is sensitive to this direction.
    """
    d = np.asarray(direction, dtype=float).reshape(-1)
    H = np.asarray(h_vectors, dtype=float)
    m = np.asarray(margins_mva, dtype=float).reshape(-1)

    if H.ndim != 2 or H.shape[0] != m.shape[0]:
        raise ValueError(
            f"h_vectors shape {H.shape} inconsistent with margins shape {m.shape}"
        )
    if d.shape[0] != H.shape[1]:
        raise ValueError(
            f"direction length {d.shape[0]} != h_vectors columns {H.shape[1]}"
        )

    sensitivities = np.abs(H @ d)
    eps = 1e-12

    tm_best = float("inf")
    lim_idx = -1
    for i in range(len(m)):
        if sensitivities[i] <= eps:
            continue
        margin_i = float(m[i])
        if margin_i <= 0.0:
            return 0.0, int(i)
        tm_i = margin_i / float(sensitivities[i])
        if tm_i < tm_best:
            tm_best = tm_i
            lim_idx = int(i)

    return tm_best, lim_idx


def directional_sensitivity(
    *,
    h_vectors: np.ndarray,
    direction: np.ndarray,
    margins_mva: np.ndarray,
) -> np.ndarray:
    """Per-line reciprocal directional margin (higher = more vulnerable).

    DS_l = |h_l^T d| / margin_l

    Lines with large DS are the ones most likely to become the bottleneck
    when injections shift along direction *d*.

    Parameters
    ----------
    h_vectors : Sensitivity matrix, shape ``(n_lines, n_vars)``.
    direction : Transfer direction vector, shape ``(n_vars,)``.
    margins_mva : Per-line margins (MVA), shape ``(n_lines,)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_lines,)``.  Returns inf for lines with margin <= 0.
    """
    d = np.asarray(direction, dtype=float).reshape(-1)
    H = np.asarray(h_vectors, dtype=float)
    m = np.asarray(margins_mva, dtype=float).reshape(-1)

    sensitivities = np.abs(H @ d)
    result = np.full(len(m), float("inf"), dtype=float)
    pos_mask = m > 0.0
    result[pos_mask] = sensitivities[pos_mask] / m[pos_mask]
    return result


# ---------------------------------------------------------------------------
# Thermal overload risk index
# ---------------------------------------------------------------------------


def thermal_risk_index(
    *,
    overload_prob: float,
    loading_ratio_val: float,
) -> float:
    """Thermal overload risk index: Pr(overload) * severity.

    R_l = Pr(overload of line l) * C_l

    where C_l = loading_ratio_l serves as a simple severity proxy: lines
    that are already more loaded get weighted higher.  This follows the
    risk-based security assessment principle of Risk = Probability × Impact.

    Parameters
    ----------
    overload_prob : Empirical overload probability from Monte Carlo [0, 1].
    loading_ratio_val : Loading ratio |S0| / Smax.

    Returns
    -------
    float
        Non-negative risk index.
    """
    p = float(overload_prob)
    lr = float(loading_ratio_val)
    if not math.isfinite(p) or p < 0.0:
        p = 0.0
    if not math.isfinite(lr) or lr < 0.0:
        lr = 0.0
    return p * lr


# ---------------------------------------------------------------------------
# Batch computation helpers
# ---------------------------------------------------------------------------


def compute_baseline_metrics(
    results: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Compute baseline metrics for every line in a results dict.

    Consumes per-line fields already present in *results*:
    ``ac_s_limit_mva``, ``ac_s0_from_mva`` / ``ac_s0_to_mva``,
    ``binding_end``, ``margin_ac_mva``, ``sigma_flow_mva``.

    Parameters
    ----------
    results
        Output of ``compute_results_for_case()`` or a loaded ``results.json``.

    Returns
    -------
    dict
        ``"line_<id>"`` -> ``{"loading_ratio", "headroom_mva",
        "cheb_prob_upper", "performance_index"}``.
    """
    out: dict[str, dict[str, float]] = {}

    for k, v in results.items():
        if not k.startswith("line_") or not isinstance(v, dict):
            continue

        s_limit = float(v.get("ac_s_limit_mva", float("nan")))
        binding_end = str(v.get("binding_end", "from"))

        if binding_end == "to":
            s0 = float(v.get("ac_s0_to_mva", float("nan")))
        else:
            s0 = float(v.get("ac_s0_from_mva", float("nan")))

        margin = float(v.get("margin_ac_mva", float("nan")))
        sigma_flow = float(v.get("sigma_flow_mva", float("nan")))

        lr = loading_ratio(s0_mva=s0, s_limit_mva=s_limit)
        hr = headroom_mva(s0_mva=s0, s_limit_mva=s_limit)
        cheb = cantelli_upper_bound(headroom=margin, sigma_flow_mva=sigma_flow)
        pi = performance_index_line(s0_mva=s0, s_limit_mva=s_limit)

        out[k] = {
            "loading_ratio": float(lr),
            "headroom_mva": float(hr),
            "cheb_prob_upper": float(cheb),
            "performance_index": float(pi),
        }

    return out


def compute_practical_metrics(
    *,
    results: dict[str, Any],
    mc_per_line_fractions: dict[str, float],
    h_vectors: np.ndarray | None = None,
    transfer_directions: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute all practical comparison metrics per line.

    This function computes metrics beyond the simple baselines: thermal
    overload risk index and (optionally) per-line directional sensitivity
    for given transfer directions.

    Parameters
    ----------
    results
        Output of ``compute_results_for_case()``.
    mc_per_line_fractions
        ``line_key`` → empirical overload fraction from MC.
    h_vectors
        Sensitivity matrix, shape ``(n_lines, n_vars)``.  Required for
        directional sensitivity / transfer margin.
    transfer_directions
        ``name`` → direction vector of shape ``(n_vars,)``.

    Returns
    -------
    dict
        ``"line_<id>"`` → ``{"thermal_risk_index": ...,
        "dir_sens_<name>": ..., ...}``.
    """
    line_keys: list[str] = []
    margins: list[float] = []
    lrs: list[float] = []

    for k, v in sorted(results.items()):
        if not k.startswith("line_") or not isinstance(v, dict):
            continue
        line_keys.append(k)

        s_limit = float(v.get("ac_s_limit_mva", float("nan")))
        binding_end = str(v.get("binding_end", "from"))
        if binding_end == "to":
            s0 = float(v.get("ac_s0_to_mva", float("nan")))
        else:
            s0 = float(v.get("ac_s0_from_mva", float("nan")))

        lr = loading_ratio(s0_mva=s0, s_limit_mva=s_limit) if s_limit > 0 else 0.0
        lrs.append(lr)
        margins.append(float(v.get("margin_ac_mva", float("nan"))))

    out: dict[str, dict[str, float]] = {}
    for i, k in enumerate(line_keys):
        prob = float(mc_per_line_fractions.get(k, float("nan")))
        lr_val = float(lrs[i])

        row: dict[str, float] = {
            "thermal_risk_index": float(
                thermal_risk_index(overload_prob=prob, loading_ratio_val=lr_val)
            ),
        }
        out[k] = row

    # Directional sensitivity (requires h-vectors)
    if (
        h_vectors is not None
        and transfer_directions is not None
        and len(transfer_directions) > 0
    ):
        m_arr = np.asarray(margins, dtype=float)
        H = np.asarray(h_vectors, dtype=float)
        if H.shape[0] == len(line_keys):
            for dir_name, d in transfer_directions.items():
                ds = directional_sensitivity(
                    h_vectors=H, direction=d, margins_mva=m_arr
                )
                for i, k in enumerate(line_keys):
                    out[k][f"dir_sens_{dir_name}"] = float(ds[i])

    return out
