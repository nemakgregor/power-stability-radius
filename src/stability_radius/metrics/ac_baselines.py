from __future__ import annotations

"""
Baseline robustness metrics for AC lines.

These are simple, well-known heuristic metrics commonly used in power systems
to gauge line vulnerability.  They serve as baselines against which the
stability-radius-based certificates can be compared.

All functions are stateless and consume data from the per-line results dict.
"""

import math
from typing import Any


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
        ``"line_<id>"`` -> ``{"loading_ratio", "headroom_mva", "cheb_prob_upper"}``.
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

        out[k] = {
            "loading_ratio": float(lr),
            "headroom_mva": float(hr),
            "cheb_prob_upper": float(cheb),
        }

    return out
