"""AC base-point feasibility check.

After solving AC PF from a DC OPF dispatch, the apparent power
|S| = sqrt(P^2 + Q^2) on some lines may exceed their thermal limits
due to reactive power flows not captured by the DC model.

This module checks whether the AC base point satisfies all line thermal
constraints.  If any constrained line has margin < 0 (i.e. |S0| > c_l),
the base point is AC-infeasible and the resulting radii on those lines
would be negative.

Mathematical background
-----------------------
For each line l, the margin at the binding (worse) end is:

    margin_l = c_l - max(|S_from|, |S_to|)

where
    |S_from| = sqrt(P_from^2 + Q_from^2)
    |S_to|   = sqrt(P_to^2 + Q_to^2)
    c_l      = thermal limit (MVA) from rateA / sn_mva / max_i_ka

If margin_l < 0 for any *constrained* line (i.e. one with a real thermal
limit, not a surrogate for rateA==0), the AC L2 radius on that line will
be negative, and the system radius r* = min_l r_l will be invalid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stability_radius.base_point.pypsa_pf import PyPSAAPFResult
from stability_radius.radii.common import estimate_line_limit_mva_with_flag

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ACLineViolation:
    """Single line thermal violation in the AC base point."""

    line_id: int
    line_pos: int
    binding_end: str  # "from" | "to"
    s0_mva: float  # |S| at binding end
    limit_mva: float  # thermal limit (MVA)
    margin_mva: float  # limit - |S0| (negative = violated)
    p_mw: float  # P at binding end
    q_mvar: float  # Q at binding end
    is_unconstrained: bool  # True if limit is a surrogate


@dataclass(frozen=True)
class ACFeasibilityResult:
    """Result of AC base-point feasibility check."""

    is_feasible: bool
    n_lines: int
    n_violated: int
    n_constrained_violated: int  # violations on real (non-surrogate) limits only
    worst_margin_mva: float
    worst_line_id: int
    violations: tuple[ACLineViolation, ...] = field(default_factory=tuple)

    def to_meta_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for __meta__."""
        d: dict[str, Any] = {
            "is_feasible": bool(self.is_feasible),
            "n_lines": int(self.n_lines),
            "n_violated": int(self.n_violated),
            "n_constrained_violated": int(self.n_constrained_violated),
            "worst_margin_mva": float(self.worst_margin_mva),
            "worst_line_id": int(self.worst_line_id),
        }
        if self.violations:
            d["violations"] = [
                {
                    "line_id": int(v.line_id),
                    "binding_end": str(v.binding_end),
                    "s0_mva": round(float(v.s0_mva), 4),
                    "limit_mva": round(float(v.limit_mva), 4),
                    "margin_mva": round(float(v.margin_mva), 4),
                    "p_mw": round(float(v.p_mw), 4),
                    "q_mvar": round(float(v.q_mvar), 4),
                    "is_unconstrained": bool(v.is_unconstrained),
                }
                for v in self.violations
            ]
        return d


def check_ac_base_point_feasibility(
    *,
    net: Any,
    base_pf: PyPSAAPFResult,
    max_violations_detail: int = 50,
) -> ACFeasibilityResult:
    """
    Check that all line thermal constraints are satisfied in the AC base point.

    For each line l:
        s_from = sqrt(p_from^2 + q_from^2)
        s_to   = sqrt(p_to^2 + q_to^2)
        s_bind = max(s_from, s_to)
        margin = c_l - s_bind

    If margin < 0 for any *constrained* line, the base point is AC-infeasible.

    Parameters
    ----------
    net :
        pandapower network.
    base_pf :
        AC PF result from ``solve_ac_pf_base_point``.
    max_violations_detail :
        Maximum number of individual violations to store (for memory/log limits).

    Returns
    -------
    ACFeasibilityResult
    """
    line_ids = [int(x) for x in sorted(net.line.index)]
    m = len(line_ids)

    p0 = np.asarray(base_pf.line_p0_mw, dtype=float).reshape(-1)
    q0 = np.asarray(base_pf.line_q0_mvar, dtype=float).reshape(-1)
    p1 = np.asarray(base_pf.line_p1_mw, dtype=float).reshape(-1)
    q1 = np.asarray(base_pf.line_q1_mvar, dtype=float).reshape(-1)

    if p0.shape != (m,) or q0.shape != (m,) or p1.shape != (m,) or q1.shape != (m,):
        raise ValueError(
            f"Base PF line flow arrays shape mismatch: expected ({m},), "
            f"got p0={p0.shape}, q0={q0.shape}, p1={p1.shape}, q1={q1.shape}"
        )

    s_from = np.sqrt(p0 * p0 + q0 * q0)
    s_to = np.sqrt(p1 * p1 + q1 * q1)

    # Extract limits and unconstrained flags.
    limits_mva = np.empty(m, dtype=float)
    is_unconstrained = np.zeros(m, dtype=bool)
    for pos, lid in enumerate(line_ids):
        lim, is_uc = estimate_line_limit_mva_with_flag(net, net.line.loc[int(lid)])
        limits_mva[pos] = float(lim)
        is_unconstrained[pos] = bool(is_uc)

    margin_from = limits_mva - s_from
    margin_to = limits_mva - s_to

    violations: list[ACLineViolation] = []
    n_violated = 0
    n_constrained_violated = 0
    worst_margin = float("inf")
    worst_line_id = -1

    for pos in range(m):
        # Binding end = the end with smaller margin (larger |S|).
        if margin_from[pos] <= margin_to[pos]:
            bind_end = "from"
            margin = float(margin_from[pos])
            s0 = float(s_from[pos])
            p_bind = float(p0[pos])
            q_bind = float(q0[pos])
        else:
            bind_end = "to"
            margin = float(margin_to[pos])
            s0 = float(s_to[pos])
            p_bind = float(p1[pos])
            q_bind = float(q1[pos])

        if margin < worst_margin:
            worst_margin = margin
            worst_line_id = line_ids[pos]

        if margin < 0:
            n_violated += 1
            if not is_unconstrained[pos]:
                n_constrained_violated += 1

            if len(violations) < max_violations_detail:
                violations.append(
                    ACLineViolation(
                        line_id=line_ids[pos],
                        line_pos=pos,
                        binding_end=bind_end,
                        s0_mva=s0,
                        limit_mva=float(limits_mva[pos]),
                        margin_mva=margin,
                        p_mw=p_bind,
                        q_mvar=q_bind,
                        is_unconstrained=bool(is_unconstrained[pos]),
                    )
                )

    is_feasible = n_constrained_violated == 0
    n_constrained = int(m - int(np.sum(is_unconstrained)))

    # Log diagnostic summary.
    if is_feasible:
        logger.info(
            "AC feasibility: PASS (%d constrained lines OK, "
            "worst_margin=%.4f MVA on line %d)",
            n_constrained,
            float(worst_margin),
            int(worst_line_id),
        )
    else:
        logger.warning(
            "AC feasibility: FAIL (%d/%d constrained lines violated, "
            "worst_margin=%.4f MVA on line %d)",
            n_constrained_violated,
            n_constrained,
            float(worst_margin),
            int(worst_line_id),
        )
        # Log top violations (sorted by margin, worst first).
        sorted_violations = sorted(violations, key=lambda v: v.margin_mva)
        for v in sorted_violations[:10]:
            q_pct = abs(v.q_mvar) / max(v.s0_mva, 1e-9) * 100
            logger.warning(
                "  Line %d (%s-end): |S0|=%.2f MVA > limit=%.2f MVA "
                "(margin=%.2f MVA, P=%.2f MW, Q=%.2f Mvar, Q/S=%.1f%%)",
                v.line_id,
                v.binding_end,
                v.s0_mva,
                v.limit_mva,
                v.margin_mva,
                v.p_mw,
                v.q_mvar,
                q_pct,
            )
        if len(sorted_violations) > 10:
            logger.warning(
                "  ... and %d more violated lines (total %d)",
                len(sorted_violations) - 10,
                n_violated,
            )

    return ACFeasibilityResult(
        is_feasible=is_feasible,
        n_lines=m,
        n_violated=n_violated,
        n_constrained_violated=n_constrained_violated,
        worst_margin_mva=float(worst_margin),
        worst_line_id=int(worst_line_id),
        violations=tuple(violations),
    )
