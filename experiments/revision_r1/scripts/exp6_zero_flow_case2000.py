"""Experiment R1-6 (reviewer point 7): zero-flow ends on case2000_goc.

The original sweep reported 34 nondifferentiable zero-flow ends on
case2000_goc, which prevented an all-constraint certificate.  With the
operator-norm extension every monitored end now receives a radius.
Also reports the ND-count sensitivity to the threshold choice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_case, save_json  # noqa: E402

from stability_radius.base_point.pandapower_opp import (  # noqa: E402
    ACFPFConfig,
    solve_ac_fpf,
)
from stability_radius.radii.ac_l2 import compute_ac_l2_radius  # noqa: E402

CASE = "pglib_opf_case2000_goc.m"
THRESHOLD_SWEEP_MVA = [1e-12, 1e-9, 1e-6, 1e-3, 1e-1]


def main() -> None:
    net, slack = load_case(CASE)
    line_ids = [int(x) for x in sorted(net.line.index)]
    # Plain runpp does not converge on the lossless case2000_goc; use the
    # AC feasibility power flow (runopp) as in the original sigma experiment.
    base_pf = solve_ac_fpf(
        net=net,
        slack_bus=slack,
        line_indices=line_ids,
        lossless=True,
        fpf_cfg=ACFPFConfig(pg0_source="case", init="dc", max_attempts=3),
    )
    ac = compute_ac_l2_radius(
        net, base_pf=base_pf, slack_bus=slack, chunk_size=64, lossless=True,
        return_h_vectors=True,
    )
    hv = ac.pop("_h_vectors")

    nd_lines = []
    opn_lines = []
    radii_all = []
    for lid in line_ids:
        d = ac[f"line_{lid}"]
        if d["is_unconstrained"]:
            continue
        r = float(d["radius_ac_l2"])
        if np.isfinite(r):
            radii_all.append(r)
        if d["ac_nondifferentiable_from"] or d["ac_nondifferentiable_to"]:
            nd_lines.append(
                {
                    "line": int(lid),
                    "s0_from": float(d["ac_s0_from_mva"]),
                    "s0_to": float(d["ac_s0_to_mva"]),
                    "operator_norm_radius_from": float(d["radius_ac_l2_from"]),
                    "operator_norm_radius_to": float(d["radius_ac_l2_to"]),
                    "status": str(d["constraint_status_ac_l2"]),
                }
            )
            if d.get("ac_operator_norm_from") or d.get("ac_operator_norm_to"):
                opn_lines.append(int(lid))

    # threshold sensitivity on |S0| counts
    s0_ends = []
    for lid in line_ids:
        d = ac[f"line_{lid}"]
        if d["is_unconstrained"]:
            continue
        s0_ends += [float(d["ac_s0_from_mva"]), float(d["ac_s0_to_mva"])]
    s0_ends = np.asarray(s0_ends)
    sensitivity = {
        str(t): int((s0_ends <= t).sum()) for t in THRESHOLD_SWEEP_MVA
    }

    finite_min = float(np.min(radii_all)) if radii_all else None
    out = {
        "experiment": "zero_flow_case2000_goc",
        "n_lines": len(line_ids),
        "n_nd_lines": len(nd_lines),
        "n_operator_norm_certified": len(opn_lines),
        "all_constraint_radius_now_defined": all(
            x["status"] in ("ok_finite_operator_norm", "ok_finite", "ok_infinite")
            for x in nd_lines
        ),
        "global_min_radius": finite_min,
        "nd_threshold_sensitivity_end_counts": sensitivity,
        "q_limit_events_at_base": len(getattr(base_pf, "q_limit_events", []) or []),
        "adjoint_residual_max": float(hv["adjoint_residual_max"]),
        "nd_lines": nd_lines,
    }
    print(
        f"case2000_goc: ND lines={len(nd_lines)}, operator-norm certified={len(opn_lines)}, "
        f"all-constraint radius defined: {out['all_constraint_radius_now_defined']}, "
        f"min radius={finite_min}"
    )
    print("threshold sensitivity (ends with |S0|<=t):", sensitivity)
    save_json("exp6_zero_flow_case2000.json", out)


if __name__ == "__main__":
    main()
