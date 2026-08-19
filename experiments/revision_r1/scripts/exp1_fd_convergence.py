"""Experiment R1-1 (reviewer point 2): finite-difference convergence of the
AC adjoint sensitivity after the slack-consistency fix.

For each case: take the top-K tightest differentiable binding ends, replay
centered finite differences along the balanced worst-case direction at a
sweep of step sizes, and report the relative error vs the linear prediction.
With a correct derivative the error must DECREASE with the step until PF
tolerance noise dominates.  Also reports the adjoint residual
||J^T h - b||_inf / max(1, ||b||_inf).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    apply_du_and_solve,
    balanced_direction,
    certificate_with_h,
    end_s_mva,
    load_case,
    save_json,
)

CASES = [
    ("pglib_opf_case14_ieee.m", "case14_ieee"),
    ("pglib_opf_case30_ieee.m", "case30_ieee"),
    ("pglib_opf_case118_ieee.m", "case118_ieee"),
    ("pglib_opf_case200_activ.m", "case200_activ"),
]
TOP_K = 10
EPS_SWEEP_MVA = [1.0, 0.1, 0.01, 0.001]


def run_case(case_file: str, label: str) -> dict:
    net, slack = load_case(case_file)
    base_pf, ac, hv, h_from, h_to, bus_ids = certificate_with_h(net, slack)
    n_bus = len(bus_ids)
    line_ids = [int(x) for x in sorted(net.line.index)]

    rows = []
    for lid in line_ids:
        d = ac[f"line_{lid}"]
        if d.get("is_unconstrained") or d.get("nondifferentiable_apparent_power"):
            continue
        r = float(d["radius_ac_l2"])
        if np.isfinite(r) and r > 0:
            rows.append((r, lid, str(d["binding_end"])))
    rows.sort()
    rows = rows[:TOP_K]

    per_end = []
    for r, lid, end in rows:
        li = line_ids.index(lid)
        h = (h_from if end == "from" else h_to)[li]
        u = balanced_direction(h, n_bus, hv["pq_mask"])
        if float(np.linalg.norm(u)) <= 0:
            continue
        pred_slope = float(np.dot(h, u))  # d|S| per unit step along u

        errs = {}
        for eps in EPS_SWEEP_MVA:
            sp, sm = [], []
            for sgn in (+1.0, -1.0):
                nn = apply_du_and_solve(net, bus_ids, sgn * eps * u, init="results")
                if nn is None:
                    break
                (sp if sgn > 0 else sm).append(end_s_mva(nn, lid, end))
            if not sp or not sm:
                errs[str(eps)] = None
                continue
            fd_slope = (sp[0] - sm[0]) / (2.0 * eps)
            rel = abs(fd_slope - pred_slope) / max(abs(pred_slope), 1e-12)
            errs[str(eps)] = float(rel)
        per_end.append(
            {"line": int(lid), "end": end, "radius": float(r), "rel_err_by_eps": errs}
        )

    def agg(eps):
        vals = [e["rel_err_by_eps"].get(str(eps)) for e in per_end]
        vals = [v for v in vals if v is not None]
        return {
            "median": float(np.median(vals)) if vals else None,
            "max": float(np.max(vals)) if vals else None,
            "n": len(vals),
        }

    out = {
        "case": label,
        "n_ends_checked": len(per_end),
        "adjoint_residual_max": float(hv["adjoint_residual_max"]),
        "eps_sweep_mva": EPS_SWEEP_MVA,
        "aggregate_by_eps": {str(e): agg(e) for e in EPS_SWEEP_MVA},
        "per_end": per_end,
    }
    a = out["aggregate_by_eps"]
    print(
        f"{label:16s} resid={out['adjoint_residual_max']:.2e}  "
        + "  ".join(
            f"eps={e}: med={a[str(e)]['median']:.2e} max={a[str(e)]['max']:.2e}"
            for e in EPS_SWEEP_MVA
            if a[str(e)]["median"] is not None
        )
    )
    return out


def main() -> None:
    results = [run_case(cf, lab) for cf, lab in CASES]
    save_json(
        "exp1_fd_convergence.json", {"experiment": "fd_convergence", "cases": results}
    )


if __name__ == "__main__":
    main()
