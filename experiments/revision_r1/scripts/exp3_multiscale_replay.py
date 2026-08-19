"""Experiment R1-3 (reviewer point 4): multi-scale nonlinear replay.

For each case: take the TOP_K tightest finite differentiable lines, replay the
balanced worst-case direction at scales alpha*r for
alpha in {0.25,...,1.5}, and record nonlinear apparent-power violation of ANY
monitored line, voltage-band violations, Q-limit activations, and PF
convergence.  Reports the empirical crossing alpha (first scale at which the
nonlinear system violates any thermal limit) and whether the nonlinear
limiting line matches the predicted one.
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
    load_case,
    save_json,
)

from stability_radius.base_point.pandapower_tools import (  # noqa: E402
    detect_q_limit_events,
)

CASES = [
    ("pglib_opf_case14_ieee.m", "case14_ieee"),
    ("pglib_opf_case30_ieee.m", "case30_ieee"),
    ("pglib_opf_case118_ieee.m", "case118_ieee"),
    ("pglib_opf_case200_activ.m", "case200_activ"),
]
TOP_K = 5
SCALES = [0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
VM_MIN, VM_MAX = 0.9, 1.1


def run_case(case_file: str, label: str) -> dict:
    net, slack = load_case(case_file)
    base_pf, ac, hv, h_from, h_to, bus_ids = certificate_with_h(net, slack)
    n_bus = len(bus_ids)
    line_ids = [int(x) for x in sorted(net.line.index)]
    limits = {lid: float(ac[f"line_{lid}"]["ac_s_limit_mva"]) for lid in line_ids}
    constrained = [lid for lid in line_ids if not ac[f"line_{lid}"]["is_unconstrained"]]

    rows = []
    for lid in constrained:
        d = ac[f"line_{lid}"]
        if d.get("nondifferentiable_apparent_power"):
            continue
        r = float(d["radius_ac_l2"])
        if np.isfinite(r) and r > 0:
            rows.append((r, lid, str(d["binding_end"])))
    rows.sort()
    rows = rows[:TOP_K]

    n_qlim_base = len(getattr(base_pf, "q_limit_events", []) or [])
    per_line = []
    for r, lid, end in rows:
        li = line_ids.index(lid)
        h = (h_from if end == "from" else h_to)[li]
        u = balanced_direction(h, n_bus, hv["pq_mask"])
        if float(np.linalg.norm(u)) <= 0:
            continue

        scale_rows = []
        crossing = None
        for alpha in SCALES:
            du = (alpha * r) * u
            nn = apply_du_and_solve(net, bus_ids, du)
            if nn is None:
                scale_rows.append({"alpha": alpha, "pf_converged": False})
                continue
            s_all = np.maximum(
                np.hypot(nn.res_line.p_from_mw.values, nn.res_line.q_from_mvar.values),
                np.hypot(nn.res_line.p_to_mw.values, nn.res_line.q_to_mvar.values),
            )
            lim_arr = np.array([limits[x] for x in line_ids])
            con_mask = np.array([x in set(constrained) for x in line_ids])
            over = (s_all > lim_arr) & con_mask
            vm = nn.res_bus.vm_pu.values
            q_events = detect_q_limit_events(nn)
            s_target = float(s_all[li])
            row = {
                "alpha": float(alpha),
                "pf_converged": True,
                "target_line_s_mva": s_target,
                "target_line_loading": s_target / limits[lid],
                "any_thermal_violation": bool(over.any()),
                "n_thermal_violations": int(over.sum()),
                "first_violated_line": int(np.array(line_ids)[over][0])
                if over.any()
                else None,
                "target_is_violated": bool(over[li]),
                "vm_min": float(vm.min()),
                "vm_max": float(vm.max()),
                "voltage_violation": bool((vm < VM_MIN).any() or (vm > VM_MAX).any()),
                "n_q_limit_events": int(len(q_events)),
                "n_q_limit_events_new": int(max(0, len(q_events) - n_qlim_base)),
            }
            scale_rows.append(row)
            if crossing is None and row["any_thermal_violation"]:
                crossing = float(alpha)

        # refine crossing by interpolation on the target line loading
        conv = [x for x in scale_rows if x.get("pf_converged")]
        alphas = [x["alpha"] for x in conv]
        loadings = [x["target_line_loading"] for x in conv]
        cross_target = None
        for a0, a1, l0, l1 in zip(alphas, alphas[1:], loadings, loadings[1:]):
            if l0 < 1.0 <= l1:
                cross_target = a0 + (1.0 - l0) / (l1 - l0) * (a1 - a0)
                break
        if cross_target is None and loadings and loadings[0] >= 1.0:
            cross_target = alphas[0]

        per_line.append(
            {
                "line": int(lid),
                "end": end,
                "radius": float(r),
                "predicted_limiting_line": int(lid),
                "crossing_alpha_any_line": crossing,
                "crossing_alpha_target_line_interp": cross_target,
                "nonlinear_limiting_matches_predicted": (
                    None
                    if crossing is None
                    else bool(
                        next(
                            (x for x in scale_rows if x.get("any_thermal_violation")),
                            {},
                        ).get("first_violated_line")
                        == lid
                    )
                ),
                "scales": scale_rows,
            }
        )

    cross_vals = [
        x["crossing_alpha_target_line_interp"]
        for x in per_line
        if x["crossing_alpha_target_line_interp"] is not None
    ]
    out = {
        "case": label,
        "scales": SCALES,
        "q_limit_events_at_base": n_qlim_base,
        "per_line": per_line,
        "crossing_alpha_target_summary": {
            "n": len(cross_vals),
            "median": float(np.median(cross_vals)) if cross_vals else None,
            "min": float(np.min(cross_vals)) if cross_vals else None,
            "max": float(np.max(cross_vals)) if cross_vals else None,
        },
    }
    cs = out["crossing_alpha_target_summary"]
    print(
        f"{label:16s} lines={len(per_line)}  crossing alpha (target-line, interp): "
        f"median={cs['median']}, range=[{cs['min']}, {cs['max']}]"
    )
    return out


def main() -> None:
    results = [run_case(cf, lab) for cf, lab in CASES]
    save_json(
        "exp3_multiscale_replay.json",
        {"experiment": "multiscale_replay", "cases": results},
    )


if __name__ == "__main__":
    main()
