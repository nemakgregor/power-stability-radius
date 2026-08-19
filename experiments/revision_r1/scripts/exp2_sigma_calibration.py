"""Experiment R1-2 (reviewer point 5): sigma-radius calibration after the fix.

case118, load-proportional uncertainty (sigma_P = 10% of bus load, floor 1e-6;
sigma_Q = sigma_P * tan(acos(0.9))).  NOTE: the UC.jl-derived hourly sigma of
the original paper is unreachable from this environment (axavier.org); the
load-proportional profile reproduces the same heterogeneous structure.

For the TOP_K lines by analytical flow sigma: compare the analytical
first-order sd of the binding-end |S| against the empirical nonlinear
Monte Carlo sd (the repo's own balanced Gaussian sampler), and run the
tightened-limit probability check on the top line: limit = s0 + 2*sd_emp,
predicted normal tail vs empirical frequency with a Wilson interval.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PF_KW, certificate_with_h, load_case, save_json  # noqa: E402

import pandapower as pp  # noqa: E402

from stability_radius.radii.ac_sigma_radius import (  # noqa: E402
    compute_ac_sigma_radius,
    overload_probability_one_sided_limit,
)
from stability_radius.verification.sampling import (  # noqa: E402
    sample_balanced_gaussian_sigma,
)
from stability_radius.workflows import extract_binding_end_data  # noqa: E402

CASE = "pglib_opf_case118_ieee.m"
SEED = 42
N_MC = 3000
TOP_K = 10


def main() -> None:
    net, slack = load_case(CASE)
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)
    line_ids = [int(x) for x in sorted(net.line.index)]

    # load-proportional sigma
    p_load = np.zeros(n_bus)
    for i in net.load.index:
        row = net.load.loc[i]
        if bool(row.get("in_service", True)):
            p_load[bus_ids.index(int(row.bus))] += float(row.p_mw)
    sig_p = np.maximum(0.10 * np.abs(p_load), 1e-6)
    sig_q = sig_p * np.tan(np.arccos(0.9))

    base_pf, ac, hv, h_from, h_to, _ = certificate_with_h(net, slack)
    h_bind, s0_mva, s_limit_mva, lids = extract_binding_end_data(
        ac_results=ac, h_from=h_from, h_to=h_to
    )
    sigma_res = compute_ac_sigma_radius(
        h_vectors=h_bind,
        s_limit_mva=s_limit_mva,
        s0_mva=s0_mva,
        sigma_p_mw=sig_p,
        sigma_q_mvar=sig_q,
        line_ids=lids,
        balance=True,
    )

    # rank by analytical flow sigma
    ranked = sorted(
        (
            (float(v["sigma_flow_mva"]), int(k.split("_")[1]))
            for k, v in sigma_res.items()
            if isinstance(v, dict) and np.isfinite(v.get("sigma_flow_mva", np.nan))
        ),
        reverse=True,
    )[:TOP_K]

    # Monte Carlo with the repo's own balanced sampler
    rng = np.random.default_rng(SEED)
    dP, dQ = sample_balanced_gaussian_sigma(rng=rng, n=N_MC, sigma_p=sig_p, sigma_q=sig_q)

    import copy

    base = copy.deepcopy(net)
    pp.runpp(base, init="dc", **PF_KW)
    sg = [pp.create_sgen(base, b, p_mw=0.0, q_mvar=0.0) for b in bus_ids]

    ends = {lid: str(ac[f"line_{lid}"]["binding_end"]) for _, lid in ranked}
    flows = {lid: [] for _, lid in ranked}
    n_fail = 0
    for k in range(N_MC):
        base.sgen.loc[sg, "p_mw"] = dP[k]
        base.sgen.loc[sg, "q_mvar"] = dQ[k]
        try:
            pp.runpp(base, init="results", **PF_KW)
        except Exception:
            n_fail += 1
            continue
        for _, lid in ranked:
            r = base.res_line.loc[lid]
            if ends[lid] == "from":
                flows[lid].append(float(np.hypot(r.p_from_mw, r.q_from_mvar)))
            else:
                flows[lid].append(float(np.hypot(r.p_to_mw, r.q_to_mvar)))

    lines_out = []
    for sflow, lid in ranked:
        arr = np.asarray(flows[lid], dtype=float)
        sd_emp = float(np.std(arr, ddof=1))
        s0 = float(ac[f"line_{lid}"][f"ac_s0_{ends[lid]}_mva"])
        # tightened-limit probability check at 2 empirical sd
        tight = s0 + 2.0 * sd_emp
        emp_p = float(np.mean(arr > tight))
        n = arr.size
        # Wilson 95%
        z = 1.959963984540054
        ph = emp_p
        den = 1 + z * z / n
        ctr = (ph + z * z / (2 * n)) / den
        hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        pred_p = float(
            overload_probability_one_sided_limit(y0=s0, limit=tight, sigma=sflow)
        )
        lines_out.append(
            {
                "line": int(lid),
                "end": ends[lid],
                "s0_mva": s0,
                "sigma_flow_analytical_mva": float(sflow),
                "sigma_flow_empirical_mva": sd_emp,
                "ratio_analytical_over_empirical": float(sflow / sd_emp) if sd_emp > 0 else None,
                "tightened_limit_mva": float(tight),
                "predicted_exceed_prob": pred_p,
                "empirical_exceed_prob": emp_p,
                "wilson95": [float(max(ctr - hw, 0.0)), float(ctr + hw)],
                "n_samples": int(n),
            }
        )
        print(
            f"line_{lid:<4d} {ends[lid]:4s} sd_ana={sflow:8.3f} sd_emp={sd_emp:8.3f} "
            f"ratio={sflow / sd_emp:6.3f}  P_pred={pred_p:.4f} P_emp={emp_p:.4f}"
        )

    ratios = [r["ratio_analytical_over_empirical"] for r in lines_out]
    out = {
        "experiment": "sigma_calibration_case118",
        "seed": SEED,
        "n_mc": N_MC,
        "n_pf_failures": n_fail,
        "sigma_model": "sigma_P = 10% bus load (floor 1e-6), sigma_Q = sigma_P*tan(acos(0.9))",
        "note": "UC.jl hourly sigma unreachable from sandbox; structure preserved",
        "ratio_summary": {
            "median": float(np.median(ratios)),
            "min": float(np.min(ratios)),
            "max": float(np.max(ratios)),
        },
        "lines": lines_out,
    }
    print(f"ratio analytical/empirical: median={out['ratio_summary']['median']:.3f} "
          f"range=[{out['ratio_summary']['min']:.3f}, {out['ratio_summary']['max']:.3f}]  PF fails={n_fail}")
    save_json("exp2_sigma_calibration.json", out)


if __name__ == "__main__":
    main()
