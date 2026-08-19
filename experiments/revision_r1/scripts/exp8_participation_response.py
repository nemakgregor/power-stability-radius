"""Experiment R1-8 (reviewer point 6): realistic admissible-response model.

case118 with a participation-factor response map: uncertainty lives ONLY at
load buses (omega = load forecast errors, MW); in-service generators respond
proportionally to their active-power headroom, alpha_g ~ (p_max - p0),
sum(alpha) = 1.  The response map is

    dP = (E - alpha 1^T) omega  =  T omega,

so the certified radius is r = margin / ||T^T h_P||_2 and the worst-case
omega* = T^T h_P / ||T^T h_P||.  We check PHYSICAL REALIZABILITY of the worst
case at the certified radius: generator responses within [p_min, p_max], and
nonlinear replay with the explicit generator redispatch (not slack pickup).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PF_KW, certificate_with_h, end_s_mva, load_case, save_json  # noqa: E402

import pandapower as pp  # noqa: E402

CASE = "pglib_opf_case118_ieee.m"
TOP_K = 5


def main() -> None:
    net, slack = load_case(CASE)
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)
    pos_of = {b: i for i, b in enumerate(bus_ids)}
    line_ids = [int(x) for x in sorted(net.line.index)]

    base_pf, ac, hv, h_from, h_to, _ = certificate_with_h(net, slack)

    # load buses (uncertainty support)
    load_bus_pos = sorted(
        {
            pos_of[int(net.load.loc[i, "bus"])]
            for i in net.load.index
            if bool(net.load.loc[i, "in_service"]) and float(net.load.loc[i, "p_mw"]) > 0
        }
    )
    E = np.zeros((n_bus, len(load_bus_pos)))
    for j, p in enumerate(load_bus_pos):
        E[p, j] = 1.0

    # generator participation ~ headroom
    gens = [
        g
        for g in net.gen.index
        if bool(net.gen.loc[g, "in_service"])
        and np.isfinite(float(net.gen.loc[g, "max_p_mw"]))
    ]
    p0g = np.array([float(net.gen.loc[g, "p_mw"]) for g in gens])
    pmax = np.array([float(net.gen.loc[g, "max_p_mw"]) for g in gens])
    pmin = np.array([float(net.gen.loc[g, "min_p_mw"]) for g in gens])
    head = np.maximum(pmax - p0g, 0.0)
    if head.sum() <= 0:
        raise RuntimeError("no generator headroom")
    alpha = head / head.sum()
    alpha_bus = np.zeros(n_bus)
    for a, g in zip(alpha, gens):
        alpha_bus[pos_of[int(net.gen.loc[g, "bus"])]] += a

    rows = []
    for lid in line_ids:
        d = ac[f"line_{lid}"]
        if d["is_unconstrained"] or d["nondifferentiable_apparent_power"]:
            continue
        r2 = float(d["radius_ac_l2"])
        if np.isfinite(r2) and r2 > 0:
            rows.append((r2, lid, str(d["binding_end"])))
    rows.sort()

    out_lines = []
    for r_bal, lid, end in rows[:TOP_K]:
        li = line_ids.index(lid)
        h_p = (h_from if end == "from" else h_to)[li][:n_bus]

        # T^T h = E^T h - (alpha^T h) * 1
        th = E.T @ h_p - float(alpha_bus @ h_p) * np.ones(len(load_bus_pos))
        denom = float(np.linalg.norm(th))
        margin = float(ac[f"line_{lid}"][f"ac_margin_{end}_mva"])
        r_part = margin / denom if denom > 1e-12 else float("inf")

        # worst-case omega at the certified radius
        omega = (r_part / denom) * th
        total = float(omega.sum())
        dgen = -alpha * total  # generator responses (MW)
        realizable = bool(
            np.all(p0g + dgen <= pmax + 1e-6) and np.all(p0g + dgen >= pmin - 1e-6)
        )

        # nonlinear replay WITH explicit generator redispatch
        nn = copy.deepcopy(net)
        for a_i, g in zip(dgen, gens):
            nn.gen.loc[g, "p_mw"] = float(nn.gen.loc[g, "p_mw"] + a_i)
        for j, p in enumerate(load_bus_pos):
            pp.create_sgen(nn, bus_ids[p], p_mw=float(omega[j]), q_mvar=0.0)
        try:
            pp.runpp(nn, init="dc", **PF_KW)
            s_repl = end_s_mva(nn, lid, end)
            conv = True
        except Exception:
            s_repl, conv = None, False

        limit = float(ac[f"line_{lid}"]["ac_s_limit_mva"])
        out_lines.append(
            {
                "line": int(lid),
                "end": end,
                "radius_balanced_pq": float(r_bal),
                "radius_participation_load_only": float(r_part),
                "ratio_participation_over_balanced": float(r_part / r_bal),
                "worst_case_total_load_error_mw": total,
                "max_single_gen_response_mw": float(np.abs(dgen).max()),
                "generator_limits_respected": realizable,
                "replay_converged": conv,
                "replay_s_over_limit": (s_repl / limit) if s_repl else None,
            }
        )
        print(
            f"line_{lid:<4d} {end:4s} r_bal={r_bal:8.3f} r_part={r_part:8.3f} "
            f"ratio={r_part / r_bal:5.2f} realizable={realizable} "
            f"replay |S|/limit={s_repl / limit if s_repl else float('nan'):.3f}"
        )

    save_json(
        "exp8_participation_response.json",
        {
            "experiment": "participation_response_case118",
            "response_model": "omega at load buses; gens respond prop. to headroom; alpha sums to 1",
            "n_load_buses": len(load_bus_pos),
            "n_gens": len(gens),
            "lines": out_lines,
        },
    )


if __name__ == "__main__":
    main()
