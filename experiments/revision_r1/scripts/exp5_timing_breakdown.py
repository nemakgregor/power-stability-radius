"""Experiment R1-5 (reviewer point 11): separated, repeated timings + memory.

Per case, REPEATS repetitions of: AC PF solve; operator build (Jacobian
assembly + sparse LU factorization); adjoint + norms (certificate);
peak resident memory after each stage; stored h-array size.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_case, peak_rss_mb, save_json  # noqa: E402

from stability_radius.ac.ac_model import build_ac_operator  # noqa: E402
from stability_radius.base_point.pypsa_pf import (  # noqa: E402
    solve_ac_pf_base_point_from_pandapower,
)
from stability_radius.radii.ac_l2 import compute_ac_l2_radius  # noqa: E402

CASES = [
    ("pglib_opf_case14_ieee.m", "case14_ieee"),
    ("pglib_opf_case30_ieee.m", "case30_ieee"),
    ("pglib_opf_case118_ieee.m", "case118_ieee"),
    ("pglib_opf_case200_activ.m", "case200_activ"),
    ("pglib_opf_case2000_goc.m", "case2000_goc"),
]
REPEATS = 7


def stat(v):
    a = np.asarray(v, dtype=float)
    return {"mean_s": float(a.mean()), "std_s": float(a.std(ddof=1)), "min_s": float(a.min()), "max_s": float(a.max()), "n": int(a.size)}


def run_case(case_file: str, label: str) -> dict:
    net, slack = load_case(case_file)
    t_pf, t_op, t_cert = [], [], []
    base_pf = None
    op = None
    for _ in range(REPEATS):
        t = time.perf_counter()
        base_pf = solve_ac_pf_base_point_from_pandapower(
            net=net, slack_bus=slack, solver="pandapower", init="dc", lossless=True
        )
        t_pf.append(time.perf_counter() - t)
    rss_pf = peak_rss_mb()
    vm = np.asarray(base_pf.v_mag_pu, dtype=float)
    va = np.asarray(base_pf.v_ang_rad, dtype=float)
    line_ids = [int(x) for x in sorted(net.line.index)]
    for _ in range(REPEATS):
        t = time.perf_counter()
        op = build_ac_operator(
            net=net, slack_bus=slack, vm_pu=vm, va_rad=va,
            line_indices=line_ids, lossless=True,
        )
        t_op.append(time.perf_counter() - t)
    rss_op = peak_rss_mb()
    ac = None
    for _ in range(REPEATS):
        t = time.perf_counter()
        ac = compute_ac_l2_radius(
            net, base_pf=base_pf, slack_bus=slack, chunk_size=64, lossless=True,
            return_h_vectors=True,
        )
        t_cert.append(time.perf_counter() - t)
    rss_cert = peak_rss_mb()
    hv = ac.pop("_h_vectors")
    h_mb = (hv["h_from"].nbytes + hv["h_to"].nbytes) / 1e6
    out = {
        "case": label,
        "n_bus": int(len(net.bus)),
        "n_line": int(len(net.line)),
        "repeats": REPEATS,
        "ac_pf": stat(t_pf),
        "operator_build_assembly_plus_lu": stat(t_op),
        "certificate_adjoint_plus_norms": stat(t_cert),
        "peak_rss_after_pf_mb": rss_pf,
        "peak_rss_after_operator_mb": rss_op,
        "peak_rss_after_certificate_mb": rss_cert,
        "stored_h_arrays_mb": float(h_mb),
        "adjoint_residual_max": float(hv["adjoint_residual_max"]),
        "n_nondiff_ends": int(
            sum(
                int(bool(ac[f"line_{lid}"]["ac_nondifferentiable_from"]))
                + int(bool(ac[f"line_{lid}"]["ac_nondifferentiable_to"]))
                for lid in line_ids
            )
        ),
    }
    print(
        f"{label:16s} PF {out['ac_pf']['mean_s']:.3f}±{out['ac_pf']['std_s']:.3f}s  "
        f"op {out['operator_build_assembly_plus_lu']['mean_s']:.3f}s  "
        f"cert {out['certificate_adjoint_plus_norms']['mean_s']:.3f}s  "
        f"RSS {rss_cert:.0f}MB  h {h_mb:.1f}MB  ND {out['n_nondiff_ends']}"
    )
    return out


def main() -> None:
    results = []
    for cf, lab in CASES:
        try:
            results.append(run_case(cf, lab))
        except Exception as e:  # noqa: BLE001
            print(f"{lab}: FAILED {type(e).__name__}: {e}")
            results.append({"case": lab, "failed": str(e)[:300]})
    save_json("exp5_timing_breakdown.json", {"experiment": "timing_breakdown", "cases": results})


if __name__ == "__main__":
    main()
