"""Repeated stage timings and peak resident-memory measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import time
from pathlib import Path

import numpy as np

from stability_radius.base_point.ac import solve_ac_pf_base_point
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.ac_l2 import compute_ac_l2_radius


def _auto_slack_bus(net) -> int:
    for index in sorted(net.ext_grid.index):
        if bool(net.ext_grid.loc[index].get("in_service", True)):
            return int(net.ext_grid.loc[index, "bus"])
    return int(sorted(net.bus.index)[0])


def _maximum_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes. The Docker experiment is Linux.
    return value / 1024.0


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_name in args.cases:
        for repetition in range(int(args.repetitions)):
            net = load_network(str(Path(args.data_dir) / case_name))
            slack_bus = _auto_slack_bus(net)
            rss_before = _maximum_rss_mb()
            pf_start = time.perf_counter()
            _base_point, base_pf = solve_ac_pf_base_point(
                net=net,
                slack_bus=slack_bus,
                pf_solver="pandapower",
                pf_init="dc",
                lossless=False,
                gen_dispatch_mw_by_name={},
                distributed_slack=False,
                trafo_model="pi",
            )
            pf_sec = time.perf_counter() - pf_start
            results = compute_ac_l2_radius(
                net,
                base_pf=base_pf,
                slack_bus=slack_bus,
                chunk_size=int(args.chunk_size),
                balance=True,
                lossless=False,
                return_h_vectors=False,
                return_timings=True,
            )
            timings = results.pop("_timings")
            rows.append(
                {
                    "case": Path(case_name).stem,
                    "repetition": int(repetition + 1),
                    "buses": int(timings["n_bus"]),
                    "line_ends": int(timings["n_line_ends"]),
                    "ac_power_flow_sec": float(pf_sec),
                    "operator_build_and_factorization_sec": float(
                        timings["operator_build_lu_sec"]
                    ),
                    "branch_gradient_rhs_sec": float(timings["line_gradient_rhs_sec"]),
                    "adjoint_solve_sec": float(timings["adjoint_solve_sec"]),
                    "support_postprocess_sec": float(timings["support_eval_sec"]),
                    "certificate_total_excluding_pf_sec": float(timings["total_sec"]),
                    "peak_resident_memory_mb": float(_maximum_rss_mb()),
                    "incremental_peak_resident_memory_mb": float(
                        max(_maximum_rss_mb() - rss_before, 0.0)
                    ),
                    "chunk_size": int(timings["chunk_size"]),
                }
            )

    with (output_dir / "performance_repetitions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    fields = [
        "ac_power_flow_sec",
        "operator_build_and_factorization_sec",
        "branch_gradient_rhs_sec",
        "adjoint_solve_sec",
        "support_postprocess_sec",
        "certificate_total_excluding_pf_sec",
        "peak_resident_memory_mb",
        "incremental_peak_resident_memory_mb",
    ]
    for case in sorted({row["case"] for row in rows}):
        subset = [row for row in rows if row["case"] == case]
        summary = {
            "case": case,
            "repetitions": len(subset),
            "buses": int(subset[0]["buses"]),
            "line_ends": int(subset[0]["line_ends"]),
        }
        for field in fields:
            values = np.asarray([float(row[field]) for row in subset], dtype=float)
            summary[f"{field}_median"] = float(np.median(values))
            summary[f"{field}_minimum"] = float(np.min(values))
            summary[f"{field}_maximum"] = float(np.max(values))
        summary_rows.append(summary)
    with (output_dir / "performance_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output_dir / "performance_summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/input")
    parser.add_argument("--output-dir", default="run_artifacts/revision1_performance")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "pglib_opf_case14_ieee.m",
            "pglib_opf_case30_ieee.m",
            "pglib_opf_case118_ieee.m",
            "pglib_opf_case200_activ.m",
            "pglib_opf_case588_sdet.m",
        ],
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
