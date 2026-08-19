"""Sensitivity of the zero-flow operator-norm radius to its relative threshold."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from stability_radius.base_point.ac import (
    solve_ac_fpf_base_point,
    solve_ac_pf_base_point,
)
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.ac_l2 import compute_ac_l2_radius


def _auto_slack_bus(net) -> int:
    for index in sorted(net.ext_grid.index):
        if bool(net.ext_grid.loc[index].get("in_service", True)):
            return int(net.ext_grid.loc[index, "bus"])
    return int(sorted(net.bus.index)[0])


def _add_near_zero_flow_spur(net) -> int:
    """Add a scale-controlled near-zero-flow line to an otherwise standard case."""

    import pandapower as pp

    source_bus = int(sorted(net.bus.index)[0])
    vn_kv = float(net.bus.loc[source_bus, "vn_kv"])
    spur_bus = int(
        pp.create_bus(net, vn_kv=vn_kv, name="revision_near_zero_flow_bus")
    )
    line_id = int(
        pp.create_line_from_parameters(
            net,
            from_bus=source_bus,
            to_bus=spur_bus,
            length_km=1.0,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.2,
            c_nf_per_km=0.0,
            max_i_ka=1.0,
            name="revision_near_zero_flow_spur",
        )
    )
    pp.create_load(
        net,
        bus=spur_bus,
        p_mw=1.0e-7,
        q_mvar=0.6e-7,
        name="revision_near_zero_flow_load",
    )
    return line_id


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net = load_network(str(Path(args.data_dir) / args.case))
    synthetic_line_id = (
        _add_near_zero_flow_spur(net) if bool(args.synthetic_zero_flow_spur) else None
    )
    slack_bus = _auto_slack_bus(net)
    if bool(args.feasible_restoration):
        _base_point, base_pf = solve_ac_fpf_base_point(
            net=net,
            slack_bus=slack_bus,
            lossless=bool(args.lossless),
        )
    else:
        last_error = None
        for pf_init in ("dc", "flat"):
            try:
                _base_point, base_pf = solve_ac_pf_base_point(
                    net=net,
                    slack_bus=slack_bus,
                    pf_solver="pandapower",
                    pf_init=pf_init,
                    lossless=bool(args.lossless),
                    gen_dispatch_mw_by_name={},
                    distributed_slack=False,
                    trafo_model="pi",
                )
                break
            except Exception as exc:  # noqa: BLE001 - benchmark fallback accounting
                last_error = exc
        else:
            raise RuntimeError("raw base power flow failed for dc and flat starts") from last_error
    rows = []
    for tolerance in args.relative_tolerances:
        results = compute_ac_l2_radius(
            net,
            base_pf=base_pf,
            slack_bus=slack_bus,
            chunk_size=int(args.chunk_size),
            balance=True,
            lossless=bool(args.lossless),
            return_timings=True,
            zero_flow_rel_tol=float(tolerance),
        )
        timings = results.pop("_timings")
        line_rows = [
            row for key, row in results.items() if key.startswith("line_")
        ]
        valid_radii = [
            float(row["certificate_radius_ac_l2"])
            for row in line_rows
            if math.isfinite(float(row["certificate_radius_ac_l2"]))
            and float(row["certificate_radius_ac_l2"]) >= 0.0
        ]
        rows.append(
            {
                "case": Path(args.case).stem,
                "lossless": bool(args.lossless),
                "base_point": "feasible_restoration"
                if bool(args.feasible_restoration)
                else "raw_initialization",
                "synthetic_near_zero_flow_line": synthetic_line_id
                if synthetic_line_id is not None
                else "",
                "zero_flow_relative_tolerance": float(tolerance),
                "zero_flow_operator_norm_ends": int(
                    timings.get("zero_flow_operator_norm_ends", 0)
                ),
                "unresolved_nondifferentiable_binding_lines": int(
                    sum(
                        1
                        for row in line_rows
                        if bool(row.get("nondifferentiable_apparent_power", False))
                        and not bool(row.get("zero_flow_operator_norm_certified", False))
                    )
                ),
                "minimum_all_constraint_radius": float(min(valid_radii))
                if valid_radii
                else float("nan"),
                "median_radius": float(np.median(valid_radii))
                if valid_radii
                else float("nan"),
                "total_time_sec": float(timings.get("total_sec", float("nan"))),
            }
        )
    with (output_dir / "zero_flow_threshold_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "zero_flow_threshold_sensitivity.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/input")
    parser.add_argument("--case", default="pglib_opf_case2000_goc.m")
    parser.add_argument("--output-dir", default="run_artifacts/revision1_zero_flow")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--relative-tolerances", nargs="+", type=float, default=[1e-8, 1e-10, 1e-12]
    )
    parser.add_argument(
        "--lossless", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--feasible-restoration", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--synthetic-zero-flow-spur",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
