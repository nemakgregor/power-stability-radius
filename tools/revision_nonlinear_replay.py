"""Multi-case, multi-scale nonlinear replay for affine AC thermal radii."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from stability_radius.base_point.ac import (
    solve_ac_fpf_base_point,
    solve_ac_pf_base_point,
)
from stability_radius.base_point.pandapower_tools import (
    apply_opp_result_to_pandapower_net,
    detect_q_limit_events,
)
from stability_radius.geometry.balanced import make_ac_block_specs, worst_case_l2_direction
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.ac_l2 import compute_ac_l2_radius
from stability_radius.radii.common import estimate_line_limit_mva_with_flag
from stability_radius.workflows import expand_h_reduced_to_full


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _auto_slack_bus(net: Any) -> int:
    for index in sorted(net.ext_grid.index):
        row = net.ext_grid.loc[index]
        if bool(row.get("in_service", True)):
            return int(row["bus"])
    return int(sorted(net.bus.index)[0])


def _scaled_case(path: Path, load_scale: float) -> Any:
    net = load_network(str(path))
    if len(net.load):
        net.load.loc[:, "p_mw"] = net.load["p_mw"].astype(float) * float(load_scale)
        net.load.loc[:, "q_mvar"] = net.load["q_mvar"].astype(float) * float(load_scale)
    return net


def _base_point_and_radii(
    net: Any,
    *,
    slack_bus: int,
) -> tuple[Any, dict[str, dict[str, Any]], str]:
    """Use the raw PF point when feasible, otherwise restore an AC-feasible point."""

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
    results = compute_ac_l2_radius(
        net,
        base_pf=base_pf,
        slack_bus=slack_bus,
        chunk_size=64,
        balance=True,
        lossless=False,
        return_h_vectors=True,
        return_timings=False,
    )
    base_infeasible = [
        key
        for key, row in results.items()
        if key.startswith("line_")
        and not bool(row.get("is_unconstrained", False))
        and (
            float(row.get("ac_margin_from_mva", float("inf"))) < -1e-7
            or float(row.get("ac_margin_to_mva", float("inf"))) < -1e-7
        )
    ]
    if not base_infeasible:
        return base_pf, results, "raw_pglib_power_flow"

    _base_point, base_pf = solve_ac_fpf_base_point(
        net=net,
        slack_bus=slack_bus,
        lossless=False,
    )
    apply_opp_result_to_pandapower_net(
        net,
        opp_gen_dispatch=getattr(base_pf, "opp_gen_dispatch", None) or {},
        opp_vm_pu=getattr(base_pf, "opp_vm_pu", None) or {},
    )
    results = compute_ac_l2_radius(
        net,
        base_pf=base_pf,
        slack_bus=slack_bus,
        chunk_size=64,
        balance=True,
        lossless=False,
        return_h_vectors=True,
        return_timings=False,
    )
    remaining_infeasible = [
        key
        for key, row in results.items()
        if key.startswith("line_")
        and not bool(row.get("is_unconstrained", False))
        and (
            float(row.get("ac_margin_from_mva", float("inf"))) < -1e-7
            or float(row.get("ac_margin_to_mva", float("inf"))) < -1e-7
        )
    ]
    if remaining_infeasible:
        raise RuntimeError(
            "restored_base_infeasible:"
            f"{len(remaining_infeasible)} monitored lines have negative margins"
        )
    return base_pf, results, "ac_opf_feasible_restoration"


def _apply_delta_and_solve(net: Any, delta_u: np.ndarray) -> tuple[Any, bool, str]:
    import pandapower as pp

    nn = copy.deepcopy(net)
    bus_ids = [int(x) for x in sorted(nn.bus.index)]
    n_bus = len(bus_ids)
    delta = np.asarray(delta_u, dtype=float).reshape(2 * n_bus)
    for pos, bus_id in enumerate(bus_ids):
        dp = float(delta[pos])
        dq = float(delta[n_bus + pos])
        if dp == 0.0 and dq == 0.0:
            continue
        pp.create_sgen(
            nn,
            bus=bus_id,
            p_mw=dp,
            q_mvar=dq,
            name=f"revision_replay_bus_{bus_id}",
            in_service=True,
        )
    last_error = ""
    for init in ("results", "dc", "flat"):
        try:
            pp.runpp(
                nn,
                algorithm="nr",
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                init=init,
                max_iteration=300,
                tolerance_mva=1e-9,
            )
            if bool(getattr(nn, "converged", False)):
                return nn, True, init
        except Exception as exc:  # noqa: BLE001 - experiment accounting
            last_error = repr(exc)
    return nn, False, last_error


def _replay_record(
    *,
    net: Any,
    delta_boundary: np.ndarray,
    scale: float,
    limits: dict[int, float],
    base_q_limit_buses: set[int],
) -> dict[str, Any]:
    nn, converged, solve_status = _apply_delta_and_solve(
        net, np.asarray(delta_boundary, dtype=float) * float(scale)
    )
    record: dict[str, Any] = {
        "scale": float(scale),
        "pf_converged": bool(converged),
        "solve_status": str(solve_status),
    }
    if not converged:
        record.update(
            {
                "thermal_violation": False,
                "voltage_violation": False,
                "q_limit_count": -1,
                "pv_pq_switch": False,
            }
        )
        return record

    worst_ratio = -float("inf")
    limiting_line = -1
    limiting_end = ""
    minimum_margin = float("inf")
    for line_id, limit in limits.items():
        for line_end, p_col, q_col in (
            ("from", "p_from_mw", "q_from_mvar"),
            ("to", "p_to_mw", "q_to_mvar"),
        ):
            p = float(nn.res_line.loc[line_id, p_col])
            q = float(nn.res_line.loc[line_id, q_col])
            apparent = math.hypot(p, q)
            ratio = apparent / float(limit)
            margin = float(limit) - apparent
            if ratio > worst_ratio:
                worst_ratio = ratio
                limiting_line = int(line_id)
                limiting_end = line_end
            minimum_margin = min(minimum_margin, margin)

    vm = nn.res_bus["vm_pu"].astype(float)
    min_bounds = nn.bus.get("min_vm_pu", 0.9)
    max_bounds = nn.bus.get("max_vm_pu", 1.1)
    if np.isscalar(min_bounds):
        min_values = np.full(len(nn.bus), float(min_bounds))
    else:
        min_values = np.asarray(min_bounds, dtype=float)
    if np.isscalar(max_bounds):
        max_values = np.full(len(nn.bus), float(max_bounds))
    else:
        max_values = np.asarray(max_bounds, dtype=float)
    min_values = np.where(np.isfinite(min_values), min_values, 0.9)
    max_values = np.where(np.isfinite(max_values), max_values, 1.1)
    voltage_count = int(np.sum((vm.to_numpy() < min_values) | (vm.to_numpy() > max_values)))

    q_events = detect_q_limit_events(nn)
    q_buses = {int(event["bus"]) for event in q_events if int(event.get("bus", -1)) >= 0}
    record.update(
        {
            "thermal_violation": bool(worst_ratio > 1.0 + 1e-8),
            "maximum_thermal_ratio": float(worst_ratio),
            "minimum_thermal_margin_mva": float(minimum_margin),
            "actual_limiting_line": int(limiting_line),
            "actual_limiting_end": str(limiting_end),
            "voltage_violation": bool(voltage_count > 0),
            "voltage_violation_count": int(voltage_count),
            "minimum_voltage_pu": float(vm.min()),
            "maximum_voltage_pu": float(vm.max()),
            "q_limit_count": int(len(q_events)),
            "pv_pq_switch": bool(q_buses != base_q_limit_buses),
        }
    )
    return record


def _first_thermal_violation_scale(
    *,
    net: Any,
    delta_boundary: np.ndarray,
    limits: dict[int, float],
    base_q_limit_buses: set[int],
    fixed_records: list[dict[str, Any]],
    tolerance: float,
    max_iterations: int,
) -> tuple[float, dict[str, Any] | None]:
    lower = 0.0
    upper = float("nan")
    upper_record: dict[str, Any] | None = None
    for record in fixed_records:
        if not bool(record["pf_converged"]):
            continue
        if bool(record["thermal_violation"]):
            upper = float(record["scale"])
            upper_record = record
            break
        lower = max(lower, float(record["scale"]))
    if not math.isfinite(upper):
        return float("inf"), None
    for _ in range(int(max_iterations)):
        if upper - lower <= float(tolerance):
            break
        midpoint = 0.5 * (lower + upper)
        record = _replay_record(
            net=net,
            delta_boundary=delta_boundary,
            scale=midpoint,
            limits=limits,
            base_q_limit_buses=base_q_limit_buses,
        )
        if bool(record["pf_converged"]) and bool(record["thermal_violation"]):
            upper = midpoint
            upper_record = record
        else:
            lower = midpoint
    return float(upper), upper_record


def _case_snapshot(
    *,
    case_path: Path,
    load_scale: float,
    top_k: int,
    fixed_scales: list[float],
    bisection_tolerance: float,
    bisection_iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    net = _scaled_case(case_path, load_scale)
    slack_bus = _auto_slack_bus(net)
    base_pf, results, base_point_source = _base_point_and_radii(
        net,
        slack_bus=slack_bus,
    )
    h_data = results.pop("_h_vectors")
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    slack_pos = bus_ids.index(slack_bus)
    h_from = expand_h_reduced_to_full(
        h_data["h_from"],
        n_bus=len(bus_ids),
        slack_pos=slack_pos,
        pq_mask=h_data["pq_mask"],
    )
    h_to = expand_h_reduced_to_full(
        h_data["h_to"],
        n_bus=len(bus_ids),
        slack_pos=slack_pos,
        pq_mask=h_data["pq_mask"],
    )
    q_bus_indices = np.where(np.asarray(h_data["pq_mask"], dtype=bool))[0]
    blocks = make_ac_block_specs(
        len(bus_ids), balance=True, q_bus_indices=q_bus_indices
    )
    line_ids = [int(x) for x in sorted(net.line.index)]
    line_pos = {line_id: pos for pos, line_id in enumerate(line_ids)}
    limits = {
        line_id: float(estimate_line_limit_mva_with_flag(net, net.line.loc[line_id])[0])
        for line_id in line_ids
    }
    base_q_limit_buses = {
        int(event["bus"])
        for event in (getattr(base_pf, "q_limit_events", ()) or ())
        if int(event.get("bus", -1)) >= 0
    }

    candidates: list[tuple[float, int, str]] = []
    affine_constraints: list[tuple[int, str, float, np.ndarray]] = []
    for line_id in line_ids:
        row = results[f"line_{line_id}"]
        if bool(row.get("is_unconstrained", False)):
            continue
        pos = line_pos[line_id]
        for line_end in ("from", "to"):
            radius = float(row.get(f"radius_ac_l2_{line_end}", float("nan")))
            if math.isfinite(radius) and radius > 0.0:
                candidates.append((radius, line_id, line_end))
            margin = float(row[f"ac_margin_{line_end}_mva"])
            h_vector = h_from[pos] if line_end == "from" else h_to[pos]
            if margin >= 0.0:
                affine_constraints.append((line_id, line_end, margin, h_vector))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    trajectory_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for rank, (seed_radius, line_id, line_end) in enumerate(
        candidates[: int(top_k)], start=1
    ):
        pos = line_pos[line_id]
        h_vector = h_from[pos] if line_end == "from" else h_to[pos]
        direction = worst_case_l2_direction(h_vector, blocks)
        directional_boundaries = []
        for constraint_line, constraint_end, margin, constraint_h in affine_constraints:
            slope = float(np.dot(constraint_h, direction))
            if slope > 1e-12:
                directional_boundaries.append(
                    (margin / slope, constraint_line, constraint_end)
                )
        if not directional_boundaries:
            continue
        affine_distance, predicted_line, predicted_end = min(
            directional_boundaries, key=lambda item: (item[0], item[1], item[2])
        )
        delta_boundary = np.asarray(direction, dtype=float) * float(affine_distance)
        fixed_records = [
            _replay_record(
                net=net,
                delta_boundary=delta_boundary,
                scale=scale,
                limits=limits,
                base_q_limit_buses=base_q_limit_buses,
            )
            for scale in fixed_scales
        ]
        for record in fixed_records:
            trajectory_rows.append(
                {
                    "case": case_path.stem,
                    "load_scale": float(load_scale),
                    "base_point_source": base_point_source,
                    "rank": int(rank),
                    "seed_line": int(line_id),
                    "seed_end": line_end,
                    "seed_line_radius_mva": float(seed_radius),
                    "predicted_line": int(predicted_line),
                    "predicted_end": predicted_end,
                    "affine_radius_mva": float(affine_distance),
                    **record,
                }
            )
        gamma, violation_record = _first_thermal_violation_scale(
            net=net,
            delta_boundary=delta_boundary,
            limits=limits,
            base_q_limit_buses=base_q_limit_buses,
            fixed_records=fixed_records,
            tolerance=bisection_tolerance,
            max_iterations=bisection_iterations,
        )
        direction_rows.append(
            {
                "case": case_path.stem,
                "load_scale": float(load_scale),
                "base_point_source": base_point_source,
                "rank": int(rank),
                "seed_line": int(line_id),
                "seed_end": line_end,
                "seed_line_radius_mva": float(seed_radius),
                "predicted_line": int(predicted_line),
                "predicted_end": predicted_end,
                "affine_radius_mva": float(affine_distance),
                "nonlinear_to_affine_distance_ratio": float(gamma),
                "censored_above_max_scale": bool(math.isinf(gamma)),
                "actual_limiting_line": int(violation_record["actual_limiting_line"])
                if violation_record
                else -1,
                "actual_limiting_end": str(violation_record["actual_limiting_end"])
                if violation_record
                else "",
                "limiting_constraint_match": bool(
                    violation_record
                    and int(violation_record["actual_limiting_line"]) == int(line_id)
                    and str(violation_record["actual_limiting_end"]) == line_end
                ),
                "voltage_violation_at_thermal_boundary": bool(
                    violation_record and violation_record["voltage_violation"]
                ),
                "pv_pq_switch_at_thermal_boundary": bool(
                    violation_record and violation_record["pv_pq_switch"]
                ),
                "pf_converged_at_thermal_boundary": bool(
                    violation_record and violation_record["pf_converged"]
                ),
            }
        )
    return trajectory_rows, direction_rows


def _summary(direction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = np.asarray(
        [
            float(row["nonlinear_to_affine_distance_ratio"])
            for row in direction_rows
            if math.isfinite(float(row["nonlinear_to_affine_distance_ratio"]))
        ],
        dtype=float,
    )
    return {
        "directions": len(direction_rows),
        "finite_violation_distances": int(finite.size),
        "censored_above_max_scale": int(len(direction_rows) - finite.size),
        "median_ratio": float(np.median(finite)) if finite.size else float("nan"),
        "p05_ratio": float(np.percentile(finite, 5)) if finite.size else float("nan"),
        "p25_ratio": float(np.percentile(finite, 25)) if finite.size else float("nan"),
        "p75_ratio": float(np.percentile(finite, 75)) if finite.size else float("nan"),
        "p95_ratio": float(np.percentile(finite, 95)) if finite.size else float("nan"),
        "minimum_ratio": float(np.min(finite)) if finite.size else float("nan"),
        "maximum_ratio": float(np.max(finite)) if finite.size else float("nan"),
        "nonconservative_count": int(np.sum(finite < 1.0)) if finite.size else 0,
        "conservative_count": int(np.sum(finite > 1.0)) if finite.size else 0,
        "constraint_match_rate": float(
            np.mean([bool(row["limiting_constraint_match"]) for row in direction_rows])
        )
        if direction_rows
        else float("nan"),
    }


def _plot(output_dir: Path, direction_rows: list[dict[str, Any]]) -> None:
    if not direction_rows:
        return
    preferred_order = [
        "pglib_opf_case14_ieee",
        "pglib_opf_case30_ieee",
        "pglib_opf_case118_ieee",
        "pglib_opf_case200_activ",
    ]
    present = {str(row["case"]) for row in direction_rows}
    cases = [case for case in preferred_order if case in present]
    cases.extend(sorted(present.difference(cases)))
    labels = {
        "pglib_opf_case14_ieee": "14-bus",
        "pglib_opf_case30_ieee": "30-bus",
        "pglib_opf_case118_ieee": "118-bus",
        "pglib_opf_case200_activ": "200-bus",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for position, case in enumerate(cases):
        finite_values = [
            float(row["nonlinear_to_affine_distance_ratio"])
            for row in direction_rows
            if row["case"] == case
            and math.isfinite(float(row["nonlinear_to_affine_distance_ratio"]))
        ]
        ax.scatter(
            [position] * len(finite_values),
            finite_values,
            alpha=0.72,
            s=28,
            color=colors[position % len(colors)],
        )
        censored_count = sum(
            bool(row["censored_above_max_scale"])
            for row in direction_rows
            if row["case"] == case
        )
        if censored_count:
            offsets = np.linspace(-0.12, 0.12, censored_count)
            ax.scatter(
                position + offsets,
                np.full(censored_count, 1.035),
                marker="^",
                s=30,
                facecolors="none",
                edgecolors=colors[position % len(colors)],
                linewidths=1.0,
            )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Nonlinear / affine thermal-violation distance")
    ax.set_xlabel("Original lossy benchmark")
    ax.set_xticks(range(len(cases)), [labels.get(case, case) for case in cases])
    ax.set_ylim(0.79, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "nonlinear_affine_distance_ratios.pdf")
    fig.savefig(output_dir / "nonlinear_affine_distance_ratios.png", dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    trajectory_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    fixed_scales = [float(value) for value in args.scales]
    errors: list[dict[str, Any]] = []
    for case_name in args.cases:
        for load_scale in args.load_scales:
            try:
                trajectories, directions = _case_snapshot(
                    case_path=Path(args.data_dir) / case_name,
                    load_scale=float(load_scale),
                    top_k=int(args.top_k),
                    fixed_scales=fixed_scales,
                    bisection_tolerance=float(args.bisection_tolerance),
                    bisection_iterations=int(args.bisection_iterations),
                )
                trajectory_rows.extend(trajectories)
                direction_rows.extend(directions)
            except Exception as exc:  # noqa: BLE001 - complete outcome accounting
                errors.append(
                    {
                        "case": Path(case_name).stem,
                        "load_scale": float(load_scale),
                        "error": repr(exc),
                    }
                )
    _write_csv(output_dir / "nonlinear_replay_trajectories.csv", trajectory_rows)
    _write_csv(output_dir / "nonlinear_violation_distances.csv", direction_rows)
    _write_csv(output_dir / "errors.csv", errors)
    summary = _summary(direction_rows)
    summary.update(
        {
            "fixed_scales": fixed_scales,
            "cases": list(args.cases),
            "load_scales": [float(value) for value in args.load_scales],
            "errors": errors,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nonlinear_replay_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot(output_dir, direction_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/input")
    parser.add_argument("--output-dir", default="run_artifacts/revision1_nonlinear")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "pglib_opf_case14_ieee.m",
            "pglib_opf_case30_ieee.m",
            "pglib_opf_case118_ieee.m",
            "pglib_opf_case200_activ.m",
        ],
    )
    parser.add_argument("--load-scales", nargs="+", type=float, default=[0.9, 1.0, 1.1])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--scales", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    )
    parser.add_argument("--bisection-tolerance", type=float, default=2.5e-3)
    parser.add_argument("--bisection-iterations", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
