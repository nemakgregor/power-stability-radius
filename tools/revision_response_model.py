"""Bounded participation-response example for a physically realizable direction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from stability_radius.base_point.ac import solve_ac_pf_base_point
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.ac_l2 import compute_ac_l2_radius
from stability_radius.workflows import expand_h_reduced_to_full


def _auto_slack_bus(net) -> int:
    for index in sorted(net.ext_grid.index):
        if bool(net.ext_grid.loc[index].get("in_service", True)):
            return int(net.ext_grid.loc[index, "bus"])
    return int(sorted(net.bus.index)[0])


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    net = load_network(str(Path(args.data_dir) / args.case))
    slack_bus = _auto_slack_bus(net)
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
    )
    h_data = results.pop("_h_vectors")
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    bus_pos = {bus_id: pos for pos, bus_id in enumerate(bus_ids)}
    slack_pos = bus_pos[slack_bus]
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
    line_ids = [int(x) for x in sorted(net.line.index)]
    candidates = []
    for pos, line_id in enumerate(line_ids):
        row = results[f"line_{line_id}"]
        radius = float(row.get("certificate_radius_ac_l2", float("nan")))
        if math.isfinite(radius) and radius > 0.0:
            candidates.append((radius, pos, line_id, str(row["binding_end"]), row))
    if not candidates:
        raise RuntimeError("No finite positive line radius is available.")
    _radius, line_pos, line_id, line_end, line_row = min(candidates)
    h = h_from[line_pos] if line_end == "from" else h_to[line_pos]

    loads = [
        row
        for _, row in net.load.iterrows()
        if bool(row.get("in_service", True)) and int(row["bus"]) in bus_pos
    ]
    n_load = len(loads)
    primitive_covariance = np.zeros((2 * n_load, 2 * n_load), dtype=float)
    primitive_bounds = np.zeros(2 * n_load, dtype=float)
    rho = float(args.pq_correlation)
    for index, row in enumerate(loads):
        p = abs(float(row.get("p_mw", 0.0)))
        q = abs(float(row.get("q_mvar", 0.0)))
        sigma_p = max(0.5, float(args.sigma_fraction) * p)
        sigma_q = max(0.25, float(args.sigma_fraction) * q)
        primitive_covariance[index, index] = sigma_p**2
        primitive_covariance[n_load + index, n_load + index] = sigma_q**2
        primitive_covariance[index, n_load + index] = rho * sigma_p * sigma_q
        primitive_covariance[n_load + index, index] = rho * sigma_p * sigma_q
        primitive_bounds[index] = max(1.0, float(args.bound_fraction) * p)
        primitive_bounds[n_load + index] = max(0.5, float(args.bound_fraction) * q)

    generator_data = []
    for index, row in net.gen.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        bus_id = int(row["bus"])
        if bus_id not in bus_pos:
            continue
        dispatch = float(row.get("p_mw", 0.0))
        upward = max(float(row.get("max_p_mw", dispatch)) - dispatch, 0.0)
        downward = max(dispatch - float(row.get("min_p_mw", 0.0)), 0.0)
        if upward > 0.0 or downward > 0.0:
            generator_data.append((int(index), bus_id, upward, downward))
    if not generator_data:
        raise RuntimeError("No generator with finite active-power headroom was found.")
    participation_weights = np.asarray(
        [max(upward, 0.0) for _, _, upward, _ in generator_data], dtype=float
    )
    if float(np.sum(participation_weights)) <= 0.0:
        participation_weights = np.ones(len(generator_data), dtype=float)
    participation_weights /= float(np.sum(participation_weights))

    response = np.zeros((2 * len(bus_ids), 2 * n_load), dtype=float)
    for load_index, row in enumerate(loads):
        load_position = bus_pos[int(row["bus"])]
        response[load_position, load_index] -= 1.0
        response[len(bus_ids) + load_position, n_load + load_index] -= 1.0
        for (_, generator_bus, _, _), alpha in zip(
            generator_data, participation_weights
        ):
            response[bus_pos[generator_bus], load_index] += float(alpha)

    scalar_response = np.asarray(h, dtype=float) @ response
    sigma_flow_sq = float(
        scalar_response @ primitive_covariance @ scalar_response.T
    )
    sigma_flow = math.sqrt(max(sigma_flow_sq, 0.0))
    margin = float(line_row["margin_ac_mva"])
    thermal_radius = margin / sigma_flow if sigma_flow > 0.0 else float("inf")
    covariance_direction = (
        primitive_covariance @ scalar_response.T / sigma_flow
        if sigma_flow > 0.0
        else np.zeros(2 * n_load)
    )

    bound_radii = []
    for index, bound in enumerate(primitive_bounds):
        coordinate_sigma = math.sqrt(max(float(primitive_covariance[index, index]), 0.0))
        if coordinate_sigma > 0.0:
            bound_radii.append(
                (float(bound / coordinate_sigma), f"primitive_bound_{index}")
            )
    active_sum_row = np.concatenate(
        [np.ones(n_load, dtype=float), np.zeros(n_load, dtype=float)]
    )
    total_error_sigma = math.sqrt(
        max(float(active_sum_row @ primitive_covariance @ active_sum_row.T), 0.0)
    )
    for (generator_index, _bus, upward, downward), alpha in zip(
        generator_data, participation_weights
    ):
        response_sigma = float(alpha) * total_error_sigma
        if response_sigma <= 0.0:
            continue
        if upward > 0.0:
            bound_radii.append(
                (float(upward / response_sigma), f"generator_{generator_index}_up")
            )
        if downward > 0.0:
            bound_radii.append(
                (float(downward / response_sigma), f"generator_{generator_index}_down")
            )
    bound_radius, binding_bound = min(bound_radii)
    physically_admissible_radius = min(float(thermal_radius), float(bound_radius))
    primitive_direction = physically_admissible_radius * covariance_direction
    injection_direction = response @ primitive_direction

    largest = np.argsort(np.abs(injection_direction))[::-1][:10]
    contribution_rows = []
    for rank, index in enumerate(largest, start=1):
        block = "P" if int(index) < len(bus_ids) else "Q"
        position = int(index) if block == "P" else int(index) - len(bus_ids)
        contribution_rows.append(
            {
                "rank": rank,
                "bus_id": int(bus_ids[position]),
                "component": block,
                "injection_change": float(injection_direction[index]),
            }
        )
    with (output_dir / "bounded_response_top_bus_contributions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contribution_rows[0]))
        writer.writeheader()
        writer.writerows(contribution_rows)

    summary = {
        "case": Path(args.case).stem,
        "line_id": int(line_id),
        "line_end": line_end,
        "active_reactive_correlation": rho,
        "generator_participation_factors": [
            {
                "generator_index": int(data[0]),
                "bus_id": int(data[1]),
                "factor": float(alpha),
                "upward_headroom_mw": float(data[2]),
                "downward_headroom_mw": float(data[3]),
            }
            for data, alpha in zip(generator_data, participation_weights)
        ],
        "load_and_reactive_bounds": int(primitive_bounds.size),
        "thermal_radius_standard_deviations": float(thermal_radius),
        "bound_radius_standard_deviations": float(bound_radius),
        "binding_physical_bound": str(binding_bound),
        "physically_admissible_radius_standard_deviations": float(
            physically_admissible_radius
        ),
        "worst_direction_is_physically_realizable": bool(
            physically_admissible_radius <= bound_radius + 1e-12
        ),
        "maximum_primitive_bound_utilization": float(
            np.max(np.abs(primitive_direction) / primitive_bounds)
        ),
        "net_active_injection_mw": float(np.sum(injection_direction[: len(bus_ids)])),
        "top_bus_contributions": contribution_rows,
    }
    (output_dir / "bounded_response_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/input")
    parser.add_argument("--case", default="pglib_opf_case118_ieee.m")
    parser.add_argument("--output-dir", default="run_artifacts/revision1_response_model")
    parser.add_argument("--pq-correlation", type=float, default=0.6)
    parser.add_argument("--sigma-fraction", type=float, default=0.02)
    parser.add_argument("--bound-fraction", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
