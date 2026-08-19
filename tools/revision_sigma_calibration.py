"""Line-wise affine-variance calibration on original lossy PGLib systems."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from stability_radius.base_point.ac import solve_ac_pf_base_point
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.ac_l2 import compute_ac_l2_radius
from stability_radius.radii.common import estimate_line_limit_mva_with_flag
from stability_radius.workflows import expand_h_reduced_to_full


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _auto_slack_bus(net: Any) -> int:
    for index in sorted(net.ext_grid.index):
        if bool(net.ext_grid.loc[index].get("in_service", True)):
            return int(net.ext_grid.loc[index, "bus"])
    return int(sorted(net.bus.index)[0])


def _response_model(
    *,
    net: Any,
    bus_ids: list[int],
    pq_mask: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return response map T, primitive covariance, and model metadata."""
    n_bus = len(bus_ids)
    bus_pos = {bus_id: pos for pos, bus_id in enumerate(bus_ids)}
    q_positions = np.where(np.asarray(pq_mask, dtype=bool))[0]

    if model == "conditioned_balanced":
        dimension = n_bus + len(q_positions)
        covariance = np.eye(dimension, dtype=float)
        p_scale = np.asarray(
            [max(0.5, 0.015 * abs(float(net.res_bus.loc[bus_id, "p_mw"]))) for bus_id in bus_ids],
            dtype=float,
        )
        q_scale = np.asarray(
            [max(0.25, 0.015 * abs(float(net.res_bus.loc[bus_ids[pos], "q_mvar"]))) for pos in q_positions],
            dtype=float,
        )
        scale = np.concatenate([p_scale, q_scale])
        covariance = np.diag(scale * scale)
        balance_rows = [
            np.concatenate(
                [np.ones(n_bus, dtype=float), np.zeros(len(q_positions), dtype=float)]
            )
        ]
        if len(q_positions):
            balance_rows.append(
                np.concatenate(
                    [
                        np.zeros(n_bus, dtype=float),
                        np.ones(len(q_positions), dtype=float),
                    ]
                )
            )
        balance = np.vstack(balance_rows)
        middle = balance @ covariance @ balance.T
        conditional_map = (
            np.eye(dimension, dtype=float)
            - covariance @ balance.T @ np.linalg.solve(middle, balance)
        )
        embed = np.zeros((2 * n_bus, dimension), dtype=float)
        embed[:n_bus, :n_bus] = np.eye(n_bus)
        for local, pos in enumerate(q_positions):
            embed[n_bus + int(pos), n_bus + local] = 1.0
        response = embed @ conditional_map
        return response, covariance, {
            "model": model,
            "mechanism": "Gaussian vector conditioned by blockwise balance projection",
            "active_reactive_correlation": 0.0,
        }

    if model != "participation_correlated":
        raise ValueError(f"Unknown response model: {model}")

    load_rows = [
        row
        for _, row in net.load.iterrows()
        if bool(row.get("in_service", True)) and int(row["bus"]) in bus_pos
    ]
    if not load_rows:
        raise ValueError("Participation model requires in-service loads.")
    primitive_dim = 2 * len(load_rows)
    covariance = np.zeros((primitive_dim, primitive_dim), dtype=float)
    rho = 0.6
    for index, row in enumerate(load_rows):
        sigma_p = max(0.5, 0.02 * abs(float(row.get("p_mw", 0.0))))
        sigma_q = max(0.25, 0.02 * abs(float(row.get("q_mvar", 0.0))))
        covariance[index, index] = sigma_p**2
        covariance[len(load_rows) + index, len(load_rows) + index] = sigma_q**2
        covariance[index, len(load_rows) + index] = rho * sigma_p * sigma_q
        covariance[len(load_rows) + index, index] = rho * sigma_p * sigma_q

    generator_buses: list[int] = []
    headrooms: list[float] = []
    for _, row in net.gen.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        bus_id = int(row["bus"])
        if bus_id not in bus_pos:
            continue
        headroom = max(float(row.get("max_p_mw", 0.0)) - float(row.get("p_mw", 0.0)), 0.0)
        if headroom > 0.0:
            generator_buses.append(bus_id)
            headrooms.append(headroom)
    for _, row in net.ext_grid.iterrows():
        bus_id = int(row["bus"])
        if bus_id in bus_pos:
            generator_buses.append(bus_id)
            headrooms.append(max(sum(headrooms), 1.0))
    if not generator_buses:
        raise ValueError("Participation model requires responsive generators.")
    alpha = np.asarray(headrooms, dtype=float)
    alpha /= float(np.sum(alpha))

    response = np.zeros((2 * n_bus, primitive_dim), dtype=float)
    for load_index, row in enumerate(load_rows):
        load_bus_pos = int(bus_pos[int(row["bus"])])
        response[load_bus_pos, load_index] -= 1.0
        response[n_bus + load_bus_pos, len(load_rows) + load_index] -= 1.0
        for generator_bus, participation in zip(generator_buses, alpha):
            response[int(bus_pos[generator_bus]), load_index] += float(participation)
    return response, covariance, {
        "model": model,
        "mechanism": "physical generator participation response to load forecast errors",
        "active_reactive_correlation": rho,
        "generator_buses": generator_buses,
        "participation_factors": alpha.tolist(),
        "minimum_generator_headroom_mw": float(min(headrooms)),
    }


def _prepare_replay_net(net: Any, bus_ids: list[int]) -> tuple[Any, list[int]]:
    import pandapower as pp

    nn = copy.deepcopy(net)
    sgen_ids: list[int] = []
    for bus_id in bus_ids:
        sgen_ids.append(
            int(
                pp.create_sgen(
                    nn,
                    bus=bus_id,
                    p_mw=0.0,
                    q_mvar=0.0,
                    name=f"sigma_replay_bus_{bus_id}",
                    in_service=True,
                )
            )
        )
    return nn, sgen_ids


def _run_samples(
    *,
    net: Any,
    bus_ids: list[int],
    sgen_ids: list[int],
    samples: np.ndarray,
    line_ids: list[int],
    binding_ends: list[str],
) -> tuple[np.ndarray, int]:
    import pandapower as pp

    flows: list[np.ndarray] = []
    failures = 0
    n_bus = len(bus_ids)
    for delta in np.asarray(samples, dtype=float):
        net.sgen.loc[sgen_ids, "p_mw"] = delta[:n_bus]
        net.sgen.loc[sgen_ids, "q_mvar"] = delta[n_bus:]
        converged = False
        for init in ("results", "dc", "flat"):
            try:
                pp.runpp(
                    net,
                    algorithm="nr",
                    calculate_voltage_angles=True,
                    enforce_q_lims=True,
                    init=init,
                    max_iteration=200,
                    tolerance_mva=1e-8,
                )
                if bool(getattr(net, "converged", False)):
                    converged = True
                    break
            except Exception:
                continue
        if not converged:
            failures += 1
            continue
        values = np.empty(len(line_ids), dtype=float)
        for pos, (line_id, line_end) in enumerate(zip(line_ids, binding_ends)):
            prefix = "from" if line_end == "from" else "to"
            p = float(net.res_line.loc[line_id, f"p_{prefix}_mw"])
            q = float(net.res_line.loc[line_id, f"q_{prefix}_mvar"])
            values[pos] = math.hypot(p, q)
        flows.append(values)
    return np.asarray(flows, dtype=float), failures


def _case_model_rows(
    *,
    case_path: Path,
    model: str,
    seeds: list[int],
    samples_per_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    net = load_network(str(case_path))
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
    # Populate res_bus for load-scaled covariance construction.
    import pandapower as pp

    pp.runpp(net, calculate_voltage_angles=True, enforce_q_lims=True, init="dc")
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
    line_ids = [int(x) for x in sorted(net.line.index)]
    binding_ends = [str(results[f"line_{line_id}"]["binding_end"]) for line_id in line_ids]
    h_binding = np.asarray(
        [
            h_from[pos] if binding_ends[pos] == "from" else h_to[pos]
            for pos in range(len(line_ids))
        ],
        dtype=float,
    )
    base_flows = np.asarray(
        [
            float(results[f"line_{line_id}"][f"ac_s0_{binding_ends[pos]}_mva"])
            for pos, line_id in enumerate(line_ids)
        ],
        dtype=float,
    )
    limits = np.asarray(
        [
            float(estimate_line_limit_mva_with_flag(net, net.line.loc[line_id])[0])
            for line_id in line_ids
        ],
        dtype=float,
    )

    response, primitive_covariance, metadata = _response_model(
        net=net,
        bus_ids=bus_ids,
        pq_mask=np.asarray(h_data["pq_mask"], dtype=bool),
        model=model,
    )
    response_covariance = response @ primitive_covariance @ response.T
    analytical_variance = np.einsum(
        "ij,jk,ik->i", h_binding, response_covariance, h_binding
    )
    analytical_sd = np.sqrt(np.maximum(analytical_variance, 0.0))

    all_flows: list[np.ndarray] = []
    total_failures = 0
    replay_net, sgen_ids = _prepare_replay_net(net, bus_ids)
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        primitive = rng.multivariate_normal(
            np.zeros(primitive_covariance.shape[0]),
            primitive_covariance,
            size=int(samples_per_seed),
        )
        samples = primitive @ response.T
        flows, failures = _run_samples(
            net=replay_net,
            bus_ids=bus_ids,
            sgen_ids=sgen_ids,
            samples=samples,
            line_ids=line_ids,
            binding_ends=binding_ends,
        )
        if flows.size:
            all_flows.append(flows)
        total_failures += int(failures)
    nonlinear_flows = np.vstack(all_flows) if all_flows else np.empty((0, len(line_ids)))
    if nonlinear_flows.shape[0] < 2:
        raise RuntimeError("Too few converged sigma-calibration samples.")
    nonlinear_changes = nonlinear_flows - base_flows[None, :]
    empirical_sd = np.std(nonlinear_changes, axis=0, ddof=1)
    normal = NormalDist()

    rows: list[dict[str, Any]] = []
    for pos, line_id in enumerate(line_ids):
        predicted_prob = (
            1.0
            - normal.cdf((limits[pos] - base_flows[pos]) / analytical_sd[pos])
            if analytical_sd[pos] > 0.0 and math.isfinite(limits[pos])
            else 0.0
        )
        empirical_prob = float(np.mean(nonlinear_flows[:, pos] > limits[pos]))
        virtual_threshold = base_flows[pos] + 2.0 * analytical_sd[pos]
        virtual_exceedance = float(np.mean(nonlinear_flows[:, pos] > virtual_threshold))
        rows.append(
            {
                "case": case_path.stem,
                "response_model": model,
                "line_id": int(line_id),
                "binding_end": binding_ends[pos],
                "analytical_sd_mva": float(analytical_sd[pos]),
                "empirical_sd_mva": float(empirical_sd[pos]),
                "analytical_to_empirical_sd_ratio": float(
                    analytical_sd[pos] / empirical_sd[pos]
                )
                if empirical_sd[pos] > 0.0
                else float("nan"),
                "predicted_rating_exceedance_probability": float(predicted_prob),
                "empirical_rating_exceedance_probability": float(empirical_prob),
                "empirical_exceedance_at_affine_two_sigma_threshold": float(
                    virtual_exceedance
                ),
                "base_flow_mva": float(base_flows[pos]),
                "rating_mva": float(limits[pos]),
                "converged_samples": int(nonlinear_flows.shape[0]),
                "failed_samples": int(total_failures),
            }
        )
    metadata.update(
        {
            "case": case_path.stem,
            "seeds": seeds,
            "samples_per_seed": int(samples_per_seed),
            "attempted_samples": int(len(seeds) * int(samples_per_seed)),
            "converged_samples": int(nonlinear_flows.shape[0]),
            "failed_samples": int(total_failures),
            "any_line_empirical_probability": float(
                np.mean(np.any(nonlinear_flows > limits[None, :], axis=1))
            ),
            "bonferroni_sum_of_marginals": float(
                min(1.0, sum(float(row["predicted_rating_exceedance_probability"]) for row in rows))
            ),
        }
    )
    return rows, metadata


def _summary(rows: list[dict[str, Any]], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = np.asarray(
        [
            float(row["analytical_to_empirical_sd_ratio"])
            for row in rows
            if math.isfinite(float(row["analytical_to_empirical_sd_ratio"]))
        ],
        dtype=float,
    )
    two_sigma = np.asarray(
        [float(row["empirical_exceedance_at_affine_two_sigma_threshold"]) for row in rows],
        dtype=float,
    )
    return {
        "case_models": len(metadata),
        "line_calibrations": len(rows),
        "median_analytical_to_empirical_sd_ratio": float(np.median(ratios)),
        "p05_sd_ratio": float(np.percentile(ratios, 5)),
        "p95_sd_ratio": float(np.percentile(ratios, 95)),
        "minimum_sd_ratio": float(np.min(ratios)),
        "maximum_sd_ratio": float(np.max(ratios)),
        "mean_empirical_exceedance_at_affine_two_sigma_threshold": float(
            np.mean(two_sigma)
        ),
        "nominal_one_sided_two_sigma_probability": 1.0 - NormalDist().cdf(2.0),
        "configurations": metadata,
    }


def _plot(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    models = sorted({str(row["response_model"]) for row in rows})
    for model in models:
        subset = [row for row in rows if row["response_model"] == model]
        ax.scatter(
            [float(row["analytical_sd_mva"]) for row in subset],
            [float(row["empirical_sd_mva"]) for row in subset],
            s=15,
            alpha=0.55,
            label=model.replace("_", " "),
        )
    maximum = max(
        max(float(row["analytical_sd_mva"]) for row in rows),
        max(float(row["empirical_sd_mva"]) for row in rows),
    )
    ax.plot([0.0, maximum], [0.0, maximum], "k--", linewidth=1.0)
    ax.set_xlabel("Analytical affine standard deviation (MVA)")
    ax.set_ylabel("Empirical nonlinear standard deviation (MVA)")
    ax.set_xlim(-0.15, maximum * 1.04)
    ax.set_ylim(-0.15, maximum * 1.04)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "sigma_calibration_scatter.pdf")
    fig.savefig(output_dir / "sigma_calibration_scatter.png", dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for case_name in args.cases:
        for model in args.response_models:
            try:
                case_rows, case_metadata = _case_model_rows(
                    case_path=Path(args.data_dir) / case_name,
                    model=str(model),
                    seeds=[int(seed) for seed in args.seeds],
                    samples_per_seed=int(args.samples_per_seed),
                )
                rows.extend(case_rows)
                metadata.append(case_metadata)
            except Exception as exc:  # noqa: BLE001 - complete outcome accounting
                errors.append(
                    {
                        "case": Path(case_name).stem,
                        "response_model": str(model),
                        "error": repr(exc),
                    }
                )
    _write_csv(output_dir / "sigma_line_calibration.csv", rows)
    _write_csv(output_dir / "errors.csv", errors)
    if rows:
        summary = _summary(rows, metadata)
        _plot(output_dir, rows)
    else:
        summary = {"case_models": 0, "line_calibrations": 0}
    summary["errors"] = errors
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sigma_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/input")
    parser.add_argument("--output-dir", default="run_artifacts/revision1_sigma_calibration")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "pglib_opf_case14_ieee.m",
            "pglib_opf_case30_ieee.m",
            "pglib_opf_case118_ieee.m",
        ],
    )
    parser.add_argument(
        "--response-models",
        nargs="+",
        default=["conditioned_balanced", "participation_correlated"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202])
    parser.add_argument("--samples-per-seed", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
