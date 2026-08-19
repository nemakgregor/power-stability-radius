"""Matched-cut lossy AC dispatch benchmarks and a preventive DC SCOPF baseline."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entry_points.n1_stability_demo import (
    _ac_n1_screen,
    _add_matpower_costs,
    _align_line_limit_proxy_with_opf_model,
    _apply_loading_limits,
    _extract_pypsa_result_from_pp,
    _iter_dispatchable_elements,
    _prepare_cost_opf_network,
    _set_default_voltage_bounds,
    _solve_cost_opf,
)
from stability_radius.dc.dc_model import build_dc_matrices
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.ac_l2 import compute_ac_l2_radius
from stability_radius.radii.ac_sigma_radius import compute_ac_sigma_radius
from stability_radius.radii.common import estimate_line_limit_mva_with_flag
from stability_radius.radii.nminus1 import (
    incidence_from_pandapower_net,
    lodf_from_ptdf,
    ptdf_for_line_transfers,
)
from stability_radius.workflows import expand_h_reduced_to_full


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _auto_slack_bus(net: Any) -> int:
    for index in sorted(net.ext_grid.index):
        if bool(net.ext_grid.loc[index].get("in_service", True)):
            return int(net.ext_grid.loc[index, "bus"])
    return int(sorted(net.bus.index)[0])


def _scaled_case(path: Path, load_scale: float) -> Any:
    net = load_network(str(path))
    if len(net.load):
        net.load.loc[:, "p_mw"] = net.load["p_mw"].astype(float) * load_scale
        net.load.loc[:, "q_mvar"] = net.load["q_mvar"].astype(float) * load_scale
    _align_line_limit_proxy_with_opf_model(net)
    return net


def _n1_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    converged = [row for row in records if bool(row.get("pf_converged", False))]
    passed = [row for row in converged if bool(row.get("n1_feasible", False))]
    overloaded = [row for row in converged if not bool(row.get("n1_feasible", False))]
    max_loading = [
        float(row["max_loading_percent"])
        for row in converged
        if math.isfinite(float(row.get("max_loading_percent", float("nan"))))
    ]
    return {
        "n1_total": len(records),
        "n1_pass": len(passed),
        "n1_converged_with_overload": len(overloaded),
        "n1_nonconverged_or_islanding": len(records) - len(converged),
        "n1_pass_rate": len(passed) / len(records) if records else float("nan"),
        "maximum_post_contingency_loading_percent": max(max_loading)
        if max_loading
        else float("nan"),
    }


def _cost_from_solved_network(net: Any) -> float:
    total = 0.0
    costs = getattr(net, "poly_cost", None)
    if costs is None or len(costs) == 0:
        return float("nan")
    result_tables = {
        "gen": "res_gen",
        "sgen": "res_sgen",
        "ext_grid": "res_ext_grid",
    }
    for _, row in costs.iterrows():
        element_type = str(row["et"])
        table_name = result_tables.get(element_type)
        if table_name is None:
            continue
        table = getattr(net, table_name, None)
        element = int(row["element"])
        if table is None or element not in table.index:
            continue
        p = float(table.loc[element, "p_mw"])
        total += (
            float(row.get("cp2_eur_per_mw2", 0.0)) * p * p
            + float(row.get("cp1_eur_per_mw", 0.0)) * p
            + float(row.get("cp0_eur", 0.0))
        )
    return float(total)


def _binding_h_vectors(
    *,
    net: Any,
    slack_bus: int,
    results: dict[str, dict[str, Any]],
    h_data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    slack_pos = bus_ids.index(int(slack_bus))
    h_from = expand_h_reduced_to_full(
        np.asarray(h_data["h_from"], dtype=float),
        n_bus=len(bus_ids),
        slack_pos=slack_pos,
        pq_mask=np.asarray(h_data["pq_mask"], dtype=bool),
    )
    h_to = expand_h_reduced_to_full(
        np.asarray(h_data["h_to"], dtype=float),
        n_bus=len(bus_ids),
        slack_pos=slack_pos,
        pq_mask=np.asarray(h_data["pq_mask"], dtype=bool),
    )
    line_ids = [int(x) for x in sorted(net.line.index)]
    binding = np.vstack(
        [
            h_from[pos]
            if str(results[f"line_{line_id}"]["binding_end"]) == "from"
            else h_to[pos]
            for pos, line_id in enumerate(line_ids)
        ]
    )
    return binding, np.asarray(h_data["pq_mask"], dtype=bool)


def _method_rankings(
    *,
    net: Any,
    slack_bus: int,
    base_pf: Any,
) -> tuple[dict[str, list[int]], dict[str, dict[str, Any]]]:
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
    line_ids = [int(x) for x in sorted(net.line.index)]
    h_binding, pq_mask = _binding_h_vectors(
        net=net,
        slack_bus=slack_bus,
        results=results,
        h_data=h_data,
    )
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    bus_pos = {bus_id: pos for pos, bus_id in enumerate(bus_ids)}
    sigma_p = np.full(len(bus_ids), 0.5, dtype=float)
    sigma_q = np.full(len(bus_ids), 0.25, dtype=float)
    for _, row in net.load.iterrows():
        pos = bus_pos[int(row["bus"])]
        sigma_p[pos] += 0.10 * abs(float(row.get("p_mw", 0.0)))
        sigma_q[pos] += 0.10 * abs(float(row.get("q_mvar", 0.0)))
    limits = np.asarray(
        [float(results[f"line_{line_id}"]["ac_s_limit_mva"]) for line_id in line_ids]
    )
    s0 = np.asarray(
        [
            float(
                results[f"line_{line_id}"][
                    "ac_s0_from_mva"
                    if results[f"line_{line_id}"]["binding_end"] == "from"
                    else "ac_s0_to_mva"
                ]
            )
            for line_id in line_ids
        ]
    )
    sigma_results = compute_ac_sigma_radius(
        h_vectors=h_binding,
        s_limit_mva=limits,
        s0_mva=s0,
        sigma_p_mw=sigma_p,
        sigma_q_mvar=sigma_q,
        line_ids=line_ids,
        balance=True,
        pq_mask=pq_mask,
    )
    h_dc, _ = build_dc_matrices(net, slack_bus=slack_bus, chunk_size=64)
    h_dc_projected = h_dc - np.mean(h_dc, axis=1, keepdims=True)
    ptdf_norm = np.linalg.norm(h_dc_projected, axis=1)

    score_rows: dict[str, dict[int, float]] = {
        "radius": {},
        "loading": {},
        "headroom": {},
        "sensitivity": {},
        "ptdf": {},
        "chance": {},
    }
    for pos, line_id in enumerate(line_ids):
        row = results[f"line_{line_id}"]
        if bool(row.get("is_unconstrained", False)):
            continue
        margin = float(row["margin_ac_mva"])
        radius = float(row["radius_ac_l2"])
        score_rows["radius"][line_id] = 1.0 / max(radius, 1e-12)
        score_rows["loading"][line_id] = float(s0[pos] / limits[pos])
        score_rows["headroom"][line_id] = -margin
        score_rows["sensitivity"][line_id] = float(row["||h||2"])
        score_rows["ptdf"][line_id] = float(ptdf_norm[pos])
        score_rows["chance"][line_id] = float(
            sigma_results[f"line_{line_id}"]["overload_probability_ac"]
        )
    rankings = {
        method: [
            line_id
            for line_id, _ in sorted(
                scores.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        for method, scores in score_rows.items()
    }
    return rankings, results


def _run_matched_cut(
    *,
    net: Any,
    case_path: Path,
    line_ids: list[int],
    selected_lines: list[int],
    cap_percent: float,
    method: str,
) -> tuple[Any, Any, float]:
    limits = {line_id: 99.0 for line_id in line_ids}
    for line_id in selected_lines:
        limits[int(line_id)] = float(cap_percent)
    return _solve_cost_opf(
        net,
        line_ids,
        input_path=str(case_path),
        max_loading_percent=99.0,
        per_line_loading_limits_pct=limits,
        label=f"matched_{method}",
    )


def _generator_model(
    net: Any,
) -> tuple[list[tuple[str, int, int]], np.ndarray, np.ndarray]:
    elements = list(_iter_dispatchable_elements(net))
    lower = []
    upper = []
    for element_type, element, _ in elements:
        table = getattr(net, element_type)
        row = table.loc[element]
        lower.append(float(row.get("min_p_mw", -1e5)))
        upper.append(float(row.get("max_p_mw", 1e5)))
    return elements, np.asarray(lower), np.asarray(upper)


def _preventive_dc_scopf(
    *,
    net: Any,
    case_path: Path,
    slack_bus: int,
) -> tuple[Any, float, int]:
    """Solve a preventive N-1 DC SCOPF with piecewise-linear generation costs."""

    import pandapower as pp

    model = copy.deepcopy(net)
    _prepare_cost_opf_network(model)
    _apply_loading_limits(model, default_loading_percent=99.0)
    _set_default_voltage_bounds(model)
    _add_matpower_costs(model, str(case_path))
    elements, p_min, p_max = _generator_model(model)
    n_gen = len(elements)
    bus_ids = [int(x) for x in sorted(model.bus.index)]
    bus_pos = {bus_id: pos for pos, bus_id in enumerate(bus_ids)}
    generator_map = np.zeros((len(bus_ids), n_gen), dtype=float)
    for column, (_, _, bus_id) in enumerate(elements):
        generator_map[bus_pos[int(bus_id)], column] = 1.0
    load = np.zeros(len(bus_ids), dtype=float)
    for _, row in model.load.iterrows():
        if bool(row.get("in_service", True)):
            load[bus_pos[int(row["bus"])]] += float(row.get("p_mw", 0.0))

    h_dc, _ = build_dc_matrices(model, slack_bus=slack_bus, chunk_size=64)
    incidence = incidence_from_pandapower_net(model)
    lodf_result = lodf_from_ptdf(
        ptdf_for_line_transfers(h_dc, incidence), islanding="skip"
    )
    limits = np.asarray(
        [
            0.99
            * float(
                estimate_line_limit_mva_with_flag(model, model.line.loc[line_id])[0]
            )
            for line_id in sorted(model.line.index)
        ],
        dtype=float,
    )
    blocks = [h_dc]
    for contingency in range(h_dc.shape[0]):
        if contingency in lodf_result.islanded_contingencies:
            continue
        post = h_dc + np.outer(lodf_result.lodf[:, contingency], h_dc[contingency])
        post[contingency, :] = 0.0
        blocks.append(post)
    flow_map = np.vstack([block @ generator_map for block in blocks])
    flow_offset = np.concatenate([-block @ load for block in blocks])
    repeated_limits = np.tile(limits, len(blocks))
    flow_ub = np.vstack([flow_map, -flow_map])
    flow_rhs = np.concatenate(
        [repeated_limits - flow_offset, repeated_limits + flow_offset]
    )

    n_var = 2 * n_gen
    cost_vector = np.concatenate([np.zeros(n_gen), np.ones(n_gen)])
    tangent_rows = []
    tangent_rhs = []
    cost_lookup: dict[tuple[str, int], tuple[float, float, float]] = {}
    for _, row in model.poly_cost.iterrows():
        cost_lookup[(str(row["et"]), int(row["element"]))] = (
            float(row.get("cp2_eur_per_mw2", 0.0)),
            float(row.get("cp1_eur_per_mw", 0.0)),
            float(row.get("cp0_eur", 0.0)),
        )
    for generator, (element_type, element, _) in enumerate(elements):
        c2, c1, c0 = cost_lookup.get((element_type, element), (0.0, 1.0, 0.0))
        grid = np.linspace(p_min[generator], p_max[generator], 9)
        for point in np.unique(grid):
            row = np.zeros(n_var, dtype=float)
            slope = 2.0 * c2 * point + c1
            row[generator] = slope
            row[n_gen + generator] = -1.0
            tangent_rows.append(row)
            tangent_rhs.append(c2 * point * point - c0)
    flow_sparse = sp.hstack(
        [sp.csr_matrix(flow_ub), sp.csr_matrix((flow_ub.shape[0], n_gen))],
        format="csr",
    )
    a_ub = sp.vstack(
        [flow_sparse, sp.csr_matrix(np.vstack(tangent_rows))], format="csr"
    )
    b_ub = np.concatenate([flow_rhs, np.asarray(tangent_rhs)])
    a_eq = np.zeros((1, n_var), dtype=float)
    a_eq[0, :n_gen] = 1.0
    result = linprog(
        cost_vector,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.asarray([float(np.sum(load))]),
        bounds=[*(zip(p_min, p_max)), *[(None, None)] * n_gen],
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-8,
            "primal_feasibility_tolerance": 1e-8,
        },
    )
    if not bool(result.success):
        raise RuntimeError(f"preventive DC SCOPF failed: {result.message}")

    solved = copy.deepcopy(model)
    dispatch = np.asarray(result.x[:n_gen], dtype=float)
    for value, (element_type, element, _) in zip(dispatch, elements):
        if element_type != "ext_grid":
            getattr(solved, element_type).at[element, "p_mw"] = float(value)
    for init in ("dc", "flat"):
        try:
            pp.runpp(
                solved,
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                init=init,
                max_iteration=300,
                tolerance_mva=1e-9,
            )
            if bool(getattr(solved, "converged", False)):
                break
        except Exception:  # noqa: BLE001 - benchmark outcome accounting
            pass
    if not bool(getattr(solved, "converged", False)):
        raise RuntimeError("preventive DC SCOPF dispatch did not converge in AC replay")
    return (
        solved,
        _cost_from_solved_network(solved),
        len(lodf_result.islanded_contingencies),
    )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for case_name in args.cases:
        case_path = Path(args.data_dir) / case_name
        for load_scale in args.load_scales:
            net = _scaled_case(case_path, float(load_scale))
            slack_bus = _auto_slack_bus(net)
            line_ids = [int(x) for x in sorted(net.line.index)]
            try:
                nn_cost, base_pf, base_cost = _solve_cost_opf(
                    net,
                    line_ids,
                    input_path=str(case_path),
                    max_loading_percent=99.0,
                    label="lossy_cost_opf",
                )
                rankings, radius_results = _method_rankings(
                    net=net,
                    slack_bus=slack_bus,
                    base_pf=base_pf,
                )
                base_records = _ac_n1_screen(nn_cost, "lossy_cost_opf")
                rows.append(
                    {
                        "case": case_path.stem,
                        "load_scale": float(load_scale),
                        "method": "cost_opf",
                        "status": "ok",
                        "cost": float(base_cost),
                        "cost_increase_percent": 0.0,
                        "selected_lines": 0,
                        "minimum_affine_radius": min(
                            float(row["radius_ac_l2"])
                            for key, row in radius_results.items()
                            if key.startswith("line_")
                            and not bool(row.get("is_unconstrained", False))
                            and float(row["radius_ac_l2"]) > 0.0
                        ),
                        **_n1_summary(base_records),
                    }
                )
                cut_count = min(int(args.cut_count), len(line_ids))
                for method in (
                    "radius",
                    "loading",
                    "headroom",
                    "sensitivity",
                    "ptdf",
                    "chance",
                ):
                    selected = rankings[method][:cut_count]
                    selected_rows.extend(
                        {
                            "case": case_path.stem,
                            "load_scale": float(load_scale),
                            "method": method,
                            "rank": rank,
                            "line_id": line_id,
                        }
                        for rank, line_id in enumerate(selected, start=1)
                    )
                    try:
                        nn_method, method_pf, cost = _run_matched_cut(
                            net=net,
                            case_path=case_path,
                            line_ids=line_ids,
                            selected_lines=selected,
                            cap_percent=float(args.cap_percent),
                            method=method,
                        )
                        method_results = compute_ac_l2_radius(
                            net,
                            base_pf=method_pf,
                            slack_bus=slack_bus,
                            chunk_size=64,
                            balance=True,
                            lossless=False,
                        )
                        n1_records = _ac_n1_screen(nn_method, f"matched_{method}")
                        rows.append(
                            {
                                "case": case_path.stem,
                                "load_scale": float(load_scale),
                                "method": method,
                                "status": "ok",
                                "cost": float(cost),
                                "cost_increase_percent": 100.0
                                * (float(cost) / base_cost - 1.0),
                                "selected_lines": cut_count,
                                "minimum_affine_radius": min(
                                    float(row["radius_ac_l2"])
                                    for key, row in method_results.items()
                                    if key.startswith("line_")
                                    and not bool(row.get("is_unconstrained", False))
                                    and float(row["radius_ac_l2"]) > 0.0
                                ),
                                **_n1_summary(n1_records),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001 - complete outcome accounting
                        rows.append(
                            {
                                "case": case_path.stem,
                                "load_scale": float(load_scale),
                                "method": method,
                                "status": "failed",
                                "error": repr(exc),
                            }
                        )
                try:
                    nn_scopf, cost_scopf, islanding = _preventive_dc_scopf(
                        net=net,
                        case_path=case_path,
                        slack_bus=slack_bus,
                    )
                    scopf_pf = _extract_pypsa_result_from_pp(nn_scopf, line_ids)
                    scopf_results = compute_ac_l2_radius(
                        net,
                        base_pf=scopf_pf,
                        slack_bus=slack_bus,
                        chunk_size=64,
                        balance=True,
                        lossless=False,
                    )
                    scopf_records = _ac_n1_screen(nn_scopf, "preventive_dc_scopf")
                    rows.append(
                        {
                            "case": case_path.stem,
                            "load_scale": float(load_scale),
                            "method": "preventive_dc_scopf",
                            "status": "ok",
                            "cost": float(cost_scopf),
                            "cost_increase_percent": 100.0
                            * (float(cost_scopf) / base_cost - 1.0),
                            "selected_lines": "all_nonislanding_n_minus_1",
                            "dc_islanding_contingencies_excluded": int(islanding),
                            "minimum_affine_radius": min(
                                float(row["radius_ac_l2"])
                                for key, row in scopf_results.items()
                                if key.startswith("line_")
                                and not bool(row.get("is_unconstrained", False))
                                and float(row["radius_ac_l2"]) > 0.0
                            ),
                            **_n1_summary(scopf_records),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        {
                            "case": case_path.stem,
                            "load_scale": float(load_scale),
                            "method": "preventive_dc_scopf",
                            "status": "failed",
                            "error": repr(exc),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "case": case_path.stem,
                        "load_scale": float(load_scale),
                        "method": "case_setup",
                        "status": "failed",
                        "error": repr(exc),
                    }
                )
            _write_csv(output_dir / "dispatch_benchmark.csv", rows)
            _write_csv(output_dir / "selected_lines.csv", selected_rows)

    successful = [row for row in rows if row.get("status") == "ok"]
    summary = {
        "case_operating_points": len(args.cases) * len(args.load_scales),
        "successful_method_runs": len(successful),
        "failed_method_runs": len(rows) - len(successful),
        "matched_cut_count": int(args.cut_count),
        "matched_branch_cap_percent": float(args.cap_percent),
        "cases": list(args.cases),
        "load_scales": [float(value) for value in args.load_scales],
    }
    (output_dir / "dispatch_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/input")
    parser.add_argument("--output-dir", default="run_artifacts/revision1_dispatch")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "pglib_opf_case14_ieee.m",
            "pglib_opf_case30_ieee.m",
            "pglib_opf_case118_ieee.m",
        ],
    )
    parser.add_argument("--load-scales", nargs="+", type=float, default=[0.9, 1.0, 1.1])
    parser.add_argument("--cut-count", type=int, default=5)
    parser.add_argument("--cap-percent", type=float, default=90.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
