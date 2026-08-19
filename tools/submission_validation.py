"""Submission validation experiments for the AC stability-radius manuscript.

This script produces the compact numerical artifacts requested in the final
review round:

- adjoint finite-difference derivative checks;
- adjoint timing/decomposition and forward dense-sensitivity comparison;
- per-case accounting for negative, degenerate, and nondifferentiable AC ends.

It intentionally writes plain CSV/JSON files so the tables can be copied into
the paper without depending on notebook state.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from stability_radius.ac.ac_model import build_ac_operator
from stability_radius.base_point.ac import solve_ac_pf_base_point
from stability_radius.base_point.pandapower_tools import (
    apply_lossless_policy_to_pandapower_net,
)
from stability_radius.geometry.balanced import (
    make_ac_block_specs,
    worst_case_l2_direction,
)
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.ac_l2 import compute_ac_l2_radius
from stability_radius.utils.download import ensure_case_file
from stability_radius.workflows import (
    expand_h_reduced_to_full,
    extract_binding_end_data,
)


DEFAULT_CASES = (
    "pglib_opf_case14_ieee.m",
    "pglib_opf_case30_ieee.m",
    "pglib_opf_case118_ieee.m",
    "pglib_opf_case200_activ.m",
    "pglib_opf_case2000_goc.m",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


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
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _case_path(data_dir: Path, case_name: str, *, allow_download: bool) -> Path:
    path = data_dir / case_name
    if allow_download:
        ensure_case_file(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing case file {path}. Re-run with --allow-download."
        )
    return path


def _auto_slack_bus(net: Any) -> int:
    if hasattr(net, "ext_grid") and net.ext_grid is not None and len(net.ext_grid):
        for eid in sorted(net.ext_grid.index):
            row = net.ext_grid.loc[eid]
            if bool(row.get("in_service", True)):
                return int(row.get("bus"))
    return int(sorted(net.bus.index)[0])


def _line_keys(ac_results: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        (key for key in ac_results if key.startswith("line_")),
        key=lambda item: int(item.split("_", 1)[1]),
    )


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _case_accounting(
    *, case: str, ac_results: dict[str, dict[str, Any]], timings: dict[str, Any]
) -> dict[str, Any]:
    line_keys = _line_keys(ac_results)
    finite_radii: list[float] = []
    negative_margins = 0
    negative_radii = 0
    degenerate_binding = 0
    nondiff_binding = 0
    nondiff_ends = 0
    unconstrained = 0
    for key in line_keys:
        row = ac_results[key]
        radius = _finite_or_nan(row.get("radius_ac_l2"))
        margin = _finite_or_nan(row.get("margin_ac_mva"))
        if math.isfinite(radius):
            finite_radii.append(radius)
        if math.isfinite(margin) and margin < 0.0:
            negative_margins += 1
        if math.isfinite(radius) and radius < 0.0:
            negative_radii += 1
        if bool(row.get("is_unconstrained", False)):
            unconstrained += 1
        norm = _finite_or_nan(row.get("||h||2"))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            degenerate_binding += 1
        if bool(row.get("nondifferentiable_apparent_power", False)):
            nondiff_binding += 1
        if bool(row.get("ac_nondifferentiable_from", False)):
            nondiff_ends += 1
        if bool(row.get("ac_nondifferentiable_to", False)):
            nondiff_ends += 1

    return {
        "case": case,
        "n_bus": int(timings.get("n_bus", 0)),
        "n_line": int(timings.get("n_line", len(line_keys))),
        "n_line_ends": int(timings.get("n_line_ends", 2 * len(line_keys))),
        "finite_radius_lines": len(finite_radii),
        "min_radius_ac_l2": min(finite_radii) if finite_radii else float("nan"),
        "median_radius_ac_l2": float(np.median(finite_radii))
        if finite_radii
        else float("nan"),
        "negative_margin_lines": negative_margins,
        "negative_radius_lines": negative_radii,
        "degenerate_binding_lines": degenerate_binding,
        "nondifferentiable_binding_lines": nondiff_binding,
        "nondifferentiable_line_ends": nondiff_ends,
        "unconstrained_lines": unconstrained,
    }


def _solve_base_pf(
    *,
    net: Any,
    slack_bus: int,
    pf_init: str,
    lossless: bool,
) -> tuple[Any, float]:
    t0 = time.perf_counter()
    try:
        _bp, base_pf = solve_ac_pf_base_point(
            net=net,
            slack_bus=int(slack_bus),
            pf_solver="pandapower",
            pf_init=str(pf_init),
            lossless=bool(lossless),
            gen_dispatch_mw_by_name={},
            distributed_slack=False,
            trafo_model="pi",
        )
    except Exception:
        if str(pf_init).strip().lower() == "flat":
            raise
        _bp, base_pf = solve_ac_pf_base_point(
            net=net,
            slack_bus=int(slack_bus),
            pf_solver="pandapower",
            pf_init="flat",
            lossless=bool(lossless),
            gen_dispatch_mw_by_name={},
            distributed_slack=False,
            trafo_model="pi",
        )
    return base_pf, time.perf_counter() - t0


def _run_line_end_pf(
    *,
    net: Any,
    delta_u: np.ndarray,
    line_id: int,
    binding_end: str,
    lossless: bool,
) -> tuple[bool, float]:
    import pandapower as pp

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)
    du = np.asarray(delta_u, dtype=float).reshape(2 * n_bus)
    nn = (
        apply_lossless_policy_to_pandapower_net(net) if lossless else copy.deepcopy(net)
    )

    dp = du[:n_bus]
    dq = du[n_bus:]
    for pos, bid in enumerate(bus_ids):
        if abs(float(dp[pos])) <= 0.0 and abs(float(dq[pos])) <= 0.0:
            continue
        pp.create_sgen(
            nn,
            bus=int(bid),
            p_mw=float(dp[pos]),
            q_mvar=float(dq[pos]),
            name=f"fd_delta_bus_{int(bid)}",
            in_service=True,
        )

    try:
        pp.runpp(
            nn,
            algorithm="nr",
            calculate_voltage_angles=True,
            enforce_q_lims=True,
            init="flat",
            max_iteration=300,
            tolerance_mva=1.0e-10,
        )
    except Exception:
        return False, float("nan")
    if not bool(getattr(nn, "converged", True)):
        return False, float("nan")

    p_col = "p_from_mw" if binding_end == "from" else "p_to_mw"
    q_col = "q_from_mvar" if binding_end == "from" else "q_to_mvar"
    p = float(nn.res_line.loc[int(line_id), p_col])
    q = float(nn.res_line.loc[int(line_id), q_col])
    return True, math.sqrt(p * p + q * q)


def _fd_rows_for_case(
    *,
    case: str,
    net: Any,
    ac_results: dict[str, dict[str, Any]],
    h_from_raw: np.ndarray,
    h_to_raw: np.ndarray,
    pq_mask: np.ndarray | None,
    slack_bus: int,
    eps_mva: float,
    top_k: int,
    random_k: int,
    all_ends_max_buses: int,
    random_seed: int,
    fd_max_buses: int,
    lossless: bool,
) -> list[dict[str, Any]]:
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)
    if n_bus > int(fd_max_buses):
        return [
            {
                "case": case,
                "status": "skipped",
                "reason": f"n_bus>{int(fd_max_buses)}",
                "n_bus": n_bus,
                "eps_mva": float(eps_mva),
            }
        ]

    slack_pos = bus_ids.index(int(slack_bus))
    h_from = expand_h_reduced_to_full(
        h_from_raw,
        n_bus=n_bus,
        slack_pos=slack_pos,
        pq_mask=pq_mask,
    )
    h_to = expand_h_reduced_to_full(
        h_to_raw,
        n_bus=n_bus,
        slack_pos=slack_pos,
        pq_mask=pq_mask,
    )
    line_ids = [int(key.split("_", 1)[1]) for key in _line_keys(ac_results)]
    line_pos = {lid: pos for pos, lid in enumerate(line_ids)}

    candidates: list[tuple[float, str, str]] = []
    for key in _line_keys(ac_results):
        row = ac_results[key]
        if bool(row.get("is_unconstrained", False)):
            continue
        for line_end in ("from", "to"):
            radius = _finite_or_nan(row.get(f"radius_ac_l2_{line_end}"))
            h_norm = _finite_or_nan(row.get(f"ac_norm_a_{line_end}"))
            s0 = _finite_or_nan(row.get(f"ac_s0_{line_end}_mva"))
            if bool(row.get(f"ac_nondifferentiable_{line_end}", False)):
                continue
            if not math.isfinite(h_norm) or h_norm <= 1.0e-12:
                continue
            if not math.isfinite(s0) or s0 <= 1.0e-8:
                continue
            rank_radius = (
                radius if math.isfinite(radius) and radius > 0.0 else float("inf")
            )
            candidates.append((rank_radius, key, line_end))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    if n_bus <= int(all_ends_max_buses):
        selected = [(radius, key, end, "all") for radius, key, end in candidates]
    else:
        critical = candidates[: int(top_k)]
        remaining = candidates[int(top_k) :]
        rng = np.random.default_rng(int(random_seed) + int(n_bus))
        if remaining and int(random_k) > 0:
            random_indices = rng.choice(
                len(remaining), size=min(int(random_k), len(remaining)), replace=False
            )
            random_selected = [remaining[int(index)] for index in random_indices]
        else:
            random_selected = []
        selected = [(*item, "critical") for item in critical]
        selected.extend((*item, "random") for item in random_selected)

    q_bus_indices = None
    if pq_mask is not None:
        q_bus_indices = np.where(np.asarray(pq_mask, dtype=bool))[0]
    blocks = make_ac_block_specs(
        n_bus,
        balance=True,
        q_bus_indices=q_bus_indices,
    )

    rows: list[dict[str, Any]] = []
    for _rank, key, line_end, sample_type in selected:
        row = ac_results[key]
        line_id = int(key.split("_", 1)[1])
        pos = int(line_pos[line_id])
        h_vec = np.asarray(
            h_from[pos, :] if line_end == "from" else h_to[pos, :], dtype=float
        )
        direction = worst_case_l2_direction(h_vec, blocks)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 0.0:
            rows.append(
                {
                    "case": case,
                    "line_id": line_id,
                    "binding_end": line_end,
                    "sample_type": sample_type,
                    "status": "skipped",
                    "reason": "zero_projected_direction",
                    "eps_mva": float(eps_mva),
                }
            )
            continue

        delta = float(eps_mva) * direction
        ok_plus, s_plus = _run_line_end_pf(
            net=net,
            delta_u=delta,
            line_id=line_id,
            binding_end=line_end,
            lossless=lossless,
        )
        ok_minus, s_minus = _run_line_end_pf(
            net=net,
            delta_u=-delta,
            line_id=line_id,
            binding_end=line_end,
            lossless=lossless,
        )
        if not ok_plus or not ok_minus:
            rows.append(
                {
                    "case": case,
                    "line_id": line_id,
                    "binding_end": line_end,
                    "sample_type": sample_type,
                    "status": "pf_failed",
                    "eps_mva": float(eps_mva),
                }
            )
            continue

        fd_derivative = (float(s_plus) - float(s_minus)) / (2.0 * float(eps_mva))
        linear_derivative = float(np.dot(h_vec, direction))
        denom = max(abs(fd_derivative), abs(linear_derivative), 1.0e-10)
        rel_error = abs(fd_derivative - linear_derivative) / denom
        absolute_error = abs(fd_derivative - linear_derivative)
        normalized_error = absolute_error / max(
            1.0, abs(fd_derivative), abs(linear_derivative)
        )
        rows.append(
            {
                "case": case,
                "line_id": line_id,
                "binding_end": line_end,
                "sample_type": sample_type,
                "status": "ok",
                "n_bus": n_bus,
                "eps_mva": float(eps_mva),
                "radius_ac_l2": row.get(
                    f"radius_ac_l2_{line_end}", float("nan")
                ),
                "h_norm_reported": row.get(
                    f"ac_norm_a_{line_end}", float("nan")
                ),
                "fd_derivative_mva_per_mva": float(fd_derivative),
                "linear_derivative_mva_per_mva": float(linear_derivative),
                "relative_error": float(rel_error),
                "absolute_error_mva_per_mva": float(absolute_error),
                "normalized_error": float(normalized_error),
                "s_plus_mva": float(s_plus),
                "s_minus_mva": float(s_minus),
            }
        )

    return rows


def _fd_summary_rows(fd_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = sorted({str(row.get("case", "")) for row in fd_rows if row.get("case")})
    out: list[dict[str, Any]] = []
    for case in cases:
        rows = [row for row in fd_rows if row.get("case") == case]
        ok = [
            float(row["relative_error"])
            for row in rows
            if row.get("status") == "ok" and math.isfinite(float(row["relative_error"]))
        ]
        absolute = [
            float(row["absolute_error_mva_per_mva"])
            for row in rows
            if row.get("status") == "ok"
            and math.isfinite(float(row.get("absolute_error_mva_per_mva", float("nan"))))
        ]
        normalized = [
            float(row["normalized_error"])
            for row in rows
            if row.get("status") == "ok"
            and math.isfinite(float(row.get("normalized_error", float("nan"))))
        ]
        eps_values = sorted(
            {
                float(row.get("eps_mva", float("nan")))
                for row in rows
                if math.isfinite(float(row.get("eps_mva", float("nan"))))
            }
        )
        out.append(
            {
                "case": case,
                "line_ends_tested": len(ok),
                "median_relative_error": float(np.median(ok)) if ok else float("nan"),
                "max_relative_error": float(np.max(ok)) if ok else float("nan"),
                "median_absolute_error": float(np.median(absolute))
                if absolute
                else float("nan"),
                "max_absolute_error": float(np.max(absolute))
                if absolute
                else float("nan"),
                "median_normalized_error": float(np.median(normalized))
                if normalized
                else float("nan"),
                "max_normalized_error": float(np.max(normalized))
                if normalized
                else float("nan"),
                "critical_ends": sum(
                    1
                    for row in rows
                    if row.get("status") == "ok"
                    and row.get("sample_type") == "critical"
                ),
                "random_ends": sum(
                    1
                    for row in rows
                    if row.get("status") == "ok"
                    and row.get("sample_type") == "random"
                ),
                "all_ends": sum(
                    1
                    for row in rows
                    if row.get("status") == "ok" and row.get("sample_type") == "all"
                ),
                "perturbation_scale_mva": eps_values[0] if eps_values else float("nan"),
                "skipped_or_failed": len(rows) - len(ok),
            }
        )
    return out


def _forward_comparison(
    *,
    net: Any,
    base_pf: Any,
    slack_bus: int,
    chunk_size: int,
    lossless: bool,
    forward_max_vars: int,
) -> dict[str, Any]:
    op = build_ac_operator(
        net=net,
        slack_bus=int(slack_bus),
        vm_pu=np.asarray(base_pf.v_mag_pu, dtype=float),
        va_rad=np.asarray(base_pf.v_ang_rad, dtype=float),
        line_indices=[int(x) for x in sorted(net.line.index)],
        lossless=bool(lossless),
        forced_pq_bus_ids={
            int(event["bus"])
            for event in (getattr(base_pf, "q_limit_events", ()) or ())
            if int(event.get("bus", -1)) >= 0
            and str(event.get("element", "")) != "ext_grid"
        },
    )
    n_vars = int(op.n_vars)
    dense_basis_mb = float(n_vars * n_vars * 8 / 1.0e6)
    rhs_chunk_mb = float(n_vars * int(chunk_size) * 8 / 1.0e6)
    row: dict[str, Any] = {
        "n_vars": n_vars,
        "forward_dense_basis_mb": dense_basis_mb,
        "adjoint_rhs_chunk_mb": rhs_chunk_mb,
    }

    if n_vars > int(forward_max_vars):
        row.update(
            {
                "forward_basis_solve_sec": float("nan"),
                "forward_status": f"skipped_n_vars>{int(forward_max_vars)}",
            }
        )
        return row

    rhs = np.eye(n_vars, dtype=float)
    t0 = time.perf_counter()
    sol = op.solve_J(rhs)
    elapsed = time.perf_counter() - t0
    row.update(
        {
            "forward_basis_solve_sec": float(elapsed),
            "forward_solution_checksum": float(np.linalg.norm(sol[:, 0], ord=2)),
            "forward_status": "ok",
        }
    )
    return row


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    performance_rows: list[dict[str, Any]] = []
    fd_rows: list[dict[str, Any]] = []
    accounting_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for case_name in args.cases:
        case_t0 = time.perf_counter()
        try:
            path = _case_path(
                data_dir, case_name, allow_download=bool(args.allow_download)
            )
            net = load_network(str(path))
            slack_bus = _auto_slack_bus(net)
            base_pf, pf_time_sec = _solve_base_pf(
                net=net,
                slack_bus=slack_bus,
                pf_init=str(args.pf_init),
                lossless=bool(args.lossless),
            )
            ac_results = compute_ac_l2_radius(
                net,
                base_pf=base_pf,
                slack_bus=slack_bus,
                chunk_size=int(args.chunk_size),
                balance=True,
                lossless=bool(args.lossless),
                return_h_vectors=True,
                return_timings=True,
            )
            h_vecs_raw = ac_results.pop("_h_vectors")
            timings = ac_results.pop("_timings")

            forward = _forward_comparison(
                net=net,
                base_pf=base_pf,
                slack_bus=slack_bus,
                chunk_size=int(args.chunk_size),
                lossless=bool(args.lossless),
                forward_max_vars=int(args.forward_max_vars),
            )
            performance_row = {
                "case": Path(case_name).stem,
                "pf_time_sec": float(pf_time_sec),
                **timings,
                **forward,
            }
            adjoint_sec = float(timings.get("adjoint_solve_sec", float("nan")))
            forward_sec = float(forward.get("forward_basis_solve_sec", float("nan")))
            performance_row["forward_to_adjoint_solve_time_ratio"] = (
                forward_sec / adjoint_sec
                if math.isfinite(forward_sec) and adjoint_sec > 0.0
                else float("nan")
            )
            performance_row["case_total_wall_sec"] = time.perf_counter() - case_t0
            performance_rows.append(performance_row)

            accounting_rows.append(
                _case_accounting(
                    case=Path(case_name).stem,
                    ac_results=ac_results,
                    timings=timings,
                )
            )

            fd_rows.extend(
                _fd_rows_for_case(
                    case=Path(case_name).stem,
                    net=net,
                    ac_results=ac_results,
                    h_from_raw=h_vecs_raw["h_from"],
                    h_to_raw=h_vecs_raw["h_to"],
                    pq_mask=h_vecs_raw.get("pq_mask"),
                    slack_bus=slack_bus,
                    eps_mva=float(args.eps_mva),
                    top_k=int(args.fd_top_k),
                    random_k=int(args.fd_random_k),
                    all_ends_max_buses=int(args.fd_all_ends_max_buses),
                    random_seed=int(args.random_seed),
                    fd_max_buses=int(args.fd_max_buses),
                    lossless=bool(args.lossless),
                )
            )
        except Exception as exc:  # noqa: BLE001 - experiment accounting
            errors.append(
                {
                    "case": Path(case_name).stem,
                    "status": "error",
                    "error": repr(exc),
                    "wall_sec": time.perf_counter() - case_t0,
                }
            )

        _write_csv(output_dir / "performance_timing.csv", performance_rows)
        _write_csv(output_dir / "fd_derivative_checks.csv", fd_rows)
        _write_csv(output_dir / "fd_derivative_summary.csv", _fd_summary_rows(fd_rows))
        _write_csv(output_dir / "case_accounting.csv", accounting_rows)
        _write_csv(output_dir / "errors.csv", errors)

    summary = {
        "config": vars(args),
        "performance_timing": performance_rows,
        "fd_derivative_checks": fd_rows,
        "fd_derivative_summary": _fd_summary_rows(fd_rows),
        "case_accounting": accounting_rows,
        "errors": errors,
    }
    (output_dir / "submission_validation.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/input")
    parser.add_argument("--output-dir", default="run_artifacts/submission_validation")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--pf-init", choices=["dc", "flat"], default="dc")
    parser.add_argument("--eps-mva", type=float, default=1.0e-2)
    parser.add_argument("--fd-top-k", type=int, default=10)
    parser.add_argument("--fd-random-k", type=int, default=10)
    parser.add_argument("--fd-all-ends-max-buses", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=20260812)
    parser.add_argument("--fd-max-buses", type=int, default=300)
    parser.add_argument("--forward-max-vars", type=int, default=800)
    parser.add_argument(
        "--lossless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the series-only AC certificate model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
