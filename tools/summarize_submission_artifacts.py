"""Build compact submission tables from experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


SIGMA_DIRS = (
    "run_sigma_radius/sigma_radius_hourly",
    "run_sigma_radius/sigma_case2000_goc",
    "run_sigma_radius/sigma_case2736sp_k",
    "run_sigma_radius/sigma_case2869_pegase",
)

N1_DIRS = (
    "n1_stability_demo/n1_demo_case118_submission",
    "n1_stability_demo/n1_demo_case118_ablation_r5",
    "n1_stability_demo/n1_demo_case118_ablation_r20",
)

REGIMES = ("cost_opf", "radius_opf", "scopf")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _wilson_interval(
    k: int, n: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = float(k) / float(n)
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_sigma(*, artifacts_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in SIGMA_DIRS:
        run_dir = artifacts_root / rel
        summary = _read_json(run_dir / "summary.json")
        mc = _read_json(run_dir / "mc_tightened_limit.json")
        validation = _read_json(run_dir / "validation.json")

        n_samples = int(mc.get("n_samples", 0) or 0)
        n_fail = int(mc.get("n_pf_failures", 0) or 0)
        n_viol = int(mc.get("n_violations", 0) or 0)
        ci_low, ci_high = _wilson_interval(n_viol, n_samples)

        rows.append(
            {
                "run_dir": rel,
                "case": summary.get("case", ""),
                "sigma_source": summary.get("sigma_source", ""),
                "n_timesteps": summary.get("n_timesteps", ""),
                "n_lines": summary.get("n_lines", ""),
                "n_lines_positive_sigma": summary.get("n_lines_positive_sigma", ""),
                "n_lines_negative_sigma": summary.get("n_lines_negative_sigma", ""),
                "min_positive_r_sigma": summary.get(
                    "global_min_positive_sigma_radius", ""
                ),
                "median_r_sigma": summary.get("global_median_sigma_radius", ""),
                "max_r_sigma": summary.get("global_max_sigma_radius", ""),
                "mc_target_line": mc.get("target_line", ""),
                "mc_target_end": mc.get("binding_end", ""),
                "mc_seed": "42" if mc else "",
                "mc_pilot_samples": mc.get("n_pilot", ""),
                "mc_attempted_samples": n_samples if mc else "",
                "mc_converged_samples": (n_samples - n_fail) if mc else "",
                "mc_pf_failures": n_fail if mc else "",
                "mc_violations": n_viol if mc else "",
                "mc_empirical_prob": mc.get("empirical_prob", ""),
                "mc_empirical_prob_wilson95_low": ci_low if mc else "",
                "mc_empirical_prob_wilson95_high": ci_high if mc else "",
                "mc_pred_prob_empirical_sigma": mc.get(
                    "analytical_prob_with_empirical_sigma", ""
                ),
                "mc_pred_prob_analytical_sigma": mc.get(
                    "analytical_prob_with_analytical_sigma", ""
                ),
                "mc_ratio_empirical_sigma": mc.get("ratio_analytE_over_empirical", ""),
                "mc_ratio_analytical_sigma": mc.get("ratio_analytA_over_empirical", ""),
                "balance_pass": validation.get("balance", {}).get("all_ok", ""),
                "sigma_floor_clamped": validation.get("sigma_floor", {}).get(
                    "n_clamped", ""
                ),
            }
        )

    _write_csv(output_dir / "sigma_mc_summary.csv", rows)
    return rows


def _extract_table_value(lines: list[str], label: str) -> list[str]:
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith(label):
            continue
        parts = re.split(r"\s{2,}", stripped)
        if len(parts) >= 4:
            return parts[-3:]
    return ["", "", ""]


def _as_float_text(value: str) -> float | str:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _radius_cut_summary(debug_log: Path) -> tuple[int, str, int]:
    if not debug_log.exists():
        return 0, "", 0
    text = debug_log.read_text(encoding="utf-8", errors="replace")
    cuts = [int(x) for x in re.findall(r"\[radius_opf\] Tightened (\d+) lines", text)]
    n_iter = len(re.findall(r"\[radius_opf\] Iteration \d+/\d+", text))
    return sum(cuts), ";".join(str(x) for x in cuts), n_iter


def summarize_dispatch(
    *, artifacts_root: Path, output_dir: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in N1_DIRS:
        run_dir = artifacts_root / rel
        summary_path = run_dir / "comparison_summary.txt"
        if not summary_path.exists():
            continue
        lines = summary_path.read_text(encoding="utf-8").splitlines()
        r_target = ""
        for line in lines:
            match = re.search(r"r_target .*: ([0-9.eE+-]+)", line)
            if match:
                r_target = float(match.group(1))
                break

        total_cuts, cuts_by_iter, radius_iterations = _radius_cut_summary(
            run_dir / "debug.log"
        )
        values_by_metric = {
            "total_cost": _extract_table_value(lines, "Total generation cost"),
            "cost_increase_pct": _extract_table_value(
                lines, "Cost increase vs Cost OPF"
            ),
            "min_radius": _extract_table_value(lines, "Min radius"),
            "median_radius": _extract_table_value(lines, "Median radius"),
            "max_overload_probability": _extract_table_value(
                lines, "Max overload probability"
            ),
            "n1_passed": _extract_table_value(lines, "N-1 passed"),
            "n1_failed": _extract_table_value(lines, "N-1 failed"),
            "n1_pass_rate_pct": _extract_table_value(lines, "N-1 pass rate"),
        }

        for idx, regime in enumerate(REGIMES):
            rows.append(
                {
                    "run_dir": rel,
                    "r_target_mw": r_target,
                    "radius_iterations_logged": radius_iterations,
                    "radius_active_cuts_total": total_cuts,
                    "radius_active_cuts_by_iter": cuts_by_iter,
                    "regime": regime,
                    "total_cost": _as_float_text(values_by_metric["total_cost"][idx]),
                    "cost_increase_pct": _as_float_text(
                        values_by_metric["cost_increase_pct"][idx]
                    ),
                    "min_radius": _as_float_text(values_by_metric["min_radius"][idx]),
                    "median_radius": _as_float_text(
                        values_by_metric["median_radius"][idx]
                    ),
                    "max_overload_probability": _as_float_text(
                        values_by_metric["max_overload_probability"][idx]
                    ),
                    "n1_passed": _as_float_text(values_by_metric["n1_passed"][idx]),
                    "n1_failed": _as_float_text(values_by_metric["n1_failed"][idx]),
                    "n1_pass_rate_pct": _as_float_text(
                        values_by_metric["n1_pass_rate_pct"][idx]
                    ),
                }
            )

    _write_csv(output_dir / "dispatch_ablation_summary.csv", rows)
    return rows


def _first_match(lines: list[str], pattern: str) -> re.Match[str] | None:
    for line in lines:
        match = re.search(pattern, line)
        if match:
            return match
    return None


def _dispatch_outcome_values(lines: list[str]) -> dict[str, Any]:
    max_prob = _extract_table_value(lines, "Max overload probability")
    n1_passed = _extract_table_value(lines, "N-1 passed")
    n1_failed = _extract_table_value(lines, "N-1 failed")
    n1_pass_rate = _extract_table_value(lines, "N-1 pass rate")
    return {
        "final_radius_opf_max_overload_probability": _as_float_text(max_prob[1]),
        "final_radius_opf_n1_passed": _as_float_text(n1_passed[1]),
        "final_radius_opf_n1_failed": _as_float_text(n1_failed[1]),
        "final_radius_opf_n1_pass_rate_pct": _as_float_text(n1_pass_rate[1]),
    }


def summarize_dispatch_iterations(
    *, artifacts_root: Path, output_dir: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in N1_DIRS:
        run_dir = artifacts_root / rel
        summary_path = run_dir / "comparison_summary.txt"
        debug_log = run_dir / "debug.log"
        if not summary_path.exists() or not debug_log.exists():
            continue

        summary_lines = summary_path.read_text(encoding="utf-8").splitlines()
        debug_lines = debug_log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        r_target = ""
        for line in summary_lines:
            match = re.search(r"r_target .*: ([0-9.eE+-]+)", line)
            if match:
                r_target = float(match.group(1))
                break

        outcome = _dispatch_outcome_values(summary_lines)
        cost_opf_cost = _first_match(
            debug_lines, r"\[cost_opf\] OPF converged .* cost=([0-9.eE+-]+)"
        )
        cost_opf_radius = _first_match(
            debug_lines,
            r"\[cost_opf\] Done .* \| constrained=(\d+) \| "
            r"min=([0-9.eE+-]+) median=([0-9.eE+-]+)",
        )
        rows.append(
            {
                "run_dir": rel,
                "r_target_mw": r_target,
                "iteration": 0,
                "phase": "initial_cost_opf",
                "active_cuts": "",
                "skipped_safe_lines": "",
                "tightened_min_pct": "",
                "tightened_mean_pct": "",
                "opf_cost": float(cost_opf_cost.group(1)) if cost_opf_cost else "",
                "min_radius_after_solve": float(cost_opf_radius.group(2))
                if cost_opf_radius
                else "",
                "median_radius_after_solve": float(cost_opf_radius.group(3))
                if cost_opf_radius
                else "",
                "status": "solved",
                **outcome,
            }
        )

        cuts_by_iter: dict[int, dict[str, Any]] = {}
        iter_no = 0
        for line in debug_lines:
            if re.search(r"\[radius_opf\] Iteration \d+/\d+", line):
                iter_no += 1
                continue
            match = re.search(
                r"\[radius_opf\] Tightened (\d+) lines "
                r"\(skipped (\d+) already safe\)"
                r"(?: \| min_pct=([0-9.eE+-]+)% "
                r"mean_pct=([0-9.eE+-]+)%)?",
                line,
            )
            if match and iter_no > 0:
                cuts_by_iter[iter_no] = {
                    "active_cuts": int(match.group(1)),
                    "skipped_safe_lines": int(match.group(2)),
                    "tightened_min_pct": float(match.group(3))
                    if match.group(3)
                    else "",
                    "tightened_mean_pct": float(match.group(4))
                    if match.group(4)
                    else "",
                }

        costs_by_iter: dict[int, float] = {}
        radii_by_iter: dict[int, tuple[float, float]] = {}
        for line in debug_lines:
            match = re.search(
                r"\[radius_opf_iter(\d+)\] OPF converged .* "
                r"cost=([0-9.eE+-]+)",
                line,
            )
            if match:
                costs_by_iter[int(match.group(1))] = float(match.group(2))
            match = re.search(
                r"\[radius_opf_iter(\d+)\] Done .* \| constrained=\d+ \| "
                r"min=([0-9.eE+-]+) median=([0-9.eE+-]+)",
                line,
            )
            if match:
                radii_by_iter[int(match.group(1))] = (
                    float(match.group(2)),
                    float(match.group(3)),
                )

        converged = any(
            "[radius_opf] All lines satisfy r >= r_target" in line
            for line in debug_lines
        )
        for iteration in sorted(cuts_by_iter):
            cut_row = cuts_by_iter[iteration]
            radii = radii_by_iter.get(iteration, ("", ""))
            status = "solved" if iteration in costs_by_iter else "converged_no_solve"
            if status == "converged_no_solve" and not converged:
                status = "no_solve_recorded"
            rows.append(
                {
                    "run_dir": rel,
                    "r_target_mw": r_target,
                    "iteration": iteration,
                    "phase": "radius_opf",
                    "active_cuts": cut_row["active_cuts"],
                    "skipped_safe_lines": cut_row["skipped_safe_lines"],
                    "tightened_min_pct": cut_row["tightened_min_pct"],
                    "tightened_mean_pct": cut_row["tightened_mean_pct"],
                    "opf_cost": costs_by_iter.get(iteration, ""),
                    "min_radius_after_solve": radii[0],
                    "median_radius_after_solve": radii[1],
                    "status": status,
                    **outcome,
                }
            )

    _write_csv(output_dir / "dispatch_iteration_summary.csv", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("run_artifacts"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("run_artifacts/submission_tables"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sigma_rows = summarize_sigma(
        artifacts_root=args.artifacts_root, output_dir=args.output_dir
    )
    dispatch_rows = summarize_dispatch(
        artifacts_root=args.artifacts_root, output_dir=args.output_dir
    )
    dispatch_iteration_rows = summarize_dispatch_iterations(
        artifacts_root=args.artifacts_root, output_dir=args.output_dir
    )
    print(f"wrote {len(sigma_rows)} sigma rows")
    print(f"wrote {len(dispatch_rows)} dispatch rows")
    print(f"wrote {len(dispatch_iteration_rows)} dispatch iteration rows")
    print(args.output_dir)


if __name__ == "__main__":
    main()
