"""Combine PGLib sweep summaries into a complete outcome-accounting table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PGLIB_ORDER = (
    "pglib_opf_case5_pjm",
    "pglib_opf_case14_ieee",
    "pglib_opf_case24_ieee_rts",
    "pglib_opf_case30_ieee",
    "pglib_opf_case57_ieee",
    "pglib_opf_case73_ieee_rts",
    "pglib_opf_case118_ieee",
    "pglib_opf_case200_activ",
    "pglib_opf_case2000_goc",
    "pglib_opf_case10000_goc",
    "pglib_opf_case588_sdet",
    "pglib_opf_case1888_rte",
    "pglib_opf_case1951_rte",
    "pglib_opf_case2853_sdet",
    "pglib_opf_case6468_rte",
    "pglib_opf_case6515_rte",
    "pglib_opf_case2383wp_k",
    "pglib_opf_case2736sp_k",
    "pglib_opf_case300_ieee",
    "pglib_opf_case1354_pegase",
    "pglib_opf_case2869_pegase",
)


def _read_summary(path: Path) -> dict[str, dict[str, Any]]:
    p = path / "summary.json"
    if not p.exists():
        return {}
    rows = json.loads(p.read_text(encoding="utf-8"))
    return {str(row["case"]): row for row in rows}


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _line_counts(json_path: Path) -> dict[str, int]:
    if not json_path.exists():
        return {
            "negative_margin_lines": 0,
            "degenerate_binding_lines": 0,
            "nondifferentiable_line_ends": 0,
            "line_rows": 0,
        }

    data = json.loads(json_path.read_text(encoding="utf-8"))
    line_rows = [
        value
        for key, value in data.items()
        if key.startswith("line_") and isinstance(value, dict)
    ]
    negative = 0
    degenerate = 0
    nondiff_ends = 0
    for row in line_rows:
        margin = _as_float(row.get("margin_ac_mva"))
        norm = _as_float(row.get("||h||2"))
        if math.isfinite(margin) and margin < 0.0:
            negative += 1
        if (not math.isfinite(norm)) or norm <= 1.0e-12:
            degenerate += 1
        if bool(row.get("ac_nondifferentiable_from", False)):
            nondiff_ends += 1
        if bool(row.get("ac_nondifferentiable_to", False)):
            nondiff_ends += 1
    return {
        "negative_margin_lines": negative,
        "degenerate_binding_lines": degenerate,
        "nondifferentiable_line_ends": nondiff_ends,
        "line_rows": len(line_rows),
    }


def combine(*, sweep_dirs: list[Path], output_dir: Path) -> list[dict[str, Any]]:
    rows_by_case: dict[str, dict[str, Any]] = {}
    source_by_case: dict[str, Path] = {}
    for sweep_dir in sweep_dirs:
        for case, row in _read_summary(sweep_dir).items():
            rows_by_case[case] = row
            source_by_case[case] = sweep_dir

    out_rows: list[dict[str, Any]] = []
    for case in PGLIB_ORDER:
        row = rows_by_case.get(case, {"case": case, "status": "not_run"})
        source = source_by_case.get(case)
        json_path = source / f"{case}.json" if source is not None else Path()
        counts = _line_counts(json_path)

        status = str(row.get("status", "unknown"))
        dc_r = _as_float(row.get("dc_r_star"))
        ac_r = _as_float(row.get("ac_r_star"))
        dc_status = "computed" if math.isfinite(dc_r) else "not_computed"
        if status in {"timeout", "dc_only_timeout"} and math.isfinite(dc_r):
            dc_status = "computed_partial"

        ac_pf_status = "ok" if status in {"ok", "ac_infeasible"} else status
        if status in {"ok", "ac_infeasible"} and math.isfinite(ac_r):
            ac_radius_status = "computed"
        elif status == "timeout":
            ac_radius_status = "timeout"
        else:
            ac_radius_status = "not_computed"

        out_rows.append(
            {
                "case": case,
                "n_buses": row.get("n_buses", 0),
                "n_lines": row.get("n_lines", 0),
                "dc_status": dc_status,
                "ac_pf_status": ac_pf_status,
                "ac_radius_status": ac_radius_status,
                "dc_r_star_mw": row.get("dc_r_star", float("nan")),
                "ac_r_star_mw": row.get("ac_r_star", float("nan")),
                "ac_feasible": row.get("ac_feasible", "n/a"),
                "ac_n_violated": row.get("ac_n_violated", 0),
                "negative_margin_lines": counts["negative_margin_lines"],
                "degenerate_binding_lines": counts["degenerate_binding_lines"],
                "nondifferentiable_line_ends": counts["nondifferentiable_line_ends"],
                "runtime_sec": row.get("time_total", float("nan")),
                "outcome": status,
                "source_dir": str(source) if source is not None else "",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pglib_outcome_accounting.csv"
    json_path = output_dir / "pglib_outcome_accounting.json"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    json_path.write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
    return out_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", action="append", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("run_artifacts/submission_validation_combined"),
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = combine(sweep_dirs=args.sweep_dir, output_dir=args.output_dir)
    print(f"combined_rows={len(rows)}")
    for row in rows:
        print(
            row["case"],
            row["outcome"],
            row["n_buses"],
            row["n_lines"],
            row["negative_margin_lines"],
            row["nondifferentiable_line_ends"],
            row["runtime_sec"],
        )


if __name__ == "__main__":
    main()
