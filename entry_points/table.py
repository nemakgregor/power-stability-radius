from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from stability_radius.utils import create_module_output_dir, setup_output_dir_logging

# AC-focused defaults: keep DC table minimal by default.
DEFAULT_DC_COLUMNS: Tuple[str, ...] = (
    "flow0_mw",
    "p0_mw",
    "p_limit_mw_est",
    "margin_mw",
    "norm_g",
    "radius_l2",
)

DEFAULT_AC_COLUMNS: Tuple[str, ...] = (
    "ac_s_limit_mva",
    "ac_s0_from_mva",
    "ac_s0_to_mva",
    "margin_ac_mva",
    "||h||2",
    "binding_end",
    "radius_ac_l2",
)

logger = logging.getLogger(__name__)


def _line_sort_key(line_key: str) -> Tuple[int, str]:
    """Sort keys like 'line_10' numerically, with a deterministic fallback."""
    try:
        return (int(line_key.split("_", 1)[1]), line_key)
    except (IndexError, ValueError):
        return (10**18, line_key)


def _is_line_key(k: str) -> bool:
    return k.startswith("line_")


def _format_float(x: Any) -> str:
    """Format numeric values consistently for terminal/CSV output."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)

    if math.isinf(xf):
        return "inf"
    if math.isnan(xf):
        return "nan"
    return f"{xf:.6g}"


def _iter_line_keys(results: dict[str, Any], *, max_rows: int | None) -> List[str]:
    line_keys = sorted(
        (k for k in results.keys() if _is_line_key(k)), key=_line_sort_key
    )
    if max_rows is not None:
        line_keys = line_keys[: int(max_rows)]
    return line_keys


def _has_any_field(results: dict[str, Any], field: str) -> bool:
    for k, v in results.items():
        if not _is_line_key(k) or not isinstance(v, dict):
            continue
        if field in v:
            return True
    return False


def infer_default_flat_columns(
    results: dict[str, Any],
    *,
    dc_columns: Sequence[str] = DEFAULT_DC_COLUMNS,
    ac_columns: Sequence[str] = DEFAULT_AC_COLUMNS,
) -> Tuple[str, ...]:
    """
    Infer a sensible default column set for "flat" table mode.

    Rationale
    ---------
    AC-only runs historically produced empty DC columns in flat mode. This function
    deterministically selects defaults based on which fields exist in results.json.

    Policy (deterministic)
    ----------------------
    - If only AC fields exist -> AC defaults.
    - If only DC fields exist -> DC defaults.
    - If both exist -> DC + AC defaults (stable concatenation).
    - If neither exists -> DC defaults (legacy fallback).
    """
    has_dc = _has_any_field(results, "radius_l2") or _has_any_field(results, "norm_g")
    has_ac = _has_any_field(results, "radius_ac_l2") or _has_any_field(
        results, "margin_ac_mva"
    )

    if has_ac and not has_dc:
        return tuple(ac_columns)
    if has_dc and not has_ac:
        return tuple(dc_columns)
    if has_ac and has_dc:
        return tuple(dc_columns) + tuple(ac_columns)
    return tuple(dc_columns)


def format_results_table(
    results: dict[str, Any],
    *,
    columns: Sequence[str] = (
        "p0_mw",
        "p_limit_mw_est",
        "margin_mw",
        "norm_g",
        "radius_l2",
    ),
    max_rows: int | None = None,
) -> str:
    """Format per-line results into a single ASCII table (flat mode)."""
    line_keys = _iter_line_keys(results, max_rows=max_rows)

    headers = ["line"] + list(columns)
    rows: List[List[str]] = []

    for k in line_keys:
        row = [k]
        data = results.get(k, {})
        if not isinstance(data, dict):
            data = {}
        for c in columns:
            row.append(_format_float(data.get(c, "")))
        rows.append(row)

    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    align_right = [False] + [True] * len(columns)

    def fmt_row(values: Sequence[str]) -> str:
        out = []
        for i, v in enumerate(values):
            out.append(v.rjust(widths[i]) if align_right[i] else v.ljust(widths[i]))
        return " | ".join(out)

    sep = "-+-".join("-" * w for w in widths)

    out_lines = [fmt_row(headers), sep]
    out_lines.extend(fmt_row(r) for r in rows)

    remaining = len([k for k in results.keys() if _is_line_key(k)]) - len(line_keys)
    if max_rows is not None and remaining > 0:
        out_lines.append(f"... ({remaining} more rows)")

    return "\n".join(out_lines)


def format_results_table_sections(
    results: dict[str, Any],
    *,
    dc_columns: Sequence[str] = DEFAULT_DC_COLUMNS,
    ac_columns: Sequence[str] = DEFAULT_AC_COLUMNS,
    max_rows: int | None = None,
) -> str:
    """
    Format results into two deterministic sections:
      - DC section
      - AC section
    Sections are shown only if the corresponding fields exist.
    """
    out: List[str] = []

    has_dc = _has_any_field(results, "radius_l2") or _has_any_field(results, "norm_g")
    has_ac = _has_any_field(results, "radius_ac_l2") or _has_any_field(
        results, "margin_ac_mva"
    )

    if has_dc:
        out.append("[DC]")
        out.append(
            format_results_table(results, columns=tuple(dc_columns), max_rows=max_rows)
        )
        out.append("")

    if has_ac:
        out.append("[AC]")
        out.append(
            format_results_table(results, columns=tuple(ac_columns), max_rows=max_rows)
        )
        out.append("")

    if not out:
        return "No per-line results found."

    return "\n".join(out).rstrip()


def format_results_csv(
    results: dict[str, Any],
    *,
    columns: Sequence[str],
    max_rows: int | None = None,
) -> str:
    """Format per-line results as CSV (flat mode, deterministic ordering)."""
    line_keys = _iter_line_keys(results, max_rows=max_rows)

    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["line", *columns])

    for k in line_keys:
        data = results.get(k, {})
        if not isinstance(data, dict):
            data = {}
        writer.writerow([k, *(_format_float(data.get(c, "")) for c in columns)])

    return buf.getvalue()


def format_results_csv_sections(
    results: dict[str, Any],
    *,
    dc_columns: Sequence[str] = DEFAULT_DC_COLUMNS,
    ac_columns: Sequence[str] = DEFAULT_AC_COLUMNS,
    max_rows: int | None = None,
) -> dict[str, str]:
    """
    Create sectioned CSV artifacts:
      - results_dc.csv (only if DC fields exist)
      - results_ac.csv (only if AC fields exist)
    Returns a dict { "dc": csv_text, "ac": csv_text } (keys may be missing).
    """
    out: dict[str, str] = {}
    has_dc = _has_any_field(results, "radius_l2") or _has_any_field(results, "norm_g")
    has_ac = _has_any_field(results, "radius_ac_l2") or _has_any_field(
        results, "margin_ac_mva"
    )

    if has_dc:
        out["dc"] = format_results_csv(
            results, columns=tuple(dc_columns), max_rows=max_rows
        )
    if has_ac:
        out["ac"] = format_results_csv(
            results, columns=tuple(ac_columns), max_rows=max_rows
        )

    return out


def _finite_radii(results: dict[str, Any], *, radius_field: str) -> List[float]:
    vals: List[float] = []
    for k, d in results.items():
        if not _is_line_key(k) or not isinstance(d, dict):
            continue
        try:
            r = float(d.get(radius_field, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(r):
            vals.append(r)
    return vals


def format_radius_summary(
    results: dict[str, Any], *, radius_field: str = "radius_l2"
) -> str:
    vals = _finite_radii(results, radius_field=radius_field)
    total = len([k for k in results.keys() if _is_line_key(k)])
    finite = len(vals)

    if finite == 0:
        return f"Summary({radius_field}): lines={total}, finite_radii=0"

    mean_v = sum(vals) / finite
    min_v = min(vals)
    max_v = max(vals)
    return (
        f"Summary({radius_field}): lines={total}, finite_radii={finite}, "
        f"mean={mean_v:.6g}, min={min_v:.6g}, max={max_v:.6g}"
    )


def _load_results_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("results.json must contain a JSON object.")
    return obj


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print/export stability radius results as a table."
    )
    parser.add_argument("results_json", type=str, help="Path to results.json")
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Limit number of rows"
    )
    parser.add_argument(
        "--format", type=str, choices=("sections", "flat"), default="sections"
    )
    parser.add_argument(
        "--radius-field",
        type=str,
        default="radius_l2",
        help="Radius field to summarize.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default="",
        help="Comma-separated list of columns (flat mode). If empty, inferred from results.",
    )
    parser.add_argument(
        "--table-out", type=str, default="", help="Write ASCII table here."
    )
    parser.add_argument(
        "--csv-out", type=str, default="", help="Write CSV here (flat mode)."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    requested_output_dir = None
    if str(args.table_out).strip():
        requested_output_dir = Path(str(args.table_out)).parent
    elif str(args.csv_out).strip():
        requested_output_dir = Path(str(args.csv_out)).parent

    artifact_dir = create_module_output_dir(
        module_name="table",
        requested_output_dir=requested_output_dir,
    )
    setup_output_dir_logging(artifact_dir)

    results = _load_results_json(Path(args.results_json))
    max_rows = int(args.max_rows) if args.max_rows is not None else None

    if str(args.format) == "flat":
        columns = (
            tuple(c.strip() for c in str(args.columns).split(",") if c.strip())
            if str(args.columns).strip()
            else infer_default_flat_columns(results)
        )
        table_str = format_results_table(results, columns=columns, max_rows=max_rows)
        csv_str = format_results_csv(results, columns=columns, max_rows=max_rows)
    else:
        table_str = format_results_table_sections(results, max_rows=max_rows)
        csv_str = ""

    print(table_str)
    print(format_radius_summary(results, radius_field=str(args.radius_field)))
    default_table_path = artifact_dir / "results_table.txt"
    default_table_path.write_text(table_str + "\n", encoding="utf-8")
    logger.info("Wrote table artifact: %s", str(default_table_path))

    if str(args.table_out).strip():
        table_path = artifact_dir / Path(str(args.table_out)).name
        table_path.write_text(
            table_str + "\n", encoding="utf-8"
        )
        logger.info("Wrote explicit table output: %s", str(table_path))

    if str(args.csv_out).strip():
        if not csv_str:
            raise ValueError("--csv-out is only supported in flat mode.")
        csv_path = artifact_dir / Path(str(args.csv_out)).name
        csv_path.write_text(csv_str, encoding="utf-8")
        logger.info("Wrote CSV output: %s", str(csv_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
