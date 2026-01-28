from __future__ import annotations

import argparse
import csv
import json
import math
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple


def _line_sort_key(line_key: str) -> Tuple[int, str]:
    """Sort keys like 'line_10' numerically, with a deterministic fallback."""
    try:
        return (int(line_key.split("_", 1)[1]), line_key)
    except (IndexError, ValueError):
        return (10**18, line_key)


def _is_line_key(k: str) -> bool:
    """Return True if the key looks like a per-line result key."""
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
    """
    Format per-line results into an ASCII table.

    Notes
    -----
    This function ignores non-line keys (e.g., "__meta__") to keep output stable.
    """
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
            if align_right[i]:
                out.append(v.rjust(widths[i]))
            else:
                out.append(v.ljust(widths[i]))
        return " | ".join(out)

    sep = "-+-".join("-" * w for w in widths)

    out_lines = [fmt_row(headers), sep]
    out_lines.extend(fmt_row(r) for r in rows)

    remaining = len([k for k in results.keys() if _is_line_key(k)]) - len(line_keys)
    if max_rows is not None and remaining > 0:
        out_lines.append(f"... ({remaining} more rows)")

    return "\n".join(out_lines)


def format_results_csv(
    results: dict[str, Any],
    *,
    columns: Sequence[str],
    max_rows: int | None = None,
) -> str:
    """Format per-line results as CSV (deterministic ordering)."""
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


def write_results_table(
    path: Path,
    results: dict[str, Any],
    *,
    columns: Sequence[str],
    max_rows: int | None = None,
) -> None:
    """Write an ASCII results table to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_results_table(results, columns=columns, max_rows=max_rows) + "\n",
        encoding="utf-8",
    )


def write_results_csv(
    path: Path,
    results: dict[str, Any],
    *,
    columns: Sequence[str],
    max_rows: int | None = None,
) -> None:
    """Write a CSV results table to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_results_csv(results, columns=columns, max_rows=max_rows),
        encoding="utf-8",
    )


def _finite_radii(results: dict[str, Any], *, radius_field: str) -> List[float]:
    vals: List[float] = []
    for k, d in results.items():
        if not _is_line_key(k):
            continue
        if not isinstance(d, dict):
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
    """Format a compact summary of radii in the provided `radius_field`."""
    vals = _finite_radii(results, radius_field=radius_field)
    total = len([k for k in results.keys() if _is_line_key(k)])
    finite = len(vals)

    if finite == 0:
        return f"Summary({radius_field}): lines={total}, finite_radii=0"

    mean_v = sum(vals) / finite
    min_v = min(vals)
    max_v = max(vals)
    return (
        f"Summary({radius_field}): "
        f"lines={total}, finite_radii={finite}, "
        f"mean={mean_v:.6g}, min={min_v:.6g}, max={max_v:.6g}"
    )


def print_results_table(
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
) -> None:
    """Print per-line results as a table to stdout."""
    print(format_results_table(results, columns=columns, max_rows=max_rows))


def print_radius_summary(
    results: dict[str, Any], *, radius_field: str = "radius_l2"
) -> None:
    """Print `format_radius_summary(...)` to stdout."""
    print(format_radius_summary(results, radius_field=radius_field))


def _load_results_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("results.json must contain a JSON object.")
    return obj


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entrypoint: print/export stability radius results as a table."""
    parser = argparse.ArgumentParser(
        description="Print/export stability radius results as a table."
    )
    parser.add_argument("results_json", type=str, help="Path to results.json")
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Limit number of rows"
    )
    parser.add_argument(
        "--radius-field",
        type=str,
        default="radius_l2",
        help="Radius field to summarize (default: radius_l2).",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default="",
        help="Comma-separated list of columns to print/export.",
    )
    parser.add_argument(
        "--table-out", type=str, default="", help="Write ASCII table here."
    )
    parser.add_argument("--csv-out", type=str, default="", help="Write CSV here.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    default_cols = ("p0_mw", "p_limit_mw_est", "margin_mw", "norm_g", "radius_l2")
    if str(args.columns).strip():
        columns = tuple(c.strip() for c in str(args.columns).split(",") if c.strip())
    else:
        columns = default_cols

    results = _load_results_json(Path(args.results_json))

    print_results_table(results, columns=columns, max_rows=args.max_rows)
    print_radius_summary(results, radius_field=str(args.radius_field))

    if str(args.table_out).strip():
        write_results_table(
            Path(str(args.table_out)),
            results,
            columns=columns,
            max_rows=args.max_rows,
        )
    if str(args.csv_out).strip():
        write_results_csv(
            Path(str(args.csv_out)),
            results,
            columns=columns,
            max_rows=args.max_rows,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
