from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _line_sort_key(line_key: str) -> Tuple[int, str]:
    """Sort keys like 'line_10' numerically, with a deterministic fallback."""
    try:
        return (int(line_key.split("_", 1)[1]), line_key)
    except Exception:
        return (10**18, line_key)


def _format_float(x: Any) -> str:
    """Format numeric values consistently for terminal output."""
    try:
        xf = float(x)
    except Exception:
        return str(x)

    if math.isinf(xf):
        return "inf"
    if math.isnan(xf):
        return "nan"
    return f"{xf:.6g}"


def format_results_table(
    results: Dict[str, Dict[str, Any]],
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

    Parameters
    ----------
    results:
        Mapping like {"line_0": {...}, ...}.
    columns:
        Fields to include in the table (in order).
    max_rows:
        If provided, show at most this many rows (deterministic order).

    Returns
    -------
    str
        A table ready to print to stdout.
    """
    line_keys = sorted(results.keys(), key=_line_sort_key)
    if max_rows is not None:
        line_keys = line_keys[:max_rows]

    headers = ["line"] + list(columns)
    rows: List[List[str]] = []

    for k in line_keys:
        row = [k]
        data = results.get(k, {})
        for c in columns:
            row.append(_format_float(data.get(c, "")))
        rows.append(row)

    # Compute column widths
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(values: Sequence[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    sep = "-+-".join("-" * w for w in widths)

    out_lines = [fmt_row(headers), sep]
    out_lines.extend(fmt_row(r) for r in rows)

    remaining = len(results) - len(line_keys)
    if max_rows is not None and remaining > 0:
        out_lines.append(f"... ({remaining} more rows)")

    return "\n".join(out_lines)


def _finite_radii(results: Dict[str, Dict[str, Any]]) -> List[float]:
    vals: List[float] = []
    for d in results.values():
        try:
            r = float(d.get("radius_l2", float("nan")))
        except Exception:
            continue
        if math.isfinite(r):
            vals.append(r)
    return vals


def print_results_table(
    results: Dict[str, Dict[str, Any]],
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
    """
    Print per-line results as a table to stdout.

    This is intended for CLI usage (demo scripts, quick inspection).
    """
    print(format_results_table(results, columns=columns, max_rows=max_rows))


def print_radius_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a compact summary of L2 radii.

    Summary includes count, finite count, mean/min/max over finite radii.
    """
    vals = _finite_radii(results)
    total = len(results)
    finite = len(vals)

    if finite == 0:
        print(f"Summary: lines={total}, finite_radii=0")
        return

    mean_v = sum(vals) / finite
    min_v = min(vals)
    max_v = max(vals)
    print(
        "Summary: "
        f"lines={total}, finite_radii={finite}, "
        f"mean={mean_v:.6g}, min={min_v:.6g}, max={max_v:.6g}"
    )


def _load_results_json(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("results.json must contain a JSON object.")
    return obj  # type: ignore[return-value]


def main(argv: Iterable[str] | None = None) -> int:
    """
    CLI entrypoint.

    Usage:
        python -m stability_radius.statistics.table runs/<timestamp>/results.json
    """
    parser = argparse.ArgumentParser(
        description="Print stability radius results as a table."
    )
    parser.add_argument("results_json", type=str, help="Path to results.json")
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Limit number of printed rows"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    results = _load_results_json(Path(args.results_json))
    print_results_table(results, max_rows=args.max_rows)
    print_radius_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
