"""Aggregate experiment JSON outputs into a CSV summary table.

Scans ``run_artifacts/`` subdirectories for ``summary.json`` and
per-case result JSON files, then produces a single CSV suitable for
inclusion in the paper.

Module usage::

    python -m stability_radius.postprocess.collect_results
    python -m stability_radius.postprocess.collect_results --output-dir run_artifacts --csv run_artifacts/collect_results/all_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from stability_radius.utils import (
    ARTIFACTS_ROOT_NAME,
    create_module_output_dir,
    setup_output_dir_logging,
)

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = Path(ARTIFACTS_ROOT_NAME)
_DEFAULT_CSV_PATH = Path(ARTIFACTS_ROOT_NAME) / "collect_results" / "all_results.csv"


def _load_json(path: Path) -> dict | list:
    """Internal helper for module-local processing."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _extract_radius_stats(results: dict) -> dict:
    """Extract DC and AC radius statistics from a per-case results dict."""
    dc_radii: list[float] = []
    ac_radii: list[float] = []
    sigma_radii: list[float] = []
    n_lines = 0

    for key, val in results.items():
        if not key.startswith("line_") or not isinstance(val, dict):
            continue
        n_lines += 1

        r_dc = val.get("radius_l2")
        if r_dc is not None and np.isfinite(r_dc):
            dc_radii.append(float(r_dc))

        r_ac = val.get("radius_ac_l2")
        if r_ac is not None and np.isfinite(r_ac):
            ac_radii.append(float(r_ac))

        r_sig = val.get("radius_ac_sigma")
        if r_sig is not None and np.isfinite(r_sig):
            sigma_radii.append(float(r_sig))

    def _stats(values: list[float]) -> dict:
        """Internal helper for module-local processing."""
        if not values:
            return {"min": "", "median": "", "mean": "", "max": "", "count": 0}
        arr = np.array(values)
        return {
            "min": f"{float(np.min(arr)):.6g}",
            "median": f"{float(np.median(arr)):.6g}",
            "mean": f"{float(np.mean(arr)):.6g}",
            "max": f"{float(np.max(arr)):.6g}",
            "count": len(values),
        }

    return {
        "n_lines": n_lines,
        "dc": _stats(dc_radii),
        "ac": _stats(ac_radii),
        "sigma": _stats(sigma_radii),
    }


def collect(output_dir: Path, csv_path: Path) -> None:
    """Execute the documented operation."""
    rows: list[dict] = []

    # Scan for per-case JSON results.
    for json_file in sorted(output_dir.rglob("*.json")):
        if json_file.name in ("summary.json", "sigma_arrays.json", "scalability.json"):
            continue
        if json_file.name.endswith("_worst_case.json"):
            continue

        try:
            data = _load_json(json_file)
        except Exception:
            logger.warning("Skipping unreadable file: %s", json_file)
            continue

        if not isinstance(data, dict) or "__meta__" not in data:
            continue

        meta = data["__meta__"]
        stats = _extract_radius_stats(data)

        row = {
            "experiment": str(json_file.parent.name),
            "case": str(json_file.stem),
            "input_path": str(meta.get("input_path", "")),
            "n_lines": stats["n_lines"],
            "compute_time_sec": meta.get("compute_time_sec", ""),
            "dc_r_min": stats["dc"]["min"],
            "dc_r_median": stats["dc"]["median"],
            "dc_r_mean": stats["dc"]["mean"],
            "dc_r_max": stats["dc"]["max"],
            "dc_r_count": stats["dc"]["count"],
            "ac_r_min": stats["ac"]["min"],
            "ac_r_median": stats["ac"]["median"],
            "ac_r_mean": stats["ac"]["mean"],
            "ac_r_max": stats["ac"]["max"],
            "ac_r_count": stats["ac"]["count"],
            "sigma_r_min": stats["sigma"]["min"],
            "sigma_r_median": stats["sigma"]["median"],
            "sigma_r_mean": stats["sigma"]["mean"],
            "sigma_r_max": stats["sigma"]["max"],
            "sigma_r_count": stats["sigma"]["count"],
        }
        rows.append(row)

    if not rows:
        logger.warning("No result files found in %s", output_dir)
        return

    # Write CSV.
    fieldnames = list(rows[0].keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("CSV summary written: %s (%d rows)", csv_path, len(rows))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Aggregate experiment JSON results into a CSV summary table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Root output directory to scan.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DEFAULT_CSV_PATH,
        help="Path for the output CSV file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    csv_dir = create_module_output_dir(
        module_name="collect_results",
        requested_output_dir=args.csv.parent,
    )
    setup_output_dir_logging(csv_dir)
    collect(args.output_dir, csv_dir / args.csv.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
