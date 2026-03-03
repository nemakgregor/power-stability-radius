"""Experiment 1: DC vs AC radius sweep across PGLib networks.

Reads ``experiments/configs/pglib_sweep.yaml``, computes DC and AC L2 radii
for each listed PGLib case, and writes per-case JSON results to
``experiments/output/pglib_sweep/``.

Usage::

    python -m experiments.run_pglib_sweep
    python -m experiments.run_pglib_sweep --config experiments/configs/pglib_sweep.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

from stability_radius.workflows import (
    ACExtensionsConfig,
    DCExtensionsConfig,
    compute_results_for_case,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "pglib_sweep.yaml"


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _numpy_serialiser(obj: object) -> object:
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run(config_path: Path) -> None:
    cfg = _load_config(config_path)
    cases = cfg["cases"]
    compute_cfg = cfg.get("compute", {})
    data_dir = Path(cfg.get("data_dir", "data/input"))
    output_dir = Path(cfg.get("output_dir", "experiments/output/pglib_sweep"))
    allow_download = bool(cfg.get("allow_download", False))

    dc_cfg = compute_cfg.get("dc", {})
    ac_cfg = compute_cfg.get("ac", {})
    base_dispatch = str(compute_cfg.get("base_dispatch", "case"))

    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []

    for case in cases:
        name = case["name"]
        filename = case["file"]
        slack_bus = int(case.get("slack_bus", 0))
        input_path = str(data_dir / filename)

        logger.info("=== Processing %s ===", name)

        try:
            results = compute_results_for_case(
                input_path=input_path,
                slack_bus=slack_bus,
                base_dispatch=base_dispatch,
                # DC
                compute_dc=True,
                dc_mode=str(dc_cfg.get("mode", "materialize")),
                dc_chunk_size=int(dc_cfg.get("chunk_size", 64)),
                dc_dtype=np.dtype(dc_cfg.get("dtype", "float64")),
                dc_inj_std_mw=float(dc_cfg.get("inj_std_mw", 10.0)),
                dc_extensions=DCExtensionsConfig(probabilistic_enabled=True),
                # AC
                compute_ac=True,
                ac_chunk_size=int(ac_cfg.get("chunk_size", 64)),
                ac_balance=bool(ac_cfg.get("balance", True)),
                ac_pf_init=str(ac_cfg.get("pf_init", "flat")),
                ac_pf_solver=str(ac_cfg.get("pf_solver", "pandapower")),
                ac_lossless=bool(ac_cfg.get("lossless", True)),
                # shared
                allow_download=allow_download,
            )
        except Exception:
            logger.exception("Failed to compute radii for %s", name)
            continue

        # Remove non-serialisable h-vectors before saving.
        results.pop("_h_vectors", None)

        case_output = output_dir / f"{name}.json"
        with case_output.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=_numpy_serialiser)
        logger.info("Results written: %s", case_output)

        # Collect summary row.
        meta = results.get("__meta__", {})
        dc_radii = []
        ac_radii = []
        for key, val in results.items():
            if not key.startswith("line_"):
                continue
            if isinstance(val, dict):
                r_dc = val.get("radius_l2")
                if r_dc is not None and np.isfinite(r_dc):
                    dc_radii.append(float(r_dc))
                r_ac = val.get("radius_ac_l2")
                if r_ac is not None and np.isfinite(r_ac):
                    ac_radii.append(float(r_ac))

        summary.append(
            {
                "case": name,
                "n_lines": len([k for k in results if k.startswith("line_")]),
                "dc_r_min": float(min(dc_radii)) if dc_radii else float("nan"),
                "dc_r_median": float(np.median(dc_radii)) if dc_radii else float("nan"),
                "ac_r_min": float(min(ac_radii)) if ac_radii else float("nan"),
                "ac_r_median": float(np.median(ac_radii)) if ac_radii else float("nan"),
                "compute_time_sec": float(meta.get("compute_time_sec", float("nan"))),
            }
        )

    # Write summary JSON.
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_numpy_serialiser)
    logger.info("Summary written: %s", summary_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Experiment 1: DC vs AC radius sweep across PGLib networks.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to pglib_sweep.yaml config.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
