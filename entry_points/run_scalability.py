"""Experiment 4: Wall-clock time vs network size (scalability analysis).

Measures compute time for DC and AC radius computation across PGLib cases
of increasing size to produce a scalability curve.

Reads ``experiments/configs/pglib_sweep.yaml`` for the case list.

Usage::

    python entry_points/run_scalability.py
    python entry_points/run_scalability.py --config experiments/configs/pglib_sweep.yaml --repeats 3
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from stability_radius.parsers.matpower import load_network
from stability_radius.utils import (
    create_module_output_dir,
    numpy_to_builtin,
    resolve_artifacts_root,
    setup_output_dir_logging,
)
from stability_radius.workflows import (
    DCExtensionsConfig,
    compute_results_for_case,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "experiments" / "configs" / "pglib_sweep.yaml"
)


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run(config_path: Path, *, repeats: int = 3) -> None:
    cfg = _load_config(config_path)
    cases = cfg["cases"]
    compute_cfg = cfg.get("compute", {})
    data_dir = Path(cfg.get("data_dir", "data/input"))
    artifacts_root = resolve_artifacts_root(cfg)
    output_dir = create_module_output_dir(
        module_name="run_scalability",
        runs_dir=artifacts_root,
        requested_output_dir=cfg.get("scalability_output_dir", None),
    )
    allow_download = bool(cfg.get("allow_download", False))
    setup_output_dir_logging(output_dir)

    dc_cfg = compute_cfg.get("dc", {})
    ac_cfg = compute_cfg.get("ac", {})
    base_dispatch = str(compute_cfg.get("base_dispatch", "case"))

    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    for case in cases:
        name = case["name"]
        filename = case["file"]
        slack_bus = int(case.get("slack_bus", 0))
        input_path = str(data_dir / filename)

        # Get network dimensions.
        try:
            net = load_network(input_path)
            n_bus = len(net.bus)
            n_line = len(net.line)
        except Exception:
            logger.warning("Cannot load %s, skipping.", name)
            continue

        logger.info("=== %s (n_bus=%d, n_line=%d) ===", name, n_bus, n_line)

        dc_times: list[float] = []
        ac_times: list[float] = []

        for rep in range(1, repeats + 1):
            logger.info("  repeat %d/%d", rep, repeats)

            # DC-only timing.
            t0 = time.perf_counter()
            try:
                compute_results_for_case(
                    input_path=input_path,
                    slack_bus=slack_bus,
                    base_dispatch=base_dispatch,
                    compute_dc=True,
                    dc_mode=str(dc_cfg.get("mode", "materialize")),
                    dc_chunk_size=int(dc_cfg.get("chunk_size", 64)),
                    dc_dtype=np.dtype(dc_cfg.get("dtype", "float64")),
                    dc_inj_std_mw=float(dc_cfg.get("inj_std_mw", 10.0)),
                    dc_extensions=DCExtensionsConfig(probabilistic_enabled=False),
                    compute_ac=False,
                    ac_chunk_size=int(ac_cfg.get("chunk_size", 64)),
                    ac_balance=bool(ac_cfg.get("balance", True)),
                    ac_pf_init=str(ac_cfg.get("pf_init", "flat")),
                    ac_pf_solver=str(ac_cfg.get("pf_solver", "pandapower")),
                    ac_lossless=bool(ac_cfg.get("lossless", True)),
                    allow_download=allow_download,
                )
                dc_times.append(time.perf_counter() - t0)
            except Exception:
                logger.exception("DC compute failed for %s", name)
                dc_times.append(float("nan"))

            # AC-only timing.
            t0 = time.perf_counter()
            try:
                compute_results_for_case(
                    input_path=input_path,
                    slack_bus=slack_bus,
                    base_dispatch=base_dispatch,
                    compute_dc=False,
                    dc_mode=str(dc_cfg.get("mode", "materialize")),
                    dc_chunk_size=int(dc_cfg.get("chunk_size", 64)),
                    dc_dtype=np.dtype(dc_cfg.get("dtype", "float64")),
                    dc_inj_std_mw=float(dc_cfg.get("inj_std_mw", 10.0)),
                    compute_ac=True,
                    ac_chunk_size=int(ac_cfg.get("chunk_size", 64)),
                    ac_balance=bool(ac_cfg.get("balance", True)),
                    ac_pf_init=str(ac_cfg.get("pf_init", "flat")),
                    ac_pf_solver=str(ac_cfg.get("pf_solver", "pandapower")),
                    ac_lossless=bool(ac_cfg.get("lossless", True)),
                    allow_download=allow_download,
                )
                ac_times.append(time.perf_counter() - t0)
            except Exception:
                logger.exception("AC compute failed for %s", name)
                ac_times.append(float("nan"))

        records.append(
            {
                "case": name,
                "n_bus": n_bus,
                "n_line": n_line,
                "repeats": repeats,
                "dc_time_sec_mean": float(np.nanmean(dc_times)),
                "dc_time_sec_std": float(np.nanstd(dc_times)),
                "dc_time_sec_all": [float(t) for t in dc_times],
                "ac_time_sec_mean": float(np.nanmean(ac_times)),
                "ac_time_sec_std": float(np.nanstd(ac_times)),
                "ac_time_sec_all": [float(t) for t in ac_times],
            }
        )

    # Write results.
    out_path = output_dir / "scalability.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, default=numpy_to_builtin)
    logger.info("Scalability results written: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 4: wall-clock time vs network size.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to pglib_sweep.yaml config.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated runs per case for timing statistics.",
    )
    args = parser.parse_args()
    run(args.config, repeats=args.repeats)


if __name__ == "__main__":
    raise SystemExit(main())
