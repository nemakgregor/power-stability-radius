"""Experiment 1: DC vs AC radius sweep across PGLib networks.

Reads ``experiments/configs/pglib_sweep.yaml``, computes DC and AC L2 radii
for each listed PGLib case, and produces:

- Per-case JSON results in ``experiments/output/pglib_sweep/``
- ``summary.json`` with aggregated metrics
- **Table 1** (printed to stdout): case, n_b, n_l, r*_DC, r*_AC, AC/DC, time, bottleneck
- **Fig. 1** (saved as PNG): bar chart comparing r*_DC and r*_AC across cases

Usage::

    python -m experiments.run_pglib_sweep
    python -m experiments.run_pglib_sweep --config experiments/configs/pglib_sweep.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
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


def _compute_case(
    *,
    input_path: str,
    slack_bus: int,
    base_dispatch: str,
    dc_cfg: dict,
    ac_cfg: dict,
    compute_dc: bool,
    compute_ac: bool,
    allow_download: bool,
) -> dict:
    """Run compute_results_for_case with the given DC/AC flags."""
    return compute_results_for_case(
        input_path=input_path,
        slack_bus=slack_bus,
        base_dispatch=base_dispatch,
        # DC
        compute_dc=compute_dc,
        dc_mode=str(dc_cfg.get("mode", "materialize")),
        dc_chunk_size=int(dc_cfg.get("chunk_size", 64)),
        dc_dtype=np.dtype(dc_cfg.get("dtype", "float64")),
        dc_inj_std_mw=float(dc_cfg.get("inj_std_mw", 10.0)),
        dc_extensions=DCExtensionsConfig(probabilistic_enabled=True),
        # AC
        compute_ac=compute_ac,
        ac_chunk_size=int(ac_cfg.get("chunk_size", 64)),
        ac_balance=bool(ac_cfg.get("balance", True)),
        ac_pf_init=str(ac_cfg.get("pf_init", "flat")),
        ac_pf_solver=str(ac_cfg.get("pf_solver", "pandapower")),
        ac_lossless=bool(ac_cfg.get("lossless", True)),
        # shared
        allow_download=allow_download,
    )


def _find_bottleneck(results: dict) -> tuple[int, float, str]:
    """Find bottleneck line: the line with the smallest finite AC L2 radius.

    Returns (line_idx, margin, mode) where mode is 'ac' if AC radius was
    available, otherwise 'dc'.  Falls back to DC if no AC radii exist.
    """
    best_line = -1
    best_radius = float("inf")
    best_margin = float("nan")
    mode = "dc"

    for key, val in results.items():
        if not key.startswith("line_") or not isinstance(val, dict):
            continue
        lid = int(key.split("_", 1)[1])

        # Prefer AC radius for bottleneck identification.
        r_ac = val.get("radius_ac_l2")
        if r_ac is not None and np.isfinite(r_ac):
            if r_ac < best_radius:
                best_radius = float(r_ac)
                best_line = lid
                best_margin = float(val.get("margin_ac_mva", float("nan")))
                mode = "ac"
        else:
            r_dc = val.get("radius_l2")
            if r_dc is not None and np.isfinite(r_dc) and mode != "ac":
                if r_dc < best_radius:
                    best_radius = float(r_dc)
                    best_line = lid
                    best_margin = float(val.get("margin_mw", float("nan")))
                    mode = "dc"

    return best_line, best_margin, mode


def _print_table(rows: list[dict]) -> None:
    """Print Table 1 to stdout in a fixed-width format."""
    header = (
        f"{'Case':<28s} {'n_b':>5s} {'n_l':>5s} "
        f"{'r*_DC (MW)':>12s} {'r*_AC (MW)':>12s} {'AC/DC':>7s} "
        f"{'T_DC (s)':>9s} {'T_AC (s)':>9s} {'T_tot (s)':>9s} "
        f"{'Bottleneck':>11s} {'Margin':>10s}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("Table 1: DC vs AC L2 Stability Radius across PGLib-OPF cases")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in rows:
        ratio_str = f"{r['ac_dc_ratio']:.3f}" if np.isfinite(r["ac_dc_ratio"]) else "n/a"
        bn_str = f"L{r['bottleneck_line']}" if r["bottleneck_line"] >= 0 else "n/a"
        margin_str = f"{r['bottleneck_margin']:.2f}" if np.isfinite(r["bottleneck_margin"]) else "n/a"
        print(
            f"{r['case']:<28s} {r['n_buses']:>5d} {r['n_lines']:>5d} "
            f"{r['dc_r_star']:>12.4f} {r['ac_r_star']:>12.4f} {ratio_str:>7s} "
            f"{r['time_dc']:>9.2f} {r['time_ac']:>9.2f} {r['time_total']:>9.2f} "
            f"{bn_str:>11s} {margin_str:>10s}"
        )

    print(sep)
    print()


def _plot_bar_chart(rows: list[dict], output_dir: Path) -> Path:
    """Generate Fig. 1: bar chart comparing r*_DC and r*_AC across cases."""
    labels = [r["case"].replace("pglib_opf_", "") for r in rows]
    dc_vals = [r["dc_r_star"] for r in rows]
    ac_vals = [r["ac_r_star"] for r in rows]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    bars_dc = ax.bar(x - width / 2, dc_vals, width, label=r"$r^*_{\mathrm{DC}}$ (L2)", color="#4C72B0")
    bars_ac = ax.bar(x + width / 2, ac_vals, width, label=r"$r^*_{\mathrm{AC}}$ (L2)", color="#DD8452")

    ax.set_xlabel("PGLib-OPF Case")
    ax.set_ylabel("Stability Radius (MW)")
    ax.set_title("Fig. 1: DC vs AC L2 Stability Radius")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = output_dir / "fig1_dc_vs_ac_radius.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    return plot_path


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

    summary_rows: list[dict] = []

    for case in cases:
        name = case["name"]
        filename = case["file"]
        slack_bus = int(case.get("slack_bus", 0))
        input_path = str(data_dir / filename)

        logger.info("=" * 60)
        logger.info("Processing %s", name)
        logger.info("=" * 60)

        # ---- DC-only run (for timing) ----
        try:
            t_dc_start = time.perf_counter()
            dc_results = _compute_case(
                input_path=input_path,
                slack_bus=slack_bus,
                base_dispatch=base_dispatch,
                dc_cfg=dc_cfg,
                ac_cfg=ac_cfg,
                compute_dc=True,
                compute_ac=False,
                allow_download=allow_download,
            )
            time_dc = time.perf_counter() - t_dc_start
        except Exception:
            logger.exception("Failed DC computation for %s", name)
            continue

        # ---- AC-only run (for timing) ----
        try:
            t_ac_start = time.perf_counter()
            ac_results = _compute_case(
                input_path=input_path,
                slack_bus=slack_bus,
                base_dispatch=base_dispatch,
                dc_cfg=dc_cfg,
                ac_cfg=ac_cfg,
                compute_dc=False,
                compute_ac=True,
                allow_download=allow_download,
            )
            time_ac = time.perf_counter() - t_ac_start
        except Exception:
            logger.exception("Failed AC computation for %s", name)
            continue

        # ---- Combined run (for merged results JSON) ----
        try:
            t_total_start = time.perf_counter()
            combined = _compute_case(
                input_path=input_path,
                slack_bus=slack_bus,
                base_dispatch=base_dispatch,
                dc_cfg=dc_cfg,
                ac_cfg=ac_cfg,
                compute_dc=True,
                compute_ac=True,
                allow_download=allow_download,
            )
            time_total = time.perf_counter() - t_total_start
        except Exception:
            logger.exception("Failed combined computation for %s", name)
            continue

        # Remove non-serialisable h-vectors before saving.
        combined.pop("_h_vectors", None)

        case_output = output_dir / f"{name}.json"
        with case_output.open("w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2, default=_numpy_serialiser)
        logger.info("Results written: %s", case_output)

        # ---- Extract metrics ----
        meta = combined.get("__meta__", {})
        bp_dc = meta.get("base_point_dc") or {}
        bp_ac = meta.get("base_point_ac") or {}
        n_buses = len(bp_dc.get("bus_ids", bp_ac.get("bus_ids", [])))
        n_lines = sum(1 for k in combined if k.startswith("line_"))

        # DC global radius (min over all lines).
        dc_radii = []
        for key, val in combined.items():
            if key.startswith("line_") and isinstance(val, dict):
                r = val.get("radius_l2")
                if r is not None and np.isfinite(r):
                    dc_radii.append(float(r))
        dc_r_star = float(min(dc_radii)) if dc_radii else float("nan")

        # AC global radius (min over all lines).
        ac_radii = []
        for key, val in combined.items():
            if key.startswith("line_") and isinstance(val, dict):
                r = val.get("radius_ac_l2")
                if r is not None and np.isfinite(r):
                    ac_radii.append(float(r))
        ac_r_star = float(min(ac_radii)) if ac_radii else float("nan")

        # AC/DC ratio.
        if np.isfinite(dc_r_star) and dc_r_star > 0 and np.isfinite(ac_r_star):
            ac_dc_ratio = ac_r_star / dc_r_star
        else:
            ac_dc_ratio = float("nan")

        # Bottleneck line.
        bn_line, bn_margin, _ = _find_bottleneck(combined)

        row = {
            "case": name,
            "n_buses": n_buses,
            "n_lines": n_lines,
            "dc_r_star": dc_r_star,
            "ac_r_star": ac_r_star,
            "ac_dc_ratio": ac_dc_ratio,
            "time_dc": time_dc,
            "time_ac": time_ac,
            "time_total": time_total,
            "bottleneck_line": bn_line,
            "bottleneck_margin": bn_margin,
        }
        summary_rows.append(row)

    if not summary_rows:
        logger.error("No cases completed successfully.")
        return

    # ---- Write summary JSON ----
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_rows, fh, indent=2, default=_numpy_serialiser)
    logger.info("Summary written: %s", summary_path)

    # ---- Print Table 1 ----
    _print_table(summary_rows)

    # ---- Generate Fig. 1 ----
    plot_path = _plot_bar_chart(summary_rows, output_dir)
    logger.info("Plot saved: %s", plot_path)
    print(f"Figure saved: {plot_path}")


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
