"""Plot sigma-radius vs compute time from the sigma-radius experiment.

Reads results from ``run_artifacts/run_sigma_radius/`` and produces a
scatter plot of per-line sigma-radius values, plus a bar chart of timing
from the scalability experiment if available.

Usage::

    python entry_points/plot_sigma_vs_time.py
    python entry_points/plot_sigma_vs_time.py --sigma-dir run_artifacts/run_sigma_radius --scalability run_artifacts/run_scalability/scalability.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stability_radius.utils import (
    ARTIFACTS_ROOT_NAME,
    create_module_output_dir,
    setup_output_dir_logging,
)

logger = logging.getLogger(__name__)

_DEFAULT_SIGMA_DIR = Path(ARTIFACTS_ROOT_NAME) / "run_sigma_radius"
_DEFAULT_SCALABILITY = (
    Path(ARTIFACTS_ROOT_NAME) / "run_scalability" / "scalability.json"
)


def _load_json(path: Path) -> dict | list:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def plot(
    sigma_dir: Path,
    scalability_path: Path | None,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sigma-radius results.
    results_files = sorted(sigma_dir.glob("*_results.json"))
    if not results_files:
        logger.warning("No result files found in %s", sigma_dir)
        return

    sigma_radii: list[float] = []
    line_ids: list[int] = []

    for rf in results_files:
        try:
            data = _load_json(rf)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for key, val in data.items():
            if not key.startswith("line_") or not isinstance(val, dict):
                continue
            r_sig = val.get("radius_ac_sigma")
            if r_sig is not None and np.isfinite(r_sig):
                sigma_radii.append(float(r_sig))
                line_ids.append(int(key.split("_")[1]))

    if not sigma_radii:
        logger.warning("No sigma-radius data found")
        return

    # Plot 1: sigma-radius distribution (sorted bar chart).
    order = np.argsort(sigma_radii)
    sorted_radii = [sigma_radii[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(sorted_radii)), sorted_radii, color="#55A868", alpha=0.8, width=1.0)
    ax.set_xlabel("Line (sorted by sigma-radius)")
    ax.set_ylabel("Sigma-radius (dimensionless)")
    ax.set_title("AC Sigma-Radius per Line (sorted)")
    ax.axhline(y=3.0, color="red", linestyle="--", linewidth=1, label="3-sigma threshold")
    ax.legend()
    fig.tight_layout()

    out_sigma = output_dir / "sigma_radius_sorted.pdf"
    fig.savefig(str(out_sigma), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", out_sigma)

    # Plot 2: scalability (if data available).
    if scalability_path is not None and scalability_path.exists():
        try:
            scal_data = _load_json(scalability_path)
        except Exception:
            logger.warning("Cannot load scalability data: %s", scalability_path)
            scal_data = []

        if isinstance(scal_data, list) and scal_data:
            names = [r["case"].replace("pglib_opf_", "") for r in scal_data]
            n_bus = [r["n_bus"] for r in scal_data]
            dc_t = [r.get("dc_time_sec_mean", float("nan")) for r in scal_data]
            ac_t = [r.get("ac_time_sec_mean", float("nan")) for r in scal_data]

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            x = np.arange(len(names))
            w = 0.35
            ax2.bar(x - w / 2, dc_t, w, label="DC", color="#4C72B0", alpha=0.8)
            ax2.bar(x + w / 2, ac_t, w, label="AC", color="#DD8452", alpha=0.8)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"{nm}\n({nb})" for nm, nb in zip(names, n_bus)],
                                rotation=45, ha="right", fontsize=8)
            ax2.set_ylabel("Wall-clock time (sec)")
            ax2.set_xlabel("Case (n_bus)")
            ax2.set_title("Compute Time: DC vs AC")
            ax2.set_yscale("log")
            ax2.legend()
            fig2.tight_layout()

            out_time = output_dir / "sigma_vs_time.pdf"
            fig2.savefig(str(out_time), dpi=150, bbox_inches="tight")
            plt.close(fig2)
            logger.info("Plot saved: %s", out_time)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Plot sigma-radius distribution and compute time.",
    )
    parser.add_argument(
        "--sigma-dir",
        type=Path,
        default=_DEFAULT_SIGMA_DIR,
        help="Directory with sigma-radius experiment results.",
    )
    parser.add_argument(
        "--scalability",
        type=Path,
        default=_DEFAULT_SCALABILITY,
        help="Path to scalability.json (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(""),
        help="Directory where plots are saved.",
    )
    args = parser.parse_args()
    output_dir = create_module_output_dir(
        module_name="plot_sigma_vs_time",
        requested_output_dir=args.output_dir,
    )
    setup_output_dir_logging(output_dir)
    plot(args.sigma_dir, args.scalability, output_dir)
