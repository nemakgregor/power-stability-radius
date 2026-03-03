"""Plot radius distributions (DC vs AC) across PGLib cases.

Reads per-case JSON results from ``experiments/output/pglib_sweep/`` and
produces box-plots / violin-plots of DC and AC L2 radii.

Usage::

    python -m experiments.plot_radius_distribution
    python -m experiments.plot_radius_distribution --input-dir experiments/output/pglib_sweep
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_INPUT_DIR = Path("experiments/output/pglib_sweep")
_DEFAULT_OUTPUT_DIR = Path("experiments/output/pglib_sweep")


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def plot(input_dir: Path, output_dir: Path) -> None:
    case_names: list[str] = []
    dc_data: list[list[float]] = []
    ac_data: list[list[float]] = []

    for json_file in sorted(input_dir.glob("*.json")):
        if json_file.name in ("summary.json",):
            continue
        try:
            data = _load_json(json_file)
        except Exception:
            continue
        if not isinstance(data, dict) or "__meta__" not in data:
            continue

        dc_radii: list[float] = []
        ac_radii: list[float] = []
        for key, val in data.items():
            if not key.startswith("line_") or not isinstance(val, dict):
                continue
            r_dc = val.get("radius_l2")
            if r_dc is not None and np.isfinite(r_dc):
                dc_radii.append(float(r_dc))
            r_ac = val.get("radius_ac_l2")
            if r_ac is not None and np.isfinite(r_ac):
                ac_radii.append(float(r_ac))

        if dc_radii or ac_radii:
            case_names.append(json_file.stem.replace("pglib_opf_", ""))
            dc_data.append(dc_radii)
            ac_data.append(ac_radii)

    if not case_names:
        logger.warning("No data found in %s", input_dir)
        return

    n = len(case_names)
    fig, axes = plt.subplots(1, 2, figsize=(7 + 2 * n, 6), sharey=False)

    # DC boxplot.
    ax = axes[0]
    ax.boxplot(dc_data, labels=case_names, vert=True, patch_artist=True,
               boxprops={"facecolor": "#4C72B0", "alpha": 0.7})
    ax.set_title("DC L2 Radius Distribution")
    ax.set_ylabel("Radius (MW)")
    ax.set_xlabel("Case")
    ax.tick_params(axis="x", rotation=45)
    ax.set_yscale("log")

    # AC boxplot.
    ax = axes[1]
    ax.boxplot(ac_data, labels=case_names, vert=True, patch_artist=True,
               boxprops={"facecolor": "#DD8452", "alpha": 0.7})
    ax.set_title("AC L2 Radius Distribution")
    ax.set_ylabel("Radius (MVA)")
    ax.set_xlabel("Case")
    ax.tick_params(axis="x", rotation=45)
    ax.set_yscale("log")

    fig.suptitle("Stability Radius Distribution: DC vs AC", fontsize=14)
    fig.tight_layout()

    out_path = output_dir / "radius_distribution.pdf"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", out_path)

    # Also save PNG for quick preview.
    out_png = output_dir / "radius_distribution.png"
    fig2, axes2 = plt.subplots(1, 2, figsize=(7 + 2 * n, 6), sharey=False)
    axes2[0].boxplot(dc_data, labels=case_names, vert=True, patch_artist=True,
                     boxprops={"facecolor": "#4C72B0", "alpha": 0.7})
    axes2[0].set_title("DC L2 Radius Distribution")
    axes2[0].set_ylabel("Radius (MW)")
    axes2[0].set_xlabel("Case")
    axes2[0].tick_params(axis="x", rotation=45)
    axes2[0].set_yscale("log")
    axes2[1].boxplot(ac_data, labels=case_names, vert=True, patch_artist=True,
                     boxprops={"facecolor": "#DD8452", "alpha": 0.7})
    axes2[1].set_title("AC L2 Radius Distribution")
    axes2[1].set_ylabel("Radius (MVA)")
    axes2[1].set_xlabel("Case")
    axes2[1].tick_params(axis="x", rotation=45)
    axes2[1].set_yscale("log")
    fig2.suptitle("Stability Radius Distribution: DC vs AC", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Plot saved: %s", out_png)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Plot DC vs AC radius distributions across PGLib cases.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_DEFAULT_INPUT_DIR,
        help="Directory with per-case JSON results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where plots are saved.",
    )
    args = parser.parse_args()
    plot(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
