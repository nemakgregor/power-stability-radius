"""Plot worst-case verification heatmaps and scatter diagnostics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from stability_radius.postprocess.cli import run_single_input_plot_cli
from stability_radius.utils import ARTIFACTS_ROOT_NAME

logger = logging.getLogger(__name__)

_DEFAULT_INPUT_DIR = Path(ARTIFACTS_ROOT_NAME) / "run_worst_case_verify"


def _load_json(path: Path) -> list | dict:
    """Internal helper for module-local processing."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def plot(input_dir: Path, output_dir: Path) -> None:
    """Execute the documented operation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for json_file in sorted(input_dir.glob("*_worst_case.json")):
        try:
            data = _load_json(json_file)
        except Exception:
            continue
        if isinstance(data, list):
            all_records.extend(data)

    if not all_records:
        logger.warning("No worst-case verification data found in %s", input_dir)
        return

    scales = sorted({r["scale"] for r in all_records})
    line_ids = sorted({r["line_id"] for r in all_records})

    if not scales or not line_ids:
        logger.warning("Insufficient data for heatmap")
        return

    n_lines = len(line_ids)
    n_scales = len(scales)
    line_id_to_pos = {lid: i for i, lid in enumerate(line_ids)}
    scale_to_pos = {s: j for j, s in enumerate(scales)}

    rel_err = np.full((n_lines, n_scales), float("nan"))
    violated = np.full((n_lines, n_scales), False)

    for r in all_records:
        i = line_id_to_pos.get(r["line_id"])
        j = scale_to_pos.get(r["scale"])
        if i is None or j is None:
            continue
        re_val = r.get("relative_error", float("nan"))
        if np.isfinite(re_val):
            rel_err[i, j] = re_val
        violated[i, j] = bool(r.get("violated", False))

    fig, ax = plt.subplots(figsize=(max(4, 2 + n_scales), max(4, 1 + 0.4 * n_lines)))
    im = ax.imshow(
        rel_err,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    ax.set_xticks(range(n_scales))
    ax.set_xticklabels([f"{s:.2f}" for s in scales])
    ax.set_yticks(range(n_lines))
    ax.set_yticklabels([str(lid) for lid in line_ids], fontsize=7)
    ax.set_xlabel("Perturbation scale")
    ax.set_ylabel("Line ID")
    ax.set_title(
        "Worst-Case Verification: Relative Error\n|predicted - actual| / actual"
    )
    fig.colorbar(im, ax=ax, label="Relative error")

    for i in range(n_lines):
        for j in range(n_scales):
            if violated[i, j]:
                ax.text(
                    j,
                    i,
                    "X",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )

    fig.tight_layout()
    out_path = output_dir / "worst_case_heatmap.pdf"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", out_path)

    predicted = [r["predicted_s_mva"] for r in all_records if r.get("pf_converged")]
    actual = [r["actual_s_mva"] for r in all_records if r.get("pf_converged")]

    if predicted and actual:
        fig2, ax2 = plt.subplots(figsize=(6, 6))

        ax2.scatter(actual, predicted, c="#4C72B0", alpha=0.6, s=30, label="Lines")

        all_vals = predicted + actual
        lo, hi = min(all_vals) * 0.9, max(all_vals) * 1.1
        ax2.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")

        ax2.set_xlabel("Actual |S| (MVA) - nonlinear PF")
        ax2.set_ylabel("Predicted |S| (MVA) - linear model")
        ax2.set_title("Worst-Case: Predicted vs Actual Apparent Power")
        ax2.legend()
        ax2.set_aspect("equal", adjustable="box")
        fig2.tight_layout()

        out_scatter = output_dir / "worst_case_scatter.pdf"
        fig2.savefig(str(out_scatter), dpi=150, bbox_inches="tight")
        plt.close(fig2)
        logger.info("Plot saved: %s", out_scatter)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""
    return run_single_input_plot_cli(
        argv=argv,
        description="Plot worst-case verification heatmap.",
        input_flag="--input-dir",
        input_default=_DEFAULT_INPUT_DIR,
        input_help="Directory with *_worst_case.json files.",
        module_name="plot_worst_case_heatmap",
        plot_func=plot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
