from __future__ import annotations

"""
Comparative evaluation: stability radii vs baseline robustness metrics.

Usage::

    python -m stability_radius.metrics_analysis \
        --input data/input/pglib_opf_case30_ieee.m \
        --slack-bus 0 \
        --sigma-p 1.0 --sigma-q 1.0 \
        --mc-samples 10000 \
        --output-dir analysis_output/case30

Pipeline:

1. ``compute_results_for_case()`` — all AC radii (L2, sigma, metric).
2. Modified MC with ``track_per_line_overloads=True`` — per-line empirical
   overload fractions.
3. ``compute_baseline_metrics()`` — loading ratio, headroom, Cantelli bound.
4. Unified DataFrame → Spearman correlations → precision-at-k → plots.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

from stability_radius.metrics.ac_baselines import compute_baseline_metrics
from stability_radius.verification.monte_carlo import run_monte_carlo_verification
from stability_radius.workflows import (
    ACExtensionsConfig,
    compute_results_for_case,
)

logger = logging.getLogger("stability_radius.metrics_analysis")

# Metrics where *lower* value means *more dangerous* (negate for Spearman).
_NEGATE_FOR_CORRELATION: set[str] = {
    "radius_ac_l2",
    "radius_ac_sigma",
    "radius_ac_metric",
    "headroom_mva",
}


# ---------------------------------------------------------------------------
# JSON encoder for numpy types
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------


def build_unified_dataframe(
    *,
    results: dict[str, Any],
    baselines: dict[str, dict[str, float]],
    mc_per_line_fractions: dict[str, float],
) -> pd.DataFrame:
    """Build a single DataFrame with one row per line, columns for all metrics.

    Parameters
    ----------
    results
        Output of ``compute_results_for_case()``.
    baselines
        Output of ``compute_baseline_metrics()``.
    mc_per_line_fractions
        ``line_key`` → empirical overload fraction from MC.
    """
    rows: list[dict[str, Any]] = []

    for k, v in sorted(results.items()):
        if not k.startswith("line_") or not isinstance(v, dict):
            continue

        binding = str(v.get("binding_end", "from"))
        s0 = float(v.get(f"ac_s0_{binding}_mva", float("nan")))

        row: dict[str, Any] = {
            "line_key": k,
            "ac_s_limit_mva": float(v.get("ac_s_limit_mva", float("nan"))),
            "s0_binding_mva": s0,
            "margin_ac_mva": float(v.get("margin_ac_mva", float("nan"))),
            "radius_ac_l2": float(v.get("radius_ac_l2", float("nan"))),
            "radius_ac_sigma": float(v.get("radius_ac_sigma", float("nan"))),
            "radius_ac_metric": float(v.get("radius_ac_metric", float("nan"))),
            "sigma_flow_mva": float(v.get("sigma_flow_mva", float("nan"))),
            "overload_probability_ac": float(
                v.get("overload_probability_ac", float("nan"))
            ),
        }

        bl = baselines.get(k, {})
        row["loading_ratio"] = float(bl.get("loading_ratio", float("nan")))
        row["headroom_mva"] = float(bl.get("headroom_mva", float("nan")))
        row["cheb_prob_upper"] = float(bl.get("cheb_prob_upper", float("nan")))

        row["empirical_overload_prob"] = float(
            mc_per_line_fractions.get(k, float("nan"))
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spearman rank correlations
# ---------------------------------------------------------------------------


def compute_rank_correlations(
    df: pd.DataFrame,
    *,
    metric_columns: list[str],
    target_column: str = "empirical_overload_prob",
) -> pd.DataFrame:
    """Compute Spearman rank correlation of each metric vs *target_column*.

    For "lower-is-more-dangerous" metrics (radii, headroom) the sign is
    flipped so that a positive rho consistently means "correctly identifies
    dangerous lines".
    """
    records: list[dict[str, Any]] = []

    for col in metric_columns:
        sub = df[[col, target_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 3:
            records.append(
                {"metric": col, "spearman_rho": float("nan"), "p_value": float("nan")}
            )
            continue

        values = sub[col].values.copy()
        if col in _NEGATE_FOR_CORRELATION:
            values = -values

        if np.ptp(values) == 0.0 or np.ptp(sub[target_column].values) == 0.0:
            records.append(
                {"metric": col, "spearman_rho": float("nan"), "p_value": float("nan")}
            )
            continue

        rho, pval = scipy_stats.spearmanr(values, sub[target_column].values)
        records.append(
            {"metric": col, "spearman_rho": float(rho), "p_value": float(pval)}
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Precision-at-k
# ---------------------------------------------------------------------------


def compute_precision_at_k(
    df: pd.DataFrame,
    *,
    metric_columns: list[str],
    target_column: str = "empirical_overload_prob",
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """For each metric, rank lines by "most dangerous" and report mean
    empirical overload probability for the top-k lines.
    """
    if k_values is None:
        k_values = [3, 5, 10]

    records: list[dict[str, Any]] = []

    for col in metric_columns:
        sub = df[[col, target_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            for kk in k_values:
                records.append(
                    {
                        "metric": col,
                        "k": kk,
                        "mean_empirical_prob": float("nan"),
                        "max_empirical_prob": float("nan"),
                    }
                )
            continue

        # Sort: small radii / headroom at top; large loading / prob at top.
        ascending = col in _NEGATE_FOR_CORRELATION
        sub_sorted = sub.sort_values(col, ascending=ascending)

        for kk in k_values:
            top = sub_sorted.head(min(kk, len(sub_sorted)))
            records.append(
                {
                    "metric": col,
                    "k": kk,
                    "mean_empirical_prob": float(top[target_column].mean()),
                    "max_empirical_prob": float(top[target_column].max()),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def generate_scatter_plots(
    df: pd.DataFrame,
    *,
    metric_columns: list[str],
    target_column: str = "empirical_overload_prob",
    output_dir: Path,
) -> list[Path]:
    """Generate scatter plots: each metric (x) vs empirical overload prob (y)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for col in metric_columns:
        sub = df[[col, target_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            sub[col], sub[target_column], alpha=0.6, edgecolors="k", linewidths=0.3
        )
        ax.set_xlabel(col)
        ax.set_ylabel(target_column)
        ax.set_title(f"{col} vs {target_column}")
        ax.grid(True, alpha=0.3)

        p = output_dir / f"scatter_{col}.png"
        fig.tight_layout()
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        paths.append(p)

    return paths


def generate_comparison_histogram(
    correlations: pd.DataFrame,
    *,
    output_dir: Path,
) -> Path:
    """Bar chart comparing Spearman correlations across all metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    valid = correlations.dropna(subset=["spearman_rho"])
    if valid.empty:
        plt.close(fig)
        p = output_dir / "spearman_bar.png"
        return p

    colors = [
        "#2196F3" if m in _NEGATE_FOR_CORRELATION else "#FF9800"
        for m in valid["metric"]
    ]
    ax.barh(
        valid["metric"],
        valid["spearman_rho"],
        color=colors,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.set_xlabel("Spearman rho (positive = correctly identifies danger)")
    ax.set_title("Spearman correlation with empirical overload probability")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    p = output_dir / "spearman_bar.png"
    fig.tight_layout()
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    return p


def generate_radius_histograms(
    df: pd.DataFrame,
    *,
    output_dir: Path,
) -> Path:
    """Histogram of AC radii distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    radius_cols = ["radius_ac_l2", "radius_ac_sigma", "radius_ac_metric"]
    available = [c for c in radius_cols if c in df.columns]

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in available:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if not vals.empty:
            ax.hist(vals, bins=30, alpha=0.5, label=col, edgecolor="k", linewidth=0.3)

    ax.set_xlabel("Radius value")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of AC stability radii")
    ax.legend()
    ax.grid(True, alpha=0.3)

    p = output_dir / "radius_histograms.png"
    fig.tight_layout()
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m stability_radius.metrics_analysis",
        description="Comparative evaluation of stability radii vs baseline metrics",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to MATPOWER/PGLib case file"
    )
    parser.add_argument(
        "--slack-bus",
        type=int,
        default=None,
        help="Slack bus ID or position (auto-detected from ext_grid if omitted)",
    )
    parser.add_argument(
        "--base-dispatch",
        type=str,
        default="case",
        choices=("case", "dc_opf", "acpf", "ac_fpf"),
    )
    parser.add_argument(
        "--sigma-p", type=float, default=1.0, help="Per-bus sigma_p_mw (uniform)"
    )
    parser.add_argument(
        "--sigma-q", type=float, default=1.0, help="Per-bus sigma_q_mvar (uniform)"
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=10000,
        help="Number of MC samples for empirical overload estimation",
    )
    parser.add_argument("--mc-seed", type=int, default=42)
    parser.add_argument(
        "--top-k",
        type=str,
        default="3,5,10",
        help="Comma-separated k values for precision-at-k",
    )
    parser.add_argument("--output-dir", type=str, default="analysis_output")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve slack bus (auto-detect from ext_grid if not specified)
    # ------------------------------------------------------------------
    if args.slack_bus is not None:
        slack_bus = int(args.slack_bus)
    else:
        from stability_radius.parsers.matpower import load_network as _load_net

        _net = _load_net(Path(args.input).expanduser().resolve())
        if (
            hasattr(_net, "ext_grid")
            and _net.ext_grid is not None
            and len(_net.ext_grid) > 0
        ):
            slack_bus = int(_net.ext_grid.bus.iloc[0])
            logger.info("Auto-detected slack bus: %d (from ext_grid)", slack_bus)
        else:
            slack_bus = int(sorted(_net.bus.index)[0])
            logger.info("Auto-detected slack bus: %d (first bus)", slack_bus)
        del _net

    # ------------------------------------------------------------------
    # Step 1: Compute radii
    # ------------------------------------------------------------------
    logger.info("Step 1/4: Computing stability radii ...")
    ac_ext = ACExtensionsConfig(
        sigma_p_mw_source="uniform",
        sigma_q_mvar_source="uniform",
        sigma_p_mw_uniform=float(args.sigma_p),
        sigma_q_mvar_uniform=float(args.sigma_q),
        metric_enabled=True,
        save_h_vectors=False,
    )
    results = compute_results_for_case(
        input_path=str(args.input),
        slack_bus=slack_bus,
        base_dispatch=str(args.base_dispatch),
        compute_dc=False,
        dc_mode="operator",
        dc_chunk_size=256,
        dc_dtype=np.float64,
        dc_inj_std_mw=float(args.sigma_p),
        compute_ac=True,
        ac_chunk_size=256,
        ac_balance=True,
        ac_pf_init="flat",
        ac_pf_solver="pandapower",
        ac_lossless=True,
        ac_extensions=ac_ext,
        allow_download=True,
    )

    # Strip numpy arrays that are not JSON-serialisable.
    results_serialisable: dict[str, Any] = {}
    for rk, rv in results.items():
        if rk == "_h_vectors":
            continue
        if isinstance(rv, dict):
            clean: dict[str, Any] = {}
            for rk2, rv2 in rv.items():
                if isinstance(rv2, np.ndarray):
                    clean[rk2] = rv2.tolist()
                else:
                    clean[rk2] = rv2
            results_serialisable[rk] = clean
        else:
            results_serialisable[rk] = rv

    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(
            results_serialisable, indent=2, ensure_ascii=False, cls=_NumpyEncoder
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("  -> results saved to %s", results_path)

    # ------------------------------------------------------------------
    # Step 2: Monte Carlo with per-line tracking
    # ------------------------------------------------------------------
    logger.info("Step 2/4: Running Monte Carlo with per-line overload tracking ...")
    vr = run_monte_carlo_verification(
        mode="ac",
        results_path=results_path,
        input_case_path=Path(args.input).resolve(),
        slack_bus=slack_bus,
        n_samples=int(args.mc_samples),
        seed=int(args.mc_seed),
        ac_sigma_p_mw=float(args.sigma_p),
        ac_sigma_q_mvar=float(args.sigma_q),
        ac_pf_solver="pandapower",
        ac_lossless=True,
        track_per_line_overloads=True,
        allow_download=True,
    )

    mc_fracs: dict[str, float] = vr.comparisons.get("per_line_overload_fractions", {})
    pf_failures = vr.comparisons.get("pf_failures_gaussian", 0)
    logger.info(
        "  -> MC done: %d samples, %d PF failures, %d lines tracked",
        args.mc_samples,
        pf_failures,
        len(mc_fracs),
    )

    mc_path = output_dir / "mc_verification.json"
    mc_path.write_text(
        json.dumps(vr.to_dict(), indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Step 3: Baseline metrics
    # ------------------------------------------------------------------
    logger.info("Step 3/4: Computing baseline metrics ...")
    baselines = compute_baseline_metrics(results)

    # ------------------------------------------------------------------
    # Step 4: Analysis and visualisation
    # ------------------------------------------------------------------
    logger.info("Step 4/4: Analysis and visualisation ...")
    df = build_unified_dataframe(
        results=results,
        baselines=baselines,
        mc_per_line_fractions=mc_fracs,
    )

    df.to_csv(output_dir / "unified_per_line_metrics.csv", index=False)

    metric_cols = [
        "radius_ac_l2",
        "radius_ac_sigma",
        "radius_ac_metric",
        "loading_ratio",
        "headroom_mva",
        "cheb_prob_upper",
        "overload_probability_ac",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    corr_df = compute_rank_correlations(df, metric_columns=metric_cols)
    corr_df.to_csv(output_dir / "spearman_correlations.csv", index=False)

    k_values = [int(x) for x in args.top_k.split(",")]
    pak_df = compute_precision_at_k(df, metric_columns=metric_cols, k_values=k_values)
    pak_df.to_csv(output_dir / "precision_at_k.csv", index=False)

    generate_scatter_plots(df, metric_columns=metric_cols, output_dir=output_dir)
    generate_comparison_histogram(corr_df, output_dir=output_dir)
    generate_radius_histograms(df, output_dir=output_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Spearman Rank Correlations vs Empirical Overload Probability ===")
    print(corr_df.to_string(index=False))
    print()

    print("=== Precision-at-k ===")
    pak_pivot = pak_df.pivot(index="metric", columns="k", values="mean_empirical_prob")
    print(pak_pivot.to_string())
    print()

    print(f"All outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
