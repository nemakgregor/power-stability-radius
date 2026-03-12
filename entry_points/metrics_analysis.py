from __future__ import annotations

"""
Comparative evaluation: stability radii vs practical robustness metrics.

Usage::

    python entry_points/metrics_analysis.py \
        --input data/input/pglib_opf_case30_ieee.m \
        --slack-bus 0 \
        --sigma-p 1.0 --sigma-q 1.0 \
        --mc-samples 10000 \
        --output-dir case30

Pipeline:

1. ``compute_results_for_case()`` — all AC radii (L2, sigma, metric) with
   h-vectors saved for directional sensitivity analysis.
2. Modified MC with ``track_per_line_overloads=True`` — per-line empirical
   overload fractions (ground truth).
3. ``compute_baseline_metrics()`` — loading ratio, headroom, Cantelli bound,
   performance index.
4. ``compute_practical_metrics()`` — thermal risk index, directional
   sensitivity for canonical transfer directions.
5. Unified DataFrame → Spearman / Kendall correlations → precision-at-k →
   hidden-danger line detection → plots.
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

from stability_radius.base_point.pandapower_tools import resolve_slack_bus_id
from stability_radius.metrics.ac_baselines import (
    compute_baseline_metrics,
    compute_practical_metrics,
    transfer_margin_linearized,
)
from stability_radius.utils import (
    NumpyJSONEncoder,
    create_module_output_dir,
    setup_output_dir_logging,
)
from stability_radius.verification.monte_carlo import run_monte_carlo_verification
from stability_radius.workflows import (
    ACExtensionsConfig,
    compute_results_for_case,
)

logger = logging.getLogger(__name__)

# Metrics where *lower* value means *more dangerous* (negate for Spearman).
_NEGATE_FOR_CORRELATION: set[str] = {
    "radius_ac_l2",
    "radius_ac_sigma",
    "radius_ac_metric",
    "headroom_mva",
}


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------


def _resolve_metrics_analysis_slack_bus(net: Any, slack_bus: int | None) -> int:
    """Resolve the slack bus using the same deterministic rule as the main workflow."""
    if slack_bus is not None:
        return int(slack_bus)
    try:
        resolved = int(resolve_slack_bus_id(net, -1))
        logger.info(
            "Auto-detected slack bus: %d (shared ext_grid tie-break rule)", resolved
        )
        return resolved
    except ValueError:
        resolved = int(sorted(net.bus.index)[0])
        logger.info(
            "Auto-detected slack bus: %d (first bus; no in-service ext_grid)",
            resolved,
        )
        return resolved


def _aggregate_bus_loads_sorted(net: Any) -> tuple[pd.Series, pd.Series]:
    """Aggregate bus loads in the project's stable sorted bus ordering."""
    bus_index = pd.Index([int(x) for x in sorted(net.bus.index)], dtype=int)
    bus_load_p = net.load.groupby("bus")["p_mw"].sum().reindex(bus_index, fill_value=0.0)
    bus_load_q = net.load.groupby("bus")["q_mvar"].sum().reindex(
        bus_index, fill_value=0.0
    )
    return bus_load_p, bus_load_q


def build_unified_dataframe(
    *,
    results: dict[str, Any],
    baselines: dict[str, dict[str, float]],
    mc_per_line_fractions: dict[str, float],
    practical: dict[str, dict[str, float]] | None = None,
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
    practical
        Output of ``compute_practical_metrics()`` (optional).
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
        row["performance_index"] = float(bl.get("performance_index", float("nan")))

        row["empirical_overload_prob"] = float(
            mc_per_line_fractions.get(k, float("nan"))
        )

        if practical is not None:
            pr = practical.get(k, {})
            for pk, pv in pr.items():
                row[pk] = float(pv)

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
# Hidden-danger line detection
# ---------------------------------------------------------------------------


def find_hidden_danger_lines(
    df: pd.DataFrame,
    *,
    radius_col: str = "radius_ac_l2",
    comparison_col: str = "loading_ratio",
    target_col: str = "empirical_overload_prob",
    rank_gap_threshold: float = 0.1,
    min_overload_prob: float = 1e-6,
    max_safe_loading_ratio: float = 0.6,
) -> pd.DataFrame:
    """Identify lines that the stability radius flags as dangerous but
    the comparison metric does not.

    A "hidden-danger" line must satisfy ALL three conditions:

    1. **Practical metric says "safe"** — the line's loading ratio is
       below *max_safe_loading_ratio* (e.g. < 0.6).  This ensures we
       only flag lines that a dispatcher would genuinely consider safe.
    2. **Radius says "dangerous"** — the line's normalized rank by
       *radius_col* is much higher (= more dangerous) than its rank by
       *comparison_col* (rank gap >= *rank_gap_threshold*).
    3. **Monte Carlo confirms the danger** — the empirical overload
       probability from AC MC >= *min_overload_prob*.

    Parameters
    ----------
    df : Unified per-line DataFrame.
    radius_col : Column for the stability radius metric.
    comparison_col : Column for the practical metric to compare against.
    target_col : Empirical ground truth column.
    rank_gap_threshold : Minimum fractional rank difference (0-1).
    min_overload_prob : Minimum empirical overload probability.
    max_safe_loading_ratio : Maximum loading_ratio for the line to be
        considered "safe" by conventional metrics.

    Returns
    -------
    pd.DataFrame
        Subset of lines identified as hidden-danger, with rank columns
        and a ``safely_loaded`` flag.
    """
    need_cols = ["line_key", radius_col, comparison_col, target_col]
    has_lr = "loading_ratio" in df.columns and "loading_ratio" not in need_cols
    if has_lr:
        need_cols = need_cols + ["loading_ratio"]

    sub = df[need_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return pd.DataFrame()

    n = len(sub)
    sub = sub.copy()

    # --- absolute safety filter ---
    lr_col = "loading_ratio" if has_lr else comparison_col
    if (
        lr_col in ("loading_ratio", comparison_col)
        and comparison_col == "loading_ratio"
    ):
        sub["safely_loaded"] = sub[comparison_col] < max_safe_loading_ratio
    elif has_lr:
        sub["safely_loaded"] = sub["loading_ratio"] < max_safe_loading_ratio
    else:
        # No loading_ratio available — use bottom half of comparison metric
        median_val = sub[comparison_col].median()
        if comparison_col in _NEGATE_FOR_CORRELATION:
            sub["safely_loaded"] = sub[comparison_col] > median_val
        else:
            sub["safely_loaded"] = sub[comparison_col] < median_val

    # --- rank comparison ---
    sub["rank_radius"] = sub[radius_col].rank(ascending=True, method="min")
    ascending_comp = comparison_col in _NEGATE_FOR_CORRELATION
    sub["rank_comparison"] = sub[comparison_col].rank(
        ascending=ascending_comp, method="min"
    )
    sub["rank_gap_norm"] = (sub["rank_comparison"] - sub["rank_radius"]) / float(n)

    hidden = sub[
        sub["safely_loaded"]
        & (sub["rank_gap_norm"] >= rank_gap_threshold)
        & (sub[target_col] >= min_overload_prob)
    ].sort_values(target_col, ascending=False)

    return hidden


def verify_worst_case_directions(
    *,
    results: dict[str, Any],
    h_matrix: np.ndarray,
    line_keys: list[str],
    transfer_directions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """For each line, compare transfer margins along canonical directions
    vs the worst-case direction identified by the stability radius.

    The worst-case direction for line *l* is ``d*_l = h_l / ||h_l||``.
    The stability radius is ``r_l = margin_l / ||h_l||``, which equals
    the transfer margin along ``d*_l``.  By construction, TM(d*_l) <= TM(d)
    for any other direction ``d``.

    This function demonstrates that the canonical transfer directions
    may miss the true bottleneck: TM along the canonical directions can
    be large, while TM along the radius worst-case direction is small.

    Returns
    -------
    pd.DataFrame
        One row per line, with columns:
        ``line_key``, ``loading_ratio``, ``radius_ac_l2``,
        ``tm_worst_case`` (= radius), and ``tm_<dir_name>`` for each
        canonical direction.
    """
    margins = []
    for k in line_keys:
        v = results.get(k, {})
        margins.append(float(v.get("margin_ac_mva", float("nan"))))
    margins_arr = np.asarray(margins, dtype=float)
    n_lines = len(line_keys)

    rows: list[dict[str, Any]] = []
    for i, k in enumerate(line_keys):
        v = results.get(k, {})
        h_l = h_matrix[i]
        h_norm = float(np.linalg.norm(h_l))

        row: dict[str, Any] = {
            "line_key": k,
            "loading_ratio": float(v.get("loading_ratio", float("nan")))
            if "loading_ratio" in v
            else float("nan"),
            "radius_ac_l2": float(v.get("radius_ac_l2", float("nan"))),
        }

        # TM along worst-case direction = radius itself
        if h_norm > 1e-12 and margins_arr[i] >= 0:
            d_worst = h_l / h_norm
            tm_wc, _ = transfer_margin_linearized(
                margins_mva=margins_arr, h_vectors=h_matrix, direction=d_worst
            )
            row["tm_worst_case"] = float(tm_wc)
        else:
            row["tm_worst_case"] = float("nan")

        # TM along each canonical direction
        for dir_name, d in transfer_directions.items():
            if d.shape[0] == h_matrix.shape[1]:
                tm_d, _ = transfer_margin_linearized(
                    margins_mva=margins_arr, h_vectors=h_matrix, direction=d
                )
                row[f"tm_{dir_name}"] = float(tm_d)
            else:
                row[f"tm_{dir_name}"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def generate_rank_comparison_plot(
    df: pd.DataFrame,
    *,
    radius_col: str = "radius_ac_l2",
    comparison_col: str = "loading_ratio",
    target_col: str = "empirical_overload_prob",
    output_dir: Path,
) -> Path:
    """Scatter plot: rank by radius vs rank by comparison metric,
    colored by empirical overload probability."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sub = (
        df[["line_key", radius_col, comparison_col, target_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    p = output_dir / f"rank_comparison_{radius_col}_vs_{comparison_col}.png"
    if sub.empty:
        fig, ax = plt.subplots()
        plt.close(fig)
        return p

    n = len(sub)
    sub = sub.copy()
    sub["rank_radius"] = sub[radius_col].rank(ascending=True, method="min")
    ascending_comp = comparison_col in _NEGATE_FOR_CORRELATION
    sub["rank_comparison"] = sub[comparison_col].rank(
        ascending=ascending_comp, method="min"
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(
        sub["rank_comparison"],
        sub["rank_radius"],
        c=sub[target_col],
        cmap="YlOrRd",
        edgecolors="k",
        linewidths=0.3,
        alpha=0.8,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(target_col)

    ax.plot([1, n], [1, n], "k--", alpha=0.3, label="equal rank")
    ax.set_xlabel(f"Rank by {comparison_col} (1 = most dangerous)")
    ax.set_ylabel(f"Rank by {radius_col} (1 = most dangerous)")
    ax.set_title(f"Rank comparison: {radius_col} vs {comparison_col}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Advanced artifact generators
# ---------------------------------------------------------------------------


def generate_pairwise_correlation_heatmap(
    df: pd.DataFrame,
    *,
    metric_columns: list[str],
    output_dir: Path,
) -> Path:
    """Pairwise Spearman correlation heatmap between all metrics.

    Shows how different metrics relate to each other and to the empirical
    overload probability.  Metrics aligned so that "more dangerous" always
    corresponds to higher values (radii and headroom are sign-flipped).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = [c for c in metric_columns if c in df.columns]
    if (
        "empirical_overload_prob" in df.columns
        and "empirical_overload_prob" not in cols
    ):
        cols.append("empirical_overload_prob")

    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    p = output_dir / "pairwise_correlation_heatmap.png"
    if len(sub) < 3 or len(cols) < 2:
        fig, ax = plt.subplots()
        plt.close(fig)
        return p

    # Align signs: negate "lower-is-dangerous" metrics
    aligned = sub.copy()
    for c in cols:
        if c in _NEGATE_FOR_CORRELATION:
            aligned[c] = -aligned[c]

    corr_matrix = aligned[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))

    # Shorten labels for readability
    short = [c.replace("radius_ac_", "r_").replace("dir_sens_", "ds_") for c in cols]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)

    # Annotate cells with values
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr_matrix.values[i, j]
            if math.isfinite(val):
                color = "white" if abs(val) > 0.7 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman rho")
    ax.set_title(
        "Pairwise Spearman Correlations\n(sign-aligned: positive = both say 'dangerous')"
    )

    fig.tight_layout()
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    return p


def generate_precision_at_k_curves(
    df: pd.DataFrame,
    *,
    metric_columns: list[str],
    target_column: str = "empirical_overload_prob",
    output_dir: Path,
) -> Path:
    """Cumulative precision-at-k curves for all metrics on one plot.

    For each k from 1..n, the top-k lines by each metric are selected
    and the mean empirical overload probability is plotted.  A metric
    that consistently identifies the most dangerous lines will have
    the highest curve.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "precision_at_k_curves.png"

    n = len(df)
    if n < 2:
        fig, ax = plt.subplots()
        plt.close(fig)
        return p

    max_k = min(n, 30)
    k_range = list(range(1, max_k + 1))

    fig, ax = plt.subplots(figsize=(9, 6))

    # Color/style presets for key metrics
    style_map = {
        "radius_ac_l2": ("#D32F2F", "-", 2.5, "Stability Radius (AC L2)"),
        "radius_ac_sigma": ("#E57373", "--", 1.5, "Stability Radius (sigma)"),
        "radius_ac_metric": ("#EF9A9A", ":", 1.5, "Stability Radius (metric)"),
        "loading_ratio": ("#1976D2", "-", 2.0, "Loading Ratio"),
        "performance_index": ("#388E3C", "-", 2.0, "Performance Index (PI)"),
        "cheb_prob_upper": ("#7B1FA2", "--", 1.5, "Cantelli Upper Bound"),
        "headroom_mva": ("#FF8F00", "--", 1.5, "Headroom (MVA)"),
        "thermal_risk_index": ("#00796B", "-", 1.5, "Thermal Risk Index"),
        "overload_probability_ac": ("#455A64", ":", 1.5, "Anal. Overload Prob"),
    }

    for col in metric_columns:
        if col not in df.columns:
            continue
        sub = df[[col, target_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue

        ascending = col in _NEGATE_FOR_CORRELATION
        sub_sorted = sub.sort_values(col, ascending=ascending)

        probs = sub_sorted[target_column].values
        means = [float(np.mean(probs[:k])) for k in k_range if k <= len(probs)]
        k_actual = k_range[: len(means)]

        color, ls, lw, label = style_map.get(
            col, (None, "-", 1.0, col.replace("dir_sens_", "DS: "))
        )
        ax.plot(k_actual, means, color=color, linestyle=ls, linewidth=lw, label=label)

    ax.set_xlabel("k (top-k lines by metric)")
    ax.set_ylabel(f"Mean {target_column} of top-k")
    ax.set_title("Precision-at-k: Which metric best identifies dangerous lines?")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    return p


def generate_hidden_danger_case_study(
    df: pd.DataFrame,
    *,
    hidden_lines: pd.DataFrame,
    output_dir: Path,
    max_lines: int = 6,
) -> Path:
    """Multi-panel bar chart comparing metric values for hidden-danger lines.

    For each hidden-danger line, shows a bar per metric (normalized 0-1)
    so the reader can see that loading_ratio / PI are low (green = safe)
    while the stability radius rank is high (red = dangerous), and the
    empirical overload probability confirms the danger.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "hidden_danger_case_study.png"

    if hidden_lines.empty or df.empty:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "No hidden-danger lines found",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        return p

    show_keys = hidden_lines["line_key"].head(max_lines).tolist()
    sub = df[df["line_key"].isin(show_keys)].copy()

    # Metrics to compare (choose the most relevant ones)
    bar_metrics = []
    candidates = [
        ("loading_ratio", "Loading\nRatio", "#1976D2"),
        ("performance_index", "Perf.\nIndex", "#388E3C"),
        ("cheb_prob_upper", "Cantelli\nBound", "#7B1FA2"),
        ("thermal_risk_index", "Thermal\nRisk", "#00796B"),
    ]
    for col, label, color in candidates:
        if col in sub.columns:
            bar_metrics.append((col, label, color))

    # Add radius rank (as percentile, so higher = more dangerous)
    radius_col = "radius_ac_l2"
    emp_col = "empirical_overload_prob"

    n_metrics = len(bar_metrics) + 2  # +radius_rank + emp_prob
    n_lines = len(show_keys)

    fig, axes = plt.subplots(1, n_lines, figsize=(3.2 * n_lines, 5), sharey=True)
    if n_lines == 1:
        axes = [axes]

    for ax_idx, line_key in enumerate(show_keys):
        ax = axes[ax_idx]
        row = sub[sub["line_key"] == line_key].iloc[0]

        labels = []
        values = []
        colors = []

        # Conventional metrics (higher = more dangerous in original scale)
        for col, label, color in bar_metrics:
            val = float(row.get(col, 0.0))
            labels.append(label)
            values.append(val)
            colors.append(color)

        # Radius rank percentile (1 - percentile of radius, so higher = more dangerous)
        if radius_col in df.columns:
            all_radii = df[radius_col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(all_radii) > 0:
                r_val = float(row.get(radius_col, float("nan")))
                percentile = float((all_radii < r_val).sum()) / len(all_radii)
                # Lower radius = more dangerous, so danger_rank = 1 - percentile
                danger_pct = 1.0 - percentile
            else:
                danger_pct = 0.0
            labels.append("Radius\nDanger %")
            values.append(danger_pct)
            colors.append("#D32F2F")

        # Empirical overload probability
        if emp_col in row:
            labels.append("Empirical\nOverload")
            values.append(float(row[emp_col]))
            colors.append("#212121")

        bars = ax.bar(
            range(len(values)),
            values,
            color=colors,
            edgecolor="k",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, fontsize=7, rotation=0)
        ax.set_title(line_key, fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate bar values
        for bar_obj, val in zip(bars, values):
            if math.isfinite(val) and val > 0:
                ax.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2,
                    bar_obj.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    axes[0].set_ylabel("Value (higher = more dangerous)")
    fig.suptitle(
        "Hidden-Danger Lines: Conventional metrics say SAFE,\n"
        "Stability Radius says DANGEROUS (confirmed by MC)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    return p


def generate_tm_gap_chart(
    wc_df: pd.DataFrame,
    *,
    output_dir: Path,
    max_lines: int = 15,
) -> Path:
    """Bar chart: canonical TM vs worst-case TM for each line.

    Lines are sorted by the gap ratio (canonical / worst-case).
    Shows that the stability radius finds a direction where TM is much
    smaller than along any of the pre-chosen canonical directions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "transfer_margin_gap.png"

    if wc_df.empty:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "No worst-case data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        return p

    tm_canon_cols = [
        c for c in wc_df.columns if c.startswith("tm_") and c != "tm_worst_case"
    ]
    if not tm_canon_cols or "tm_worst_case" not in wc_df.columns:
        fig, ax = plt.subplots()
        plt.close(fig)
        return p

    show = wc_df.copy()
    show["min_canonical_tm"] = show[tm_canon_cols].min(axis=1)
    show = show.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["tm_worst_case", "min_canonical_tm"]
    )
    show = show[show["tm_worst_case"] > 0]
    show["gap_ratio"] = show["min_canonical_tm"] / show["tm_worst_case"]
    show = show.nlargest(max_lines, "gap_ratio")

    if show.empty:
        fig, ax = plt.subplots()
        plt.close(fig)
        return p

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(show))
    bar_h = 0.35

    ax.barh(
        y_pos - bar_h / 2,
        show["min_canonical_tm"].values,
        height=bar_h,
        color="#1976D2",
        edgecolor="k",
        linewidth=0.5,
        label="Min Canonical TM",
        alpha=0.85,
    )
    ax.barh(
        y_pos + bar_h / 2,
        show["tm_worst_case"].values,
        height=bar_h,
        color="#D32F2F",
        edgecolor="k",
        linewidth=0.5,
        label="Worst-case TM (= Stability Radius)",
        alpha=0.85,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(show["line_key"].values, fontsize=8)
    ax.set_xlabel("Transfer Margin (MVA)")
    ax.set_title(
        "Transfer Margin Gap: Canonical directions miss the worst-case\n"
        "(sorted by gap ratio, descending)"
    )
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate gap ratios
    for i, (_, row) in enumerate(show.iterrows()):
        ratio = row["gap_ratio"]
        if math.isfinite(ratio):
            ax.text(
                max(row["min_canonical_tm"], row["tm_worst_case"]) * 1.02,
                i,
                f"{ratio:.1f}x",
                va="center",
                fontsize=7,
                color="#B71C1C",
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    return p


def generate_danger_decomposition_plot(
    df: pd.DataFrame,
    *,
    results: dict[str, Any],
    h_matrix: np.ndarray | None,
    output_dir: Path,
) -> Path:
    """Scatter plot of margin vs ||h|| colored by empirical overload probability.

    Since radius = margin / ||h||, lines with LOW margin AND HIGH ||h||
    have small radius (dangerous).  This plot reveals WHY a low-loaded
    line can be dangerous: it has high sensitivity (large ||h||).

    Hidden-danger lines cluster in the "moderate margin + high ||h||" region.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "danger_decomposition.png"

    if h_matrix is None or df.empty:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "No h-vectors available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        return p

    line_keys = [
        k
        for k in sorted(results.keys())
        if k.startswith("line_") and isinstance(results[k], dict)
    ]

    if len(line_keys) != h_matrix.shape[0]:
        fig, ax = plt.subplots()
        plt.close(fig)
        return p

    margins = []
    h_norms = []
    emp_probs = []
    radii = []
    lr_vals = []
    keys = []

    for i, k in enumerate(line_keys):
        v = results[k]
        margin = float(v.get("margin_ac_mva", float("nan")))
        h_norm = float(np.linalg.norm(h_matrix[i]))
        radius = float(v.get("radius_ac_l2", float("nan")))

        row_match = df[df["line_key"] == k]
        if row_match.empty:
            continue
        emp_prob = float(row_match["empirical_overload_prob"].iloc[0])
        lr = (
            float(row_match["loading_ratio"].iloc[0])
            if "loading_ratio" in row_match.columns
            else float("nan")
        )

        if math.isfinite(margin) and math.isfinite(h_norm) and h_norm > 0:
            margins.append(margin)
            h_norms.append(h_norm)
            emp_probs.append(emp_prob)
            radii.append(radius)
            lr_vals.append(lr)
            keys.append(k)

    if not margins:
        fig, ax = plt.subplots()
        plt.close(fig)
        return p

    margins_arr = np.array(margins)
    h_norms_arr = np.array(h_norms)
    emp_probs_arr = np.array(emp_probs)
    lr_arr = np.array(lr_vals)
    keys_arr = np.array(keys)

    # Separate overloaded (prob > 0) from non-overloaded lines
    overloaded = emp_probs_arr > 0
    safe_mask = ~overloaded

    # Log-scale normalization for overloaded lines so low-probability
    # lines (0.0002) are visually distinct from high-probability (0.48)
    overloaded_probs = emp_probs_arr[overloaded]
    if len(overloaded_probs) > 0:
        vmin = max(overloaded_probs.min(), 1e-4)
        vmax = overloaded_probs.max()
        if vmin >= vmax:
            vmin = vmax / 10
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def _draw_panel(ax, x_arr, xlabel, title):
        # Non-overloaded lines: gray dots
        if safe_mask.any():
            ax.scatter(
                x_arr[safe_mask],
                h_norms_arr[safe_mask],
                c="lightgray",
                edgecolors="gray",
                linewidths=0.3,
                alpha=0.5,
                s=40,
                label="No overload",
                zorder=2,
            )

        # Overloaded lines: colored by log-scaled probability
        sc = None
        if overloaded.any():
            sc = ax.scatter(
                x_arr[overloaded],
                h_norms_arr[overloaded],
                c=emp_probs_arr[overloaded],
                cmap="YlOrRd",
                norm=norm,
                edgecolors="k",
                linewidths=0.5,
                alpha=0.9,
                s=80,
                zorder=3,
            )
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Empirical overload probability")

            # Label the top overloaded lines
            top_n = min(5, int(overloaded.sum()))
            top_idx = np.argsort(emp_probs_arr[overloaded])[-top_n:]
            ov_x = x_arr[overloaded]
            ov_h = h_norms_arr[overloaded]
            ov_keys = keys_arr[overloaded]
            ov_probs = emp_probs_arr[overloaded]
            for idx in top_idx:
                ax.annotate(
                    f"{ov_keys[idx]}\n({ov_probs[idx]:.1%})",
                    (ov_x[idx], ov_h[idx]),
                    fontsize=6,
                    fontweight="bold",
                    color="#B71C1C",
                    textcoords="offset points",
                    xytext=(5, 5),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                )

        n_ov = int(overloaded.sum())
        ax.set_xlabel(xlabel)
        ax.set_ylabel("||h|| (sensitivity norm)")
        ax.set_title(f"{title}\n({n_ov} overloaded lines out of {len(emp_probs_arr)})")
        ax.grid(True, alpha=0.3)
        return sc

    # Panel 1: margin vs ||h||
    sc1 = _draw_panel(
        axes[0],
        margins_arr,
        "Margin (MVA) = S_limit - |S_0|",
        "Danger Decomposition: radius = margin / ||h||",
    )

    # Draw iso-radius contours
    if len(margins_arr) > 0 and margins_arr.max() > 0 and h_norms_arr.max() > 0:
        m_range = np.linspace(0.01, margins_arr.max() * 1.1, 100)
        for r_iso in np.percentile(np.array(radii), [10, 25, 50]):
            if r_iso > 0 and math.isfinite(r_iso):
                h_contour = m_range / r_iso
                valid = h_contour <= h_norms_arr.max() * 1.3
                axes[0].plot(
                    m_range[valid],
                    h_contour[valid],
                    "--",
                    color="gray",
                    alpha=0.4,
                    linewidth=0.8,
                )
                label_idx = np.searchsorted(h_contour[valid], h_norms_arr.max() * 0.5)
                if label_idx < len(m_range[valid]):
                    axes[0].text(
                        m_range[valid][label_idx],
                        h_contour[valid][label_idx],
                        f"r={r_iso:.1f}",
                        fontsize=7,
                        color="gray",
                        alpha=0.7,
                    )

    # Panel 2: loading_ratio vs ||h||
    _draw_panel(
        axes[1],
        lr_arr,
        "Loading Ratio = |S_0| / S_limit",
        "Hidden Danger: Low LR does NOT mean safe",
    )

    # Mark the "safe by LR" region
    if any(math.isfinite(lr) for lr in lr_arr):
        axes[1].axvline(x=0.6, color="green", linestyle="--", linewidth=1.5, alpha=0.6)
        axes[1].text(
            0.58,
            h_norms_arr.max() * 0.95,
            "LR < 0.6\n'safe'",
            fontsize=8,
            color="green",
            ha="right",
            va="top",
        )

    fig.suptitle(
        "WHY the Stability Radius catches danger that Loading Ratio misses",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def generate_classification_scatter(
    df: pd.DataFrame,
    *,
    metric_columns: list[str],
    target_col: str = "empirical_overload_prob",
    output_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    """For each predictive metric generate a scatter plot that classifies
    every line into one of four categories:

    - **TP** (True Positive): metric predicts danger AND MC confirms overload.
    - **FP** (False Positive): metric predicts danger BUT MC shows no overload.
    - **FN** (False Negative / Hidden Danger): metric predicts safe BUT MC
      confirms overload.  These are the lines the metric *misses*.
    - **TN** (True Negative): metric predicts safe AND MC shows no overload.

    The "danger" threshold for each metric is chosen so that exactly *K*
    lines are predicted dangerous, where *K* = number of MC-overloaded lines.
    This makes precision/recall directly comparable across metrics.

    Returns
    -------
    (output_dir, classification_df)
        ``classification_df`` has one row per line and a column per metric
        with the label TP / FP / FN / TN.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    mc_positive = df[target_col] > 0
    k = int(mc_positive.sum())

    if k == 0 or df.empty:
        return output_dir, pd.DataFrame()

    # Exclude a-posteriori metrics that use MC output.
    skip = {"thermal_risk_index"}
    predictive = [c for c in metric_columns if c in df.columns and c not in skip]

    # Build classification table for ALL lines.
    cls_df = df[["line_key", target_col]].copy()
    cls_df["mc_overloaded"] = mc_positive

    for col in predictive:
        ascending = col in _NEGATE_FOR_CORRELATION
        ranked = df[col].rank(ascending=ascending, method="first")
        predicted_danger = ranked <= k
        labels = pd.Series("TN", index=df.index)
        labels[predicted_danger & mc_positive] = "TP"
        labels[predicted_danger & ~mc_positive] = "FP"
        labels[~predicted_danger & mc_positive] = "FN"
        cls_df[f"cls_{col}"] = labels
        cls_df[col] = df[col]

    # Save full classification table.
    cls_df.to_csv(output_dir / "classification_all_lines.csv", index=False)

    # ---------- per-metric scatter plots ----------
    colors = {"TP": "#4CAF50", "FP": "#FF9800", "FN": "#E53935", "TN": "#BDBDBD"}
    labels_nice = {
        "TP": "True Positive (danger confirmed)",
        "FP": "False Positive (false alarm)",
        "FN": "False Negative (missed danger)",
        "TN": "True Negative (safe confirmed)",
    }

    for col in predictive:
        cls_col = f"cls_{col}"
        ascending = col in _NEGATE_FOR_CORRELATION

        fig, ax = plt.subplots(figsize=(9, 7))

        # Draw points by category (TN first so they stay behind).
        for cat in ["TN", "FP", "FN", "TP"]:
            mask = cls_df[cls_col] == cat
            if not mask.any():
                continue
            size = 30 if cat == "TN" else 70
            alpha = 0.4 if cat == "TN" else 0.85
            ax.scatter(
                cls_df.loc[mask, col],
                cls_df.loc[mask, target_col],
                c=colors[cat],
                s=size,
                alpha=alpha,
                edgecolors="k" if cat != "TN" else "gray",
                linewidths=0.4,
                label=labels_nice[cat],
                zorder=2 if cat == "TN" else 3,
            )

        # Draw threshold line.
        threshold_vals = df[col].sort_values(ascending=ascending)
        threshold = float(threshold_vals.iloc[k - 1])
        ax.axvline(
            x=threshold,
            color="#1565C0",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        side = "left" if ascending else "right"
        ax.text(
            threshold,
            ax.get_ylim()[1] * 0.95,
            f"  top-{k} threshold",
            fontsize=8,
            color="#1565C0",
            ha="left" if ascending else "right",
            va="top",
        )

        # Label FN lines (hidden dangers).
        fn_mask = cls_df[cls_col] == "FN"
        fn_rows = cls_df[fn_mask].nlargest(5, target_col)
        for _, row in fn_rows.iterrows():
            ax.annotate(
                row["line_key"],
                (row[col], row[target_col]),
                fontsize=6,
                color="#B71C1C",
                fontweight="bold",
                textcoords="offset points",
                xytext=(5, 5),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )

        # Confusion matrix counts in legend title.
        tp = int((cls_df[cls_col] == "TP").sum())
        fp = int((cls_df[cls_col] == "FP").sum())
        fn = int((cls_df[cls_col] == "FN").sum())
        tn = int((cls_df[cls_col] == "TN").sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        col_label = col.replace("_", " ").title()
        direction = "(lower = danger)" if ascending else "(higher = danger)"
        ax.set_xlabel(f"{col_label} {direction}")
        ax.set_ylabel("Empirical Overload Probability (MC)")
        ax.set_title(
            f"Classification by {col_label}\n"
            f"TP={tp}  FP={fp}  FN={fn}  TN={tn}"
            f"  |  Precision={precision:.0%}  Recall={recall:.0%}"
        )
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(
            str(output_dir / f"classification_{col}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    return output_dir, cls_df


def generate_summary_comparison_table(
    df: pd.DataFrame,
    *,
    hidden_lines_lr: pd.DataFrame,
    corr_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Generate a summary comparison table as a formatted CSV.

    The table has two sections:
    1. Per-metric summary row: Spearman rho, precision-at-3, precision-at-5.
    2. Per-line detail for hidden-danger lines with all metric values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Section 1: Per-metric summary
    summary_rows: list[dict[str, Any]] = []
    for _, cr in corr_df.iterrows():
        metric = cr["metric"]
        rho = cr["spearman_rho"]
        pval = cr["p_value"]

        # Compute precision-at-3 and precision-at-5
        if metric in df.columns:
            sub = (
                df[[metric, "empirical_overload_prob"]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            ascending = metric in _NEGATE_FOR_CORRELATION
            sub_sorted = sub.sort_values(metric, ascending=ascending)
            p3 = (
                float(
                    sub_sorted.head(min(3, len(sub_sorted)))[
                        "empirical_overload_prob"
                    ].mean()
                )
                if not sub_sorted.empty
                else float("nan")
            )
            p5 = (
                float(
                    sub_sorted.head(min(5, len(sub_sorted)))[
                        "empirical_overload_prob"
                    ].mean()
                )
                if not sub_sorted.empty
                else float("nan")
            )
        else:
            p3 = float("nan")
            p5 = float("nan")

        summary_rows.append(
            {
                "metric": metric,
                "spearman_rho": float(rho) if math.isfinite(float(rho)) else None,
                "p_value": float(pval) if math.isfinite(float(pval)) else None,
                "precision_at_3": p3,
                "precision_at_5": p5,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_metric_comparison.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")

    # Section 2: Hidden-danger lines full detail
    if not hidden_lines_lr.empty and not df.empty:
        hd_keys = hidden_lines_lr["line_key"].tolist()
        detail = df[df["line_key"].isin(hd_keys)].copy()

        # Add ranks
        radius_col = "radius_ac_l2"
        if radius_col in df.columns:
            all_ranked = df.copy()
            all_ranked["rank_by_radius"] = all_ranked[radius_col].rank(
                ascending=True, method="min"
            )
            if "loading_ratio" in all_ranked.columns:
                all_ranked["rank_by_loading_ratio"] = all_ranked["loading_ratio"].rank(
                    ascending=False, method="min"
                )
            if "performance_index" in all_ranked.columns:
                all_ranked["rank_by_perf_index"] = all_ranked["performance_index"].rank(
                    ascending=False, method="min"
                )

            detail = all_ranked[all_ranked["line_key"].isin(hd_keys)]

        detail_path = output_dir / "hidden_danger_lines_detail.csv"
        detail.to_csv(detail_path, index=False, float_format="%.6f")

    return summary_path


# ---------------------------------------------------------------------------
# Transfer direction generation
# ---------------------------------------------------------------------------


def _generate_canonical_transfer_directions(
    results: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Generate canonical transfer directions from the base-point metadata.

    Several directions are generated to cover different transfer scenarios,
    since transfer margin / ATC can only assess danger along pre-chosen
    directions.  The stability radius, by contrast, finds the worst
    direction automatically.

    Directions returned
    -------------------
    - ``max_gen_to_max_load``: inject +1 MW at the largest-generation bus,
      withdraw -1 MW at the largest-load bus.
    - ``second_gen_to_second_load``: same pattern for the second-largest
      gen and load buses, providing a different transfer corridor.
    - ``uniform_stress``: +1 MW at every non-slack bus (uniform load
      increase), balanced by the slack.

    All directions are in the *reduced* theta-variable space (n_bus - 1)
    and must be zero-padded to n_vars when h-vectors include V-variables.

    Returns
    -------
    dict
        ``name`` -> direction vector of shape ``(n_bus - 1,)``.
    """
    meta = results.get("__meta__", {})
    bp_ac = meta.get("base_point_ac", {})
    bus_ids = bp_ac.get("bus_ids", [])
    n_bus = len(bus_ids)
    if n_bus < 2:
        return {}

    slack_bus_id = meta.get("slack_bus", bus_ids[0])
    bus_p_mw_raw = bp_ac.get("bus_p_mw", None)

    directions: dict[str, np.ndarray] = {}

    if bus_p_mw_raw is not None and len(bus_p_mw_raw) == n_bus:
        bus_p = np.asarray(bus_p_mw_raw, dtype=float)

        # Sort non-slack buses by injection: generators > 0, loads < 0
        non_slack = [(i, bus_p[i]) for i in range(n_bus) if bus_ids[i] != slack_bus_id]

        gens = sorted([(i, p) for i, p in non_slack if p > 0], key=lambda x: -x[1])
        loads = sorted([(i, p) for i, p in non_slack if p < 0], key=lambda x: x[1])

        # Direction 1: max gen -> max load
        if gens and loads and gens[0][0] != loads[0][0]:
            directions["max_gen_to_max_load"] = _make_two_bus_direction(
                n_bus=n_bus,
                inject_pos=gens[0][0],
                withdraw_pos=loads[0][0],
                slack_bus_id=slack_bus_id,
                bus_ids=bus_ids,
            )

        # Direction 2: second-largest gen -> second-largest load
        if len(gens) >= 2 and len(loads) >= 2 and gens[1][0] != loads[1][0]:
            directions["second_gen_to_second_load"] = _make_two_bus_direction(
                n_bus=n_bus,
                inject_pos=gens[1][0],
                withdraw_pos=loads[1][0],
                slack_bus_id=slack_bus_id,
                bus_ids=bus_ids,
            )

    # Direction 3: uniform stress (all non-slack buses +1, slack absorbs)
    d_uniform = np.ones(n_bus, dtype=float)
    slack_pos = 0
    for i, bid in enumerate(bus_ids):
        if bid == slack_bus_id:
            slack_pos = i
            break
    d_uniform[slack_pos] = 0.0
    d_uniform -= d_uniform.mean()
    mask = np.ones(n_bus, dtype=bool)
    mask[slack_pos] = False
    directions["uniform_stress"] = d_uniform[mask]

    return directions


def _make_two_bus_direction(
    *,
    n_bus: int,
    inject_pos: int,
    withdraw_pos: int,
    slack_bus_id: int,
    bus_ids: list[int],
) -> np.ndarray:
    """Create a sum-zero P-injection direction for a source-sink pair.

    Returns a vector in the *reduced* theta-variable space (n_bus - 1).
    """
    slack_pos = 0
    for i, bid in enumerate(bus_ids):
        if bid == slack_bus_id:
            slack_pos = i
            break

    d_full = np.zeros(n_bus, dtype=float)
    d_full[inject_pos] = 1.0
    d_full[withdraw_pos] = -1.0

    # Project to sum-zero
    d_full -= d_full.mean()

    # Reduce: remove slack bus position
    mask = np.ones(n_bus, dtype=bool)
    mask[slack_pos] = False
    d_red = d_full[mask]

    return d_red


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python entry_points/metrics_analysis.py",
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
        "--sigma-p-scale",
        type=float,
        default=None,
        help=(
            "If set, overrides --sigma-p with load-proportional sigma: "
            "sigma_p[i] = max(sigma_p_min, load_p[i] * scale). "
            "E.g. --sigma-p-scale 0.05 uses 5%% of bus load as std-dev."
        ),
    )
    parser.add_argument(
        "--sigma-p-min",
        type=float,
        default=1.0,
        help="Minimum sigma_p_mw per bus when using --sigma-p-scale (default 1.0 MW)",
    )
    parser.add_argument(
        "--sigma-q-scale",
        type=float,
        default=None,
        help="Like --sigma-p-scale but for Q: sigma_q[i] = max(sigma_q_min, load_q[i] * scale)",
    )
    parser.add_argument(
        "--sigma-q-min",
        type=float,
        default=1.0,
        help="Minimum sigma_q_mvar per bus when using --sigma-q-scale (default 1.0 MVAr)",
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional artifact subdirectory name under run_artifacts/metrics_analysis/",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    output_dir = create_module_output_dir(
        module_name="metrics_analysis",
        requested_output_dir=args.output_dir,
    )
    setup_output_dir_logging(
        output_dir,
        level_console=str(args.log_level),
        level_file="DEBUG",
    )

    # ------------------------------------------------------------------
    # Resolve slack bus (auto-detect from ext_grid if not specified)
    # ------------------------------------------------------------------
    from stability_radius.parsers.matpower import load_network as _load_net

    _net = _load_net(Path(args.input).expanduser().resolve())
    slack_bus = _resolve_metrics_analysis_slack_bus(_net, args.slack_bus)

    # ------------------------------------------------------------------
    # Build per-bus sigma arrays (uniform or load-proportional)
    # ------------------------------------------------------------------
    n_bus_total = int(len(_net.bus))

    # Sum load at each bus once (used for both P and Q scale modes)
    bus_load_p, bus_load_q = _aggregate_bus_loads_sorted(_net)

    if args.sigma_p_scale is not None:
        scale_p = float(args.sigma_p_scale)
        min_p = float(args.sigma_p_min)
        sigma_p_array = np.maximum(min_p, bus_load_p.values * scale_p)
        logger.info(
            "Load-proportional sigma_p: scale=%.2f, min=%.1f -> range [%.2f, %.2f] MW",
            scale_p, min_p, float(sigma_p_array.min()), float(sigma_p_array.max()),
        )
    else:
        sigma_p_array = None

    if args.sigma_q_scale is not None:
        scale_q = float(args.sigma_q_scale)
        min_q = float(args.sigma_q_min)
        sigma_q_array = np.maximum(min_q, bus_load_q.values * scale_q)
        logger.info(
            "Load-proportional sigma_q: scale=%.2f, min=%.1f -> range [%.2f, %.2f] MVAr",
            scale_q, min_q, float(sigma_q_array.min()), float(sigma_q_array.max()),
        )
    elif sigma_p_array is not None:
        # Q uniform when only P scale is given
        sigma_q_array = np.full(n_bus_total, float(args.sigma_q))
    else:
        sigma_q_array = None

    sigma_p_for_radius = (
        sigma_p_array
        if sigma_p_array is not None
        else np.full(n_bus_total, float(args.sigma_p))
    )
    sigma_q_for_radius = (
        sigma_q_array
        if sigma_q_array is not None
        else np.full(n_bus_total, float(args.sigma_q))
    )
    del _net

    # ------------------------------------------------------------------
    # Step 1: Compute radii (with h-vectors for directional analysis)
    # ------------------------------------------------------------------
    logger.info("Step 1/5: Computing stability radii ...")
    _use_nonuniform = sigma_p_array is not None or sigma_q_array is not None
    if _use_nonuniform:
        ac_ext = ACExtensionsConfig(
            sigma_p_mw_source="uc_jl",
            sigma_q_mvar_source="uc_jl",
            sigma_p_mw_array=sigma_p_for_radius.tolist(),
            sigma_q_mvar_array=sigma_q_for_radius.tolist(),
            metric_enabled=True,
            save_h_vectors=True,
        )
    else:
        ac_ext = ACExtensionsConfig(
            sigma_p_mw_source="uniform",
            sigma_q_mvar_source="uniform",
            sigma_p_mw_uniform=float(args.sigma_p),
            sigma_q_mvar_uniform=float(args.sigma_q),
            metric_enabled=True,
            save_h_vectors=True,
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

    # Normalise pf_solver for MC compatibility: ac_fpf uses runopp then
    # validates with runpp, so the base-point flows are pandapower-compatible.
    _meta_ser = results_serialisable.get("__meta__", {})
    _bp_ser = _meta_ser.get("base_point_ac", {})
    if str(_bp_ser.get("pf_solver", "")).lower() == "pandapower_opp":
        _bp_ser["pf_solver"] = "pandapower"

    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(
            results_serialisable, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("  -> results saved to %s", results_path)

    # ------------------------------------------------------------------
    # Step 2: Monte Carlo with per-line tracking
    # ------------------------------------------------------------------
    logger.info("Step 2/5: Running Monte Carlo with per-line overload tracking ...")
    vr = run_monte_carlo_verification(
        mode="ac",
        results_path=results_path,
        input_case_path=Path(args.input).resolve(),
        slack_bus=slack_bus,
        n_samples=int(args.mc_samples),
        seed=int(args.mc_seed),
        ac_sigma_p_mw=sigma_p_for_radius,
        ac_sigma_q_mvar=sigma_q_for_radius,
        ac_pf_solver="pandapower",
        ac_lossless=True,
        ac_basepoint_s_tol_mva=0.01,
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
        json.dumps(vr.to_dict(), indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
        + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Step 3: Baseline metrics (loading ratio, headroom, Cantelli, PI)
    # ------------------------------------------------------------------
    logger.info("Step 3/5: Computing baseline metrics ...")
    baselines = compute_baseline_metrics(results)

    # ------------------------------------------------------------------
    # Step 4: Practical metrics (thermal risk, directional sensitivity)
    # ------------------------------------------------------------------
    logger.info("Step 4/5: Computing practical metrics ...")

    # Extract h-vectors for directional analysis
    h_vecs = results.get("_h_vectors", None)
    h_matrix: np.ndarray | None = None
    transfer_dirs: dict[str, np.ndarray] | None = None

    if h_vecs is not None:
        h_from = h_vecs.get("h_from", None)
        h_to = h_vecs.get("h_to", None)
        if h_from is not None and h_to is not None:
            # Use binding-end h-vectors per line
            line_keys_ordered = [
                k
                for k in sorted(results.keys())
                if k.startswith("line_") and isinstance(results[k], dict)
            ]
            h_list = []
            for i, k in enumerate(line_keys_ordered):
                binding = str(results[k].get("binding_end", "from"))
                if binding == "to":
                    h_list.append(h_to[i])
                else:
                    h_list.append(h_from[i])
            h_matrix = np.array(h_list, dtype=float)

            # Generate canonical transfer directions
            raw_dirs = _generate_canonical_transfer_directions(results)
            if raw_dirs:
                # Pad theta-only directions with zeros for V-block
                n_vars = h_matrix.shape[1]
                transfer_dirs = {}
                for name, d_red in raw_dirs.items():
                    if d_red.shape[0] == n_vars:
                        transfer_dirs[name] = d_red
                    elif d_red.shape[0] < n_vars:
                        d_padded = np.zeros(n_vars, dtype=float)
                        d_padded[: d_red.shape[0]] = d_red
                        transfer_dirs[name] = d_padded
                    else:
                        logger.warning(
                            "Skipping direction '%s': dimension mismatch "
                            "(%d vs n_vars=%d)",
                            name,
                            d_red.shape[0],
                            n_vars,
                        )

    practical = compute_practical_metrics(
        results=results,
        mc_per_line_fractions=mc_fracs,
        h_vectors=h_matrix,
        transfer_directions=transfer_dirs,
    )

    # ------------------------------------------------------------------
    # Step 5: Analysis and visualisation
    # ------------------------------------------------------------------
    logger.info("Step 5/5: Analysis and visualisation ...")
    df = build_unified_dataframe(
        results=results,
        baselines=baselines,
        mc_per_line_fractions=mc_fracs,
        practical=practical,
    )

    df.to_csv(output_dir / "unified_per_line_metrics.csv", index=False)

    metric_cols = [
        "radius_ac_l2",
        "radius_ac_sigma",
        "radius_ac_metric",
        "loading_ratio",
        "headroom_mva",
        "cheb_prob_upper",
        "performance_index",
        "overload_probability_ac",
        "thermal_risk_index",
    ]
    # Add directional sensitivity columns dynamically
    dir_sens_cols = [c for c in df.columns if c.startswith("dir_sens_")]
    metric_cols.extend(dir_sens_cols)
    metric_cols = [c for c in metric_cols if c in df.columns]

    corr_df = compute_rank_correlations(df, metric_columns=metric_cols)
    corr_df.to_csv(output_dir / "spearman_correlations.csv", index=False)

    k_values = [int(x) for x in args.top_k.split(",")]
    pak_df = compute_precision_at_k(df, metric_columns=metric_cols, k_values=k_values)
    pak_df.to_csv(output_dir / "precision_at_k.csv", index=False)

    generate_scatter_plots(df, metric_columns=metric_cols, output_dir=output_dir)
    generate_comparison_histogram(corr_df, output_dir=output_dir)
    generate_radius_histograms(df, output_dir=output_dir)

    # Hidden-danger line detection
    hidden_lr = find_hidden_danger_lines(
        df, radius_col="radius_ac_l2", comparison_col="loading_ratio"
    )
    if not hidden_lr.empty:
        hidden_lr.to_csv(output_dir / "hidden_danger_vs_loading_ratio.csv", index=False)

    hidden_pi = find_hidden_danger_lines(
        df, radius_col="radius_ac_l2", comparison_col="performance_index"
    )
    if not hidden_pi.empty:
        hidden_pi.to_csv(
            output_dir / "hidden_danger_vs_performance_index.csv", index=False
        )

    # Rank comparison plots
    generate_rank_comparison_plot(
        df,
        radius_col="radius_ac_l2",
        comparison_col="loading_ratio",
        output_dir=output_dir,
    )
    generate_rank_comparison_plot(
        df,
        radius_col="radius_ac_l2",
        comparison_col="performance_index",
        output_dir=output_dir,
    )

    # Worst-case direction verification
    wc_df = pd.DataFrame()
    if h_matrix is not None and transfer_dirs:
        line_keys_ordered = [
            k
            for k in sorted(results.keys())
            if k.startswith("line_") and isinstance(results[k], dict)
        ]
        wc_df = verify_worst_case_directions(
            results=results,
            h_matrix=h_matrix,
            line_keys=line_keys_ordered,
            transfer_directions=transfer_dirs,
        )
        wc_df.to_csv(output_dir / "worst_case_direction_verification.csv", index=False)

    # --- Advanced artifacts ---
    generate_pairwise_correlation_heatmap(
        df,
        metric_columns=metric_cols,
        output_dir=output_dir,
    )
    generate_precision_at_k_curves(
        df,
        metric_columns=metric_cols,
        output_dir=output_dir,
    )
    generate_hidden_danger_case_study(
        df,
        hidden_lines=hidden_lr,
        output_dir=output_dir,
    )
    if not wc_df.empty:
        generate_tm_gap_chart(wc_df, output_dir=output_dir)
    generate_danger_decomposition_plot(
        df,
        results=results,
        h_matrix=h_matrix,
        output_dir=output_dir,
    )
    generate_summary_comparison_table(
        df,
        hidden_lines_lr=hidden_lr,
        corr_df=corr_df,
        output_dir=output_dir,
    )
    generate_classification_scatter(
        df,
        metric_columns=metric_cols,
        output_dir=output_dir,
    )

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

    if not hidden_lr.empty:
        print(
            f"=== Hidden-danger lines (radius vs loading_ratio): "
            f"{len(hidden_lr)} found ==="
        )
        print(
            hidden_lr[
                [
                    "line_key",
                    "radius_ac_l2",
                    "loading_ratio",
                    "empirical_overload_prob",
                    "rank_radius",
                    "rank_comparison",
                    "rank_gap_norm",
                ]
            ].to_string(index=False)
        )
        print()

    if not hidden_pi.empty:
        print(
            f"=== Hidden-danger lines (radius vs performance_index): "
            f"{len(hidden_pi)} found ==="
        )
        print(
            hidden_pi[
                [
                    "line_key",
                    "radius_ac_l2",
                    "performance_index",
                    "empirical_overload_prob",
                    "rank_radius",
                    "rank_comparison",
                    "rank_gap_norm",
                ]
            ].to_string(index=False)
        )
        print()

    if not wc_df.empty:
        print("=== Worst-case direction verification (sample) ===")
        # Show lines where canonical TMs are large but worst-case TM is small
        tm_canon_cols = [
            c for c in wc_df.columns if c.startswith("tm_") and c != "tm_worst_case"
        ]
        if tm_canon_cols:
            wc_show = wc_df.copy()
            wc_show["min_canonical_tm"] = wc_show[tm_canon_cols].min(axis=1)
            wc_show["canonical_vs_worst"] = wc_show["min_canonical_tm"] / wc_show[
                "tm_worst_case"
            ].replace(0, np.nan)
            # Show lines where canonical TM >> worst-case TM (ratio > 2)
            interesting = wc_show[wc_show["canonical_vs_worst"] > 2.0].nsmallest(
                10, "tm_worst_case"
            )
            if not interesting.empty:
                show_cols = [
                    "line_key",
                    "radius_ac_l2",
                    "tm_worst_case",
                    "min_canonical_tm",
                    "canonical_vs_worst",
                ]
                print(interesting[show_cols].to_string(index=False))
                print(
                    "\n  (canonical_vs_worst > 1 means canonical transfer directions"
                    "\n   underestimate the danger found by the stability radius)"
                )
            else:
                print("  No lines with canonical TM >> worst-case TM found.")
        print()

    print(f"All outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

