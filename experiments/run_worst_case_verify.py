"""Experiment 3: Worst-case perturbation verification via nonlinear AC PF.

For the bottleneck line (smallest r*_AC) of each PGLib case, constructs
worst-case perturbations at multiple scale factors and verifies them
with full nonlinear pandapower AC power flow.

Produces:
- Table 3: verification summary per case (crossing alpha, linearization error)
- Figure 3: predicted vs actual flow curves (multi-panel, one per case)
- Per-case JSON results with all scale-factor details
- Validation checks (crossing >= 0.95, PF divergence detection)

Requires a prior run of ``run_pglib_sweep.py`` with ``save_h_vectors=true``
(or uses ``--recompute`` flag to regenerate h-vectors on the fly).

Usage::

    python -m experiments.run_worst_case_verify \\
        --sweep-dir experiments/output/pglib_sweep_good_v2
    python -m experiments.run_worst_case_verify \\
        --results experiments/output/pglib_sweep/pglib_opf_case30_ieee.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from stability_radius.base_point.ac import solve_ac_fpf_base_point
from stability_radius.base_point.pandapower_tools import resolve_slack_bus_id
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.ac_l2 import compute_ac_l2_radius
from stability_radius.verification.verify_worst_case import verify_worst_case
from stability_radius.workflows import _expand_h_reduced_to_full

logger = logging.getLogger(__name__)

_DEFAULT_SCALES = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5]


def _numpy_serialiser(obj: object) -> object:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# h-vector loading / recomputation
# ---------------------------------------------------------------------------


def _load_h_vectors_npz(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Load h-vectors from an NPZ file.

    Returns (h_from_full, h_to_full, bus_ids, line_ids).
    """
    hvecs = np.load(str(npz_path))
    return (
        hvecs["h_from"],
        hvecs["h_to"],
        hvecs["bus_ids"].tolist(),
        hvecs["line_ids"].tolist(),
    )


def _recompute_h_vectors(
    input_path: str,
    slack_bus: int,
    *,
    lossless: bool = True,
    balance: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], Any]:
    """Recompute h-vectors by running AC OPF + AC L2 with return_h_vectors.

    Returns (h_from_full, h_to_full, bus_ids, line_ids, net).
    """
    net = load_network(input_path)
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)

    _bp_ac, base_pf = solve_ac_fpf_base_point(
        net=net,
        slack_bus=slack_bus,
        lossless=lossless,
    )

    ac_l2 = compute_ac_l2_radius(
        net,
        base_pf=base_pf,
        slack_bus=slack_bus,
        chunk_size=64,
        balance=balance,
        lossless=lossless,
        return_h_vectors=True,
    )

    h_vecs_raw = ac_l2.pop("_h_vectors", None)
    if h_vecs_raw is None:
        raise RuntimeError("AC L2 did not return h-vectors")

    slack_bus_id = resolve_slack_bus_id(net, slack_bus)
    slack_pos = bus_ids.index(slack_bus_id)

    h_from = _expand_h_reduced_to_full(
        h_vecs_raw["h_from"],
        n_bus=n_bus,
        slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )
    h_to = _expand_h_reduced_to_full(
        h_vecs_raw["h_to"],
        n_bus=n_bus,
        slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )

    line_ids = [int(x) for x in sorted(net.line.index)]
    return h_from, h_to, bus_ids, line_ids, net


# ---------------------------------------------------------------------------
# Per-case multi-scale verification
# ---------------------------------------------------------------------------


def _verify_case(
    *,
    case_name: str,
    results: dict,
    net: Any,
    h_from_full: np.ndarray,
    h_to_full: np.ndarray,
    line_ids: list[int],
    scales: list[float],
    lossless: bool = True,
    balance: bool = True,
) -> dict[str, Any]:
    """Run multi-scale verification for the bottleneck line of one case.

    Returns a dict with per-scale results, crossing alpha, and summary.
    """
    # Find bottleneck line (smallest finite AC L2 radius)
    best_lid = -1
    best_r_ac = float("inf")
    best_data: dict[str, Any] = {}

    for pos, lid in enumerate(line_ids):
        key = f"line_{lid}"
        row = results.get(key, {})
        r_ac = row.get("radius_ac_l2", float("inf"))
        if not np.isfinite(r_ac) or r_ac <= 0:
            continue
        if r_ac < best_r_ac:
            best_r_ac = float(r_ac)
            best_lid = int(lid)
            best_data = dict(row)
            best_data["_pos"] = pos

    if best_lid < 0:
        logger.warning("Case %s: no lines with finite AC L2 radius.", case_name)
        return {"case": case_name, "status": "no_finite_radius"}

    binding_end = str(best_data.get("binding_end", "from"))
    pos = best_data["_pos"]
    h_vec = h_from_full[pos] if binding_end == "from" else h_to_full[pos]

    s0_mva = float(best_data.get(f"ac_s0_{binding_end}_mva", float("nan")))
    limit_mva = float(best_data.get("ac_s_limit_mva", float("nan")))
    margin_mva = (
        limit_mva - s0_mva
        if np.isfinite(limit_mva) and np.isfinite(s0_mva)
        else float("nan")
    )

    logger.info(
        "Case %s: bottleneck=line_%d, r_AC=%.4f MVA, s0=%.2f, limit=%.2f, margin=%.2f",
        case_name,
        best_lid,
        best_r_ac,
        s0_mva,
        limit_mva,
        margin_mva,
    )

    # Run verification at each scale
    scale_results: list[dict] = []
    for scale in sorted(scales):
        vr = verify_worst_case(
            net=net,
            line_id=best_lid,
            h_vec=h_vec,
            radius=best_r_ac,
            s0_mva=s0_mva,
            limit_mva=limit_mva,
            scale=scale,
            balance=balance,
            lossless=lossless,
        )

        # Predicted flow from linearized model: s0 + alpha * (c - s0) = s0 + alpha * margin
        predicted_normalized = (
            s0_mva / limit_mva + scale * (1.0 - s0_mva / limit_mva)
            if limit_mva > 0
            else float("nan")
        )
        actual_normalized = (
            vr.actual_s_mva / limit_mva
            if vr.pf_converged and limit_mva > 0
            else float("nan")
        )

        scale_results.append(
            {
                "scale": scale,
                "pf_converged": vr.pf_converged,
                "actual_s_mva": vr.actual_s_mva,
                "predicted_s_mva": vr.predicted_s_mva,
                "actual_normalized": actual_normalized,
                "predicted_normalized": predicted_normalized,
                "violated": vr.violated,
                "relative_error": vr.relative_error,
            }
        )

        logger.info(
            "  scale=%.2f: converged=%s actual=%.4f predicted=%.4f violated=%s",
            scale,
            vr.pf_converged,
            vr.actual_s_mva,
            vr.predicted_s_mva,
            vr.violated,
        )

    # Compute scale_at_crossing: interpolate where |S_actual| / c = 1.0
    crossing_alpha = _interpolate_crossing(scale_results, limit_mva)

    # Linearization error at alpha=1.0
    alpha_1_result = next((r for r in scale_results if r["scale"] == 1.0), None)
    if alpha_1_result and alpha_1_result["pf_converged"] and limit_mva > 0:
        lin_error_pct = (
            (alpha_1_result["predicted_s_mva"] - alpha_1_result["actual_s_mva"])
            / limit_mva
            * 100.0
        )
    else:
        lin_error_pct = float("nan")

    n_pf_failures = sum(1 for r in scale_results if not r["pf_converged"])

    return {
        "case": case_name,
        "bottleneck_line": best_lid,
        "binding_end": binding_end,
        "radius_ac_l2": best_r_ac,
        "s0_mva": s0_mva,
        "limit_mva": limit_mva,
        "margin_mva": margin_mva,
        "crossing_alpha": crossing_alpha,
        "linearization_error_pct": lin_error_pct,
        "n_pf_failures": n_pf_failures,
        "actual_s_at_alpha1": alpha_1_result["actual_s_mva"]
        if alpha_1_result
        else float("nan"),
        "predicted_s_at_alpha1": alpha_1_result["predicted_s_mva"]
        if alpha_1_result
        else float("nan"),
        "scale_results": scale_results,
        "status": "ok",
    }


def _interpolate_crossing(
    scale_results: list[dict],
    limit_mva: float,
) -> float:
    """Interpolate the scale factor where |S_actual| first crosses the limit.

    Returns the smallest alpha where |S_actual| >= limit_mva.
    Uses linear interpolation between adjacent converged scale points.
    """
    if limit_mva <= 0:
        return float("nan")

    converged = [
        (r["scale"], r["actual_s_mva"]) for r in scale_results if r["pf_converged"]
    ]
    converged.sort(key=lambda x: x[0])

    if not converged:
        return float("nan")

    # Check if any point is already at or above the limit
    for alpha, s_actual in converged:
        if s_actual >= limit_mva:
            # Find the previous point for interpolation
            prev = None
            for a2, s2 in converged:
                if a2 < alpha and s2 < limit_mva:
                    prev = (a2, s2)
            if prev is not None:
                a_lo, s_lo = prev
                a_hi, s_hi = alpha, s_actual
                # Linear interpolation: solve s_lo + t*(s_hi - s_lo) = limit
                denom = s_hi - s_lo
                if abs(denom) > 1e-15:
                    t = (limit_mva - s_lo) / denom
                    return a_lo + t * (a_hi - a_lo)
            return alpha

    return float("nan")


# ---------------------------------------------------------------------------
# Table 3: Verification Summary
# ---------------------------------------------------------------------------


def _print_table3(case_results: list[dict]) -> None:
    """Print Table 3 to stdout."""
    header = (
        f"{'Case':>28s}  {'Line':>5s}  {'r*_AC':>8s}  {'Margin':>8s}  "
        f"{'a_cross':>8s}  {'S_act':>8s}  {'S_pred':>8s}  {'Err%':>7s}  {'PF_fail':>7s}"
    )
    width = len(header)
    print()
    print("=" * width)
    print("Table 3: Worst-Case Verification Summary")
    print("=" * width)
    print(header)
    print("-" * width)

    for cr in case_results:
        if cr.get("status") != "ok":
            print(
                f"{cr['case']:>28s}  {'--':>5s}  {'--':>8s}  {'--':>8s}  "
                f"{'--':>8s}  {'--':>8s}  {'--':>8s}  {'--':>7s}  {'--':>7s}"
            )
            continue

        crossing = cr["crossing_alpha"]
        crossing_str = f"{crossing:>8.3f}" if np.isfinite(crossing) else "    n/a"
        err = cr["linearization_error_pct"]
        err_str = f"{err:>7.2f}" if np.isfinite(err) else "    n/a"
        act = cr["actual_s_at_alpha1"]
        act_str = f"{act:>8.2f}" if np.isfinite(act) else "     n/a"
        pred = cr["predicted_s_at_alpha1"]
        pred_str = f"{pred:>8.2f}" if np.isfinite(pred) else "     n/a"

        print(
            f"{cr['case']:>28s}  {cr['bottleneck_line']:>5d}  "
            f"{cr['radius_ac_l2']:>8.2f}  {cr['margin_mva']:>8.2f}  "
            f"{crossing_str}  {act_str}  {pred_str}  {err_str}  {cr['n_pf_failures']:>7d}"
        )

    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Figure 3: Predicted vs Actual Flow Curves
# ---------------------------------------------------------------------------


def _plot_figure3(
    case_results: list[dict],
    *,
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Multi-panel plot: predicted vs actual |S|/c at each scale factor."""
    ok_cases = [cr for cr in case_results if cr.get("status") == "ok"]
    if not ok_cases:
        logger.warning("No valid cases for Figure 3.")
        return

    n = len(ok_cases)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols) if ncols > 0 else 1

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False
    )

    for idx, cr in enumerate(ok_cases):
        row_i = idx // ncols
        col_i = idx % ncols
        ax = axes[row_i][col_i]

        alphas_actual: list[float] = []
        s_actual_norm: list[float] = []
        alphas_pred: list[float] = []
        s_pred_norm: list[float] = []
        pf_fail_alphas: list[float] = []

        for sr in cr["scale_results"]:
            alpha = sr["scale"]
            if sr["pf_converged"]:
                alphas_actual.append(alpha)
                s_actual_norm.append(sr["actual_normalized"])
            else:
                pf_fail_alphas.append(alpha)
            alphas_pred.append(alpha)
            s_pred_norm.append(sr["predicted_normalized"])

        # Shaded regions
        ax.axhspan(
            0,
            1.0,
            xmin=0,
            xmax=1.0 / max(max(alphas_pred, default=1.5), 1),
            alpha=0.05,
            color="green",
            zorder=0,
        )

        # Reference lines
        ax.axhline(
            y=1.0,
            color="red",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            label="$c$ (limit)",
        )
        ax.axvline(
            x=1.0,
            color="gray",
            linestyle="--",
            linewidth=0.6,
            alpha=0.5,
            label="$\\alpha=1$",
        )

        # Actual (solid)
        if alphas_actual:
            ax.plot(
                alphas_actual,
                s_actual_norm,
                "o-",
                color="#4C72B0",
                linewidth=1.5,
                markersize=4,
                label="actual (NL PF)",
                zorder=2,
            )

        # Predicted (dashed)
        if alphas_pred:
            ax.plot(
                alphas_pred,
                s_pred_norm,
                "s--",
                color="#DD8452",
                linewidth=1.2,
                markersize=3,
                label="predicted (linear)",
                zorder=2,
            )

        # PF failures
        if pf_fail_alphas:
            ax.scatter(
                pf_fail_alphas,
                [1.0] * len(pf_fail_alphas),
                marker="x",
                color="red",
                s=50,
                zorder=3,
                label="PF failed",
            )

        # Case label
        case_short = cr["case"].replace("pglib_opf_", "")
        ax.set_title(
            f"{case_short}\nL{cr['bottleneck_line']}, r={cr['radius_ac_l2']:.1f}",
            fontsize=9,
        )
        ax.set_xlabel("$\\alpha$ (scale factor)", fontsize=9)
        ax.set_ylabel("$|S|/c$", fontsize=9)
        ax.tick_params(labelsize=8)

        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row_i = idx // ncols
        col_i = idx % ncols
        axes[row_i][col_i].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"fig3_worst_case_verify.{ext}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    logger.info("Figure 3 saved.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def _run_validation_checks(case_results: list[dict], output_dir: Path) -> dict:
    """Run validation checks and print summary."""
    checks: dict[str, Any] = {}

    ok_cases = [cr for cr in case_results if cr.get("status") == "ok"]

    # 1. Check crossing >= 0.95 for all cases
    crossing_details: list[dict] = []
    for cr in ok_cases:
        crossing = cr["crossing_alpha"]
        if np.isfinite(crossing):
            is_sound = crossing >= 0.95
            is_dangerous = crossing < 0.9
        else:
            is_sound = True  # no crossing found = conservative
            is_dangerous = False
        crossing_details.append(
            {
                "case": cr["case"],
                "crossing_alpha": crossing,
                "is_sound": is_sound,
                "is_dangerous": is_dangerous,
            }
        )

    all_sound = all(d["is_sound"] for d in crossing_details)
    any_dangerous = any(d["is_dangerous"] for d in crossing_details)
    checks["crossing"] = {
        "all_sound": all_sound,
        "any_dangerous": any_dangerous,
        "details": crossing_details,
    }

    # 2. PF divergence at alpha=1.0
    divergence_at_1: list[dict] = []
    for cr in ok_cases:
        alpha_1 = next((r for r in cr["scale_results"] if r["scale"] == 1.0), None)
        if alpha_1 is not None:
            divergence_at_1.append(
                {
                    "case": cr["case"],
                    "pf_converged": alpha_1["pf_converged"],
                }
            )
    checks["pf_divergence_at_1"] = {
        "any_diverged": any(not d["pf_converged"] for d in divergence_at_1),
        "details": divergence_at_1,
    }

    # Print summary
    print()
    print("=" * 60)
    print("Validation Checks (Step 6.3)")
    print("=" * 60)
    print(f"  All crossings >= 0.95: {'PASS' if all_sound else 'FAIL'}")
    if any_dangerous:
        print("  WARNING: Some crossings < 0.90 (dangerously optimistic linearization)")
        for d in crossing_details:
            if d["is_dangerous"]:
                print(f"    {d['case']}: crossing_alpha={d['crossing_alpha']:.3f}")
    pf_div = checks["pf_divergence_at_1"]
    if pf_div["any_diverged"]:
        print("  WARNING: PF diverged at alpha=1.0 for some cases:")
        for d in pf_div["details"]:
            if not d["pf_converged"]:
                print(f"    {d['case']}")
    else:
        print("  PF converged at alpha=1.0: PASS (all cases)")
    print("=" * 60)
    print()

    val_path = output_dir / "validation_worst_case.json"
    with val_path.open("w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2, default=_numpy_serialiser)
    logger.info("Validation checks saved: %s", val_path)

    return checks


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run(
    *,
    results_paths: list[Path],
    output_dir: Path,
    scales: list[float],
    top_k: int = 1,
    recompute: bool = False,
    lossless: bool = True,
    balance: bool = True,
) -> None:
    """Run multi-scale worst-case verification across one or more cases.

    Parameters
    ----------
    results_paths:
        Paths to per-case results JSONs from ``run_pglib_sweep``.
    output_dir:
        Where to write output files.
    scales:
        List of perturbation scale factors.
    top_k:
        Number of lines per case to verify (1 = bottleneck only).
    recompute:
        If True, recompute h-vectors from scratch instead of loading from NPZ.
    lossless:
        Whether the certificate used lossless (r=0) lines.
    balance:
        Whether the certificate used balanced projections.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_case_results: list[dict] = []

    for rpath in results_paths:
        logger.info("Processing: %s", rpath.name)

        with rpath.open(encoding="utf-8") as fh:
            results = json.load(fh)

        meta = results.get("__meta__", {})
        input_path = meta.get("input_path")
        slack_bus = int(meta.get("slack_bus", 0))

        if input_path is None:
            logger.warning("Skipping %s: no __meta__.input_path", rpath.name)
            all_case_results.append(
                {"case": rpath.stem, "status": "missing_input_path"}
            )
            continue

        # Resolve input path (may have been saved from a different machine)
        if not Path(input_path).exists():
            # Try relative to data/input
            alt = Path("data/input") / Path(input_path).name
            if alt.exists():
                input_path = str(alt)
            else:
                from stability_radius.utils.download import ensure_case_file

                try:
                    ensure_case_file(str(alt))
                    input_path = str(alt)
                except Exception:
                    logger.warning(
                        "Skipping %s: cannot find %s", rpath.name, input_path
                    )
                    all_case_results.append(
                        {"case": rpath.stem, "status": "missing_network"}
                    )
                    continue

        # Load or recompute h-vectors
        npz_path = rpath.parent / "h_vectors.npz"
        case_npz = rpath.with_suffix(".npz")

        if not recompute and npz_path.exists():
            h_from, h_to, bus_ids, line_ids = _load_h_vectors_npz(npz_path)
            net = load_network(input_path)
        elif not recompute and case_npz.exists():
            h_from, h_to, bus_ids, line_ids = _load_h_vectors_npz(case_npz)
            net = load_network(input_path)
        else:
            logger.info("  Recomputing h-vectors for %s", rpath.stem)
            try:
                h_from, h_to, bus_ids, line_ids, net = _recompute_h_vectors(
                    input_path,
                    slack_bus,
                    lossless=lossless,
                    balance=balance,
                )
            except Exception:
                logger.warning(
                    "  Failed to recompute h-vectors for %s", rpath.stem, exc_info=True
                )
                all_case_results.append(
                    {"case": rpath.stem, "status": "recompute_failed"}
                )
                continue

        case_result = _verify_case(
            case_name=rpath.stem,
            results=results,
            net=net,
            h_from_full=h_from,
            h_to_full=h_to,
            line_ids=line_ids,
            scales=scales,
            lossless=lossless,
            balance=balance,
        )
        all_case_results.append(case_result)

        # Save per-case results
        case_out = output_dir / f"{rpath.stem}_worst_case.json"
        with case_out.open("w", encoding="utf-8") as fh:
            json.dump(case_result, fh, indent=2, default=_numpy_serialiser)

    # Table 3
    _print_table3(all_case_results)

    # Figure 3
    _plot_figure3(all_case_results, output_dir=output_dir)

    # Validation checks
    _run_validation_checks(all_case_results, output_dir)

    # Save combined results
    combined_path = output_dir / "table3_summary.json"
    with combined_path.open("w", encoding="utf-8") as fh:
        json.dump(all_case_results, fh, indent=2, default=_numpy_serialiser)
    logger.info("All results saved: %s", combined_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Experiment 3: worst-case perturbation verification (multi-scale).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sweep-dir",
        type=Path,
        default=None,
        help="Directory with per-case JSON results from run_pglib_sweep.",
    )
    group.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=None,
        help="One or more per-case results JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/output/worst_case_verify"),
        help="Output directory.",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=_DEFAULT_SCALES,
        help="Perturbation scale factors to test.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of lines per case to verify (default: 1 = bottleneck only).",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute h-vectors from network instead of loading from NPZ.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        nargs="*",
        default=None,
        help="Filter to specific case names (without .json extension).",
    )
    args = parser.parse_args()

    # Collect results paths
    if args.sweep_dir is not None:
        json_files = sorted(args.sweep_dir.glob("*.json"))
        results_paths = [
            f for f in json_files if f.name not in ("summary.json", "debug.log")
        ]
        if args.cases:
            case_set = set(args.cases)
            results_paths = [f for f in results_paths if f.stem in case_set]
    else:
        results_paths = list(args.results)

    if not results_paths:
        logger.error("No results files found.")
        return

    logger.info("Processing %d case(s)", len(results_paths))

    run(
        results_paths=results_paths,
        output_dir=args.output_dir,
        scales=args.scales,
        top_k=args.top_k,
        recompute=args.recompute,
    )


if __name__ == "__main__":
    main()
