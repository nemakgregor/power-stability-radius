"""Experiment 2: Multi-hour AC sigma-radius with UC.jl data.

Runs AC OPF for each hourly timestep in a UC.jl instance, computes AC
sigma-radius at every operating point, aggregates across hours (worst-case),
and produces:
- Table 2: full 12-column sigma-radius results (top-k tightest lines)
- Figure 2: scatter plot of L2 vs sigma-radius
- Figure 2b: per-bus sigma heatmap
- Figure 6: network topology graph
- Worst-case verification for top-k lines
- Monte Carlo validation for empirical overload probability
- Validation checks (balance, Gaussian consistency, sigma floor)
- CSV + JSON + NPZ exports for reproducibility

Usage::

    python -m experiments.run_sigma_radius
    python -m experiments.run_sigma_radius --config experiments/configs/uc_jl_case118.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandapower.topology as pt
import yaml

from stability_radius.base_point.ac import solve_ac_fpf_base_point
from stability_radius.base_point.pandapower_opp import ACFPFConfig
from stability_radius.base_point.pandapower_tools import resolve_slack_bus_id
from stability_radius.parsers.matpower import load_network
from stability_radius.parsers.uc_jl import load_hourly_profiles, load_sigma
from stability_radius.radii.ac_feasibility import check_ac_base_point_feasibility
from stability_radius.radii.ac_l2 import compute_ac_l2_radius
from stability_radius.radii.ac_sigma_radius import compute_ac_sigma_radius
from stability_radius.utils.download import download_uc_jl_instance
from stability_radius.verification.ac_monte_carlo_sigma import run_ac_monte_carlo_sigma
from stability_radius.verification.verify_worst_case import verify_worst_case
from stability_radius.workflows import (
    _expand_h_reduced_to_full,
    _extract_binding_end_data,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "uc_jl_case118.yaml"
_SIGMA_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _numpy_serialiser(obj: object) -> object:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _clamp_sigma(sigma: np.ndarray, floor: float = _SIGMA_FLOOR) -> np.ndarray:
    out = sigma.copy()
    out[out < floor] = floor
    return out


# ---------------------------------------------------------------------------
# Per-hour computation
# ---------------------------------------------------------------------------


def _set_loads_for_hour(
    net: Any,
    *,
    hour: int,
    load_p_mw: np.ndarray,
    load_q_mvar: np.ndarray,
    bus_ids: list[int],
) -> None:
    """Set net.load P/Q from the UC.jl hourly profile (in-place)."""
    bus_to_pos = {bid: pos for pos, bid in enumerate(bus_ids)}
    for load_idx in net.load.index:
        load_bus = int(net.load.at[load_idx, "bus"])
        if load_bus not in bus_to_pos:
            continue
        pos = bus_to_pos[load_bus]
        net.load.at[load_idx, "p_mw"] = float(load_p_mw[pos, hour])
        net.load.at[load_idx, "q_mvar"] = float(load_q_mvar[pos, hour])


def _compute_hour(
    *,
    net_template: Any,
    hour: int,
    load_p_mw: np.ndarray,
    load_q_mvar: np.ndarray,
    bus_ids: list[int],
    slack_bus: int,
    lossless: bool,
    fpf_cfg: ACFPFConfig | None,
    ac_chunk_size: int,
    ac_balance: bool,
    sigma_p_mw: np.ndarray,
    sigma_q_mvar: np.ndarray,
) -> dict | None:
    """Run AC OPF + sigma-radius for one hour. Returns result dict or None."""
    net = copy.deepcopy(net_template)
    _set_loads_for_hour(
        net,
        hour=hour,
        load_p_mw=load_p_mw,
        load_q_mvar=load_q_mvar,
        bus_ids=bus_ids,
    )

    total_load = float(net.load.p_mw.sum())
    logger.info("  Hour %d: total load = %.1f MW", hour, total_load)

    # AC OPF
    try:
        _bp_ac, base_pf = solve_ac_fpf_base_point(
            net=net,
            slack_bus=slack_bus,
            lossless=lossless,
            fpf_cfg=fpf_cfg,
        )
    except Exception:
        logger.warning("  Hour %d: AC OPF failed", hour, exc_info=True)
        return None

    # AC L2 radius + h-vectors
    try:
        ac_l2 = compute_ac_l2_radius(
            net,
            base_pf=base_pf,
            slack_bus=slack_bus,
            chunk_size=ac_chunk_size,
            balance=ac_balance,
            lossless=lossless,
            return_h_vectors=True,
        )
    except Exception:
        logger.warning("  Hour %d: AC L2 radius failed", hour, exc_info=True)
        return None

    h_vecs_raw = ac_l2.pop("_h_vectors", None)
    if h_vecs_raw is None:
        logger.warning("  Hour %d: no h-vectors returned", hour)
        return None

    n_bus = len(bus_ids)
    slack_bus_id = resolve_slack_bus_id(net, slack_bus)
    slack_pos = bus_ids.index(slack_bus_id)

    h_from = _expand_h_reduced_to_full(
        h_vecs_raw["h_from"], n_bus=n_bus, slack_pos=slack_pos
    )
    h_to = _expand_h_reduced_to_full(
        h_vecs_raw["h_to"], n_bus=n_bus, slack_pos=slack_pos
    )

    h_bind, s0_mva, s_limit_mva, line_ids = _extract_binding_end_data(
        ac_results=ac_l2,
        h_from=h_from,
        h_to=h_to,
    )

    # AC feasibility check
    feasibility = check_ac_base_point_feasibility(net=net, base_pf=base_pf)
    n_violated = feasibility.n_constrained_violated
    if not feasibility.is_feasible:
        logger.warning(
            "  Hour %d: AC base point infeasible (%d constrained lines violated, "
            "worst margin=%.2f MVA on line %d). "
            "Sigma-radii on those lines will be negative.",
            hour,
            n_violated,
            feasibility.worst_margin_mva,
            feasibility.worst_line_id,
        )

    # Sigma-radius
    sigma_res = compute_ac_sigma_radius(
        h_vectors=h_bind,
        s_limit_mva=s_limit_mva,
        s0_mva=s0_mva,
        sigma_p_mw=sigma_p_mw,
        sigma_q_mvar=sigma_q_mvar,
        line_ids=line_ids,
        balance=ac_balance,
    )

    n_finite = sum(
        1
        for v in sigma_res.values()
        if isinstance(v, dict) and np.isfinite(v.get("radius_ac_sigma", float("nan")))
    )
    n_negative = sum(
        1
        for v in sigma_res.values()
        if isinstance(v, dict)
        and np.isfinite(v.get("radius_ac_sigma", float("nan")))
        and v.get("radius_ac_sigma", float("nan")) < 0
    )
    logger.info(
        "  Hour %d: OK, %d lines with finite sigma-radius (%d negative/infeasible)",
        hour,
        n_finite,
        n_negative,
    )

    return {
        "ac_l2_results": ac_l2,
        "sigma_results": sigma_res,
        "h_bind": h_bind,
        "h_from_full": h_from,
        "h_to_full": h_to,
        "s0_mva": s0_mva,
        "s_limit_mva": s_limit_mva,
        "line_ids": line_ids,
        "total_load_mw": total_load,
        "ac_feasibility": feasibility,
    }


# ---------------------------------------------------------------------------
# Aggregation across hours
# ---------------------------------------------------------------------------


def _aggregate_across_hours(
    hourly_results: dict[int, dict],
) -> dict[str, Any]:
    """Take min sigma-radius per line across all hours, tracking details.

    Negative sigma-radii (base infeasible) are kept and will sort to the top
    as the "worst" lines.  They are flagged with ``base_infeasible=True``.
    """
    min_sigma: dict[str, float] = {}
    worst_hour: dict[str, int] = {}
    worst_h: dict[str, np.ndarray] = {}
    all_hours: dict[str, dict[int, float]] = {}

    # Per-line details at the worst hour
    worst_hour_ac_l2_radius: dict[str, float] = {}
    worst_hour_s0_mva: dict[str, float] = {}
    worst_hour_s_limit_mva: dict[str, float] = {}
    worst_hour_sigma_flow: dict[str, float] = {}
    worst_hour_binding_end: dict[str, str] = {}
    worst_hour_overload_prob: dict[str, float] = {}
    base_infeasible: dict[str, bool] = {}

    line_ids: list[int] = []
    for hour, res in sorted(hourly_results.items()):
        if not line_ids:
            line_ids = res["line_ids"]
        for lk, v in res["sigma_results"].items():
            if not isinstance(v, dict):
                continue
            r_sig = v.get("radius_ac_sigma", float("nan"))
            if not np.isfinite(r_sig):
                continue

            all_hours.setdefault(lk, {})[hour] = float(r_sig)

            if lk not in min_sigma or float(r_sig) < min_sigma[lk]:
                min_sigma[lk] = float(r_sig)
                worst_hour[lk] = hour
                # Find position of this line in h_bind
                lid = int(lk.split("_", 1)[1])
                if lid in res["line_ids"]:
                    pos = res["line_ids"].index(lid)
                    worst_h[lk] = res["h_bind"][pos, :].copy()

                # Track details at worst hour
                worst_hour_sigma_flow[lk] = float(v.get("sigma_flow_mva", float("nan")))
                worst_hour_overload_prob[lk] = float(
                    v.get("overload_probability_ac", float("nan"))
                )

                # Flag lines with negative r_sigma as base-infeasible
                base_infeasible[lk] = float(r_sig) < 0

                ac_l2 = res["ac_l2_results"]
                if lk in ac_l2 and isinstance(ac_l2[lk], dict):
                    worst_hour_ac_l2_radius[lk] = float(
                        ac_l2[lk].get("radius_ac_l2", float("nan"))
                    )
                    worst_hour_binding_end[lk] = str(ac_l2[lk].get("binding_end", "?"))

                if lid in res["line_ids"]:
                    worst_hour_s0_mva[lk] = float(res["s0_mva"][pos])
                    worst_hour_s_limit_mva[lk] = float(res["s_limit_mva"][pos])

    return {
        "min_sigma_radius": min_sigma,
        "worst_hour": worst_hour,
        "worst_h_bind": worst_h,
        "all_hours": all_hours,
        "line_ids": line_ids,
        "worst_hour_ac_l2_radius": worst_hour_ac_l2_radius,
        "worst_hour_s0_mva": worst_hour_s0_mva,
        "worst_hour_s_limit_mva": worst_hour_s_limit_mva,
        "worst_hour_sigma_flow": worst_hour_sigma_flow,
        "worst_hour_binding_end": worst_hour_binding_end,
        "worst_hour_overload_prob": worst_hour_overload_prob,
        "base_infeasible": base_infeasible,
    }


# ---------------------------------------------------------------------------
# Table 2: Full 12-column sigma-radius results
# ---------------------------------------------------------------------------


def _build_table2_rows(
    agg: dict[str, Any],
    *,
    top_k: int = 10,
) -> list[dict]:
    """Build Table 2 rows (top-k tightest lines) with all columns.

    Lines with negative sigma-radius (base infeasible) are included and
    flagged.  MC violation rate and verified status are left as None --
    filled later.
    """
    candidates = sorted(agg["min_sigma_radius"].items(), key=lambda kv: kv[1])
    top = candidates[:top_k]

    rows: list[dict] = []
    for lk, r_sig in top:
        lid = int(lk.split("_", 1)[1])
        s0 = agg["worst_hour_s0_mva"].get(lk, float("nan"))
        c = agg["worst_hour_s_limit_mva"].get(lk, float("nan"))
        infeasible = agg.get("base_infeasible", {}).get(lk, False)
        rows.append(
            {
                "line_id": lid,
                "line_key": lk,
                "binding_end": agg["worst_hour_binding_end"].get(lk, "?"),
                "s0_mva": s0,
                "limit_mva": c,
                "margin_mva": c - s0
                if np.isfinite(c) and np.isfinite(s0)
                else float("nan"),
                "sigma_flow_mva": agg["worst_hour_sigma_flow"].get(lk, float("nan")),
                "r_sigma": r_sig,
                "p_overload": agg["worst_hour_overload_prob"].get(lk, float("nan")),
                "r_l2_uniform": agg["worst_hour_ac_l2_radius"].get(lk, float("nan")),
                "mc_violation_rate": None,  # filled after MC step
                "worst_hour": agg["worst_hour"].get(lk, -1),
                "verified": None,  # filled after verification step
                "base_infeasible": infeasible,
            }
        )
    return rows


def _print_table2(rows: list[dict]) -> None:
    """Print full Table 2 to stdout."""
    header = (
        f"{'Line':>6s}  {'End':>4s}  {'S0':>8s}  {'Limit':>8s}  {'Margin':>8s}  "
        f"{'sig_flow':>8s}  {'r_sig':>8s}  {'P_over':>10s}  {'r_L2':>8s}  "
        f"{'MC_viol':>8s}  {'Hour':>4s}  {'OK?':>4s}  {'Feas':>5s}"
    )
    width = len(header)
    print()
    print("=" * width)
    print("Table 2: AC Sigma-Radius (top-k tightest lines, worst hour)")
    print("=" * width)
    print(header)
    print("-" * width)

    for r in rows:
        mc = (
            f"{r['mc_violation_rate']:.2e}"
            if r["mc_violation_rate"] is not None
            else "   --"
        )
        ver = (
            " Y"
            if r["verified"] is True
            else (" N" if r["verified"] is False else " --")
        )
        p_ov = r["p_overload"]
        p_str = f"{p_ov:.2e}" if np.isfinite(p_ov) else "      --"
        feas = "  NO" if r.get("base_infeasible", False) else "  ok"
        print(
            f"{r['line_id']:>6d}  {r['binding_end']:>4s}  "
            f"{r['s0_mva']:>8.2f}  {r['limit_mva']:>8.2f}  {r['margin_mva']:>8.2f}  "
            f"{r['sigma_flow_mva']:>8.4f}  {r['r_sigma']:>8.4f}  {p_str:>10s}  "
            f"{r['r_l2_uniform']:>8.2f}  {mc:>8s}  {r['worst_hour']:>4d}  {ver:>4s}  {feas:>5s}"
        )

    # Summary counts
    n_infeasible = sum(1 for r in rows if r.get("base_infeasible", False))
    if n_infeasible > 0:
        print(
            f"\nNOTE: {n_infeasible}/{len(rows)} lines have negative r_sigma "
            f"(base flow exceeds thermal limit at worst hour)"
        )

    print("=" * width)
    print()


def _export_table2_csv(rows: list[dict], output_dir: Path) -> None:
    """Export Table 2 rows to CSV."""
    csv_path = output_dir / "table2_sigma_radius.csv"
    fieldnames = [
        "line_id",
        "binding_end",
        "s0_mva",
        "limit_mva",
        "margin_mva",
        "sigma_flow_mva",
        "r_sigma",
        "p_overload",
        "r_l2_uniform",
        "mc_violation_rate",
        "worst_hour",
        "verified",
        "base_infeasible",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("Table 2 CSV exported: %s", csv_path)


# ---------------------------------------------------------------------------
# Worst-case verification
# ---------------------------------------------------------------------------


def _run_worst_case_verification(
    *,
    net: Any,
    agg: dict[str, Any],
    table_rows: list[dict],
    bus_ids: list[int],
    load_p_mw: np.ndarray,
    load_q_mvar: np.ndarray,
    slack_bus: int,
    lossless: bool,
    fpf_cfg: ACFPFConfig | None,
    scales: list[float],
    output_dir: Path,
) -> list[dict]:
    """Verify worst-case perturbation for each line in table_rows.

    Multi-scale verification: for each line, run verification at each scale
    factor in ``scales``.  Lines with negative r_L2 (base infeasible) are
    skipped since their perturbation direction is inverted.
    """
    verification_results: list[dict] = []

    for row in table_rows:
        lk = row["line_key"]
        lid = row["line_id"]
        wh = row["worst_hour"]
        h_vec = agg["worst_h_bind"].get(lk)
        if h_vec is None:
            logger.warning("No h-vector for %s, skipping verification.", lk)
            row["verified"] = None
            continue

        r_l2 = agg["worst_hour_ac_l2_radius"].get(lk, float("nan"))
        s0 = row["s0_mva"]
        limit = row["limit_mva"]

        if not np.isfinite(r_l2) or not np.isfinite(s0) or not np.isfinite(limit):
            logger.warning("Non-finite values for %s, skipping verification.", lk)
            row["verified"] = None
            continue

        # Skip lines with negative r_L2 (base infeasible: s0 > limit already)
        if r_l2 <= 0:
            logger.warning(
                "Line %s has r_L2=%.4f <= 0 (base infeasible), skipping verification.",
                lk,
                r_l2,
            )
            row["verified"] = None
            verification_results.append(
                {
                    "line_id": lid,
                    "line_key": lk,
                    "status": "skipped_infeasible",
                    "r_l2": r_l2,
                    "s0_mva": s0,
                    "limit_mva": limit,
                }
            )
            continue

        # Set up network at worst hour
        net_wh = copy.deepcopy(net)
        _set_loads_for_hour(
            net_wh,
            hour=wh,
            load_p_mw=load_p_mw,
            load_q_mvar=load_q_mvar,
            bus_ids=bus_ids,
        )

        # Multi-scale verification
        scale_results: list[dict] = []
        any_verified = False
        for scale in sorted(scales):
            try:
                result = verify_worst_case(
                    net=net_wh,
                    line_id=lid,
                    h_vec=h_vec,
                    radius=r_l2,
                    s0_mva=s0,
                    limit_mva=limit,
                    scale=scale,
                    balance=True,
                    lossless=lossless,
                )
                sr = result.to_dict()
                sr["scale"] = scale
                scale_results.append(sr)
                if scale == 1.0 and result.pf_converged:
                    row["verified"] = result.violated
                    any_verified = True
            except Exception:
                logger.warning(
                    "Verification failed for %s at scale=%.2f", lk, scale, exc_info=True
                )
                scale_results.append(
                    {"line_id": lid, "scale": scale, "error": "exception"}
                )

        if not any_verified:
            row["verified"] = None

        verification_results.append(
            {
                "line_id": lid,
                "line_key": lk,
                "worst_hour": wh,
                "r_l2": r_l2,
                "s0_mva": s0,
                "limit_mva": limit,
                "scale_results": scale_results,
                "status": "ok",
            }
        )

    # Save
    vr_path = output_dir / "verification_results.json"
    with vr_path.open("w", encoding="utf-8") as fh:
        json.dump(verification_results, fh, indent=2, default=_numpy_serialiser)
    logger.info("Verification results saved: %s", vr_path)

    return verification_results


# ---------------------------------------------------------------------------
# Monte Carlo validation
# ---------------------------------------------------------------------------


def _run_monte_carlo_validation(
    *,
    net: Any,
    agg: dict[str, Any],
    table_rows: list[dict],
    bus_ids: list[int],
    load_p_mw: np.ndarray,
    load_q_mvar: np.ndarray,
    sigma_p_mw: np.ndarray,
    sigma_q_mvar: np.ndarray,
    slack_bus: int,
    lossless: bool,
    fpf_cfg: ACFPFConfig | None,
    n_samples: int,
    seed: int,
    output_dir: Path,
) -> dict | None:
    """Run MC validation at the worst hour for the global-tightest feasible line.

    Only runs MC for a line with r_sigma > 0 (base feasible).  Per-line
    empirical overload probabilities are populated into all table rows
    regardless.
    """
    if not table_rows:
        return None

    # Find the tightest *feasible* line (r_sigma > 0) for the MC sigma-ball check
    top_row = None
    for row in table_rows:
        if row["r_sigma"] > 0 and np.isfinite(row["r_sigma"]):
            top_row = row
            break

    if top_row is None:
        logger.warning(
            "All top-k lines have r_sigma <= 0 (base infeasible). "
            "Running MC anyway for per-line empirical overload rates, "
            "but soundness_inside_sigma_ball will be N/A."
        )
        # Fall back to the absolute tightest line for hour selection
        top_row = table_rows[0]

    wh = top_row["worst_hour"]
    r_sig = top_row["r_sigma"]

    # Use a positive r_sigma for the sigma-ball check; if all are negative,
    # use abs(r_sigma) so MC still runs for per-line rates
    r_sigma_for_ball = r_sig if r_sig > 0 else float("inf")

    # Set up network at worst hour
    net_wh = copy.deepcopy(net)
    _set_loads_for_hour(
        net_wh,
        hour=wh,
        load_p_mw=load_p_mw,
        load_q_mvar=load_q_mvar,
        bus_ids=bus_ids,
    )

    logger.info(
        "Running MC validation: n_samples=%d, r_sigma=%.4f, worst_hour=%d, "
        "target_line=%s",
        n_samples,
        r_sig,
        wh,
        top_row["line_key"],
    )

    try:
        mc_result = run_ac_monte_carlo_sigma(
            net=net_wh,
            sigma_p_mw=sigma_p_mw,
            sigma_q_mvar=sigma_q_mvar,
            r_sigma=r_sigma_for_ball,
            n_samples=n_samples,
            seed=seed,
            lossless=lossless,
        )
    except Exception:
        logger.warning("MC validation failed", exc_info=True)
        return None

    # Populate per-line MC violation rates into table rows
    for row in table_rows:
        mc_key = row["line_key"]
        if mc_key in mc_result.empirical_overload_probability:
            row["mc_violation_rate"] = mc_result.empirical_overload_probability[mc_key]

    # Save
    mc_out = {
        "worst_hour": wh,
        "r_sigma": r_sig,
        "r_sigma_for_ball": r_sigma_for_ball,
        "target_line": top_row["line_key"],
        "n_samples": mc_result.n_samples,
        "n_violations": mc_result.n_violations,
        "n_pf_failures": mc_result.n_pf_failures,
        "soundness_inside_sigma_ball": mc_result.soundness_inside_sigma_ball,
        "empirical_overload_probability": mc_result.empirical_overload_probability,
    }
    mc_path = output_dir / "mc_results.json"
    with mc_path.open("w", encoding="utf-8") as fh:
        json.dump(mc_out, fh, indent=2, default=_numpy_serialiser)
    logger.info("MC results saved: %s", mc_path)

    # Print MC summary
    n_infeasible_lines = sum(
        1
        for lk, prob in mc_result.empirical_overload_probability.items()
        if prob > 0.5
    )
    print()
    print("=" * 60)
    print("Monte Carlo Validation Summary")
    print("=" * 60)
    print(f"  Worst hour:               {wh}")
    print(f"  Target line:              {top_row['line_key']}")
    print(f"  r_sigma:                  {r_sig:.4f}")
    print(f"  Samples:                  {mc_result.n_samples}")
    print(f"  Samples with violations:  {mc_result.n_violations}")
    print(f"  PF failures:              {mc_result.n_pf_failures}")
    print(f"  Soundness (in sigma-ball): {mc_result.soundness_inside_sigma_ball:.4f}")
    print(f"  Lines with >50% overload: {n_infeasible_lines}")
    if n_infeasible_lines > 0:
        print(
            "  NOTE: Lines with >50% MC overload are likely base-infeasible "
            "(S0 > c at this operating point)"
        )
    print("=" * 60)
    print()

    return mc_out


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def _run_validation_checks(
    *,
    agg: dict[str, Any],
    hourly_results: dict[int, dict],
    table_rows: list[dict],
    mc_results: dict | None,
    sigma_p_mw_raw: np.ndarray,
    n_bus: int,
    output_dir: Path,
) -> dict:
    """Run validation checks and save results."""
    checks: dict[str, Any] = {}

    # 0. Feasibility summary: how many lines are base-infeasible?
    n_infeasible = sum(1 for r in table_rows if r.get("base_infeasible", False))
    n_total_lines_with_sigma = len(agg["min_sigma_radius"])
    n_negative_sigma = sum(
        1 for v in agg["min_sigma_radius"].values() if v < 0
    )
    checks["feasibility"] = {
        "n_lines_total": n_total_lines_with_sigma,
        "n_lines_negative_sigma": n_negative_sigma,
        "n_top_k_infeasible": n_infeasible,
        "note": "Lines with negative sigma-radius have S0 > c at worst hour",
    }

    # 1. Balance check: |sum(worst_case_dp_mw)| < 1e-6 for top lines
    balance_ok = True
    balance_details: list[dict] = []
    for row in table_rows:
        lk = row["line_key"]
        wh = row["worst_hour"]
        if wh not in hourly_results:
            continue
        sigma_res = hourly_results[wh]["sigma_results"]
        if lk not in sigma_res or not isinstance(sigma_res[lk], dict):
            continue
        dp = sigma_res[lk].get("worst_case_dp_mw")
        if dp is None:
            continue
        dp_arr = np.asarray(dp, dtype=float)
        dp_sum = float(np.abs(np.sum(dp_arr)))
        ok = dp_sum < 1e-6
        if not ok:
            balance_ok = False
        balance_details.append({"line": lk, "sum_dp_mw": dp_sum, "ok": ok})
    checks["balance"] = {"all_ok": balance_ok, "details": balance_details}

    # 2. Sigma floor impact
    n_clamped = int(np.sum(sigma_p_mw_raw < _SIGMA_FLOOR))
    floor_frac = n_clamped / max(n_bus, 1)
    checks["sigma_floor"] = {
        "n_clamped": n_clamped,
        "n_bus": n_bus,
        "fraction": floor_frac,
        "warn": floor_frac > 0.5,
    }

    # 3. Gaussian consistency (analytical vs MC)
    gauss_details: list[dict] = []
    if mc_results is not None:
        mc_probs = mc_results.get("empirical_overload_probability", {})
        for row in table_rows:
            lk = row["line_key"]
            analytical = row["p_overload"]
            mc_emp = mc_probs.get(lk)
            if mc_emp is None or not np.isfinite(analytical):
                continue
            # Avoid division by zero
            if mc_emp > 0 and analytical > 0:
                ratio = analytical / mc_emp
            elif mc_emp == 0 and analytical == 0:
                ratio = 1.0
            else:
                ratio = float("inf")
            ok = 0.5 <= ratio <= 2.0 if np.isfinite(ratio) else False
            gauss_details.append(
                {
                    "line": lk,
                    "analytical": analytical,
                    "mc_empirical": mc_emp,
                    "ratio": ratio,
                    "ok": ok,
                }
            )
    checks["gaussian_consistency"] = {
        "details": gauss_details,
        "all_ok": all(d["ok"] for d in gauss_details) if gauss_details else True,
    }

    # Print summary
    print()
    print("=" * 60)
    print("Validation Checks")
    print("=" * 60)
    print(
        f"  Feasibility:           {n_negative_sigma}/{n_total_lines_with_sigma} "
        f"lines have negative sigma-radius"
    )
    if n_infeasible > 0:
        print(
            f"                         {n_infeasible} of top-k lines are base-infeasible"
        )
    print(
        f"  Balance check:         {'PASS' if checks['balance']['all_ok'] else 'FAIL'}"
    )
    print(
        f"  Sigma floor impact:    {n_clamped}/{n_bus} buses clamped "
        f"({'WARN' if checks['sigma_floor']['warn'] else 'OK'})"
    )
    if gauss_details:
        print(
            f"  Gaussian consistency:  {'PASS' if checks['gaussian_consistency']['all_ok'] else 'WARN'}"
        )
        for d in gauss_details:
            print(
                f"    {d['line']}: analytical={d['analytical']:.2e}, "
                f"mc={d['mc_empirical']:.2e}, ratio={d['ratio']:.2f} "
                f"{'OK' if d['ok'] else 'MISMATCH'}"
            )
    else:
        print("  Gaussian consistency:  N/A (no MC data)")
    print("=" * 60)
    print()

    # Save
    val_path = output_dir / "validation.json"
    with val_path.open("w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2, default=_numpy_serialiser)
    logger.info("Validation results saved: %s", val_path)

    return checks


# ---------------------------------------------------------------------------
# Figure 2: L2 vs sigma-radius scatter plot
# ---------------------------------------------------------------------------


def _plot_scatter_l2_vs_sigma(
    agg: dict[str, Any],
    *,
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Scatter plot: AC L2 radius vs min sigma-radius across all lines.

    Only lines with r_sigma > 0 AND r_L2 > 0 are included (strict filter
    for log-log axes).  Lines with negative radii are base-infeasible and
    omitted from this plot.
    """
    line_keys = sorted(agg["min_sigma_radius"].keys())

    r_l2_vals: list[float] = []
    r_sig_vals: list[float] = []
    loading_fracs: list[float] = []
    binding_ends: list[str] = []
    line_labels: list[str] = []

    n_skipped_negative = 0
    for lk in line_keys:
        r_sig = agg["min_sigma_radius"].get(lk, float("nan"))
        r_l2 = agg["worst_hour_ac_l2_radius"].get(lk, float("nan"))
        s0 = agg["worst_hour_s0_mva"].get(lk, float("nan"))
        c = agg["worst_hour_s_limit_mva"].get(lk, float("nan"))
        be = agg["worst_hour_binding_end"].get(lk, "?")

        # Strict filter: both must be positive and finite for log-log
        if not np.isfinite(r_sig) or not np.isfinite(r_l2):
            continue
        if r_l2 <= 0 or r_sig <= 0:
            n_skipped_negative += 1
            continue

        r_l2_vals.append(r_l2)
        r_sig_vals.append(r_sig)
        loading_fracs.append(
            s0 / c if np.isfinite(s0) and np.isfinite(c) and c > 0 else 0.5
        )
        binding_ends.append(be)
        line_labels.append(lk)

    if n_skipped_negative > 0:
        logger.info(
            "Scatter plot: skipped %d lines with r_sig<=0 or r_L2<=0 (base infeasible)",
            n_skipped_negative,
        )

    if len(r_l2_vals) < 2:
        logger.warning("Not enough data for L2 vs sigma scatter plot.")
        return

    r_l2_arr = np.array(r_l2_vals)
    r_sig_arr = np.array(r_sig_vals)
    loading_arr = np.array(loading_fracs)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by binding end
    colors = ["#4C72B0" if be == "from" else "#DD8452" for be in binding_ends]
    sizes = 5 + 15 * loading_arr  # range 5-20

    ax.scatter(
        r_l2_arr,
        r_sig_arr,
        c=colors,
        s=sizes**2,
        alpha=0.6,
        edgecolors="gray",
        linewidths=0.3,
    )

    # Diagonal reference line (uniform-sigma equivalence): r_sig = r_l2 / mean_sigma
    mean_sigma = np.mean(r_l2_arr / r_sig_arr)
    ref_x = np.array([r_l2_arr.min(), r_l2_arr.max()])
    ax.plot(
        ref_x,
        ref_x / mean_sigma,
        "--",
        color="gray",
        linewidth=0.8,
        alpha=0.6,
        label=f"uniform-$\\sigma$ ref ($\\bar{{\\sigma}}$={mean_sigma:.1f})",
    )

    # Label top-3 tightest lines
    sorted_idx = np.argsort(r_sig_arr)[:3]
    for idx in sorted_idx:
        lid = int(line_labels[idx].split("_", 1)[1])
        ax.annotate(
            f"L{lid}",
            (r_l2_arr[idx], r_sig_arr[idx]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$r_{L2}$ (MVA)", fontsize=12)
    ax.set_ylabel("$r_\\sigma$ (number of $\\sigma$)", fontsize=12)
    ax.tick_params(labelsize=10)

    # Legend for binding end colors and size
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#4C72B0",
            markersize=8,
            label="from end",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#DD8452",
            markersize=8,
            label="to end",
        ),
    ]
    # Add size legend entries for loading fraction
    for load_frac, label in [(0.3, "$S_0/c$=30%"), (0.7, "$S_0/c$=70%"), (0.95, "$S_0/c$=95%")]:
        sz = 5 + 15 * load_frac
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=sz,
                label=label,
            )
        )
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"fig2_l2_vs_sigma.{ext}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    logger.info("Figure 2 (L2 vs sigma scatter) saved.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2b: Per-bus sigma heatmap
# ---------------------------------------------------------------------------


def _plot_sigma_heatmap(
    sigma_p_mw: np.ndarray,
    sigma_q_mvar: np.ndarray,
    *,
    bus_ids: list[int],
    agg: dict[str, Any],
    top_k_critical: int,
    output_dir: Path,
    dpi: int = 300,
    max_buses: int = 30,
) -> None:
    """Heatmap of per-bus sigma_P and sigma_Q, sorted by total sigma."""
    n_bus = len(bus_ids)
    total_sigma = np.sqrt(sigma_p_mw**2 + sigma_q_mvar**2)
    sorted_idx = np.argsort(total_sigma)[::-1]  # descending by total

    # Show at most max_buses rows
    show_idx = sorted_idx[:max_buses]
    show_bus_ids = [bus_ids[i] for i in show_idx]

    data = np.column_stack([sigma_p_mw[show_idx], sigma_q_mvar[show_idx]])

    # Find buses that contribute to top-k critical lines
    sorted_lines = sorted(agg["min_sigma_radius"].items(), key=lambda kv: kv[1])
    top_critical = sorted_lines[:top_k_critical]
    critical_bus_positions: set[int] = set()
    for lk, _r in top_critical:
        h = agg["worst_h_bind"].get(lk)
        if h is None:
            continue
        h_p = h[:n_bus]
        h_q = h[n_bus:]
        contrib = h_p**2 + h_q**2
        # Mark buses with top-5 contribution for this line
        top_bus_idx = np.argsort(contrib)[-5:]
        critical_bus_positions.update(top_bus_idx.tolist())

    fig, ax = plt.subplots(figsize=(6, max(6, len(show_idx) * 0.3)))

    # Log-normalize for color scale
    data_log = data.copy()
    data_log[data_log <= 0] = _SIGMA_FLOOR
    vmin = (
        float(np.min(data_log[data_log > _SIGMA_FLOOR]))
        if np.any(data_log > _SIGMA_FLOOR)
        else _SIGMA_FLOOR
    )
    vmax = float(np.max(data_log))

    norm = mcolors.LogNorm(
        vmin=max(vmin, _SIGMA_FLOOR), vmax=max(vmax, _SIGMA_FLOOR * 10)
    )
    im = ax.imshow(
        data_log, aspect="auto", cmap="Reds", norm=norm, interpolation="nearest"
    )

    # Star markers for buses in critical directions
    for row_pos, orig_idx in enumerate(show_idx):
        if orig_idx in critical_bus_positions:
            for col in range(2):
                ax.plot(col, row_pos, marker="*", color="blue", markersize=8, zorder=3)

    ax.set_yticks(range(len(show_idx)))
    ax.set_yticklabels([f"bus {bid}" for bid in show_bus_ids], fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$\\sigma_P$ (MW)", "$\\sigma_Q$ (MVAr)"], fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Injection std dev (MW / MVAr)", fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"fig2b_sigma_heatmap.{ext}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    logger.info("Figure 2b (sigma heatmap) saved.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save h-vectors to NPZ
# ---------------------------------------------------------------------------


def _save_hvectors_npz(
    agg: dict[str, Any],
    *,
    output_dir: Path,
) -> None:
    """Save aggregated worst-hour h-vectors to NPZ for downstream experiments."""
    npz_data: dict[str, np.ndarray] = {}
    for lk, h in agg["worst_h_bind"].items():
        # Use line key as array name (e.g., "line_42")
        npz_data[lk] = np.asarray(h, dtype=float)

    # Also save line_ids as metadata
    npz_data["line_ids"] = np.array(agg["line_ids"], dtype=int)

    npz_path = output_dir / "hvectors.npz"
    np.savez_compressed(str(npz_path), **npz_data)
    logger.info("h-vectors saved: %s (%d lines)", npz_path, len(agg["worst_h_bind"]))


# ---------------------------------------------------------------------------
# Topology plot
# ---------------------------------------------------------------------------


def _plot_topology(
    net: Any,
    *,
    agg: dict[str, Any],
    bus_ids: list[int],
    top_k_critical: int,
    n_hours_ok: int,
    output_dir: Path,
    figsize: tuple[float, float] = (16, 12),
    dpi: int = 200,
) -> None:
    """Network graph: lines by min sigma-radius, buses by threat contribution."""
    n_bus = len(bus_ids)

    # --- Build networkx graph and layout ---
    G = pt.create_nxgraph(net, respect_switches=False)
    pos = nx.kamada_kawai_layout(G)

    # --- Per-line sigma-radius for coloring ---
    line_indices = list(net.line.index)
    min_sr = agg["min_sigma_radius"]
    line_radii = np.full(len(line_indices), float("nan"), dtype=float)
    for i, lid in enumerate(line_indices):
        lk = f"line_{lid}"
        if lk in min_sr:
            line_radii[i] = min_sr[lk]

    finite_mask = np.isfinite(line_radii)
    if not np.any(finite_mask):
        logger.warning("No finite sigma-radii for topology plot.")
        return

    vmin = float(np.nanmin(line_radii[finite_mask]))
    vmax = float(np.nanmax(line_radii[finite_mask]))
    if vmin == vmax:
        vmax = vmin + 1.0
    line_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    line_cmap = plt.colormaps["RdYlGn"]

    # --- Per-bus threat contribution from top-k critical lines ---
    sorted_lines = sorted(min_sr.items(), key=lambda kv: kv[1])
    top_critical = sorted_lines[:top_k_critical]

    threat = np.zeros(n_bus, dtype=float)
    for lk, _r in top_critical:
        h = agg["worst_h_bind"].get(lk)
        if h is None:
            continue
        h_p = h[:n_bus]
        h_q = h[n_bus:]
        threat += h_p**2 + h_q**2

    threat_max = float(np.max(threat)) if np.max(threat) > 0 else 1.0
    threat_norm = threat / threat_max  # [0, 1]

    # Map bus_ids to bus positions for coloring
    bus_id_to_pos = {bid: i for i, bid in enumerate(bus_ids)}
    bus_colors = []
    bus_sizes = []
    bus_cmap = plt.colormaps["YlOrRd"]
    for node in G.nodes():
        if int(node) in bus_id_to_pos:
            t = threat_norm[bus_id_to_pos[int(node)]]
        else:
            t = 0.0
        bus_colors.append(bus_cmap(t))
        bus_sizes.append(30 + 300 * t)  # range [30, 330]

    # --- Draw ---
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw edges (lines) colored by sigma-radius
    from_buses = net.line["from_bus"].values
    to_buses = net.line["to_bus"].values
    edge_list = list(zip(from_buses, to_buses))

    edge_colors = []
    edge_widths = []
    for i, lid in enumerate(line_indices):
        r = line_radii[i]
        if np.isfinite(r):
            c = line_cmap(line_norm(r))
            w = 1.0 + 3.0 * (1.0 - line_norm(r))  # thicker for smaller radius
        else:
            c = (0.8, 0.8, 0.8, 0.5)  # gray for inf/nan
            w = 0.5
        edge_colors.append(c)
        edge_widths.append(w)

    # Draw edges individually (nx.draw_networkx_edges doesn't support per-edge width easily)
    for idx, (u, v) in enumerate(edge_list):
        if u in pos and v in pos:
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax.plot(
                x,
                y,
                color=edge_colors[idx],
                linewidth=edge_widths[idx],
                solid_capstyle="round",
                zorder=1,
            )

    # Draw nodes
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    scatter = ax.scatter(
        node_x,
        node_y,
        c=bus_colors,
        s=bus_sizes,
        edgecolors="black",
        linewidths=0.3,
        zorder=2,
    )

    # Label top-k critical lines
    for lk, r_sig in top_critical:
        lid = int(lk.split("_", 1)[1])
        if lid not in net.line.index:
            continue
        row = net.line.loc[lid]
        fb, tb = int(row["from_bus"]), int(row["to_bus"])
        if fb in pos and tb in pos:
            mx = (pos[fb][0] + pos[tb][0]) / 2
            my = (pos[fb][1] + pos[tb][1]) / 2
            ax.annotate(
                f"L{lid}\n({r_sig:.1f})",
                (mx, my),
                fontsize=6,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.8),
                zorder=3,
            )

    # Colorbars
    sm_line = plt.cm.ScalarMappable(cmap=line_cmap, norm=line_norm)
    sm_line.set_array([])
    cbar_line = fig.colorbar(sm_line, ax=ax, fraction=0.03, pad=0.01)
    cbar_line.set_label("Min sigma-radius across hours (sigma)", fontsize=10)

    sm_bus = plt.cm.ScalarMappable(
        cmap=bus_cmap,
        norm=mcolors.Normalize(vmin=0, vmax=threat_max),
    )
    sm_bus.set_array([])
    cbar_bus = fig.colorbar(sm_bus, ax=ax, fraction=0.03, pad=0.04)
    cbar_bus.set_label(
        "Bus threat contribution (top-%d lines)" % top_k_critical, fontsize=10
    )

    ax.set_title(
        "Case118: Hourly AC Sigma-Radius (min across %d hours)" % n_hours_ok,
        fontsize=14,
    )
    ax.set_axis_off()
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = output_dir / f"topology_sigma_radius.{ext}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        logger.info("Topology plot saved: %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run(config_path: Path) -> None:
    cfg = _load_config(config_path)
    case_cfg = cfg["case"]
    uc_cfg = cfg["uc_jl"]
    compute_cfg = cfg.get("compute", {})
    ac_cfg = compute_cfg.get("ac", {})
    fpf_cfg_dict = ac_cfg.get("fpf", {})
    plot_cfg = cfg.get("plot", {})
    data_dir = Path(cfg.get("data_dir", "data/input"))
    output_dir = Path(cfg.get("output_dir", "experiments/output/sigma_radius_hourly"))
    allow_download = bool(cfg.get("allow_download", False))

    lossless = bool(ac_cfg.get("lossless", True))
    ac_chunk_size = int(ac_cfg.get("chunk_size", 64))
    ac_balance = bool(ac_cfg.get("balance", True))
    top_k_critical = int(plot_cfg.get("top_k_critical", 5))
    figsize = tuple(plot_cfg.get("figsize", [16, 12]))
    plot_dpi = int(plot_cfg.get("dpi", 200))

    input_path = str(data_dir / case_cfg["matpower_file"])
    slack_bus = int(case_cfg.get("slack_bus", 0))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build AC FPF config
    fpf = ACFPFConfig(
        pg0_source=str(fpf_cfg_dict.get("pg0_source", "case")),
        vm_min_pu=float(fpf_cfg_dict.get("vm_min_pu", 0.9)),
        vm_max_pu=float(fpf_cfg_dict.get("vm_max_pu", 1.1)),
        max_iteration=int(fpf_cfg_dict.get("max_iteration", 300)),
        max_loading_percent=float(fpf_cfg_dict.get("max_loading_percent", 99.0)),
        init=str(fpf_cfg_dict.get("init", "dc")),
        max_attempts=int(fpf_cfg_dict.get("max_attempts", 2)),
        per_attempt_timeout=float(fpf_cfg_dict.get("per_attempt_timeout", 0)),
    )

    # ------------------------------------------------------------------
    # Step 1: Download UC.jl + extract sigma and hourly profiles.
    # ------------------------------------------------------------------
    uc_dest = Path(uc_cfg.get("dest_dir", "data/uc_jl"))
    uc_case_name = str(uc_cfg["case_name"])
    uc_date = str(uc_cfg.get("date", "2017-01-01"))
    power_factor = float(uc_cfg.get("power_factor", 0.9))

    logger.info("Downloading UC.jl instance: %s (date=%s)", uc_case_name, uc_date)
    uc_path = download_uc_jl_instance(uc_case_name, dest_dir=uc_dest, date=uc_date)

    # Sigma (overall variability)
    sigma_data = load_sigma(uc_path, power_factor=power_factor)
    sigma_p_mw = _clamp_sigma(sigma_data["sigma_p_mw"])
    sigma_q_mvar = _clamp_sigma(sigma_data["sigma_q_mvar"])
    n_clamped = int(np.sum(sigma_data["sigma_p_mw"] < _SIGMA_FLOOR))

    logger.info(
        "Sigma loaded: n_bus=%d, sigma_P range=[%.4g, %.4g] MW (%d buses clamped)",
        len(sigma_p_mw),
        float(np.min(sigma_p_mw)),
        float(np.max(sigma_p_mw)),
        n_clamped,
    )

    # Hourly profiles
    hourly_data = load_hourly_profiles(uc_path, power_factor=power_factor)
    load_p_mw = hourly_data["load_p_mw"]
    load_q_mvar = hourly_data["load_q_mvar"]
    n_hours = hourly_data["n_timesteps"]

    logger.info(
        "Hourly profiles: %d timesteps, total_P range=[%.1f, %.1f] MW",
        n_hours,
        float(np.sum(load_p_mw, axis=0).min()),
        float(np.sum(load_p_mw, axis=0).max()),
    )

    # Save sigma arrays
    sigma_out = output_dir / "sigma_arrays.json"
    with sigma_out.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "sigma_p_mw": sigma_p_mw.tolist(),
                "sigma_q_mvar": sigma_q_mvar.tolist(),
                "n_clamped": n_clamped,
                "sigma_floor": _SIGMA_FLOOR,
                "metadata": sigma_data["metadata"],
            },
            fh,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Step 2: Load network.
    # ------------------------------------------------------------------
    from stability_radius.utils.download import ensure_case_file

    if not Path(input_path).exists() and allow_download:
        ensure_case_file(input_path)

    net = load_network(input_path)
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)

    if len(sigma_p_mw) != n_bus:
        raise ValueError(
            f"Sigma array length ({len(sigma_p_mw)}) != network bus count ({n_bus})"
        )
    if load_p_mw.shape[0] != n_bus:
        raise ValueError(
            f"Hourly profile bus count ({load_p_mw.shape[0]}) != network ({n_bus})"
        )

    logger.info("Network loaded: %d buses, %d lines", n_bus, len(net.line))

    # ------------------------------------------------------------------
    # Step 3: Per-hour loop.
    # ------------------------------------------------------------------
    hourly_results: dict[int, dict] = {}
    for hour in range(n_hours):
        logger.info("--- Hour %d/%d ---", hour + 1, n_hours)
        result = _compute_hour(
            net_template=net,
            hour=hour,
            load_p_mw=load_p_mw,
            load_q_mvar=load_q_mvar,
            bus_ids=bus_ids,
            slack_bus=slack_bus,
            lossless=lossless,
            fpf_cfg=fpf,
            ac_chunk_size=ac_chunk_size,
            ac_balance=ac_balance,
            sigma_p_mw=sigma_p_mw,
            sigma_q_mvar=sigma_q_mvar,
        )
        if result is not None:
            hourly_results[hour] = result

    n_ok = len(hourly_results)
    n_fail = n_hours - n_ok
    n_infeasible_hours = sum(
        1
        for res in hourly_results.values()
        if not res["ac_feasibility"].is_feasible
    )
    logger.info(
        "Hourly loop done: %d/%d hours succeeded, %d failed, %d infeasible",
        n_ok,
        n_hours,
        n_fail,
        n_infeasible_hours,
    )

    if not hourly_results:
        logger.error("All hours failed. Cannot produce results.")
        return

    # ------------------------------------------------------------------
    # Step 4: Aggregate across hours.
    # ------------------------------------------------------------------
    agg = _aggregate_across_hours(hourly_results)

    # ------------------------------------------------------------------
    # Step 5: Build full Table 2 (pre-populate; MC/verification later).
    # ------------------------------------------------------------------
    table_rows = _build_table2_rows(agg, top_k=top_k_critical)

    # ------------------------------------------------------------------
    # Step 6: Save h-vectors to NPZ.
    # ------------------------------------------------------------------
    _save_hvectors_npz(agg, output_dir=output_dir)

    # ------------------------------------------------------------------
    # Step 7: Worst-case verification for top-k lines.
    # ------------------------------------------------------------------
    verify_cfg = cfg.get("verification", {})
    verify_top_k = int(verify_cfg.get("top_k", 10))
    verify_scales = [float(s) for s in verify_cfg.get("scales", [1.0])]

    if table_rows:
        _run_worst_case_verification(
            net=net,
            agg=agg,
            table_rows=table_rows[:verify_top_k],
            bus_ids=bus_ids,
            load_p_mw=load_p_mw,
            load_q_mvar=load_q_mvar,
            slack_bus=slack_bus,
            lossless=lossless,
            fpf_cfg=fpf,
            scales=verify_scales,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Step 8: Monte Carlo validation.
    # ------------------------------------------------------------------
    mc_cfg = cfg.get("monte_carlo", {})
    mc_enabled = bool(mc_cfg.get("enabled", True))
    mc_n_samples = int(mc_cfg.get("n_samples", 5000))
    mc_seed = int(mc_cfg.get("seed", 42))

    mc_results = None
    if mc_enabled and table_rows:
        mc_results = _run_monte_carlo_validation(
            net=net,
            agg=agg,
            table_rows=table_rows,
            bus_ids=bus_ids,
            load_p_mw=load_p_mw,
            load_q_mvar=load_q_mvar,
            sigma_p_mw=sigma_p_mw,
            sigma_q_mvar=sigma_q_mvar,
            slack_bus=slack_bus,
            lossless=lossless,
            fpf_cfg=fpf,
            n_samples=mc_n_samples,
            seed=mc_seed,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Step 9: Print and export full Table 2.
    # ------------------------------------------------------------------
    _print_table2(table_rows)
    _export_table2_csv(table_rows, output_dir)

    # ------------------------------------------------------------------
    # Step 10: Validation checks.
    # ------------------------------------------------------------------
    _run_validation_checks(
        agg=agg,
        hourly_results=hourly_results,
        table_rows=table_rows,
        mc_results=mc_results,
        sigma_p_mw_raw=sigma_data["sigma_p_mw"],
        n_bus=n_bus,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    # Step 11: Save results JSON.
    # ------------------------------------------------------------------
    # Per-hour summary
    hourly_summary: dict[str, Any] = {}
    for hour, res in sorted(hourly_results.items()):
        sigma_vals = [
            float(v["radius_ac_sigma"])
            for v in res["sigma_results"].values()
            if isinstance(v, dict)
            and np.isfinite(v.get("radius_ac_sigma", float("nan")))
        ]
        hourly_summary[str(hour)] = {
            "total_load_mw": res["total_load_mw"],
            "n_finite_sigma_radii": len(sigma_vals),
            "min_sigma_radius": min(sigma_vals) if sigma_vals else float("nan"),
            "median_sigma_radius": float(np.median(sigma_vals))
            if sigma_vals
            else float("nan"),
            "ac_feasible": res["ac_feasibility"].is_feasible,
            "n_constrained_violated": res["ac_feasibility"].n_constrained_violated,
        }

    results_out = {
        "case": case_cfg["name"],
        "n_hours_total": n_hours,
        "n_hours_ok": n_ok,
        "n_hours_failed": n_fail,
        "n_hours_infeasible": n_infeasible_hours,
        "min_sigma_radius": {k: v for k, v in sorted(agg["min_sigma_radius"].items())},
        "worst_hour": {k: v for k, v in sorted(agg["worst_hour"].items())},
        "per_hour_summary": hourly_summary,
        "table_rows": table_rows,
    }
    results_path = output_dir / "hourly_results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results_out, fh, indent=2, default=_numpy_serialiser)
    logger.info("Results written: %s", results_path)

    # Summary
    all_min = list(agg["min_sigma_radius"].values())
    finite_min = [v for v in all_min if np.isfinite(v)]
    positive_min = [v for v in finite_min if v > 0]
    negative_min = [v for v in finite_min if v < 0]
    summary = {
        "case": case_cfg["name"],
        "n_hours_total": n_hours,
        "n_hours_ok": n_ok,
        "n_hours_infeasible": n_infeasible_hours,
        "n_lines": len(all_min),
        "n_lines_finite": len(finite_min),
        "n_lines_positive_sigma": len(positive_min),
        "n_lines_negative_sigma": len(negative_min),
        "global_min_sigma_radius": min(finite_min) if finite_min else float("nan"),
        "global_min_positive_sigma_radius": min(positive_min)
        if positive_min
        else float("nan"),
        "global_median_sigma_radius": float(np.median(finite_min))
        if finite_min
        else float("nan"),
        "global_max_sigma_radius": max(finite_min) if finite_min else float("nan"),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_numpy_serialiser)
    logger.info("Summary written: %s", summary_path)

    # ------------------------------------------------------------------
    # Step 12: Plots.
    # ------------------------------------------------------------------
    scatter_dpi = int(plot_cfg.get("scatter_dpi", 300))
    heatmap_dpi = int(plot_cfg.get("heatmap_dpi", 300))

    _plot_topology(
        net,
        agg=agg,
        bus_ids=bus_ids,
        top_k_critical=top_k_critical,
        n_hours_ok=n_ok,
        output_dir=output_dir,
        figsize=figsize,
        dpi=plot_dpi,
    )

    _plot_scatter_l2_vs_sigma(
        agg,
        output_dir=output_dir,
        dpi=scatter_dpi,
    )

    _plot_sigma_heatmap(
        sigma_p_mw,
        sigma_q_mvar,
        bus_ids=bus_ids,
        agg=agg,
        top_k_critical=top_k_critical,
        output_dir=output_dir,
        dpi=heatmap_dpi,
    )

    logger.info("Experiment 2 (multi-hour) complete. Output: %s", output_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Experiment 2: multi-hour sigma-radius with UC.jl data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to uc_jl_case118.yaml config.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
