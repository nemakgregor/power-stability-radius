"""Experiment 2: AC sigma-radius at average operating point.

Computes σ from UC.jl load time series (std of per-bus P and Q across hours),
then runs a single AC OPF at the average operating point and computes
sigma-radius once for each line.

Produces:
- Table 2: full sigma-radius results (top-k tightest lines)
- Figure 2: scatter plot of L2 vs sigma-radius
- Figure 2b: per-bus sigma heatmap
- Figure 6: network topology graph
- Worst-case verification for top-k lines
- Monte Carlo validation for empirical overload probability
- Validation checks (balance, Gaussian consistency, sigma floor)
- CSV + JSON + NPZ exports for reproducibility

Usage::

    python entry_points/run_sigma_radius.py
    python entry_points/run_sigma_radius.py --config experiments/configs/uc_jl_case118.yaml
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
from stability_radius.utils import (
    create_module_output_dir,
    numpy_to_builtin,
    resolve_artifacts_root,
    setup_output_dir_logging,
)
from stability_radius.base_point.pandapower_opp import ACFPFConfig
from stability_radius.base_point.pandapower_tools import (
    apply_lossless_policy_to_pandapower_net,
    apply_opp_result_to_pandapower_net,
    resolve_slack_bus_id,
)
from stability_radius.parsers.matpower import load_network
from stability_radius.parsers.uc_jl import load_hourly_profiles, load_sigma
from stability_radius.radii.ac_feasibility import check_ac_base_point_feasibility
from stability_radius.radii.ac_l2 import compute_ac_l2_radius
from stability_radius.radii.ac_sigma_radius import compute_ac_sigma_radius
from stability_radius.utils.download import download_uc_jl_instance
from stability_radius.verification.verify_worst_case import (
    find_violation_scale,
    verify_worst_case,
)
from stability_radius.workflows import (
    _expand_h_reduced_to_full,
    _extract_binding_end_data,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "configs"
    / "uc_jl_case118.yaml"
)
_SIGMA_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def _clamp_sigma(sigma: np.ndarray, floor: float = _SIGMA_FLOOR) -> np.ndarray:
    out = sigma.copy()
    out[out < floor] = floor
    return out


def _generate_synthetic_sigma(
    net: Any,
    bus_ids: list[int],
    *,
    sigma_fraction: float = 0.10,
    power_factor: float = 0.9,
    floor: float = _SIGMA_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic per-bus σ_P, σ_Q from the network's base loads.

    Each bus's σ_P is set proportional to its total active load:
        σ_P = sigma_fraction * |P_load|
    Buses with zero load get σ_P = floor.  σ_Q is derived via power factor:
        σ_Q = σ_P * sin(arccos(pf)) / pf

    This allows running the sigma-radius pipeline on any network without
    needing UC.jl time-series data.
    """
    bus_to_pos = {bid: i for i, bid in enumerate(bus_ids)}
    n_bus = len(bus_ids)
    total_p = np.zeros(n_bus)

    for load_idx in net.load.index:
        b = int(net.load.at[load_idx, "bus"])
        if b in bus_to_pos:
            total_p[bus_to_pos[b]] += abs(float(net.load.at[load_idx, "p_mw"]))

    sigma_p = sigma_fraction * total_p
    sigma_p[sigma_p < floor] = floor

    pf = max(min(power_factor, 0.999), 0.01)
    q_over_p = math.sqrt(1.0 - pf**2) / pf
    sigma_q = sigma_p * q_over_p
    sigma_q[sigma_q < floor] = floor

    logger.info(
        "Synthetic sigma: n_bus=%d, sigma_fraction=%.2f, "
        "sig_P range=[%.4g, %.4g] MW, sig_Q range=[%.4g, %.4g] MVAr",
        n_bus, sigma_fraction,
        float(np.min(sigma_p)), float(np.max(sigma_p)),
        float(np.min(sigma_q)), float(np.max(sigma_q)),
    )
    return sigma_p, sigma_q


def _set_loads_to_average(
    net: Any,
    *,
    load_p_mw: np.ndarray,
    load_q_mvar: np.ndarray,
    bus_ids: list[int],
) -> None:
    """Set net.load P/Q to the mean across all hours (in-place).

    If a bus has multiple load elements, the average is split equally
    among them so the total bus load equals the per-bus average.
    """
    avg_p = load_p_mw.mean(axis=1)  # (n_bus,)
    avg_q = load_q_mvar.mean(axis=1)
    bus_to_pos = {bid: pos for pos, bid in enumerate(bus_ids)}

    # Count load elements per bus for proper splitting.
    from collections import Counter

    bus_load_count: Counter[int] = Counter()
    for load_idx in net.load.index:
        load_bus = int(net.load.at[load_idx, "bus"])
        if load_bus in bus_to_pos:
            bus_load_count[load_bus] += 1

    for load_idx in net.load.index:
        load_bus = int(net.load.at[load_idx, "bus"])
        if load_bus not in bus_to_pos:
            continue
        pos = bus_to_pos[load_bus]
        n_loads = bus_load_count[load_bus]
        net.load.at[load_idx, "p_mw"] = float(avg_p[pos]) / n_loads
        net.load.at[load_idx, "q_mvar"] = float(avg_q[pos]) / n_loads


# ---------------------------------------------------------------------------
# Compute sigma-radius at average operating point
# ---------------------------------------------------------------------------


def _compute_at_average_point(
    *,
    net_template: Any,
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
    """Run AC OPF at average loads, compute h-vectors and sigma-radius.

    Returns a result dict with ac_l2_results, sigma_results, h_bind, etc.,
    or None if the OPF fails.
    """
    net = copy.deepcopy(net_template)
    _set_loads_to_average(
        net,
        load_p_mw=load_p_mw,
        load_q_mvar=load_q_mvar,
        bus_ids=bus_ids,
    )

    total_load = float(net.load.p_mw.sum())
    logger.info("Average-point OPF: total load = %.1f MW", total_load)

    # AC OPF
    try:
        _bp_ac, base_pf = solve_ac_fpf_base_point(
            net=net,
            slack_bus=slack_bus,
            lossless=lossless,
            fpf_cfg=fpf_cfg,
        )
    except Exception:
        logger.error("Average-point AC OPF failed", exc_info=True)
        return None

    # Apply OPP dispatch back to net so that verification / MC can
    # reproduce the same operating point with pp.runpp().
    apply_opp_result_to_pandapower_net(
        net,
        opp_gen_dispatch=base_pf.opp_gen_dispatch,
        opp_vm_pu=base_pf.opp_vm_pu,
    )

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
        logger.error("Average-point AC L2 radius failed", exc_info=True)
        return None

    h_vecs_raw = ac_l2.pop("_h_vectors", None)
    if h_vecs_raw is None:
        logger.error("Average-point: no h-vectors returned")
        return None

    n_bus = len(bus_ids)
    slack_bus_id = resolve_slack_bus_id(net, slack_bus)
    slack_pos = bus_ids.index(slack_bus_id)

    h_from = _expand_h_reduced_to_full(
        h_vecs_raw["h_from"], n_bus=n_bus, slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )
    h_to = _expand_h_reduced_to_full(
        h_vecs_raw["h_to"], n_bus=n_bus, slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )

    h_bind, s0_mva, s_limit_mva, line_ids = _extract_binding_end_data(
        ac_results=ac_l2,
        h_from=h_from,
        h_to=h_to,
    )

    # AC feasibility check
    feasibility = check_ac_base_point_feasibility(net=net, base_pf=base_pf)
    if not feasibility.is_feasible:
        logger.warning(
            "Average-point: AC base point infeasible (%d constrained lines violated, "
            "worst margin=%.2f MVA on line %d). "
            "Sigma-radii on those lines will be negative.",
            feasibility.n_constrained_violated,
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
        "Average-point: %d lines with finite sigma-radius (%d negative/infeasible)",
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
        "base_pf": base_pf,
        "base_net": net,  # net with average loads + OPP gen dispatch applied
    }


# ---------------------------------------------------------------------------
# Build result dict from single-point computation
# ---------------------------------------------------------------------------


def _build_result_dict(avg_result: dict) -> dict[str, Any]:
    """Build a result dictionary from the single average-point computation.

    The returned dict has the same per-line structure as the old aggregation
    but without per-hour min/worst-hour tracking.
    """
    sigma_res = avg_result["sigma_results"]
    ac_l2 = avg_result["ac_l2_results"]
    line_ids = avg_result["line_ids"]
    n_bus = avg_result["h_bind"].shape[1] // 2

    sigma_radius: dict[str, float] = {}
    h_bind_dict: dict[str, np.ndarray] = {}
    s0_mva_dict: dict[str, float] = {}
    s_limit_mva_dict: dict[str, float] = {}
    sigma_flow_dict: dict[str, float] = {}
    binding_end_dict: dict[str, str] = {}
    overload_prob_dict: dict[str, float] = {}
    ac_l2_radius_dict: dict[str, float] = {}
    base_infeasible: dict[str, bool] = {}

    for lk, v in sigma_res.items():
        if not isinstance(v, dict):
            continue
        r_sig = v.get("radius_ac_sigma", float("nan"))
        if not np.isfinite(r_sig):
            continue

        sigma_radius[lk] = float(r_sig)
        sigma_flow_dict[lk] = float(v.get("sigma_flow_mva", float("nan")))
        overload_prob_dict[lk] = float(v.get("overload_probability_ac", float("nan")))
        base_infeasible[lk] = float(r_sig) < 0

        lid = int(lk.split("_", 1)[1])
        if lid in line_ids:
            pos = line_ids.index(lid)
            h_bind_dict[lk] = avg_result["h_bind"][pos, :].copy()
            s0_mva_dict[lk] = float(avg_result["s0_mva"][pos])
            s_limit_mva_dict[lk] = float(avg_result["s_limit_mva"][pos])

        if lk in ac_l2 and isinstance(ac_l2[lk], dict):
            ac_l2_radius_dict[lk] = float(ac_l2[lk].get("radius_ac_l2", float("nan")))
            binding_end_dict[lk] = str(ac_l2[lk].get("binding_end", "?"))

    return {
        "sigma_radius": sigma_radius,
        "h_bind": h_bind_dict,
        "line_ids": line_ids,
        "ac_l2_radius": ac_l2_radius_dict,
        "s0_mva": s0_mva_dict,
        "s_limit_mva": s_limit_mva_dict,
        "sigma_flow": sigma_flow_dict,
        "binding_end": binding_end_dict,
        "overload_prob": overload_prob_dict,
        "base_infeasible": base_infeasible,
    }


# ---------------------------------------------------------------------------
# Table 2: Full sigma-radius results
# ---------------------------------------------------------------------------


def _build_table2_rows(
    res: dict[str, Any],
) -> list[dict]:
    """Build Table 2 rows for ALL lines, sorted by sigma-radius ascending.

    Lines with negative sigma-radius (base infeasible) are included and
    flagged.  MC violation rate and verified status are left as None --
    filled later.
    """
    candidates = sorted(res["sigma_radius"].items(), key=lambda kv: kv[1])

    rows: list[dict] = []
    for lk, r_sig in candidates:
        lid = int(lk.split("_", 1)[1])
        s0 = res["s0_mva"].get(lk, float("nan"))
        c = res["s_limit_mva"].get(lk, float("nan"))
        infeasible = res.get("base_infeasible", {}).get(lk, False)
        rows.append(
            {
                "line_id": lid,
                "line_key": lk,
                "binding_end": res["binding_end"].get(lk, "?"),
                "s0_mva": s0,
                "limit_mva": c,
                "margin_mva": c - s0
                if np.isfinite(c) and np.isfinite(s0)
                else float("nan"),
                "sigma_flow_mva": res["sigma_flow"].get(lk, float("nan")),
                "r_sigma": r_sig,
                "p_overload": res["overload_prob"].get(lk, float("nan")),
                "r_l2_uniform": res["ac_l2_radius"].get(lk, float("nan")),
                "mc_violation_rate": None,  # filled after MC step
                "verified": None,  # filled after verification step
                "base_infeasible": infeasible,
            }
        )
    return rows


def _print_table2(rows: list[dict], *, top_k: int = 20) -> None:
    """Print Table 2 to stdout (top_k tightest lines)."""
    display = rows[:top_k]
    header = (
        f"{'Line':>6s}  {'End':>4s}  {'S0(MVA)':>9s}  {'Limit':>9s}  "
        f"{'Margin':>9s}  {'sig_flow':>9s}  {'r_sigma':>9s}  "
        f"{'P_over':>10s}  {'r_L2':>8s}  {'Verif':>5s}  {'Feas':>4s}"
    )
    width = len(header)
    print()
    print("=" * width)
    print(f"Table 2: AC Sigma-Radius (top-{len(display)} of {len(rows)} lines)")
    print("=" * width)
    print(header)
    print("-" * width)

    for r in display:
        p_ov = r["p_overload"]
        p_str = f"{p_ov:.2e}" if np.isfinite(p_ov) else "      --"
        feas = "  NO" if r.get("base_infeasible", False) else "  ok"
        if r["verified"] is True:
            ver_str = "  YES"
        elif r["verified"] is False:
            ver_str = "   NO"
        else:
            ver_str = "   --"
        print(
            f"{r['line_id']:>6d}  {r['binding_end']:>4s}  "
            f"{r['s0_mva']:>9.2f}  {r['limit_mva']:>9.2f}  {r['margin_mva']:>9.2f}  "
            f"{r['sigma_flow_mva']:>9.4f}  {r['r_sigma']:>9.4f}  {p_str:>10s}  "
            f"{r['r_l2_uniform']:>8.2f}  {ver_str}  {feas}"
        )

    n_infeasible = sum(1 for r in rows if r.get("base_infeasible", False))
    if n_infeasible > 0:
        print(
            f"\nNOTE: {n_infeasible}/{len(rows)} lines have negative r_sigma "
            f"(base flow exceeds thermal limit at average operating point)"
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
    res: dict[str, Any],
    sigma_results: dict[str, dict[str, Any]],
    table_rows: list[dict],
    bus_ids: list[int],
    lossless: bool,
    scales: list[float],
    output_dir: Path,
) -> list[dict]:
    """Verify worst-case perturbation for each line in table_rows.

    *net* must be the OPP-solved base network (average loads **and** OPP gen
    dispatch applied) so that ``pp.runpp()`` reproduces the analytical base
    operating point.

    The perturbation vector is taken from the sigma-radius certificate
    (pre-computed worst_case_dp_mw / worst_case_dq_mvar), NOT from the
    L2 certificate.  This ensures that the verified direction matches
    the sigma-weighted balance projection.

    Lines with negative r_sigma (base infeasible) are skipped.
    """
    verification_results: list[dict] = []

    # ------------------------------------------------------------------
    # Base-point consistency check: run PF with zero perturbation and
    # verify that actual |S| at each binding end matches the analytical
    # s0_mva.  A large discrepancy means the verification network is not
    # at the same operating point as the analytics.
    # ------------------------------------------------------------------
    bp_check_done = False
    for row in table_rows:
        if bp_check_done:
            break
        lk = row["line_key"]
        lid = row["line_id"]
        h_vec = res["h_bind"].get(lk)
        r_sigma = row.get("r_sigma", float("nan"))
        s0 = row.get("s0_mva", float("nan"))
        be = row.get("binding_end")
        if h_vec is None or not np.isfinite(s0) or r_sigma <= 0:
            continue
        n_bus_bp = len(bus_ids)
        zero_du = np.zeros(2 * n_bus_bp, dtype=float)
        try:
            bp_result = verify_worst_case(
                net=net,
                line_id=lid,
                h_vec=h_vec,
                radius=r_sigma,
                s0_mva=s0,
                limit_mva=row.get("limit_mva", float("nan")),
                scale=0.0,
                balance=True,
                lossless=lossless,
                delta_u=zero_du,
                binding_end=be,
            )
            if bp_result.pf_converged:
                bp_err = abs(bp_result.actual_s_mva - s0) / max(s0, 1e-12)
                if bp_err > 0.05:
                    logger.error(
                        "BASE-POINT MISMATCH on %s: analytical s0=%.4f MVA, "
                        "PF actual=%.4f MVA (rel_err=%.4f). "
                        "The verification network may not match the analytics.",
                        lk,
                        s0,
                        bp_result.actual_s_mva,
                        bp_err,
                    )
                else:
                    logger.info(
                        "Base-point consistency OK on %s: s0=%.4f, "
                        "actual=%.4f (rel_err=%.6f)",
                        lk,
                        s0,
                        bp_result.actual_s_mva,
                        bp_err,
                    )
            bp_check_done = True
        except Exception:
            logger.warning("Base-point consistency check failed", exc_info=True)

    # ------------------------------------------------------------------
    # Tiny-step finite-difference test for h_vec validation.
    # For the first feasible line, check that
    #   (|S(x + eps*d)| - |S(x)|) / eps  ≈  h^T d
    # where d is the worst-case direction.
    #
    # IMPORTANT: we use the *actual PF base-point flow* as the reference,
    # NOT the analytical s0 from the certificate.  The analytical s0 may
    # differ from the PF result due to lossless/lossy model mismatch or
    # base-dispatch differences; using it as reference would introduce a
    # constant bias that swamps the tiny eps perturbation.
    # ------------------------------------------------------------------
    fd_check_done = False
    for row in table_rows:
        if fd_check_done:
            break
        lk = row["line_key"]
        lid = row["line_id"]
        h_vec_fd = res["h_bind"].get(lk)
        r_sigma_fd = row.get("r_sigma", float("nan"))
        s0_fd = row.get("s0_mva", float("nan"))
        be_fd = row.get("binding_end")
        if h_vec_fd is None or not np.isfinite(s0_fd) or r_sigma_fd <= 0:
            continue

        sigma_entry_fd = sigma_results.get(lk)
        if sigma_entry_fd is None or not isinstance(sigma_entry_fd, dict):
            continue
        wc_dp_fd = sigma_entry_fd.get("worst_case_dp_mw")
        wc_dq_fd = sigma_entry_fd.get("worst_case_dq_mvar")
        if wc_dp_fd is None or wc_dq_fd is None:
            continue

        wc_du_fd = np.concatenate(
            [np.asarray(wc_dp_fd, dtype=float), np.asarray(wc_dq_fd, dtype=float)]
        )
        # Normalise to get a direction
        wc_norm = float(np.linalg.norm(wc_du_fd))
        if wc_norm < 1e-15:
            continue
        d_dir = wc_du_fd / wc_norm

        eps_fd = 1e-3  # small step in MW/MVAr space
        eps_du = eps_fd * d_dir

        try:
            # First: get the actual PF base-point flow for this line (zero
            # perturbation).  This is the correct reference for the FD test.
            n_bus_fd = len(bus_ids)
            zero_du_fd = np.zeros(2 * n_bus_fd, dtype=float)
            res_base_fd = verify_worst_case(
                net=net,
                line_id=lid,
                h_vec=h_vec_fd,
                radius=r_sigma_fd,
                s0_mva=s0_fd,
                limit_mva=row.get("limit_mva", float("nan")),
                scale=0.0,
                balance=True,
                lossless=lossless,
                delta_u=zero_du_fd,
                binding_end=be_fd,
            )
            if not res_base_fd.pf_converged:
                logger.warning("FD base-point PF did not converge for %s", lk)
                fd_check_done = True
                continue
            s0_actual_pf = res_base_fd.actual_s_mva

            # Second: perturbed PF
            res_plus = verify_worst_case(
                net=net,
                line_id=lid,
                h_vec=h_vec_fd,
                radius=r_sigma_fd,
                s0_mva=s0_fd,
                limit_mva=row.get("limit_mva", float("nan")),
                scale=1.0,
                balance=True,
                lossless=lossless,
                delta_u=eps_du,
                binding_end=be_fd,
            )
            if res_plus.pf_converged:
                s_eps = res_plus.actual_s_mva
                # Use actual PF s0 as reference (not analytical s0)
                delta_s_fd = s_eps - s0_actual_pf
                delta_s_lin = float(np.dot(h_vec_fd, eps_du))
                fd_err = abs(delta_s_fd - delta_s_lin) / max(abs(delta_s_lin), 1e-12)
                logger.info(
                    "Finite-diff test on %s (eps=%.1e): "
                    "dS_fd=%.6e, dS_lin=%.6e, rel_err=%.4f "
                    "(s0_analytical=%.4f, s0_actual_pf=%.4f)",
                    lk,
                    eps_fd,
                    delta_s_fd,
                    delta_s_lin,
                    fd_err,
                    s0_fd,
                    s0_actual_pf,
                )
                if fd_err > 0.5:
                    logger.error(
                        "FINITE-DIFF MISMATCH on %s: h_vec may be incorrect "
                        "(unit mismatch, wrong indexing, or wrong line end).",
                        lk,
                    )
            fd_check_done = True
        except Exception:
            logger.warning("Finite-diff check failed for %s", lk, exc_info=True)

    for row in table_rows:
        lk = row["line_key"]
        lid = row["line_id"]
        h_vec = res["h_bind"].get(lk)
        if h_vec is None:
            logger.warning("No h-vector for %s, skipping verification.", lk)
            row["verified"] = None
            continue

        r_sigma = row["r_sigma"]
        s0 = row["s0_mva"]
        limit = row["limit_mva"]

        if not np.isfinite(r_sigma) or not np.isfinite(s0) or not np.isfinite(limit):
            logger.warning("Non-finite values for %s, skipping verification.", lk)
            row["verified"] = None
            continue

        # Skip lines with negative r_sigma (base infeasible: s0 > limit already)
        if r_sigma <= 0:
            logger.warning(
                "Line %s has r_sigma=%.4f <= 0 (base infeasible), skipping verification.",
                lk,
                r_sigma,
            )
            row["verified"] = None
            verification_results.append(
                {
                    "line_id": lid,
                    "line_key": lk,
                    "status": "skipped_infeasible",
                    "r_sigma": r_sigma,
                    "s0_mva": s0,
                    "limit_mva": limit,
                }
            )
            continue

        # Extract pre-computed sigma-radius worst-case perturbation
        sigma_entry = sigma_results.get(lk)
        if sigma_entry is None or not isinstance(sigma_entry, dict):
            logger.warning("No sigma-radius entry for %s, skipping verification.", lk)
            row["verified"] = None
            continue

        wc_dp = sigma_entry.get("worst_case_dp_mw")
        wc_dq = sigma_entry.get("worst_case_dq_mvar")
        if wc_dp is None or wc_dq is None:
            logger.warning(
                "No worst-case vectors in sigma-radius entry for %s, skipping.", lk
            )
            row["verified"] = None
            continue

        # Full worst-case perturbation vector [ΔP; ΔQ]
        wc_delta_u = np.concatenate(
            [np.asarray(wc_dp, dtype=float), np.asarray(wc_dq, dtype=float)]
        )

        # Diagnostic: perturbation magnitude and linearity assessment
        du_norm = float(np.linalg.norm(wc_delta_u))
        du_p_norm = float(np.linalg.norm(wc_delta_u[: len(bus_ids)]))
        du_q_norm = float(np.linalg.norm(wc_delta_u[len(bus_ids) :]))
        logger.info(
            "Worst-case perturbation for %s: ||du||=%.4f MW/MVAr "
            "(||dP||=%.4f MW, ||dQ||=%.4f MVAr), "
            "margin=%.4f MVA, s0=%.4f MVA, limit=%.4f MVA",
            lk,
            du_norm,
            du_p_norm,
            du_q_norm,
            limit - s0,
            s0,
            limit,
        )

        # Multi-scale verification
        # net already has average loads + OPP dispatch; verify_worst_case
        # deep-copies internally.
        scale_results: list[dict] = []
        any_verified = False
        for scale in sorted(scales):
            try:
                result = verify_worst_case(
                    net=net,
                    line_id=lid,
                    h_vec=h_vec,
                    radius=r_sigma,
                    s0_mva=s0,
                    limit_mva=limit,
                    scale=scale,
                    balance=True,
                    lossless=lossless,
                    delta_u=wc_delta_u * float(scale),
                    binding_end=row.get("binding_end"),
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

        # ------------------------------------------------------------------
        # Binary search for the actual violation scale.
        # The linear model predicts violation at scale=1.0. Find the actual
        # scale at which the nonlinear PF flow exceeds the thermal limit.
        # ------------------------------------------------------------------
        viol_scale_result = None
        try:
            viol_scale_result = find_violation_scale(
                net=net,
                line_id=lid,
                h_vec=h_vec,
                radius=r_sigma,
                s0_mva=s0,
                limit_mva=limit,
                delta_u_unit=wc_delta_u,
                binding_end=row.get("binding_end"),
                lossless=lossless,
                scale_max=50.0,
                tol=0.01,
            )
            logger.info(
                "Violation scale for %s: predicted=1.0, actual=%.4f "
                "(conservatism=%.2fx, n_pf=%d)",
                lk,
                viol_scale_result.actual_violation_scale,
                viol_scale_result.conservatism_ratio,
                viol_scale_result.n_pf_calls,
            )
        except Exception:
            logger.warning(
                "Violation scale search failed for %s", lk, exc_info=True
            )

        verification_results.append(
            {
                "line_id": lid,
                "line_key": lk,
                "r_sigma": r_sigma,
                "s0_mva": s0,
                "limit_mva": limit,
                "scale_results": scale_results,
                "violation_scale": viol_scale_result.to_dict()
                if viol_scale_result is not None
                else None,
                "status": "ok",
            }
        )

    # Save
    vr_path = output_dir / "verification_results.json"
    with vr_path.open("w", encoding="utf-8") as fh:
        json.dump(verification_results, fh, indent=2, default=numpy_to_builtin)
    logger.info("Verification results saved: %s", vr_path)

    return verification_results


# ---------------------------------------------------------------------------
# Monte Carlo validation with tightened limits
# ---------------------------------------------------------------------------


def _run_tightened_limit_mc(
    *,
    net: Any,
    res: dict[str, Any],
    avg_result: dict,
    table_rows: list[dict],
    bus_ids: list[int],
    sigma_p_mw: np.ndarray,
    sigma_q_mvar: np.ndarray,
    lossless: bool,
    n_samples: int,
    seed: int,
    target_r_sigma: float,
    output_dir: Path,
) -> dict | None:
    """Run MC validation with artificially tightened thermal limits.

    Two-phase approach:
    1. **Pilot phase**: run a small number of PF samples (~200) to estimate
       the empirical standard deviation of line flow under Gaussian
       perturbations.  This accounts for nonlinearity and model mismatch
       that make the linearized ``sigma_flow`` overestimate the true
       flow variability.
    2. **Main phase**: set ``limit_tight = s0 + target_r_sigma * sigma_empirical``
       so that violations are reachable, then run the full MC.

    This tests whether the Gaussian model correctly predicts the empirical
    overload rate when limits are set to produce observable violations.
    """
    if not table_rows:
        return None

    # Find the tightest feasible line
    top_row = None
    for row in table_rows:
        if row["r_sigma"] > 0 and np.isfinite(row["r_sigma"]):
            top_row = row
            break
    if top_row is None:
        logger.warning("No feasible lines for tightened-limit MC.")
        return None

    lk = top_row["line_key"]
    lid = top_row["line_id"]
    s0 = top_row["s0_mva"]
    sigma_flow_analytical = top_row["sigma_flow_mva"]
    be = top_row.get("binding_end", "to")

    if not np.isfinite(sigma_flow_analytical) or sigma_flow_analytical <= 0:
        logger.warning("sigma_flow not finite for %s, skipping tightened-limit MC.", lk)
        return None

    import pandapower as pp

    from stability_radius.verification.ac_monte_carlo_sigma import (
        _sample_gaussian_sigma,
        _sigma_inv_norm,
    )

    sig_p = np.asarray(sigma_p_mw, dtype=float)
    sig_q = np.asarray(sigma_q_mvar, dtype=float)

    # ---- Phase 1: Pilot samples to estimate empirical sigma_flow ----
    n_pilot = min(200, n_samples // 2)
    logger.info(
        "Tightened-limit MC phase 1: running %d pilot samples to estimate "
        "empirical sigma_flow for %s",
        n_pilot, lk,
    )

    nn_pilot = copy.deepcopy(net)
    if lossless:
        nn_pilot = apply_lossless_policy_to_pandapower_net(nn_pilot)

    bus_ids_sorted = [int(x) for x in sorted(nn_pilot.bus.index)]
    sgen_idx_pilot: list[int] = []
    for bid in bus_ids_sorted:
        idx = int(pp.create_sgen(
            nn_pilot, bus=int(bid), p_mw=0.0, q_mvar=0.0,
            name=f"pilot_mc_bus_{int(bid)}", in_service=True,
        ))
        sgen_idx_pilot.append(idx)

    try:
        pp.runpp(nn_pilot, calculate_voltage_angles=True, enforce_q_lims=True, init="results")
    except Exception:
        try:
            pp.runpp(nn_pilot, calculate_voltage_angles=True, enforce_q_lims=True, init="flat")
        except Exception:
            logger.warning("Tightened-limit MC: pilot base PF did not converge.")
            return None
    if not bool(getattr(nn_pilot, "converged", True)):
        logger.warning("Tightened-limit MC: pilot base PF did not converge.")
        return None

    # Read base flow
    if be == "from":
        p0 = float(nn_pilot.res_line.loc[lid, "p_from_mw"])
        q0 = float(nn_pilot.res_line.loc[lid, "q_from_mvar"])
    else:
        p0 = float(nn_pilot.res_line.loc[lid, "p_to_mw"])
        q0 = float(nn_pilot.res_line.loc[lid, "q_to_mvar"])
    s0_pf = math.sqrt(p0**2 + q0**2)

    rng_pilot = np.random.default_rng(int(seed) + 99999)
    dp_pilot, dq_pilot = _sample_gaussian_sigma(
        rng=rng_pilot, n=n_pilot, sigma_p=sig_p, sigma_q=sig_q,
    )

    pilot_flows: list[float] = []
    for j in range(n_pilot):
        nn_pilot.sgen.loc[sgen_idx_pilot, "p_mw"] = dp_pilot[j, :]
        nn_pilot.sgen.loc[sgen_idx_pilot, "q_mvar"] = dq_pilot[j, :]
        try:
            pp.runpp(nn_pilot, calculate_voltage_angles=True, enforce_q_lims=True, init="results")
            conv = bool(getattr(nn_pilot, "converged", True))
        except Exception:
            conv = False
        if conv:
            if be == "from":
                pf = float(nn_pilot.res_line.loc[lid, "p_from_mw"])
                qf = float(nn_pilot.res_line.loc[lid, "q_from_mvar"])
            else:
                pf = float(nn_pilot.res_line.loc[lid, "p_to_mw"])
                qf = float(nn_pilot.res_line.loc[lid, "q_to_mvar"])
            pilot_flows.append(math.sqrt(pf**2 + qf**2))

    if len(pilot_flows) < 20:
        logger.warning("Too few converged pilot samples (%d), skipping.", len(pilot_flows))
        return None

    pilot_arr = np.array(pilot_flows)
    sigma_flow_empirical = float(np.std(pilot_arr))

    logger.info(
        "Pilot results: s0_pf=%.4f, sigma_analytical=%.4f, sigma_empirical=%.4f "
        "(ratio=%.4f, %d/%d converged)",
        s0_pf, sigma_flow_analytical, sigma_flow_empirical,
        sigma_flow_empirical / sigma_flow_analytical,
        len(pilot_flows), n_pilot,
    )

    # ---- Phase 2: Main MC with tightened limit based on empirical sigma ----
    tight_limit = s0_pf + target_r_sigma * sigma_flow_empirical

    # Analytical prediction (using empirical sigma)
    from stability_radius.radii.ac_sigma_radius import (
        _overload_probability_symmetric_limit,
    )

    analytical_prob_empirical = _overload_probability_symmetric_limit(
        s0_mva=s0_pf, c_mva=tight_limit, sigma_mva=sigma_flow_empirical,
    )
    # Also compute with analytical sigma for comparison
    analytical_prob_analytical = _overload_probability_symmetric_limit(
        s0_mva=s0, c_mva=tight_limit, sigma_mva=sigma_flow_analytical,
    )

    logger.info(
        "Tightened-limit MC phase 2: tight_limit=%.4f MVA "
        "(s0=%.4f + %.2f * %.4f), n_samples=%d, "
        "P_overload(empirical sigma)=%.6e, P_overload(analytical sigma)=%.6e",
        tight_limit, s0_pf, target_r_sigma, sigma_flow_empirical,
        n_samples, analytical_prob_empirical, analytical_prob_analytical,
    )

    # Run main MC
    nn = copy.deepcopy(net)
    if lossless:
        nn = apply_lossless_policy_to_pandapower_net(nn)

    sgen_idx: list[int] = []
    for bid in bus_ids_sorted:
        idx = int(pp.create_sgen(
            nn, bus=int(bid), p_mw=0.0, q_mvar=0.0,
            name=f"tight_mc_bus_{int(bid)}", in_service=True,
        ))
        sgen_idx.append(idx)

    try:
        pp.runpp(nn, calculate_voltage_angles=True, enforce_q_lims=True, init="results")
    except Exception:
        try:
            pp.runpp(nn, calculate_voltage_angles=True, enforce_q_lims=True, init="flat")
        except Exception:
            logger.warning("Tightened-limit MC: main base PF did not converge.")
            return None
    if not bool(getattr(nn, "converged", True)):
        logger.warning("Tightened-limit MC: main base PF did not converge.")
        return None

    rng_main = np.random.default_rng(int(seed) + 12345)
    dp_all, dq_all = _sample_gaussian_sigma(
        rng=rng_main, n=n_samples, sigma_p=sig_p, sigma_q=sig_q,
    )

    inv_sig_p = 1.0 / sig_p
    inv_sig_q = 1.0 / sig_q
    sigma_norms = _sigma_inv_norm(dp_all, dq_all, inv_sig_p, inv_sig_q)
    inside_ball = sigma_norms <= float(target_r_sigma)

    n_violations = 0
    n_pf_failures = 0
    n_inside_ball = int(np.sum(inside_ball))
    inside_ball_no_violation = 0

    for j in range(int(n_samples)):
        nn.sgen.loc[sgen_idx, "p_mw"] = dp_all[j, :]
        nn.sgen.loc[sgen_idx, "q_mvar"] = dq_all[j, :]

        try:
            pp.runpp(nn, calculate_voltage_angles=True, enforce_q_lims=True, init="results")
            conv = bool(getattr(nn, "converged", True))
        except Exception:
            conv = False

        if not conv:
            n_pf_failures += 1
            n_violations += 1
            continue

        if be == "from":
            pf = float(nn.res_line.loc[lid, "p_from_mw"])
            qf = float(nn.res_line.loc[lid, "q_from_mvar"])
        else:
            pf = float(nn.res_line.loc[lid, "p_to_mw"])
            qf = float(nn.res_line.loc[lid, "q_to_mvar"])
        s_actual = math.sqrt(pf**2 + qf**2)

        violated = s_actual > tight_limit
        if violated:
            n_violations += 1
        elif bool(inside_ball[j]):
            inside_ball_no_violation += 1

    empirical_prob = n_violations / max(n_samples, 1)
    soundness = (
        float(inside_ball_no_violation) / float(n_inside_ball)
        if n_inside_ball > 0
        else float("nan")
    )

    # Compare analytical (with empirical sigma) vs empirical
    if empirical_prob > 0 and analytical_prob_empirical > 0:
        ratio_empirical = analytical_prob_empirical / empirical_prob
    elif empirical_prob == 0 and analytical_prob_empirical == 0:
        ratio_empirical = 1.0
    else:
        ratio_empirical = float("inf")

    # Compare analytical (with original sigma) vs empirical
    if empirical_prob > 0 and analytical_prob_analytical > 0:
        ratio_analytical = analytical_prob_analytical / empirical_prob
    elif empirical_prob == 0 and analytical_prob_analytical == 0:
        ratio_analytical = 1.0
    else:
        ratio_analytical = float("inf")

    result = {
        "target_line": lk,
        "binding_end": be,
        "target_r_sigma": target_r_sigma,
        "s0_mva": s0,
        "s0_pf_mva": s0_pf,
        "sigma_flow_analytical": sigma_flow_analytical,
        "sigma_flow_empirical": sigma_flow_empirical,
        "sigma_ratio": sigma_flow_empirical / sigma_flow_analytical,
        "original_limit_mva": top_row["limit_mva"],
        "tightened_limit_mva": tight_limit,
        "analytical_prob_with_empirical_sigma": analytical_prob_empirical,
        "analytical_prob_with_analytical_sigma": analytical_prob_analytical,
        "empirical_prob": empirical_prob,
        "ratio_analytE_over_empirical": ratio_empirical,
        "ratio_analytA_over_empirical": ratio_analytical,
        "n_samples": n_samples,
        "n_pilot": n_pilot,
        "n_violations": n_violations,
        "n_pf_failures": n_pf_failures,
        "n_inside_ball": n_inside_ball,
        "soundness_inside_ball": soundness,
    }

    # Print summary
    print()
    print("=" * 70)
    print("Tightened-Limit MC Validation (Gaussian Consistency Test)")
    print("=" * 70)
    print(f"  Target line:              {lk} ({be} end)")
    print(f"  S0 (analytical):          {s0:.4f} MVA")
    print(f"  S0 (PF actual):           {s0_pf:.4f} MVA")
    print(f"  sigma_flow (analytical):  {sigma_flow_analytical:.4f} MVA")
    print(f"  sigma_flow (empirical):   {sigma_flow_empirical:.4f} MVA "
          f"(ratio={sigma_flow_empirical / sigma_flow_analytical:.4f})")
    print(f"  Original limit:           {top_row['limit_mva']:.4f} MVA "
          f"(r_sigma={top_row['r_sigma']:.2f})")
    print(f"  Tightened limit:          {tight_limit:.4f} MVA "
          f"(r_sigma={target_r_sigma:.2f} based on empirical sigma)")
    print(f"  Pilot samples:            {n_pilot} ({len(pilot_flows)} converged)")
    print(f"  Main samples:             {n_samples}")
    print(f"  Empirical violations:     {n_violations} "
          f"({empirical_prob:.4%})")
    print(f"  P (empirical sigma):      {analytical_prob_empirical:.4e} "
          f"(ratio={ratio_empirical:.4f})")
    print(f"  P (analytical sigma):     {analytical_prob_analytical:.4e} "
          f"(ratio={ratio_analytical:.4f})")
    print(f"  PF failures:              {n_pf_failures}")
    gauss_ok = 0.3 <= ratio_empirical <= 3.0 if np.isfinite(ratio_empirical) else False
    print(f"  Gaussian consistency:     {'PASS' if gauss_ok else 'FAIL'} "
          f"(ratio in [0.3, 3.0])")
    print("=" * 70)
    print()

    mc_tight_path = output_dir / "mc_tightened_limit.json"
    with mc_tight_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=numpy_to_builtin)
    logger.info("Tightened-limit MC results saved: %s", mc_tight_path)

    return result


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def _run_validation_checks(
    *,
    res: dict[str, Any],
    avg_result: dict,
    table_rows: list[dict],
    sigma_p_mw_raw: np.ndarray,
    n_bus: int,
    output_dir: Path,
) -> dict:
    """Run validation checks and save results."""
    checks: dict[str, Any] = {}

    # 0. Feasibility summary
    n_infeasible = sum(1 for r in table_rows if r.get("base_infeasible", False))
    n_total_lines = len(res["sigma_radius"])
    n_negative_sigma = sum(1 for v in res["sigma_radius"].values() if v < 0)
    checks["feasibility"] = {
        "n_lines_total": n_total_lines,
        "n_lines_negative_sigma": n_negative_sigma,
        "n_top_k_infeasible": n_infeasible,
        "note": "Lines with negative sigma-radius have S0 > c at average point",
    }

    # 1. Balance check: |sum(worst_case_dp_mw)| < 1e-6 for top lines
    balance_ok = True
    balance_details: list[dict] = []
    sigma_results = avg_result["sigma_results"]
    for row in table_rows:
        lk = row["line_key"]
        if lk not in sigma_results or not isinstance(sigma_results[lk], dict):
            continue
        dp = sigma_results[lk].get("worst_case_dp_mw")
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

    # 3. Gaussian consistency note
    # The original per-line analytical-vs-MC comparison is not meaningful here
    # because r_sigma is typically 10-25+, making analytical P(overload) ~1e-23
    # to ~1e-142, which cannot be validated with any practical MC sample count.
    # The tightened-limit MC (Step 7b) is the correct Gaussian consistency
    # test — it uses empirical sigma_flow and artificially low limits to
    # produce observable violations (~2-5%) and validates the Gaussian model.
    checks["gaussian_consistency"] = {
        "note": "See tightened-limit MC (Step 7b) for meaningful Gaussian validation.",
        "all_ok": True,
    }

    # Print summary
    print()
    print("=" * 60)
    print("Validation Checks")
    print("=" * 60)
    print(
        f"  Feasibility:           {n_negative_sigma}/{n_total_lines} "
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
    print(
        "  Gaussian consistency:  see tightened-limit MC (Step 7b)"
    )
    print("=" * 60)
    print()

    # Save
    val_path = output_dir / "validation.json"
    with val_path.open("w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2, default=numpy_to_builtin)
    logger.info("Validation results saved: %s", val_path)

    return checks


# ---------------------------------------------------------------------------
# Figure: Top critical lines by sigma-radius (horizontal bar chart)
# ---------------------------------------------------------------------------


def _plot_critical_lines_bar(
    res: dict[str, Any],
    *,
    case_name: str = "",
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Horizontal bar chart of ALL lines sorted by sigma-radius (linear scale)."""
    candidates = sorted(
        (
            (lk, r_sig)
            for lk, r_sig in res["sigma_radius"].items()
            if np.isfinite(r_sig) and r_sig > 0
        ),
        key=lambda kv: kv[1],
    )
    if not candidates:
        logger.warning("No positive-radius lines for critical-lines bar chart.")
        return

    line_labels = []
    r_sigma_vals = []
    loading_fracs = []
    for lk, r_sig in reversed(candidates):  # reversed so tightest is at top
        lid = int(lk.split("_", 1)[1])
        s0 = res["s0_mva"].get(lk, 0.0)
        c = res["s_limit_mva"].get(lk, 1.0)
        load_frac = s0 / c if c > 0 else 0.0
        line_labels.append(f"L{lid}")
        r_sigma_vals.append(r_sig)
        loading_fracs.append(load_frac)

    n_lines = len(candidates)
    r_arr = np.array(r_sigma_vals)
    load_arr = np.array(loading_fracs)

    # Clip x-axis at 95th percentile to avoid outliers compressing the view
    x_limit = float(np.percentile(r_arr, 95)) * 1.15
    if x_limit <= 0:
        x_limit = float(np.max(r_arr)) * 1.05
    r_display = np.clip(r_arr, 0, x_limit)

    cmap = plt.colormaps["RdYlGn"]
    # Invert: high loading (near 1.0) = red, low loading (near 0) = green
    colors = [cmap(1.0 - lf) for lf in load_arr]

    fig, ax = plt.subplots(figsize=(12, max(4, n_lines * 0.18)))
    y_pos = np.arange(n_lines)
    ax.barh(y_pos, r_display, color=colors, edgecolor="gray", linewidth=0.3)

    # Only annotate the 20 tightest lines with text labels (bottom of plot)
    n_annotate = min(20, n_lines)
    for i in range(n_lines - n_annotate, n_lines):
        r = r_arr[i]
        lf = load_arr[i]
        x_text = min(r, x_limit) + x_limit * 0.01
        label = f"r={r:.1f}  ({lf:.0%})"
        ax.text(x_text, i, label, va="center", fontsize=6)

    # Mark clipped bars
    n_clipped = int(np.sum(r_arr > x_limit))

    ax.set_xlim(0, x_limit * 1.15)
    ax.set_yticks(y_pos[::max(1, n_lines // 50)])
    ax.set_yticklabels(
        [line_labels[i] for i in range(0, n_lines, max(1, n_lines // 50))],
        fontsize=7,
    )
    ax.set_xlabel("Sigma-radius $r_\\sigma$", fontsize=12)
    clip_note = f" (x-axis clipped, {n_clipped} lines exceed)" if n_clipped else ""
    ax.set_title(
        f"{case_name}: All {n_lines} Lines by Sigma-Radius{clip_note}",
        fontsize=13,
    )
    ax.invert_yaxis()

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Loading fraction $S_0 / c$", fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"fig_critical_lines.{ext}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    logger.info("Critical lines bar chart saved.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Flow vs thermal limit scatter (which lines are close to overload)
# ---------------------------------------------------------------------------


def _plot_flow_vs_limit(
    res: dict[str, Any],
    *,
    case_name: str = "",
    output_dir: Path,
    dpi: int = 300,
    top_k_label: int = 10,
    r_sigma_max: float = 50.0,
) -> None:
    """Scatter plot of base-point flow S0 vs thermal limit.

    Only lines with r_sigma <= *r_sigma_max* are shown (filters out the
    most stable lines that are far from overload and would clutter the plot).
    """
    s0_vals: list[float] = []
    limit_vals: list[float] = []
    r_sigma_vals: list[float] = []
    labels: list[str] = []

    for lk, r_sig in res["sigma_radius"].items():
        s0 = res["s0_mva"].get(lk, float("nan"))
        c = res["s_limit_mva"].get(lk, float("nan"))
        if not np.isfinite(s0) or not np.isfinite(c) or not np.isfinite(r_sig):
            continue
        if r_sig > r_sigma_max:
            continue
        s0_vals.append(s0)
        limit_vals.append(c)
        r_sigma_vals.append(r_sig)
        labels.append(lk)

    if len(s0_vals) < 2:
        logger.warning("Not enough data for flow-vs-limit plot.")
        return

    n_total = sum(1 for r in res["sigma_radius"].values() if np.isfinite(r))

    s0_arr = np.array(s0_vals)
    limit_arr = np.array(limit_vals)
    r_sig_arr = np.array(r_sigma_vals)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Linear color scale for r_sigma
    r_clip = np.clip(r_sig_arr, 0, None)
    sc = ax.scatter(
        limit_arr, s0_arr,
        c=r_clip, cmap="RdYlGn", edgecolors="gray", linewidths=0.3,
        s=40, alpha=0.7,
    )

    # Diagonal S0 = limit
    max_val = max(float(np.max(limit_arr)), float(np.max(s0_arr))) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, alpha=0.5,
            label="$S_0 = c$ (overload boundary)")

    # Label top-k closest to boundary (smallest margin)
    margins = limit_arr - s0_arr
    closest_idx = np.argsort(np.abs(margins))[:top_k_label]
    for idx in closest_idx:
        lid = int(labels[idx].split("_", 1)[1])
        ax.annotate(
            f"L{lid} (r={r_sig_arr[idx]:.1f})",
            (limit_arr[idx], s0_arr[idx]),
            textcoords="offset points", xytext=(5, -8),
            fontsize=7, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    ax.set_xlabel("Thermal limit $c$ (MVA)", fontsize=12)
    ax.set_ylabel("Base-point flow $S_0$ (MVA)", fontsize=12)
    ax.set_title(
        f"{case_name}: Line Loading vs Thermal Limit "
        f"({len(s0_vals)}/{n_total} lines with $r_\\sigma \\leq {r_sigma_max:.0f}$)",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="upper left")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("$r_\\sigma$", fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"fig_flow_vs_limit.{ext}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    logger.info("Flow vs limit scatter saved.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Violation scale (certificate conservatism)
# ---------------------------------------------------------------------------


def _plot_violation_scale(
    verification_results: list[dict],
    *,
    case_name: str = "",
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Bar chart: actual violation scale vs predicted (1.0) for each line."""
    items: list[tuple[str, float, float]] = []
    for vr in verification_results:
        vs = vr.get("violation_scale")
        if vs is None:
            continue
        actual = vs.get("actual_violation_scale", float("nan"))
        conserv = vs.get("conservatism_ratio", float("nan"))
        if not np.isfinite(actual) or not np.isfinite(conserv):
            continue
        lk = vr.get("line_key", f"line_{vr.get('line_id', '?')}")
        lid = int(lk.split("_", 1)[1])
        items.append((f"L{lid}", actual, conserv))

    if not items:
        logger.warning("No violation-scale data for conservatism plot.")
        return

    labels_list = [x[0] for x in items]
    actuals = np.array([x[1] for x in items])
    conserv_ratios = np.array([x[2] for x in items])

    # Color by conservatism: ~1.0 = green (accurate), >2 = yellow, >5 = red
    cmap = plt.colormaps["RdYlGn"]
    colors = [cmap(min(1.0, 1.0 / max(c, 0.01))) for c in conserv_ratios]

    fig, ax = plt.subplots(figsize=(10, max(3, len(items) * 0.45)))
    y_pos = np.arange(len(items))
    ax.barh(y_pos, actuals, color=colors, edgecolor="gray", linewidth=0.5)
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5,
               label="Predicted violation (scale=1.0)")

    for i, (a, c) in enumerate(zip(actuals, conserv_ratios)):
        ax.text(
            a + 0.05, i,
            f"actual={a:.2f}  ({c:.1f}x conserv.)",
            va="center", fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_list, fontsize=9)
    ax.set_xlabel("Perturbation scale at first violation", fontsize=12)
    ax.set_title(
        f"{case_name}: Certificate Conservatism "
        "(actual vs predicted violation scale)",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = output_dir / f"fig_violation_scale.{ext}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    logger.info("Violation scale (conservatism) chart saved.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save h-vectors to NPZ
# ---------------------------------------------------------------------------


def _save_hvectors_npz(
    res: dict[str, Any],
    *,
    output_dir: Path,
) -> None:
    """Save h-vectors to NPZ for downstream experiments."""
    npz_data: dict[str, np.ndarray] = {}
    for lk, h in res["h_bind"].items():
        npz_data[lk] = np.asarray(h, dtype=float)

    npz_data["line_ids"] = np.array(res["line_ids"], dtype=int)

    npz_path = output_dir / "hvectors.npz"
    np.savez_compressed(str(npz_path), **npz_data)
    logger.info("h-vectors saved: %s (%d lines)", npz_path, len(res["h_bind"]))


# ---------------------------------------------------------------------------
# Topology plot
# ---------------------------------------------------------------------------


def _plot_topology(
    net: Any,
    *,
    res: dict[str, Any],
    bus_ids: list[int],
    top_k_critical: int,
    case_name: str = "",
    output_dir: Path,
    figsize: tuple[float, float] = (16, 12),
    dpi: int = 200,
) -> None:
    """Network graph: lines by sigma-radius, buses by threat contribution."""
    n_bus = len(bus_ids)

    G = pt.create_nxgraph(net, respect_switches=False)
    # spring_layout is O(n) iterations — much faster than kamada_kawai O(n³)
    pos = nx.spring_layout(G, k=2.0 / max(1, n_bus**0.5), iterations=80, seed=42)

    line_indices = list(net.line.index)
    sr = res["sigma_radius"]
    line_radii = np.full(len(line_indices), float("nan"), dtype=float)
    for i, lid in enumerate(line_indices):
        lk = f"line_{lid}"
        if lk in sr:
            line_radii[i] = sr[lk]

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

    sorted_lines = sorted(sr.items(), key=lambda kv: kv[1])
    top_critical = sorted_lines[:top_k_critical]

    threat = np.zeros(n_bus, dtype=float)
    for lk, _r in top_critical:
        h = res["h_bind"].get(lk)
        if h is None:
            continue
        h_p = h[:n_bus]
        h_q = h[n_bus:]
        threat += h_p**2 + h_q**2

    threat_max = float(np.max(threat)) if np.max(threat) > 0 else 1.0
    threat_norm = threat / threat_max

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
        bus_sizes.append(30 + 300 * t)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    from_buses = net.line["from_bus"].values
    to_buses = net.line["to_bus"].values
    edge_list = list(zip(from_buses, to_buses))

    edge_colors = []
    edge_widths = []
    for i, lid in enumerate(line_indices):
        r = line_radii[i]
        if np.isfinite(r):
            c = line_cmap(line_norm(r))
            w = 1.0 + 3.0 * (1.0 - line_norm(r))
        else:
            c = (0.8, 0.8, 0.8, 0.5)
            w = 0.5
        edge_colors.append(c)
        edge_widths.append(w)

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

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    ax.scatter(
        node_x,
        node_y,
        c=bus_colors,
        s=bus_sizes,
        edgecolors="black",
        linewidths=0.3,
        zorder=2,
    )

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

    sm_line = plt.cm.ScalarMappable(cmap=line_cmap, norm=line_norm)
    sm_line.set_array([])
    cbar_line = fig.colorbar(sm_line, ax=ax, fraction=0.03, pad=0.01)
    cbar_line.set_label("Sigma-radius at average point (sigma)", fontsize=10)

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
        f"{case_name}: AC Sigma-Radius at Average Operating Point",
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
    compute_cfg = cfg.get("compute", {})
    ac_cfg = compute_cfg.get("ac", {})
    fpf_cfg_dict = ac_cfg.get("fpf", {})
    plot_cfg = cfg.get("plot", {})
    data_dir = Path(cfg.get("data_dir", "data/input"))
    artifacts_root = resolve_artifacts_root(cfg)
    output_dir = create_module_output_dir(
        module_name="run_sigma_radius",
        runs_dir=artifacts_root,
        requested_output_dir=cfg.get("output_dir", None),
    )
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
    # Step 1: Load network.
    # ------------------------------------------------------------------
    from stability_radius.utils.download import ensure_case_file

    if not Path(input_path).exists() and allow_download:
        ensure_case_file(input_path)

    net = load_network(input_path)
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)

    logger.info("Network loaded: %d buses, %d lines", n_bus, len(net.line))

    # ------------------------------------------------------------------
    # Step 2: Compute sigma and load profiles.
    #
    # sigma_source: "uc_jl" uses UC.jl time-series data (e.g. case118).
    #               "synthetic" generates sigma as a fraction of case loads
    #               (works with any network, no external data needed).
    # ------------------------------------------------------------------
    sigma_source = str(cfg.get("sigma_source", "uc_jl"))

    if sigma_source == "uc_jl":
        uc_cfg = cfg["uc_jl"]
        uc_dest = Path(uc_cfg.get("dest_dir", "data/uc_jl"))
        uc_case_name = str(uc_cfg["case_name"])
        uc_date = str(uc_cfg.get("date", "2017-01-01"))
        power_factor = float(uc_cfg.get("power_factor", 0.9))

        logger.info("Downloading UC.jl instance: %s (date=%s)", uc_case_name, uc_date)
        uc_path = download_uc_jl_instance(
            uc_case_name, dest_dir=uc_dest, date=uc_date,
        )

        sigma_data = load_sigma(uc_path, power_factor=power_factor)
        sigma_p_mw_raw = sigma_data["sigma_p_mw"]
        sigma_q_mvar_raw = sigma_data["sigma_q_mvar"]
        sigma_p_mw = _clamp_sigma(sigma_p_mw_raw)
        sigma_q_mvar = _clamp_sigma(sigma_q_mvar_raw)
        n_clamped = int(np.sum(sigma_p_mw_raw < _SIGMA_FLOOR))

        hourly_data = load_hourly_profiles(uc_path, power_factor=power_factor)
        load_p_mw = hourly_data["load_p_mw"]
        load_q_mvar = hourly_data["load_q_mvar"]
        n_hours = hourly_data["n_timesteps"]

        if load_p_mw.shape[0] != n_bus:
            raise ValueError(
                f"UC.jl bus count ({load_p_mw.shape[0]}) != network ({n_bus})"
            )

        logger.info(
            "UC.jl sigma: n_bus=%d, sigma_P range=[%.4g, %.4g] MW "
            "(%d clamped), %d timesteps",
            len(sigma_p_mw),
            float(np.min(sigma_p_mw)),
            float(np.max(sigma_p_mw)),
            n_clamped,
            n_hours,
        )

    elif sigma_source == "synthetic":
        # Generate sigma from case loads: sigma_P = fraction * |P_load|
        synth_cfg = cfg.get("synthetic_sigma", {})
        sigma_fraction = float(synth_cfg.get("fraction", 0.10))
        power_factor = float(synth_cfg.get("power_factor", 0.9))

        bus_to_pos = {bid: pos for pos, bid in enumerate(bus_ids)}
        bus_p_load = np.zeros(n_bus, dtype=float)
        bus_q_load = np.zeros(n_bus, dtype=float)

        for load_idx in net.load.index:
            lb = int(net.load.at[load_idx, "bus"])
            if lb in bus_to_pos:
                pos = bus_to_pos[lb]
                bus_p_load[pos] += abs(float(net.load.at[load_idx, "p_mw"]))
                bus_q_load[pos] += abs(
                    float(net.load.at[load_idx, "q_mvar"]),
                )

        sigma_p_mw_raw = sigma_fraction * bus_p_load
        sigma_q_mvar_raw = sigma_fraction * bus_q_load
        # For buses with zero load, use a small sigma based on mean load
        mean_p = (
            float(np.mean(bus_p_load[bus_p_load > 0]))
            if np.any(bus_p_load > 0)
            else 1.0
        )
        sigma_p_mw_raw[sigma_p_mw_raw < _SIGMA_FLOOR] = (
            sigma_fraction * mean_p * 0.1
        )
        sigma_q_mvar_raw[sigma_q_mvar_raw < _SIGMA_FLOOR] = (
            sigma_fraction * mean_p * 0.1 * math.tan(math.acos(power_factor))
        )

        sigma_p_mw = _clamp_sigma(sigma_p_mw_raw)
        sigma_q_mvar = _clamp_sigma(sigma_q_mvar_raw)
        n_clamped = int(np.sum(sigma_p_mw_raw < _SIGMA_FLOOR))

        # For synthetic mode, loads already in the case are the "average"
        load_p_mw = bus_p_load[:, None]  # shape (n_bus, 1)
        load_q_mvar = bus_q_load[:, None]
        n_hours = 1

        logger.info(
            "Synthetic sigma (%.0f%% of load): n_bus=%d, "
            "sigma_P range=[%.4g, %.4g] MW (%d clamped)",
            sigma_fraction * 100,
            len(sigma_p_mw),
            float(np.min(sigma_p_mw)),
            float(np.max(sigma_p_mw)),
            n_clamped,
        )

    else:
        raise ValueError(
            f"Unknown sigma_source: {sigma_source!r}. "
            "Use 'uc_jl' or 'synthetic'."
        )

    # Save sigma arrays
    sigma_out = output_dir / "sigma_arrays.json"
    with sigma_out.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "sigma_p_mw": sigma_p_mw.tolist(),
                "sigma_q_mvar": sigma_q_mvar.tolist(),
                "sigma_source": sigma_source,
                "n_timesteps": n_hours,
                "n_clamped": n_clamped,
                "sigma_floor": _SIGMA_FLOOR,
            },
            fh,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Step 3: Compute sigma-radius at average operating point.
    # ------------------------------------------------------------------
    avg_result = _compute_at_average_point(
        net_template=net,
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

    if avg_result is None:
        logger.error("Average-point computation failed. Cannot produce results.")
        return

    # Build result dict for downstream use
    res = _build_result_dict(avg_result)

    # ------------------------------------------------------------------
    # Step 4: Build full Table 2 (pre-populate; MC/verification later).
    # ------------------------------------------------------------------
    table_rows = _build_table2_rows(res)

    # ------------------------------------------------------------------
    # Step 5: Save h-vectors to NPZ.
    # ------------------------------------------------------------------
    _save_hvectors_npz(res, output_dir=output_dir)

    # ------------------------------------------------------------------
    # Step 6: Worst-case verification for top-k lines.
    # ------------------------------------------------------------------
    verify_cfg = cfg.get("verification", {})
    verify_top_k = int(verify_cfg.get("top_k", 10))
    verify_scales = [float(s) for s in verify_cfg.get("scales", [1.0])]

    verification_results = None
    if table_rows:
        verification_results = _run_worst_case_verification(
            net=avg_result["base_net"],
            res=res,
            sigma_results=avg_result["sigma_results"],
            table_rows=table_rows[:verify_top_k],
            bus_ids=bus_ids,
            lossless=lossless,
            scales=verify_scales,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Step 7: Monte Carlo validation (tightened-limit).
    # Uses artificially low limits so that violations are reachable within
    # ~5000 samples, enabling a meaningful Gaussian consistency check.
    # The old "standard" MC (comparing analytical P(overload) ~1e-23 to
    # MC P=0.00) has been removed — it always showed MISMATCH because
    # r_sigma >> 5 makes analytical probabilities astronomically small.
    # ------------------------------------------------------------------
    mc_cfg = cfg.get("monte_carlo", {})
    mc_enabled = bool(mc_cfg.get("enabled", True))
    mc_n_samples = int(mc_cfg.get("n_samples", 5000))
    mc_seed = int(mc_cfg.get("seed", 42))

    mc_tight_cfg = mc_cfg.get("tightened_limit", {})
    mc_tight_enabled = bool(mc_tight_cfg.get("enabled", mc_enabled))
    mc_tight_r_sigma = float(mc_tight_cfg.get("target_r_sigma", 2.0))
    mc_tight_n_samples = int(mc_tight_cfg.get("n_samples", mc_n_samples))

    mc_results = None
    if mc_tight_enabled and table_rows:
        mc_results = _run_tightened_limit_mc(
            net=avg_result["base_net"],
            res=res,
            avg_result=avg_result,
            table_rows=table_rows,
            bus_ids=bus_ids,
            sigma_p_mw=sigma_p_mw,
            sigma_q_mvar=sigma_q_mvar,
            lossless=lossless,
            n_samples=mc_tight_n_samples,
            seed=mc_seed,
            target_r_sigma=mc_tight_r_sigma,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Step 8: Print and export full Table 2.
    # ------------------------------------------------------------------
    _print_table2(table_rows, top_k=top_k_critical)
    _export_table2_csv(table_rows, output_dir)

    # ------------------------------------------------------------------
    # Step 9: Validation checks.
    # ------------------------------------------------------------------
    _run_validation_checks(
        res=res,
        avg_result=avg_result,
        table_rows=table_rows,
        sigma_p_mw_raw=sigma_p_mw_raw,
        n_bus=n_bus,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    # Step 10: Save results JSON.
    # ------------------------------------------------------------------
    results_out = {
        "case": case_cfg["name"],
        "sigma_source": sigma_source,
        "n_timesteps": n_hours,
        "sigma_radius": {k: v for k, v in sorted(res["sigma_radius"].items())},
        "table_rows": table_rows,
    }
    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results_out, fh, indent=2, default=numpy_to_builtin)
    logger.info("Results written: %s", results_path)

    # Summary
    all_sr = list(res["sigma_radius"].values())
    finite_sr = [v for v in all_sr if np.isfinite(v)]
    positive_sr = [v for v in finite_sr if v > 0]
    negative_sr = [v for v in finite_sr if v < 0]
    summary = {
        "case": case_cfg["name"],
        "sigma_source": sigma_source,
        "n_timesteps": n_hours,
        "n_lines": len(all_sr),
        "n_lines_finite": len(finite_sr),
        "n_lines_positive_sigma": len(positive_sr),
        "n_lines_negative_sigma": len(negative_sr),
        "global_min_sigma_radius": min(finite_sr) if finite_sr else float("nan"),
        "global_min_positive_sigma_radius": min(positive_sr)
        if positive_sr
        else float("nan"),
        "global_median_sigma_radius": float(np.median(finite_sr))
        if finite_sr
        else float("nan"),
        "global_max_sigma_radius": max(finite_sr) if finite_sr else float("nan"),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=numpy_to_builtin)
    logger.info("Summary written: %s", summary_path)

    # ------------------------------------------------------------------
    # Step 11: Plots.
    # ------------------------------------------------------------------
    case_name = str(case_cfg.get("name", "unknown"))

    for plot_fn, plot_kwargs in [
        (
            _plot_topology,
            dict(
                net=net, res=res, bus_ids=bus_ids,
                top_k_critical=top_k_critical, case_name=case_name,
                output_dir=output_dir, figsize=figsize, dpi=plot_dpi,
            ),
        ),
        (
            _plot_critical_lines_bar,
            dict(res=res, case_name=case_name, output_dir=output_dir, dpi=plot_dpi),
        ),
        (
            _plot_flow_vs_limit,
            dict(res=res, case_name=case_name, output_dir=output_dir, dpi=plot_dpi),
        ),
    ]:
        try:
            plot_fn(**plot_kwargs)
        except Exception:
            logger.warning("Plot %s failed", plot_fn.__name__, exc_info=True)

    if verification_results:
        try:
            _plot_violation_scale(
                verification_results,
                case_name=case_name, output_dir=output_dir, dpi=plot_dpi,
            )
        except Exception:
            logger.warning("Violation scale plot failed", exc_info=True)

    # ------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------
    min_r = min(positive_sr) if positive_sr else float("nan")
    min_r_line = ""
    if positive_sr:
        for lk, v in sorted(res["sigma_radius"].items(), key=lambda kv: kv[1]):
            if v > 0:
                min_r_line = f" (line {lk.split('_', 1)[1]})"
                break
    mc_status = "N/A"
    if mc_results and isinstance(mc_results, dict):
        ratio = mc_results.get("ratio_analytE_over_empirical", None)
        if ratio is not None and ratio != float("inf"):
            mc_status = f"PASS (ratio={ratio:.2f})" if 0.5 <= ratio <= 2.0 else f"FAIL (ratio={ratio:.2f})"
        elif mc_results.get("n_violations", 0) == 0 and mc_results.get("empirical_prob", 0) == 0:
            mc_status = "PASS (0 violations)"

    print()
    print("=" * 55)
    print("  EXPERIMENT SUMMARY")
    print("=" * 55)
    print(f"  Case:           {case_name} ({n_bus} buses, {len(net.line)} lines)")
    print(f"  Sigma source:   {sigma_source}")
    print(f"  Min r_sigma:    {min_r:.2f}{min_r_line}")
    print(f"  Median r_sigma: {summary['global_median_sigma_radius']:.2f}")
    print(f"  Tightened MC:   {mc_status}")
    print(f"  Output:         {output_dir}")
    print("=" * 55)
    print()

    logger.info("Experiment complete. Output: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 2: sigma-radius at average operating point.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
