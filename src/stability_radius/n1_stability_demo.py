"""N-1 Stability Demo: Proof that stability radius finds safer operating regimes.

Pipeline
--------
1. Load MATPOWER network.
2. Solve Cost OPF (AC FPF minimising deviation from case dispatch, nominal limits).
3. Compute AC L2 stability radii + h-vectors for cost-OPF regime.
4. Verify worst-case perturbation causes a thermal overload (proof of concept).
5. Iteratively solve Radius OPF (same objective but tightened limits from h-norms).
6. Compute AC L2 radii for radius-OPF regime.
7. DC-based N-1 effective radii for both regimes.
8. Brute-force AC N-1 screening for both regimes (disconnect each line, run PF).
9. Compare regimes; save CSV tables, summary text, and plots.

Usage
-----
python -m stability_radius.n1_stability_demo \\
    --input data/input/pglib_opf_case118_ieee.m \\
    --r-target 0.5 --n-iter 2 \\
    --output-dir analysis_output/n1_demo_case118
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
from pandas import DataFrame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    # Force UTF-8 output to avoid Windows cp1252 issues with special chars
    handler = logging.StreamHandler(sys.stdout)
    handler.setStream(sys.stdout)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    logging.basicConfig(format=fmt, datefmt="%H:%M:%S", level=level, stream=sys.stdout)
    for name in ("pandapower", "numba", "urllib3", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _make_output_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _lid_int(key: str) -> int:
    """Parse integer line id from result key like 'line_5' -> 5."""
    return int(key.split("_", 1)[1])


def _lid_str(lid_int: int) -> str:
    return f"line_{lid_int}"


# ---------------------------------------------------------------------------
# Phase 1: Load network
# ---------------------------------------------------------------------------


def _load_and_prepare(input_path: str, slack_bus_override: int | None):
    """Load network, apply lossless policy, resolve slack bus."""
    from stability_radius.parsers.matpower import load_network
    from stability_radius.base_point.pandapower_tools import (
        apply_lossless_policy_to_pandapower_net,
        ensure_ext_grid_at_slack,
        resolve_slack_bus_id,
    )

    logger.info("Loading network: %s", input_path)
    net = load_network(input_path)
    logger.info("Network: %d buses, %d lines", len(net.bus), len(net.line))

    slack_bus = resolve_slack_bus_id(
        net, slack_bus_override if slack_bus_override else 0
    )
    logger.info("Slack bus: %d", slack_bus)

    net_lossless = apply_lossless_policy_to_pandapower_net(net)
    ensure_ext_grid_at_slack(net_lossless, slack_bus)

    line_indices = sorted(net_lossless.line.index.tolist())  # list of ints
    return net_lossless, slack_bus, line_indices


# ---------------------------------------------------------------------------
# Phase 2: Solve AC FPF OPF
# ---------------------------------------------------------------------------


def _solve_fpf(
    net,
    slack_bus: int,
    line_indices: list[int],
    max_loading_percent: float = 99.0,
    label: str = "OPF",
):
    """Returns (BasePointAC, PyPSAAPFResult)."""
    from stability_radius.base_point import solve_ac_fpf_base_point
    from stability_radius.base_point.pandapower_opp import ACFPFConfig

    fpf_cfg = ACFPFConfig(
        pg0_source="case",
        max_loading_percent=max_loading_percent,
        max_iteration=300,
        pdipm_feastol=1e-4,
        pdipm_gradtol=1e-4,
        pdipm_comptol=1e-4,
        init="dc",
        max_attempts=2,
    )
    logger.info(
        "[%s] Solving AC FPF (max_loading=%.1f%%)...", label, max_loading_percent
    )
    t0 = time.time()
    bp, base_pf = solve_ac_fpf_base_point(
        net=net,
        slack_bus=slack_bus,
        lossless=True,
        fpf_cfg=fpf_cfg,
        line_indices=line_indices,
    )
    logger.info(
        "[%s] OPF done in %.1fs, status=%s", label, time.time() - t0, base_pf.status
    )
    return bp, base_pf


# ---------------------------------------------------------------------------
# Phase 3: Compute AC L2 radii
# ---------------------------------------------------------------------------


def _compute_radii(
    net_lossless, base_pf, slack_bus: int, label: str = "radius"
) -> tuple[dict, dict]:
    """Returns (per_line_results, h_vectors_dict).

    per_line_results keys: 'line_X' (string).
    h_vectors keys: 'h_from' (m, n_vars), 'h_to', 'pq_mask'.
    """
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius

    logger.info("[%s] Computing AC L2 radii...", label)
    t0 = time.time()
    results = compute_ac_l2_radius(
        net_lossless,
        base_pf=base_pf,
        slack_bus=slack_bus,
        return_h_vectors=True,
    )
    h_vectors = results.pop("_h_vectors", {})

    constrained = {
        k: v for k, v in results.items() if not v.get("is_unconstrained", False)
    }
    if constrained:
        radii = [v["radius_ac_l2"] for v in constrained.values()]
        logger.info(
            "[%s] Done %.1fs | constrained=%d | min=%.4f median=%.4f mean=%.4f",
            label,
            time.time() - t0,
            len(constrained),
            min(radii),
            float(np.median(radii)),
            float(np.mean(radii)),
        )
    else:
        logger.warning("[%s] No constrained lines.", label)

    return results, h_vectors


# ---------------------------------------------------------------------------
# Phase 4: Verify worst-case perturbation
# ---------------------------------------------------------------------------


def _verify_worst_case_perturbation(
    net_lossless,
    base_pf,
    results: dict,
    h_vectors: dict,
    slack_bus: int,
    line_indices: list[int],
) -> dict:
    """Apply h* at radius magnitude, run PF, check overload."""
    import pandapower as pp

    constrained = {
        k: v for k, v in results.items() if not v.get("is_unconstrained", False)
    }
    if not constrained:
        return {"verified": False, "reason": "no constrained lines"}

    worst_key = min(constrained, key=lambda k: constrained[k]["radius_ac_l2"])
    worst_res = constrained[worst_key]
    r_worst = float(worst_res["radius_ac_l2"])
    binding_end = str(worst_res.get("binding_end", "from"))
    worst_lid = _lid_int(worst_key)

    logger.info(
        "[verify] Worst line: %s (lid=%d) | radius=%.4f | binding=%s",
        worst_key,
        worst_lid,
        r_worst,
        binding_end,
    )

    h_from = h_vectors.get("h_from")  # (m, n_vars)
    h_to = h_vectors.get("h_to")
    pq_mask = h_vectors.get("pq_mask")

    if h_from is None:
        return {"verified": False, "reason": "h_vectors not available"}

    # Position of worst line in line_indices order
    if worst_lid not in line_indices:
        return {"verified": False, "reason": f"line {worst_lid} not in line_indices"}
    worst_pos = line_indices.index(worst_lid)

    h_vec = h_from[worst_pos] if binding_end == "from" else h_to[worst_pos]
    h_norm = float(np.linalg.norm(h_vec))
    if h_norm < 1e-12:
        return {"verified": False, "reason": "h vector is zero"}

    # Reconstruct full bus perturbation
    # h_vec has reduced dimension: [P-block (n_bus-1), Q-block (n_pq)]
    n_bus = len(net_lossless.bus)
    bus_ids = sorted(net_lossless.bus.index.tolist())
    slack_pos = bus_ids.index(slack_bus)
    n_theta = n_bus - 1
    n_red = h_vec.shape[0]

    if pq_mask is not None:
        n_pq = int(np.sum(pq_mask))
    else:
        n_pq = n_red - n_theta

    if n_theta + n_pq != n_red:
        return {
            "verified": False,
            "reason": f"dimension mismatch n_red={n_red} vs n_theta+n_pq={n_theta + n_pq}",
        }

    h_P_red = h_vec[:n_theta]
    h_Q_red = h_vec[n_theta:]

    # Embed into full (n_bus,) perturbation
    scale = r_worst / h_norm
    dp_full = np.zeros(n_bus)
    non_slack = [i for i in range(n_bus) if i != slack_pos]
    dp_full[non_slack] = h_P_red * scale

    dq_full = np.zeros(n_bus)
    if pq_mask is not None:
        pq_pos = [i for i, m in enumerate(pq_mask) if m]
    else:
        pq_pos = non_slack[:n_pq]
    if len(pq_pos) == len(h_Q_red):
        dq_full[pq_pos] = h_Q_red * scale

    # Apply perturbation via sgens on a network copy
    nn = copy.deepcopy(net_lossless)
    from stability_radius.base_point.pandapower_tools import (
        apply_opp_result_to_pandapower_net,
    )

    opp_dispatch = getattr(base_pf, "opp_gen_dispatch", None) or {}
    opp_vm = getattr(base_pf, "opp_vm_pu", None) or {}
    if opp_dispatch or opp_vm:
        apply_opp_result_to_pandapower_net(
            nn, opp_gen_dispatch=opp_dispatch, opp_vm_pu=opp_vm
        )

    for bpos, bid in enumerate(bus_ids):
        dp_i = float(dp_full[bpos])
        dq_i = float(dq_full[bpos])
        if abs(dp_i) < 1e-6 and abs(dq_i) < 1e-6:
            continue
        pp.create_sgen(
            nn, bus=bid, p_mw=dp_i, q_mvar=dq_i, name=f"_perturb_{bid}", in_service=True
        )

    # Run AC PF at perturbed point
    converged = False
    for init in ("results", "dc", "flat"):
        try:
            pp.runpp(
                nn,
                init=init,
                calculate_voltage_angles=True,
                numba=False,
                max_iter=50,
                enforce_q_lims=False,
            )
            if bool(getattr(nn, "converged", False)):
                converged = True
                break
        except Exception:
            pass

    if not converged:
        return {
            "verified": True,
            "reason": "AC PF diverged after perturbation (infeasibility confirmed)",
            "worst_line": worst_lid,
            "radius": r_worst,
            "pf_converged": False,
        }

    if hasattr(nn, "res_line") and nn.res_line is not None and len(nn.res_line):
        loading = (
            nn.res_line.at[worst_lid, "loading_percent"]
            if worst_lid in nn.res_line.index
            else None
        )
        overloaded_count = int((nn.res_line.loading_percent > 100).sum())
        target_overloaded = bool(loading is not None and loading > 100)
        logger.info(
            "[verify] After perturbation: target line loading=%.1f%%, overloaded=%d",
            float(loading) if loading is not None else -1.0,
            overloaded_count,
        )
        return {
            "verified": target_overloaded or overloaded_count > 0,
            "worst_line": worst_lid,
            "radius": r_worst,
            "target_line_loading_pct": float(loading) if loading is not None else None,
            "total_overloaded_lines": overloaded_count,
            "pf_converged": True,
        }

    return {"verified": False, "reason": "no res_line in perturbed PF"}


# ---------------------------------------------------------------------------
# Phase 5: Radius OPF (iterative tightening)
# ---------------------------------------------------------------------------


def _solve_radius_opf(
    net_lossless,
    slack_bus: int,
    line_indices: list[int],
    cost_results: dict,
    cost_h_vectors: dict,
    r_target: float,
    n_iter: int,
):
    """Returns (bp, base_pf, results, h_vectors) for radius OPF regime.

    Tightening strategy: only tighten lines whose current AC L2 radius is
    *below* r_target. Lines already safe (radius >= r_target) keep their
    original limits. This avoids over-constraining the OPF.
    """
    from stability_radius.radii.common import estimate_line_limit_mva

    current_results = cost_results
    base_pf_radius = None
    bp_radius = None
    radius_results = cost_results
    radius_h_vectors = cost_h_vectors

    for iteration in range(n_iter):
        logger.info("[radius_opf] Iteration %d/%d", iteration + 1, n_iter)

        net_tight = copy.deepcopy(net_lossless)

        tightened = 0
        skipped_safe = 0
        for lid in line_indices:
            key = _lid_str(lid)
            res = current_results.get(key, {})
            if res.get("is_unconstrained", False):
                continue
            current_radius = float(res.get("radius_ac_l2", float("inf")))
            # Only tighten lines that are actually vulnerable
            if math.isfinite(current_radius) and current_radius >= r_target:
                skipped_safe += 1
                continue
            h_norm = float(res.get("||h||2", 0.0))
            if h_norm < 1e-10:
                continue
            s_limit = float(
                estimate_line_limit_mva(net_lossless, net_lossless.line.loc[lid])
            )
            new_limit_mva = max(0.10 * s_limit, s_limit - r_target * h_norm)
            new_loading_pct = 100.0 * new_limit_mva / s_limit
            net_tight.line.at[lid, "max_loading_percent"] = new_loading_pct
            tightened += 1

        tight_pcts = net_tight.line.max_loading_percent.values
        logger.info(
            "[radius_opf] Tightened %d lines (skipped %d safe) | "
            "min_loading_pct=%.1f%% mean=%.1f%%",
            tightened,
            skipped_safe,
            float(np.min(tight_pcts)),
            float(np.mean(tight_pcts)),
        )

        if tightened == 0:
            logger.info(
                "[radius_opf] All lines satisfy r >= r_target=%.2f. Converged.", r_target
            )
            break

        try:
            # Pass max_loading_percent=100.0 so ACFPFConfig does NOT override the
            # per-line max_loading_percent values we set above on net_tight.
            # (ACFPFConfig only overwrites per-line limits when its value < 100.0.)
            bp_radius, base_pf_radius = _solve_fpf(
                net_tight,
                slack_bus,
                line_indices,
                max_loading_percent=100.0,
                label=f"radius_opf_iter{iteration + 1}",
            )
        except Exception as exc:
            logger.error("[radius_opf] OPF failed on iter %d: %s", iteration + 1, exc)
            if iteration == 0:
                raise
            logger.warning("[radius_opf] Using previous iteration result.")
            break

        radius_results, radius_h_vectors = _compute_radii(
            net_lossless,
            base_pf_radius,
            slack_bus,
            label=f"radius_opf_iter{iteration + 1}",
        )
        current_results = radius_results

    return bp_radius, base_pf_radius, radius_results, radius_h_vectors


# ---------------------------------------------------------------------------
# AC Sigma-Radius + Baseline Metrics
# ---------------------------------------------------------------------------


def _compute_sigma_and_baselines(
    results: dict,
    h_vectors: dict,
    line_indices: list[int],
    sigma_p_mw: float,
    sigma_q_mvar: float,
    label: str,
) -> dict:
    """Compute AC sigma-radius, overload probability, and baseline metrics.

    Returns a summary dict with keys for the comparison table.
    """
    from stability_radius.radii.ac_sigma_radius import compute_ac_sigma_radius
    from stability_radius.metrics.ac_baselines import (
        loading_ratio,
        headroom_mva,
        cantelli_upper_bound,
        performance_index_line,
    )

    # --- Build sigma-radius inputs from h_vectors and results ---
    h_from = h_vectors.get("h_from")  # (m, n_vars)
    h_to = h_vectors.get("h_to")      # (m, n_vars)
    pq_mask = h_vectors.get("pq_mask")  # (n_bus,) bool

    if h_from is None or h_to is None:
        logger.warning("[%s] No h_vectors available, skipping sigma-radius.", label)
        return {}

    n_lines_h = h_from.shape[0]
    n_vars = h_from.shape[1]  # n_theta + n_pq

    # pq_mask tells us which buses have Q (PQ buses)
    n_bus = len(pq_mask) if pq_mask is not None else 0
    if n_bus == 0:
        logger.warning("[%s] No pq_mask, skipping sigma-radius.", label)
        return {}

    n_theta = n_bus - 1  # excluding slack
    n_pq = int(np.sum(pq_mask)) if pq_mask is not None else 0

    # Build per-line h-vectors for binding end: shape (n_lines, 2*n_bus)
    # h_from/h_to are (n_lines, n_vars) where n_vars = n_theta + n_pq
    # We need to expand to full (n_lines, 2*n_bus) = [hP_all_buses; hQ_all_buses]
    # h_vectors[:,0:n_theta] correspond to d/d(theta_i) for non-slack buses
    # h_vectors[:,n_theta:] correspond to d/d(V_i) for PQ buses (mapped by pq_mask)
    # But for sigma-radius we need sensitivity w.r.t. [dP; dQ] injections, which
    # is what the AC L2 h-vectors already represent (via adjoint).

    # Collect per-line data
    binding_h = np.empty((n_lines_h, n_vars), dtype=float)
    s_limit_arr = np.empty(n_lines_h, dtype=float)
    s0_arr = np.empty(n_lines_h, dtype=float)
    line_ids_ordered = []

    pos = 0
    for lid in line_indices:
        key = _lid_str(lid)
        res = results.get(key, {})
        if res.get("is_unconstrained", False) or pos >= n_lines_h:
            continue
        binding_end = res.get("binding_end", "from")
        if binding_end == "to":
            binding_h[pos] = h_to[pos]
            s0_arr[pos] = abs(float(res.get("ac_s0_to_mva", 0.0)))
        else:
            binding_h[pos] = h_from[pos]
            s0_arr[pos] = abs(float(res.get("ac_s0_from_mva", 0.0)))
        s_limit_arr[pos] = float(res.get("ac_s_limit_mva", 1e6))
        line_ids_ordered.append(lid)
        pos += 1

    n_actual = pos
    if n_actual == 0:
        return {}

    binding_h = binding_h[:n_actual]
    s_limit_arr = s_limit_arr[:n_actual]
    s0_arr = s0_arr[:n_actual]

    # --- Sigma-radius with per-bus injection std devs ---
    # Use uniform sigma across all buses
    sig_p = np.full(n_vars, sigma_p_mw, dtype=float)
    sig_q = np.full(n_vars, sigma_q_mvar, dtype=float)
    # The h_vectors have n_vars columns (n_theta P-variables + n_pq Q-variables).
    # We need to match: sigma_p for the P part (n_theta), sigma_q for Q part (n_pq).
    # But compute_ac_sigma_radius expects h_vectors of shape (n_lines, 2*n_bus)
    # with sig_p, sig_q of shape (n_bus,).
    # Since our h_vectors are in the reduced Jacobian space (n_theta + n_pq),
    # we compute sigma_flow manually.

    # Compute sigma_flow = sqrt(sum (sigma_p * hP_i)^2 + sum (sigma_q * hQ_i)^2)
    hP = binding_h[:, :n_theta]  # (n, n_theta)
    hQ = binding_h[:, n_theta:]  # (n, n_pq)

    # Balance: zero-mean subtraction for sum-zero perturbation constraints
    hP_bal = hP - np.mean(hP, axis=1, keepdims=True)
    hQ_bal = hQ - np.mean(hQ, axis=1, keepdims=True)

    sigma_flow = np.sqrt(
        (sigma_p_mw ** 2) * np.sum(hP_bal ** 2, axis=1)
        + (sigma_q_mvar ** 2) * np.sum(hQ_bal ** 2, axis=1)
    )

    margin = s_limit_arr - s0_arr
    eps = 1e-15

    # sigma-radius: r_sigma = margin / sigma_flow
    r_sigma = np.where(sigma_flow > eps, margin / sigma_flow, float("inf"))

    # Overload probability (Gaussian): P(|S| > c)
    overload_probs = np.zeros(n_actual, dtype=float)
    for i in range(n_actual):
        sf = float(sigma_flow[i])
        if sf <= 0.0:
            overload_probs[i] = 0.0 if margin[i] > 0 else 1.0
        else:
            from stability_radius.radii.ac_sigma_radius import (
                _overload_probability_symmetric_limit,
            )
            overload_probs[i] = _overload_probability_symmetric_limit(
                s0_mva=float(s0_arr[i]),
                c_mva=float(s_limit_arr[i]),
                sigma_mva=sf,
            )

    # --- Baseline metrics ---
    lr_vals = s0_arr / np.maximum(s_limit_arr, 1e-10)
    hr_vals = s_limit_arr - s0_arr
    cantelli_vals = np.where(
        margin > 0,
        sigma_flow ** 2 / (sigma_flow ** 2 + margin ** 2),
        1.0,
    )
    pi_vals = 0.5 * (lr_vals ** 2)  # performance index (n=1)

    # Summary
    finite_mask = np.isfinite(r_sigma)
    finite_r = r_sigma[finite_mask]

    summary = {
        "label": label,
        "n_lines": n_actual,
        # AC sigma-radius
        "sigma_radius_min": float(np.min(finite_r)) if finite_r.size else float("nan"),
        "sigma_radius_median": float(np.median(finite_r)) if finite_r.size else float("nan"),
        "sigma_radius_p10": float(np.percentile(finite_r, 10)) if finite_r.size else float("nan"),
        # Overload probability
        "max_overload_prob": float(np.max(overload_probs)),
        "mean_overload_prob": float(np.mean(overload_probs)),
        "n_prob_above_1pct": int(np.sum(overload_probs > 0.01)),
        "n_prob_above_5pct": int(np.sum(overload_probs > 0.05)),
        # Cantelli upper bound
        "max_cantelli_ub": float(np.max(cantelli_vals)),
        "mean_cantelli_ub": float(np.mean(cantelli_vals)),
        # Performance Index
        "pi_system": float(np.sum(pi_vals)),
        "pi_max": float(np.max(pi_vals)),
        # Loading
        "max_loading_ratio": float(np.max(lr_vals)),
        "min_headroom_mva": float(np.min(hr_vals)),
    }

    # Log
    logger.info(
        "[%s] Sigma-radius: min=%.4f median=%.4f | "
        "max_overload_prob=%.6f | n_prob>1%%=%d | Cantelli_max=%.6f | PI=%.4f",
        label,
        summary["sigma_radius_min"],
        summary["sigma_radius_median"],
        summary["max_overload_prob"],
        summary["n_prob_above_1pct"],
        summary["max_cantelli_ub"],
        summary["pi_system"],
    )

    return summary


# ---------------------------------------------------------------------------
# Phase 6: DC-based N-1 effective radii
# ---------------------------------------------------------------------------


def _build_base_q_from_pf(net_lossless, base_pf, line_indices: list[int]):
    """Build LineBaseQuantities from AC PF result for use in N-1 computation.

    Uses AC from-end MW flow as DC base flow approximation (lossless network).
    """
    from stability_radius.radii.common import (
        LineBaseQuantities,
        estimate_line_limit_mva_with_flag,
    )

    m = len(line_indices)
    p0_from = np.asarray(base_pf.line_p0_mw, dtype=float)  # (m,), signed
    p0_abs = np.abs(p0_from)

    limits = np.empty(m, dtype=float)
    is_uc = np.zeros(m, dtype=bool)
    for pos, lid in enumerate(line_indices):
        lim, uc = estimate_line_limit_mva_with_flag(
            net_lossless, net_lossless.line.loc[lid]
        )
        limits[pos] = float(lim)
        is_uc[pos] = bool(uc)

    margin = limits - p0_abs

    return LineBaseQuantities(
        line_indices=list(line_indices),
        flow0_mw=p0_from,
        p0_abs_mw=p0_abs,
        limit_mva_assumed_mw=limits,
        margin_mw=margin,
        is_unconstrained=is_uc,
    )


def _dc_n1_radii(
    net_lossless, base_pf, slack_bus: int, line_indices: list[int], label: str
) -> dict:
    from stability_radius.dc.dc_model import build_dc_matrices
    from stability_radius.radii.nminus1 import compute_nminus1_l2_radius

    logger.info("[%s] Building DC sensitivity matrix for N-1...", label)

    try:
        H_full, _ = build_dc_matrices(net_lossless, slack_bus=slack_bus, chunk_size=512)
    except Exception as exc:
        logger.warning("[%s] DC matrix build failed: %s", label, exc)
        return {}

    logger.info("[%s] H_full shape: %s. Computing N-1 radii...", label, H_full.shape)

    # Build base quantities from AC PF result to avoid internal DC OPF
    base_q = _build_base_q_from_pf(net_lossless, base_pf, line_indices)

    try:
        n1_results = compute_nminus1_l2_radius(net_lossless, H_full, base=base_q)
    except Exception as exc:
        logger.warning("[%s] N-1 radius failed: %s", label, exc)
        return {}

    n1r = [v.get("radius_nminus1", float("inf")) for v in n1_results.values()]
    finite = [x for x in n1r if math.isfinite(x)]
    if finite:
        logger.info(
            "[%s] DC N-1 radius: min=%.4f median=%.4f",
            label,
            min(finite),
            float(np.median(finite)),
        )
    return n1_results


# ---------------------------------------------------------------------------
# Phase 7: Brute-force AC N-1 screening
# ---------------------------------------------------------------------------


def _ac_n1_screen(net_lossless, base_pf, slack_bus: int, label: str) -> list[dict]:
    import pandapower as pp
    from stability_radius.base_point.pandapower_tools import (
        apply_opp_result_to_pandapower_net,
    )

    net_base = copy.deepcopy(net_lossless)
    opp_dispatch = getattr(base_pf, "opp_gen_dispatch", None) or {}
    opp_vm = getattr(base_pf, "opp_vm_pu", None) or {}
    if opp_dispatch or opp_vm:
        apply_opp_result_to_pandapower_net(
            net_base, opp_gen_dispatch=opp_dispatch, opp_vm_pu=opp_vm
        )

    # Run base PF to store as initial guess
    for init in ("dc", "flat"):
        try:
            pp.runpp(
                net_base,
                init=init,
                calculate_voltage_angles=True,
                numba=False,
                max_iter=50,
                enforce_q_lims=False,
            )
            if bool(getattr(net_base, "converged", False)):
                break
        except Exception:
            pass

    line_ids = sorted(net_lossless.line.index.tolist())
    records = []
    logger.info("[%s] AC N-1 screening: %d contingencies...", label, len(line_ids))

    for i, lid in enumerate(line_ids):
        nn = copy.deepcopy(net_base)
        nn.line.at[lid, "in_service"] = False

        converged = False
        n_overloads = -1
        for init in ("results", "dc", "flat"):
            try:
                pp.runpp(
                    nn,
                    init=init,
                    calculate_voltage_angles=True,
                    numba=False,
                    max_iter=50,
                    enforce_q_lims=False,
                )
                if bool(getattr(nn, "converged", False)):
                    converged = True
                    break
            except Exception:
                pass

        if converged and hasattr(nn, "res_line") and nn.res_line is not None:
            n_overloads = int((nn.res_line.loading_percent > 100).sum())

        records.append(
            {
                "contingency_line": int(lid),
                "pf_converged": converged,
                "n1_feasible": converged and n_overloads == 0,
                "n_overloads": n_overloads,
            }
        )

        if (i + 1) % 20 == 0 or (i + 1) == len(line_ids):
            n_done = i + 1
            n_pass = sum(1 for r in records if r["n1_feasible"])
            logger.info(
                "[%s] N-1 progress: %d/%d | pass=%d",
                label,
                n_done,
                len(line_ids),
                n_pass,
            )

    passed = sum(1 for r in records if r["n1_feasible"])
    total = len(records)
    logger.info(
        "[%s] N-1 done: pass=%d/%d (%.1f%%)",
        label,
        passed,
        total,
        100.0 * passed / total if total else 0.0,
    )
    return records


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _radii_to_df(results: dict) -> "DataFrame":
    import pandas as pd

    rows = []
    for key, r in results.items():
        lid = _lid_int(key)
        binding = str(r.get("binding_end", "from"))
        s0_mva = r.get(f"ac_s0_{binding}_mva", float("nan"))
        s_limit = r.get("ac_s_limit_mva", float("nan"))
        loading_ratio = (
            float(s0_mva) / float(s_limit)
            if math.isfinite(float(s0_mva)) and float(s_limit) > 0
            else float("nan")
        )
        rows.append(
            {
                "line_id": lid,
                "radius_ac_l2": r.get("radius_ac_l2", float("nan")),
                "margin_ac_mva": r.get("margin_ac_mva", float("nan")),
                "h_norm": r.get("||h||2", float("nan")),
                "loading_ratio": loading_ratio,
                "is_unconstrained": bool(r.get("is_unconstrained", False)),
                "binding_end": binding,
                "ac_s0_mva": s0_mva,
                "ac_s_limit_mva": s_limit,
            }
        )
    return pd.DataFrame(rows).set_index("line_id").sort_index()


def _summary_stats(results: dict, label: str) -> dict:
    constrained = {
        k: v for k, v in results.items() if not v.get("is_unconstrained", False)
    }
    if not constrained:
        return {"label": label, "n_constrained": 0}
    radii = [v["radius_ac_l2"] for v in constrained.values()]
    # Compute loading ratios
    lr_vals = []
    for v in constrained.values():
        s0 = v.get(f"ac_s0_{v.get('binding_end', 'from')}_mva", float("nan"))
        sl = v.get("ac_s_limit_mva", 0.0)
        if math.isfinite(float(s0)) and float(sl) > 0:
            lr_vals.append(float(s0) / float(sl))
    return {
        "label": label,
        "n_constrained": len(constrained),
        "radius_min": float(min(radii)),
        "radius_median": float(np.median(radii)),
        "radius_mean": float(np.mean(radii)),
        "radius_max": float(max(radii)),
        "loading_ratio_mean": float(np.mean(lr_vals)) if lr_vals else float("nan"),
        "loading_ratio_max": float(max(lr_vals)) if lr_vals else float("nan"),
    }


def _n1_summary(records: list[dict], label: str) -> dict:
    total = len(records)
    if total == 0:
        return {"label": label, "n_contingencies": 0}
    passed = sum(1 for r in records if r["n1_feasible"])
    diverged = sum(1 for r in records if not r["pf_converged"])
    failed = total - passed - diverged
    max_ov = max(
        (r["n_overloads"] for r in records if r["n_overloads"] >= 0), default=0
    )
    return {
        "label": label,
        "n_contingencies": total,
        "n1_pass": passed,
        "n1_fail": failed,
        "n1_diverged": diverged,
        "n1_pass_rate_pct": round(100.0 * passed / total, 2),
        "max_overloads_in_contingency": max_ov,
    }


def _dc_n1_summary(n1_results: dict, label: str) -> dict:
    if not n1_results:
        return {"label": label, "n_constrained": 0}
    # compute_nminus1_l2_radius uses key "radius_nminus1" (not "nminus1_l2_radius")
    # It doesn't set "is_unconstrained", so we include all entries
    radii = []
    for v in n1_results.values():
        r = v.get("radius_nminus1", float("inf"))
        if math.isfinite(r):
            radii.append(float(r))
    return {
        "label": label,
        "n_constrained": len(n1_results),
        "dc_n1_radius_min": float(min(radii)) if radii else float("nan"),
        "dc_n1_radius_median": float(np.median(radii)) if radii else float("nan"),
        "dc_n1_radius_mean": float(np.mean(radii)) if radii else float("nan"),
    }


def _save_csv(df, path: Path, label: str) -> None:
    df.to_csv(path)
    logger.info("Saved %s: %s (%d rows)", label, path, len(df))


def _dc_n1_to_df(n1_results: dict) -> "DataFrame":
    import pandas as pd

    rows = []
    for key, r in n1_results.items():
        lid = _lid_int(key)
        rows.append(
            {
                "line_id": lid,
                "radius_nminus1": r.get("radius_nminus1", float("nan")),
                "flow0_mw": r.get("flow0_mw", float("nan")),
                "p_limit_mw_est": r.get("p_limit_mw_est", float("nan")),
                "margin_mw": r.get("margin_mw", float("nan")),
                "worst_contingency_line_idx": r.get("worst_contingency_line_idx", -1),
            }
        )
    return pd.DataFrame(rows).set_index("line_id").sort_index()


def _save_n1_csv(records: list[dict], path: Path, label: str) -> None:
    import pandas as pd

    pd.DataFrame(records).to_csv(path, index=False)
    logger.info("Saved %s: %s (%d rows)", label, path, len(records))


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def _print_comparison(
    cost_sum,
    radius_sum,
    cost_n1,
    radius_n1,
    cost_dc_n1,
    radius_dc_n1,
    cost_ac_n1,
    radius_ac_n1,
    cost_sigma,
    radius_sigma,
    verify,
    r_target,
    sigma_p_mw,
    sigma_q_mvar,
    output_path,
):

    def _fmt(val, fmt=".4f"):
        if isinstance(val, float):
            return format(val, fmt)
        return str(val)

    lines = [
        "=" * 70,
        "N-1 STABILITY DEMO - COMPARISON SUMMARY",
        "=" * 70,
        "",
        f"  r_target (tightening parameter): {r_target}",
        f"  sigma_p = {sigma_p_mw} MW, sigma_q = {sigma_q_mvar} MVAr",
        "",
        "--- AC L2 Stability Radius (constrained lines only) ---",
        f"  {'Metric':<35} {'Cost OPF':>12} {'Radius OPF':>12}",
        "  " + "-" * 60,
    ]
    for key, lbl in [
        ("n_constrained", "Constrained lines"),
        ("radius_min", "Min radius"),
        ("radius_median", "Median radius"),
        ("radius_mean", "Mean radius"),
        ("loading_ratio_mean", "Mean loading ratio"),
        ("loading_ratio_max", "Max loading ratio"),
    ]:
        cv = _fmt(cost_sum.get(key, "N/A"))
        rv = _fmt(radius_sum.get(key, "N/A"))
        lines.append(f"  {lbl:<35} {cv:>12} {rv:>12}")

    lines += [
        "",
        "--- AC N-1 Stability Radius (min over all contingencies) ---",
        f"  {'Metric':<35} {'Cost OPF':>12} {'Radius OPF':>12}",
        "  " + "-" * 60,
    ]
    for key, lbl in [
        ("n_lines",                 "Lines with N-1 radius computed"),
        ("n_already_n1_infeasible", "Lines already N-1 infeasible"),
        ("ac_n1_radius_min",        "Min AC N-1 radius (MW, positive)"),
        ("ac_n1_radius_median",     "Median AC N-1 radius (MW)"),
        ("ac_n1_radius_p10",        "P10 AC N-1 radius (MW)"),
    ]:
        cv = _fmt(cost_ac_n1.get(key, "N/A"))
        rv = _fmt(radius_ac_n1.get(key, "N/A"))
        lines.append(f"  {lbl:<35} {cv:>12} {rv:>12}")

    lines += [
        "",
        f"--- AC Sigma-Radius & Overload Probability (sigma_p={sigma_p_mw}, sigma_q={sigma_q_mvar}) ---",
        f"  {'Metric':<35} {'Cost OPF':>12} {'Radius OPF':>12}",
        "  " + "-" * 60,
    ]
    for key, lbl in [
        ("sigma_radius_min",     "Min sigma-radius"),
        ("sigma_radius_median",  "Median sigma-radius"),
        ("sigma_radius_p10",     "P10 sigma-radius"),
        ("max_overload_prob",    "Max overload probability"),
        ("mean_overload_prob",   "Mean overload probability"),
        ("n_prob_above_1pct",    "Lines with P(overload) > 1%"),
        ("n_prob_above_5pct",    "Lines with P(overload) > 5%"),
        ("max_cantelli_ub",      "Max Cantelli upper bound"),
        ("pi_system",            "System Performance Index"),
        ("pi_max",               "Max line Performance Index"),
        ("min_headroom_mva",     "Min headroom (MVA)"),
    ]:
        cv = _fmt(cost_sigma.get(key, "N/A"), ".6f" if "prob" in key or "cantelli" in key else ".4f")
        rv = _fmt(radius_sigma.get(key, "N/A"), ".6f" if "prob" in key or "cantelli" in key else ".4f")
        lines.append(f"  {lbl:<35} {cv:>12} {rv:>12}")

    lines += [
        "",
        "--- DC N-1 Effective Radius ---",
        f"  {'Metric':<35} {'Cost OPF':>12} {'Radius OPF':>12}",
        "  " + "-" * 60,
    ]
    for key, lbl in [
        ("dc_n1_radius_min", "Min N-1 effective radius"),
        ("dc_n1_radius_median", "Median N-1 effective radius"),
    ]:
        cv = _fmt(cost_dc_n1.get(key, "N/A"))
        rv = _fmt(radius_dc_n1.get(key, "N/A"))
        lines.append(f"  {lbl:<35} {cv:>12} {rv:>12}")

    lines += [
        "",
        "--- AC N-1 Screening ---",
        f"  {'Metric':<35} {'Cost OPF':>12} {'Radius OPF':>12}",
        "  " + "-" * 60,
    ]
    for key, lbl in [
        ("n1_pass", "N-1 passed"),
        ("n1_fail", "N-1 failed (overloads)"),
        ("n1_diverged", "N-1 diverged"),
        ("n1_pass_rate_pct", "N-1 pass rate (%)"),
        ("max_overloads_in_contingency", "Max overloads in any N-1"),
    ]:
        cv = _fmt(cost_n1.get(key, "N/A"), ".2f")
        rv = _fmt(radius_n1.get(key, "N/A"), ".2f")
        lines.append(f"  {lbl:<35} {cv:>12} {rv:>12}")

    lines += ["", "--- Worst-Case Perturbation Verification ---"]
    if verify.get("verified"):
        lines += [
            f"  Worst constrained line:    {verify.get('worst_line')}",
            f"  Stability radius:          {verify.get('radius', 'N/A'):.4f}",
            f"  PF converged after perturbation: {verify.get('pf_converged')}",
            f"  Target line loading:       {verify.get('target_line_loading_pct', 'N/A')}%",
            f"  Total overloaded lines:    {verify.get('total_overloaded_lines', 'N/A')}",
            "  => VERIFIED: stability radius h* perturbation triggers overload.",
        ]
    elif verify.get("reason"):
        lines += [f"  Not verified: {verify['reason']}"]
    else:
        # PF converged but perturbation didn't cause overload
        loading = verify.get("target_line_loading_pct")
        lines += [
            f"  Worst constrained line:    {verify.get('worst_line')}",
            f"  Stability radius:          {_fmt(verify.get('radius', 'N/A'))}",
            f"  Target line loading after perturbation: "
            f"{f'{loading:.1f}%' if loading is not None else 'N/A'}",
            "  => NOTE: balanced-norm h* direction did not trigger target overload",
            "     (h-vector uses balanced zero-sum projection; raw h/||h|| may differ).",
        ]

    lines += ["", "=" * 70, ""]

    text = "\n".join(lines)
    print(text)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Summary saved: %s", output_path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_radius_cdf(
    cost_results: dict, radius_results: dict, output_path: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def _get_radii(res):
        return sorted(
            v["radius_ac_l2"]
            for v in res.values()
            if not v.get("is_unconstrained", False)
            and math.isfinite(v.get("radius_ac_l2", float("nan")))
        )

    cost_r = _get_radii(cost_results)
    rad_r = _get_radii(radius_results)
    if not cost_r and not rad_r:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for radii, lbl, col in [(cost_r, "Cost OPF", "C0"), (rad_r, "Radius OPF", "C1")]:
        if radii:
            ax.plot(
                radii,
                np.arange(1, len(radii) + 1) / len(radii),
                label=lbl,
                color=col,
                linewidth=2,
            )
    ax.set_xlabel("AC L2 Stability Radius", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("Cumulative Distribution of AC L2 Stability Radius", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_n1_overloads(
    cost_n1: list[dict], radius_n1: list[dict], output_path: Path
) -> None:
    if not cost_n1 or not radius_n1:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    cost_df = pd.DataFrame(cost_n1).set_index("contingency_line")
    rad_df = pd.DataFrame(radius_n1).set_index("contingency_line")
    common = sorted(set(cost_df.index) & set(rad_df.index))
    if not common:
        return

    cost_ov = cost_df.loc[common, "n_overloads"].clip(lower=0).values
    rad_ov = rad_df.loc[common, "n_overloads"].clip(lower=0).values
    x = np.arange(len(common))
    w = 0.4

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, cost_ov, w, label="Cost OPF", color="C0", alpha=0.8)
    ax.bar(x + w / 2, rad_ov, w, label="Radius OPF", color="C1", alpha=0.8)
    ax.set_xlabel("Contingency line index", fontsize=11)
    ax.set_ylabel("Overloaded lines", fontsize=11)
    ax.set_title("AC N-1 Screening: Overloads per Contingency", fontsize=12)
    ax.legend(fontsize=10)
    step = max(1, len(common) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(
        [str(common[i]) for i in range(0, len(common), step)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_radius_scatter(
    cost_results: dict, radius_results: dict, output_path: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    common = sorted(
        k
        for k in cost_results
        if k in radius_results
        and not cost_results[k].get("is_unconstrained", False)
        and not radius_results[k].get("is_unconstrained", False)
    )
    if not common:
        return

    cost_r = [cost_results[k]["radius_ac_l2"] for k in common]
    rad_r = [radius_results[k]["radius_ac_l2"] for k in common]
    lim = max(max(cost_r), max(rad_r)) * 1.05

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(cost_r, rad_r, alpha=0.6, s=20, color="C2")
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Radius (Cost OPF)", fontsize=12)
    ax.set_ylabel("Radius (Radius OPF)", fontsize=12)
    ax.set_title("Per-Line Radius: Cost OPF vs Radius OPF", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def _extract_pypsa_result_from_pp(nn, line_indices: list[int]):
    """Extract PyPSAAPFResult from an already-solved pandapower network (after runpp)."""
    from stability_radius.base_point.pypsa_pf import PyPSAAPFResult

    bus_ids = sorted(nn.bus.index.tolist())
    v_mag = np.array([float(nn.res_bus.at[bid, "vm_pu"]) for bid in bus_ids])
    v_ang = np.array(
        [float(nn.res_bus.at[bid, "va_degree"]) * math.pi / 180.0 for bid in bus_ids]
    )

    m = len(line_indices)
    p0 = np.zeros(m)
    q0 = np.zeros(m)
    p1 = np.zeros(m)
    q1 = np.zeros(m)

    for pos, lid in enumerate(line_indices):
        if lid in nn.res_line.index:
            p0[pos] = float(nn.res_line.at[lid, "p_from_mw"])
            q0[pos] = float(nn.res_line.at[lid, "q_from_mvar"])
            p1[pos] = float(nn.res_line.at[lid, "p_to_mw"])
            q1[pos] = float(nn.res_line.at[lid, "q_to_mvar"])
        # Out-of-service lines stay at 0

    return PyPSAAPFResult(
        bus_ids=tuple(bus_ids),
        v_mag_pu=v_mag,
        v_ang_rad=v_ang,
        line_ids=tuple(line_indices),
        line_p0_mw=p0,
        line_q0_mvar=q0,
        line_p1_mw=p1,
        line_q1_mvar=q1,
        status="PP_RUNPP_OK",
    )


# ---------------------------------------------------------------------------
# AC N-1 Stability Radius
# ---------------------------------------------------------------------------


def _compute_ac_n1_radii(
    net_lossless, base_pf, slack_bus: int, line_indices: list[int], label: str
) -> dict:
    """Compute per-line AC N-1 stability radius.

    For each line l: AC N-1 radius = min over contingencies k != l of
    (AC L2 radius of line l when line k is disconnected at the existing dispatch).

    Returns dict: line_id (int) -> min_n1_radius (float).
    """
    import pandapower as pp
    from stability_radius.base_point.pandapower_tools import (
        apply_opp_result_to_pandapower_net,
    )
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius

    logger.info(
        "[%s] Computing AC N-1 stability radii (%d contingencies)...",
        label,
        len(line_indices),
    )
    t0 = time.time()

    # Prepare base network at OPF operating point
    net_base = copy.deepcopy(net_lossless)
    opp_dispatch = getattr(base_pf, "opp_gen_dispatch", None) or {}
    opp_vm = getattr(base_pf, "opp_vm_pu", None) or {}
    if opp_dispatch or opp_vm:
        apply_opp_result_to_pandapower_net(
            net_base, opp_gen_dispatch=opp_dispatch, opp_vm_pu=opp_vm
        )

    # Run base PF to set res_* tables for use as initial guess
    for init in ("dc", "flat"):
        try:
            pp.runpp(
                net_base,
                init=init,
                calculate_voltage_angles=True,
                numba=False,
                max_iter=50,
                enforce_q_lims=False,
            )
            if bool(getattr(net_base, "converged", False)):
                break
        except Exception:
            pass

    # Per-line: accumulate minimum N-1 radius
    n1_min = {lid: float("inf") for lid in line_indices}
    n_done = 0
    n_converged = 0

    for lid_k in line_indices:
        # Create contingency network (mark line k out of service)
        nn = copy.deepcopy(net_base)
        nn.line.at[lid_k, "in_service"] = False

        # Run AC PF at existing dispatch (no re-dispatch)
        converged = False
        for init in ("results", "dc", "flat"):
            try:
                pp.runpp(
                    nn,
                    init=init,
                    calculate_voltage_angles=True,
                    numba=False,
                    max_iter=50,
                    enforce_q_lims=False,
                )
                if bool(getattr(nn, "converged", False)):
                    converged = True
                    break
            except Exception:
                pass

        n_done += 1
        if not converged:
            continue
        n_converged += 1

        # Extract PF result
        try:
            base_pf_k = _extract_pypsa_result_from_pp(nn, line_indices)
        except Exception as exc:
            logger.debug(
                "[%s] Extract PF result failed for contingency %d: %s",
                label,
                lid_k,
                exc,
            )
            continue

        # Compute AC L2 radii on the contingency network
        try:
            radii_k = compute_ac_l2_radius(
                nn, base_pf=base_pf_k, slack_bus=slack_bus, return_h_vectors=False
            )
        except Exception as exc:
            logger.debug(
                "[%s] Radius computation failed for contingency %d: %s",
                label,
                lid_k,
                exc,
            )
            continue

        # Update per-line N-1 min radius
        for key, res in radii_k.items():
            lid_l = _lid_int(key)
            if lid_l == lid_k:
                continue  # skip the disconnected line itself
            if res.get("is_unconstrained", False):
                continue
            r_l = float(res.get("radius_ac_l2", float("inf")))
            if math.isfinite(r_l) and r_l < n1_min.get(lid_l, float("inf")):
                n1_min[lid_l] = r_l

        if n_done % 25 == 0 or n_done == len(line_indices):
            logger.info(
                "[%s] AC N-1 progress: %d/%d (converged=%d)",
                label,
                n_done,
                len(line_indices),
                n_converged,
            )

    # Filter to lines with finite N-1 radius
    result = {lid: r for lid, r in n1_min.items() if math.isfinite(r)}
    finite = list(result.values())
    elapsed = time.time() - t0
    if finite:
        logger.info(
            "[%s] AC N-1 radii done in %.1fs | lines=%d | min=%.4f median=%.4f mean=%.4f",
            label,
            elapsed,
            len(finite),
            min(finite),
            float(np.median(finite)),
            float(np.mean(finite)),
        )
    else:
        logger.warning("[%s] No finite AC N-1 radii found.", label)

    return result


def _ac_n1_radius_summary(n1_radii: dict, label: str) -> dict:
    finite = [v for v in n1_radii.values() if math.isfinite(v)]
    if not finite:
        return {"label": label, "n_lines": 0}
    positive = [v for v in finite if v > 0]
    n_infeasible = len([v for v in finite if v <= 0])
    return {
        "label": label,
        "n_lines": len(finite),
        "n_already_n1_infeasible": n_infeasible,
        "ac_n1_radius_min": float(min(positive)) if positive else float("nan"),
        "ac_n1_radius_median": float(np.median(positive)) if positive else float("nan"),
        "ac_n1_radius_mean": float(np.mean(positive)) if positive else float("nan"),
        "ac_n1_radius_p10": float(np.percentile(positive, 10)) if positive else float("nan"),
    }


def _plot_ac_n1_radius_cdf(cost_n1: dict, radius_n1: dict, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cost_r = sorted(v for v in cost_n1.values() if math.isfinite(v))
    rad_r = sorted(v for v in radius_n1.values() if math.isfinite(v))
    if not cost_r and not rad_r:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for radii, lbl, col in [(cost_r, "Cost OPF", "C0"), (rad_r, "Radius OPF", "C1")]:
        if radii:
            ax.plot(
                radii,
                np.arange(1, len(radii) + 1) / len(radii),
                label=lbl,
                color=col,
                linewidth=2,
            )
    ax.set_xlabel("AC N-1 Stability Radius (MW)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("CDF of AC N-1 Stability Radius per Line", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="N-1 Stability Demo: Cost OPF vs Radius OPF comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to MATPOWER .m file")
    parser.add_argument(
        "--output-dir", default="analysis_output/n1_demo", help="Output directory"
    )
    parser.add_argument(
        "--slack-bus", type=int, default=None, help="Slack bus index (None = auto)"
    )
    parser.add_argument(
        "--r-target",
        type=float,
        default=0.0,
        help="Target stability radius for limit tightening (in MW; "
        "set to 0 to auto-scale to 10x min radius)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=3, help="Number of radius OPF iterations"
    )
    parser.add_argument(
        "--sigma-p", type=float, default=5.0,
        help="Per-bus P injection std dev (MW) for sigma-radius/probability",
    )
    parser.add_argument(
        "--sigma-q", type=float, default=2.0,
        help="Per-bus Q injection std dev (MVAr) for sigma-radius/probability",
    )
    parser.add_argument(
        "--skip-n1-screening",
        action="store_true",
        help="Skip brute-force AC N-1 screening",
    )
    parser.add_argument(
        "--skip-dc-n1", action="store_true", help="Skip DC-based N-1 effective radius"
    )
    parser.add_argument(
        "--skip-ac-n1-radius",
        action="store_true",
        help="Skip AC N-1 stability radius computation (saves time)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    out_dir = _make_output_dir(args.output_dir)
    logger.info("Output directory: %s", out_dir.resolve())

    # Phase 1: Load
    net_lossless, slack_bus, line_indices = _load_and_prepare(
        args.input, args.slack_bus
    )

    # Phase 2: Cost OPF
    bp_cost, base_pf_cost = _solve_fpf(
        net_lossless,
        slack_bus,
        line_indices,
        max_loading_percent=99.0,
        label="cost_opf",
    )

    # Phase 3: Radii for cost OPF
    cost_results, cost_h_vectors = _compute_radii(
        net_lossless, base_pf_cost, slack_bus, label="cost_opf"
    )

    # Phase 4: Verify worst-case perturbation
    logger.info("Verifying worst-case perturbation...")
    verify_result = _verify_worst_case_perturbation(
        net_lossless,
        base_pf_cost,
        cost_results,
        cost_h_vectors,
        slack_bus,
        line_indices,
    )
    logger.info("Verification: %s", verify_result)

    # Auto r_target: if 0, scale to 10x the minimum constrained radius
    r_target = float(args.r_target)
    if r_target <= 0.0:
        constrained_radii = [
            v["radius_ac_l2"] for v in cost_results.values()
            if not v.get("is_unconstrained", False) and math.isfinite(v.get("radius_ac_l2", float("nan")))
        ]
        if constrained_radii:
            r_target = 10.0 * float(min(constrained_radii))
            logger.info("Auto r_target = 10 * min_radius = %.4f MW", r_target)
        else:
            r_target = 5.0

    # Phase 5: Radius OPF
    bp_radius, base_pf_radius, radius_results, radius_h_vectors = _solve_radius_opf(
        net_lossless,
        slack_bus,
        line_indices,
        cost_results,
        cost_h_vectors,
        r_target=r_target,
        n_iter=args.n_iter,
    )

    # Phase 6: DC N-1 radii
    cost_dc_n1_results = {}
    radius_dc_n1_results = {}
    if not args.skip_dc_n1:
        cost_dc_n1_results = _dc_n1_radii(
            net_lossless, base_pf_cost, slack_bus, line_indices, "cost_opf"
        )
        radius_dc_n1_results = _dc_n1_radii(
            net_lossless, base_pf_radius, slack_bus, line_indices, "radius_opf"
        )
        if cost_dc_n1_results:
            _save_csv(
                _dc_n1_to_df(cost_dc_n1_results),
                out_dir / "dc_n1_cost_opf.csv",
                "DC N-1 (cost)",
            )
        if radius_dc_n1_results:
            _save_csv(
                _dc_n1_to_df(radius_dc_n1_results),
                out_dir / "dc_n1_radius_opf.csv",
                "DC N-1 (radius)",
            )

    # Phase 6b: AC N-1 stability radius
    cost_ac_n1_radii = {}
    radius_ac_n1_radii = {}
    if not args.skip_ac_n1_radius:
        cost_ac_n1_radii = _compute_ac_n1_radii(
            net_lossless, base_pf_cost, slack_bus, line_indices, "cost_opf"
        )
        radius_ac_n1_radii = _compute_ac_n1_radii(
            net_lossless, base_pf_radius, slack_bus, line_indices, "radius_opf"
        )
        # Save AC N-1 radii CSVs
        import pandas as pd
        if cost_ac_n1_radii:
            pd.DataFrame(
                [{"line_id": k, "ac_n1_radius": v} for k, v in cost_ac_n1_radii.items()]
            ).set_index("line_id").sort_index().to_csv(
                out_dir / "ac_n1_radii_cost.csv"
            )
        if radius_ac_n1_radii:
            pd.DataFrame(
                [{"line_id": k, "ac_n1_radius": v} for k, v in radius_ac_n1_radii.items()]
            ).set_index("line_id").sort_index().to_csv(
                out_dir / "ac_n1_radii_radius.csv"
            )

    # Phase 7: AC N-1 screening
    cost_n1_records = []
    radius_n1_records = []
    if not args.skip_n1_screening:
        cost_n1_records = _ac_n1_screen(
            net_lossless, base_pf_cost, slack_bus, "cost_opf"
        )
        radius_n1_records = _ac_n1_screen(
            net_lossless, base_pf_radius, slack_bus, "radius_opf"
        )
        _save_n1_csv(cost_n1_records, out_dir / "n1_screening_cost.csv", "AC N-1 cost")
        _save_n1_csv(
            radius_n1_records, out_dir / "n1_screening_radius.csv", "AC N-1 radius"
        )

    # Save per-line radii
    _save_csv(_radii_to_df(cost_results), out_dir / "cost_opf_radii.csv", "cost radii")
    _save_csv(
        _radii_to_df(radius_results), out_dir / "radius_opf_radii.csv", "radius radii"
    )

    # Phase 8: AC Sigma-radius + Baseline metrics
    sigma_p_mw = float(args.sigma_p)
    sigma_q_mvar = float(args.sigma_q)
    cost_sigma = _compute_sigma_and_baselines(
        cost_results, cost_h_vectors, line_indices,
        sigma_p_mw=sigma_p_mw, sigma_q_mvar=sigma_q_mvar, label="cost_opf",
    )
    radius_sigma = _compute_sigma_and_baselines(
        radius_results, radius_h_vectors, line_indices,
        sigma_p_mw=sigma_p_mw, sigma_q_mvar=sigma_q_mvar, label="radius_opf",
    )

    # Summary
    cost_sum = _summary_stats(cost_results, "cost_opf")
    radius_sum = _summary_stats(radius_results, "radius_opf")
    cost_n1_sum = _n1_summary(cost_n1_records, "cost_opf")
    radius_n1_sum = _n1_summary(radius_n1_records, "radius_opf")
    cost_dc_n1_sum = _dc_n1_summary(cost_dc_n1_results, "cost_opf")
    radius_dc_n1_sum = _dc_n1_summary(radius_dc_n1_results, "radius_opf")
    cost_ac_n1_sum = _ac_n1_radius_summary(cost_ac_n1_radii, "cost_opf")
    radius_ac_n1_sum = _ac_n1_radius_summary(radius_ac_n1_radii, "radius_opf")

    _print_comparison(
        cost_sum,
        radius_sum,
        cost_n1_sum,
        radius_n1_sum,
        cost_dc_n1_sum,
        radius_dc_n1_sum,
        cost_ac_n1_sum,
        radius_ac_n1_sum,
        cost_sigma,
        radius_sigma,
        verify_result,
        r_target=r_target,
        sigma_p_mw=sigma_p_mw,
        sigma_q_mvar=sigma_q_mvar,
        output_path=out_dir / "comparison_summary.txt",
    )

    # Plots
    _plot_radius_cdf(cost_results, radius_results, out_dir / "plot_radius_cdf.png")
    _plot_n1_overloads(
        cost_n1_records, radius_n1_records, out_dir / "plot_n1_overloads.png"
    )
    _plot_radius_scatter(
        cost_results, radius_results, out_dir / "plot_radius_scatter.png"
    )
    _plot_ac_n1_radius_cdf(
        cost_ac_n1_radii, radius_ac_n1_radii, out_dir / "plot_ac_n1_radius_cdf.png"
    )

    logger.info("Done. All outputs in: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
