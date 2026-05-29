"""N-1 Stability Demo: Proof that stability radius finds safer operating regimes.

Pipeline
--------
1. Load MATPOWER network.
2. Solve validated cost-minimising AC OPF.
3. Compute AC L2 stability radii + h-vectors for cost-OPF regime.
4. Verify worst-case perturbation causes a thermal overload (proof of concept).
5. Iteratively solve Radius OPF (same objective but tightened limits from h-norms).
6. Compute AC L2 radii for radius-OPF regime.
7. Solve a screening-based SCOPF proxy and compare all three regimes.
8. DC/AC N-1 post-processing and brute-force AC N-1 screening.
9. Save CSV tables, summary text, plots, and `debug.log` under `run_artifacts/`.

Usage
-----
python entry_points/n1_stability_demo.py \\
    --input data/input/pglib_opf_case118_ieee.m \\
    --r-target 0.5 --n-iter 2 --scopf-iter 2 \\
    --output-dir n1_demo_case118
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from pandas import DataFrame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool, *, log_file: Path | None = None) -> None:
    """Internal helper for module-local processing."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    logging.basicConfig(
        format=fmt,
        datefmt="%H:%M:%S",
        level=level,
        handlers=handlers,
        force=True,
    )
    for name in ("pandapower", "numba", "urllib3", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _resolve_output_dir(requested_output_dir: str | None) -> Path:
    """Internal helper for module-local processing."""
    from stability_radius.utils import create_module_output_dir

    return create_module_output_dir(
        module_name="n1_stability_demo",
        requested_output_dir=requested_output_dir,
    )


def _lid_int(key: str) -> int:
    """Parse integer line id from result key like 'line_5' -> 5."""
    return int(key.split("_", 1)[1])


def _lid_str(lid_int: int) -> str:
    """Internal helper for module-local processing."""
    return f"line_{lid_int}"


def _line_loading_limit_pct(net, line_id: int) -> float:
    """Internal helper for module-local processing."""
    if (
        not hasattr(net, "line")
        or net.line is None
        or int(line_id) not in net.line.index
    ):
        return float("nan")

    row = net.line.loc[int(line_id)]
    try:
        max_loading_percent = float(row.get("max_loading_percent", 100.0))
    except (TypeError, ValueError):
        max_loading_percent = 100.0
    if not math.isfinite(max_loading_percent) or max_loading_percent <= 0.0:
        max_loading_percent = 100.0
    return float(max_loading_percent)


def _line_opf_nominal_limit_mva(net, line_id: int) -> float:
    """Return the pandapower OPF nominal line limit before max_loading scaling.

    This mirrors pandapower's branch RATE_A construction for lines:
    sqrt(3) * vn_kv(from_bus) * max_i_ka * df * parallel.
    """
    if (
        not hasattr(net, "line")
        or net.line is None
        or int(line_id) not in net.line.index
    ):
        return float("nan")

    row = net.line.loc[int(line_id)]
    try:
        max_i_ka = float(row.get("max_i_ka", float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(max_i_ka) or max_i_ka <= 0.0:
        return float("nan")

    try:
        from_bus = int(row.get("from_bus"))
        vn_kv = float(net.bus.at[from_bus, "vn_kv"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    if not math.isfinite(vn_kv) or vn_kv <= 0.0:
        return float("nan")

    try:
        df = float(row.get("df", 1.0))
    except (TypeError, ValueError):
        df = 1.0
    if not math.isfinite(df) or df <= 0.0:
        df = 1.0

    try:
        parallel = float(row.get("parallel", 1.0))
    except (TypeError, ValueError):
        parallel = 1.0
    if not math.isfinite(parallel) or parallel <= 0.0:
        parallel = 1.0

    return float(math.sqrt(3.0) * vn_kv * max_i_ka * df * parallel)


def _line_opf_effective_limit_mva(net, line_id: int) -> float:
    """Return the pandapower OPF effective line limit after max_loading scaling."""
    nominal_limit = _line_opf_nominal_limit_mva(net, line_id)
    if not math.isfinite(nominal_limit) or nominal_limit <= 0.0:
        return float("nan")

    max_loading_percent = _line_loading_limit_pct(net, line_id)
    if not math.isfinite(max_loading_percent) or max_loading_percent <= 0.0:
        return float("nan")

    return float(nominal_limit * max_loading_percent / 100.0)


def _align_line_limit_proxy_with_opf_model(net) -> dict:
    """Align demo proxy line limits with pandapower OPF branch limits.

    The stability-radius utilities read explicit MVA ratings first (``rateA`` /
    ``rate_a_mva``), while pandapower OPF for lines builds branch RATE_A from
    current ratings. On imported MATPOWER cases these can differ materially.
    For the demo, overwrite the explicit line MVA rating with the OPF-equivalent
    current-based nominal limit so security metrics and OPF constraints refer to
    the same branch model.
    """
    from stability_radius.radii.common import estimate_line_limit_mva_with_flag

    if not hasattr(net, "line") or net.line is None or len(net.line) == 0:
        return {
            "n_lines_checked": 0,
            "n_lines_aligned": 0,
            "max_pre_align_abs_diff_mva": float("nan"),
            "median_pre_align_abs_diff_mva": float("nan"),
            "worst_line_id": None,
        }

    if "rateA" not in net.line.columns:
        net.line["rateA"] = np.nan
    if "rate_a_mva" not in net.line.columns:
        net.line["rate_a_mva"] = np.nan

    abs_diffs: list[float] = []
    worst_line_id: int | None = None
    worst_abs_diff = -1.0
    aligned = 0
    checked = 0

    for lid in sorted(int(x) for x in net.line.index):
        opf_nominal_limit = _line_opf_nominal_limit_mva(net, lid)
        if not math.isfinite(opf_nominal_limit) or opf_nominal_limit <= 0.0:
            continue

        row = net.line.loc[lid]
        max_loading_percent = _line_loading_limit_pct(net, lid)
        mult = max_loading_percent / 100.0

        proxy_limit_mva, _ = estimate_line_limit_mva_with_flag(net, row)
        proxy_nominal_limit = (
            float(proxy_limit_mva) / mult
            if mult > 0.0 and math.isfinite(proxy_limit_mva)
            else float("nan")
        )
        abs_diff = abs(proxy_nominal_limit - opf_nominal_limit)
        abs_diffs.append(abs_diff)
        checked += 1
        if abs_diff > worst_abs_diff:
            worst_abs_diff = abs_diff
            worst_line_id = int(lid)
        if abs_diff > 1e-6:
            aligned += 1

        net.line.at[lid, "rateA"] = float(opf_nominal_limit)
        net.line.at[lid, "rate_a_mva"] = float(opf_nominal_limit)

    summary = {
        "n_lines_checked": checked,
        "n_lines_aligned": aligned,
        "max_pre_align_abs_diff_mva": float(max(abs_diffs))
        if abs_diffs
        else float("nan"),
        "median_pre_align_abs_diff_mva": float(np.median(abs_diffs))
        if abs_diffs
        else float("nan"),
        "worst_line_id": worst_line_id,
    }
    if checked:
        logger.info(
            "[limits] Aligned %d/%d line proxy limits with pandapower OPF branch limits; "
            "max pre-align diff=%.4f MVA (line=%s).",
            aligned,
            checked,
            summary["max_pre_align_abs_diff_mva"],
            worst_line_id,
        )
    return summary


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
    _align_line_limit_proxy_with_opf_model(net_lossless)

    line_indices = sorted(net_lossless.line.index.tolist())  # list of ints
    return net_lossless, slack_bus, line_indices


# ---------------------------------------------------------------------------
# Phase 2: True cost-minimising AC OPF
# ---------------------------------------------------------------------------


def _add_matpower_costs(nn, input_path: str) -> int:
    """Parse MATPOWER gencost and add poly_cost entries to pandapower network.

    Returns the number of generators that received a non-zero cost.
    Matches generators by bus number (robust to ordering differences).
    Also assigns cost to ext_grid (slack) when the reference bus has costs.
    """
    import re
    import pandapower as pp

    with open(input_path, encoding="utf-8", errors="replace") as _f:
        txt = _f.read()

    def _parse(name):
        """Internal helper for module-local processing."""
        m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", txt, re.DOTALL)
        if not m:
            return []
        rows = []
        for line in m.group(1).strip().split("\n"):
            line = line.strip().split("%")[0].strip().rstrip(";").strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
        return rows

    gen_rows = _parse("gen")
    cost_rows = _parse("gencost")
    if not gen_rows or not cost_rows:
        logger.warning("[costs] No gencost data in %s", input_path)
        return 0

    bus_costs: dict[int, list[tuple[float, float, float]]] = {}
    for gr, cr in zip(gen_rows, cost_rows):
        bus = int(gr[0])
        ncost = int(cr[3]) if len(cr) > 3 else 3
        if ncost == 3 and len(cr) >= 7:
            c2, c1, c0 = cr[4], cr[5], cr[6]
        elif ncost >= 2 and len(cr) >= 6:
            c2, c1, c0 = 0.0, cr[4], cr[5]
        else:
            c2, c1, c0 = 0.0, 0.0, 0.0
        bus_costs.setdefault(bus, []).append((float(c2), float(c1), float(c0)))

    _clear_existing_costs(nn)

    elements_by_bus: dict[int, list[tuple[str, int]]] = {}
    for element_type, idx, bus in _iter_dispatchable_elements(nn):
        elements_by_bus.setdefault(int(bus), []).append((element_type, int(idx)))

    added = 0
    for bus, elements in elements_by_bus.items():
        costs = list(bus_costs.get(int(bus), []))
        if not costs:
            continue
        if len(costs) != len(elements):
            agg = np.sum(np.asarray(costs, dtype=float), axis=0) / float(len(elements))
            logger.warning(
                "[costs] Bus %d has %d MATPOWER cost rows but %d pandapower elements; using equal-share assignment.",
                int(bus),
                len(costs),
                len(elements),
            )
            assigned_costs = [tuple(float(x) for x in agg)] * len(elements)
        else:
            assigned_costs = costs

        for (element_type, idx), (c2, c1, c0) in zip(elements, assigned_costs):
            if c0 == 0.0 and c1 == 0.0 and c2 == 0.0:
                continue
            pp.create_poly_cost(
                nn,
                idx,
                element_type,
                cp1_eur_per_mw=c1,
                cp2_eur_per_mw2=c2,
                cp0_eur=c0,
            )
            added += 1

    logger.info("[costs] Added %d poly_cost entries from %s", added, input_path)
    return added


def _set_default_voltage_bounds(nn) -> None:
    """Set bus voltage bounds for OPF, preserving existing values and clipping to safe range."""
    nn.bus["min_vm_pu"] = nn.bus.get("min_vm_pu", 0.9).fillna(0.9).clip(lower=0.85)
    nn.bus["max_vm_pu"] = nn.bus.get("max_vm_pu", 1.1).fillna(1.1).clip(upper=1.15)


def _clear_existing_costs(nn) -> None:
    """Internal helper for module-local processing."""
    if hasattr(nn, "poly_cost") and nn.poly_cost is not None and len(nn.poly_cost):
        nn.poly_cost.drop(nn.poly_cost.index, inplace=True)
    if hasattr(nn, "pwl_cost") and nn.pwl_cost is not None and len(nn.pwl_cost):
        nn.pwl_cost.drop(nn.pwl_cost.index, inplace=True)


def _iter_dispatchable_elements(nn):
    """Internal helper for module-local processing."""
    for table_name, element_type in (
        ("ext_grid", "ext_grid"),
        ("gen", "gen"),
        ("sgen", "sgen"),
    ):
        table = getattr(nn, table_name, None)
        if table is None or len(table) == 0:
            continue
        for idx in sorted(int(x) for x in table.index):
            row = table.loc[idx]
            if "in_service" in row and not bool(row.get("in_service", True)):
                continue
            yield element_type, int(idx), int(row["bus"])


def _prepare_cost_opf_network(nn) -> None:
    """Internal helper for module-local processing."""
    from stability_radius.base_point.pandapower_opp import (
        _set_line_thermal_limits,
        _setup_gen_for_opp,
    )

    _clear_existing_costs(nn)
    _set_line_thermal_limits(nn)
    _setup_gen_for_opp(nn, pg0_source="case")
    _clear_existing_costs(nn)


def _apply_loading_limits(
    nn,
    *,
    default_loading_percent: float,
    per_line_loading_limits_pct: Mapping[int, float] | None = None,
) -> None:
    """Internal helper for module-local processing."""
    default_pct = float(default_loading_percent)
    if hasattr(nn, "line") and nn.line is not None and len(nn.line):
        nn.line["max_loading_percent"] = default_pct
        if per_line_loading_limits_pct:
            for lid, pct in per_line_loading_limits_pct.items():
                if int(lid) in nn.line.index:
                    nn.line.at[int(lid), "max_loading_percent"] = float(pct)
    if hasattr(nn, "trafo") and nn.trafo is not None and len(nn.trafo):
        nn.trafo["max_loading_percent"] = default_pct


def _extract_opp_state(nn) -> tuple[dict[str, float], dict[int, float]]:
    """Internal helper for module-local processing."""
    opp_dispatch: dict[str, float] = {}
    if hasattr(nn, "res_gen") and nn.res_gen is not None and len(nn.res_gen):
        for gid in sorted(int(x) for x in nn.res_gen.index):
            opp_dispatch[f"gen_{gid}"] = float(nn.res_gen.loc[gid, "p_mw"])
    if hasattr(nn, "res_sgen") and nn.res_sgen is not None and len(nn.res_sgen):
        for sid in sorted(int(x) for x in nn.res_sgen.index):
            opp_dispatch[f"sgen_{sid}"] = float(nn.res_sgen.loc[sid, "p_mw"])
    opp_vm_pu = {
        int(bid): float(nn.res_bus.loc[bid, "vm_pu"])
        for bid in sorted(int(x) for x in nn.res_bus.index)
    }
    return opp_dispatch, opp_vm_pu


def _validate_opf_with_pf(nn, label: str) -> tuple[bool, float]:
    """Replay the solved OPF point with AC PF.

    Returns
    -------
    (pf_converged, max_current_loading_gap_pct)
        ``pf_converged`` is the actual acceptance criterion.
        ``max_current_loading_gap_pct`` is diagnostic-only and compares
        post-PF ``loading_percent`` against ``max_loading_percent``.

    Notes
    -----
    ``runopp`` is configured with ``OPF_FLOW_LIM=0``, i.e. apparent-power branch
    limits. The post-PF current-loading gap is therefore informative, but not
    the branch-feasibility criterion used by the solver.
    """
    import pandapower as pp

    from stability_radius.base_point.pandapower_tools import (
        apply_opp_result_to_pandapower_net,
    )

    opp_dispatch, opp_vm_pu = _extract_opp_state(nn)
    apply_opp_result_to_pandapower_net(
        nn,
        opp_gen_dispatch=opp_dispatch,
        opp_vm_pu=opp_vm_pu,
    )

    pf_ok = False
    for init, enforce_q_lims in (("results", True), ("dc", True), ("flat", False)):
        try:
            pp.runpp(
                nn,
                calculate_voltage_angles=True,
                init=init,
                enforce_q_lims=enforce_q_lims,
                numba=False,
                max_iteration=100,
                tolerance_mva=1e-8,
            )
            if bool(getattr(nn, "converged", False)):
                pf_ok = True
                break
        except Exception as exc:
            logger.debug(
                "[%s] Post-OPF PF attempt (init=%s) failed: %s", label, init, exc
            )

    if not pf_ok:
        logger.warning("[%s] Post-OPF PF validation failed to converge.", label)
        return False, float("inf")

    max_loading_violation_pct = 0.0
    if hasattr(nn, "res_line") and nn.res_line is not None and len(nn.res_line):
        for lid in nn.res_line.index:
            target = float(nn.line.at[lid, "max_loading_percent"])
            actual = float(nn.res_line.at[lid, "loading_percent"])
            max_loading_violation_pct = max(max_loading_violation_pct, actual - target)
    if hasattr(nn, "res_trafo") and nn.res_trafo is not None and len(nn.res_trafo):
        for tid in nn.res_trafo.index:
            target = float(nn.trafo.at[tid, "max_loading_percent"])
            actual = float(nn.res_trafo.at[tid, "loading_percent"])
            max_loading_violation_pct = max(max_loading_violation_pct, actual - target)

    if max_loading_violation_pct > 0.25:
        logger.warning(
            "[%s] Post-OPF PF replay shows current-based loading above "
            "max_loading_percent by %.3f%%. Diagnostic only: runopp here "
            "enforces apparent-power branch limits (OPF_FLOW_LIM=0).",
            label,
            max_loading_violation_pct,
        )

    return True, max_loading_violation_pct


def _run_cost_opf(nn, label: str = "opf") -> float:
    """Run pandapower runopp on an already-configured network.

    Assumes poly_cost, line limits, and voltage bounds are set.
    Returns total generation cost ($/h).
    Raises RuntimeError if OPF does not converge.
    """
    import pandapower as pp
    import warnings

    for init in ("dc", "flat"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp.runopp(
                    nn,
                    calculate_voltage_angles=True,
                    OPF_FLOW_LIM=0,
                    OPF_VIOLATION=1e-4,
                    init=init,
                    max_iteration=300,
                    numba=False,
                )
            logger.info(
                "[%s] OPF converged (init=%s): cost=%.2f $/h, max_loading=%.1f%%",
                label,
                init,
                float(nn.res_cost),
                float(nn.res_line.loading_percent.max()),
            )
            return float(nn.res_cost)
        except Exception as exc:
            logger.warning("[%s] OPF attempt (init=%s) failed: %s", label, init, exc)

    raise RuntimeError(
        f"[{label}] AC cost OPF did not converge with any init strategy."
    )


def _solve_cost_opf(
    net_lossless,
    line_indices: list[int],
    input_path: str,
    max_loading_percent: float = 99.0,
    per_line_loading_limits_pct: Mapping[int, float] | None = None,
    label: str = "cost_opf",
):
    """Solve true cost-minimising AC OPF.

    Uses actual generator cost curves from the MATPOWER file, not the
    feasibility (min-deviation) objective.

    Returns
    -------
    (nn_solved, base_pf, total_cost_eur_h)
        nn_solved      : pandapower network with res_* populated
        base_pf        : PyPSAAPFResult extracted from res_*
        total_cost     : total generation cost in $/h
    """
    import pandapower as pp

    for attempt_idx, scale in enumerate((1.0, 0.995, 0.99), start=1):
        nn = copy.deepcopy(net_lossless)
        _prepare_cost_opf_network(nn)
        _apply_loading_limits(
            nn,
            default_loading_percent=float(max_loading_percent) * float(scale),
            per_line_loading_limits_pct=(
                {
                    int(lid): float(pct) * float(scale)
                    for lid, pct in per_line_loading_limits_pct.items()
                }
                if per_line_loading_limits_pct
                else None
            ),
        )
        _set_default_voltage_bounds(nn)

        n_added = _add_matpower_costs(nn, input_path)
        if n_added == 0:
            logger.warning(
                "[%s] No cost data found; falling back to unit costs.", label
            )
            for element_type, idx, _ in _iter_dispatchable_elements(nn):
                if element_type == "ext_grid":
                    pp.create_poly_cost(nn, idx, element_type, cp1_eur_per_mw=1.0)
                    continue
                table_name = "gen" if element_type == "gen" else "sgen"
                p_max = float(getattr(nn, table_name).at[idx, "max_p_mw"])
                if p_max > 0:
                    pp.create_poly_cost(nn, idx, element_type, cp1_eur_per_mw=1.0)

        logger.info(
            "[%s] Solving true cost-minimising AC OPF (attempt=%d, loading_scale=%.3f)...",
            label,
            attempt_idx,
            scale,
        )
        t0 = time.time()
        total_cost = _run_cost_opf(nn, label=label)
        pf_replayed, max_violation_pct = _validate_opf_with_pf(nn, label=label)
        logger.info(
            "[%s] Done in %.1fs, cost=%.2f $/h, pf_replayed=%s, current_gap=%.3f%%",
            label,
            time.time() - t0,
            total_cost,
            pf_replayed,
            max_violation_pct,
        )
        if pf_replayed:
            base_pf = _extract_pypsa_result_from_pp(nn, line_indices)
            return nn, base_pf, total_cost
        logger.warning(
            "[%s] Retrying with tighter loading limits after PF replay failure.",
            label,
        )

    raise RuntimeError(f"[{label}] AC cost OPF could not be PF-validated.")


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
    r_target: float,
    n_iter: int,
    input_path: str,
):
    """Cost-minimising OPF with stability-radius-guided thermal tightening.

    Uses the SAME generator cost curves as the cost OPF — the only
    difference from the cost OPF is that lines with radius < r_target
    have their thermal limits tightened by r_target * ||h_l||.
    Iterates n_iter times, updating h-norms after each round.

    Returns
    -------
    (nn_solved, base_pf, results, h_vectors, total_cost)
    """
    from stability_radius.radii.common import estimate_line_limit_mva

    current_results = cost_results
    nn_radius = None
    base_pf_radius = None
    radius_results = cost_results
    radius_h_vectors = {}
    total_cost = float("nan")

    for iteration in range(n_iter):
        logger.info("[radius_opf] Iteration %d/%d", iteration + 1, n_iter)

        # Tighten limits only for vulnerable lines (radius < r_target)
        tight_limits_pct: dict[int, float] = {}
        tightened = 0
        skipped_safe = 0
        for lid in line_indices:
            key = _lid_str(lid)
            res = current_results.get(key, {})
            current_radius = float(res.get("radius_ac_l2", float("inf")))
            base_pct = 99.0
            if res.get("is_unconstrained", False) or not math.isfinite(current_radius):
                tight_limits_pct[int(lid)] = base_pct
                continue
            if current_radius >= r_target:
                skipped_safe += 1
                tight_limits_pct[int(lid)] = base_pct
                continue
            h_norm = float(res.get("||h||2", 0.0))
            if h_norm < 1e-10:
                tight_limits_pct[int(lid)] = base_pct
                continue
            s_limit = float(
                estimate_line_limit_mva(net_lossless, net_lossless.line.loc[lid])
            )
            new_limit_mva = max(0.10 * s_limit, s_limit - r_target * h_norm)
            tight_limits_pct[int(lid)] = 100.0 * new_limit_mva / s_limit
            tightened += 1

        tight_pcts = np.asarray(list(tight_limits_pct.values()), dtype=float)
        logger.info(
            "[radius_opf] Tightened %d lines (skipped %d already safe) | "
            "min_pct=%.1f%% mean_pct=%.1f%%",
            tightened,
            skipped_safe,
            float(np.min(tight_pcts)),
            float(np.mean(tight_pcts)),
        )

        if tightened == 0:
            logger.info(
                "[radius_opf] All lines satisfy r >= r_target=%.2f. Converged.",
                r_target,
            )
            break

        try:
            nn_radius, base_pf_radius, total_cost = _solve_cost_opf(
                net_lossless,
                line_indices,
                input_path=input_path,
                max_loading_percent=99.0,
                per_line_loading_limits_pct=tight_limits_pct,
                label=f"radius_opf_iter{iteration + 1}",
            )
        except Exception as exc:
            logger.error("[radius_opf] OPF failed on iter %d: %s", iteration + 1, exc)
            if iteration == 0:
                logger.warning(
                    "[radius_opf] First tightened solve failed; falling back to cost OPF baseline."
                )
                break
            logger.warning("[radius_opf] Keeping previous iteration result.")
            break

        radius_results, radius_h_vectors = _compute_radii(
            net_lossless,
            base_pf_radius,
            slack_bus,
            label=f"radius_opf_iter{iteration + 1}",
        )
        current_results = radius_results

    return nn_radius, base_pf_radius, radius_results, radius_h_vectors, total_cost


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
    h_to = h_vectors.get("h_to")  # (m, n_vars)
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
        (sigma_p_mw**2) * np.sum(hP_bal**2, axis=1)
        + (sigma_q_mvar**2) * np.sum(hQ_bal**2, axis=1)
    )

    margin = s_limit_arr - s0_arr
    eps = 1e-15

    # sigma-radius: r_sigma = margin / sigma_flow
    r_sigma = np.where(sigma_flow > eps, margin / sigma_flow, float("inf"))

    # Overload probability (Gaussian): P(S0 + X > c) for apparent-power limits.
    overload_probs = np.zeros(n_actual, dtype=float)
    for i in range(n_actual):
        sf = float(sigma_flow[i])
        if sf <= 0.0:
            overload_probs[i] = 0.0 if margin[i] > 0 else 1.0
        else:
            from stability_radius.radii.ac_sigma_radius import (
                overload_probability_one_sided_limit,
            )

            overload_probs[i] = overload_probability_one_sided_limit(
                y0=float(s0_arr[i]),
                limit=float(s_limit_arr[i]),
                sigma=sf,
            )

    # --- Baseline metrics ---
    lr_vals = s0_arr / np.maximum(s_limit_arr, 1e-10)
    hr_vals = s_limit_arr - s0_arr
    cantelli_vals = np.where(
        margin > 0,
        sigma_flow**2 / (sigma_flow**2 + margin**2),
        1.0,
    )
    pi_vals = 0.5 * (lr_vals**2)  # performance index (n=1)

    # Summary
    finite_mask = np.isfinite(r_sigma)
    finite_r = r_sigma[finite_mask]

    summary = {
        "label": label,
        "n_lines": n_actual,
        # AC sigma-radius
        "sigma_radius_min": float(np.min(finite_r)) if finite_r.size else float("nan"),
        "sigma_radius_median": float(np.median(finite_r))
        if finite_r.size
        else float("nan"),
        "sigma_radius_p10": float(np.percentile(finite_r, 10))
        if finite_r.size
        else float("nan"),
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
    """Internal helper for module-local processing."""
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


def _ac_n1_screen(
    nn_solved,
    label: str,
    *,
    return_peak_loadings: bool = False,
) -> list[dict] | tuple[list[dict], dict[int, float]]:
    """Brute-force AC N-1 screening from a solved pandapower network.

    nn_solved must have res_gen and res_bus populated (from runopp or runpp).
    Generator dispatch and voltage setpoints are read from the OPF result so
    the contingency PF starts from the correct operating point.
    """
    import pandapower as pp

    # Build base network at the OPF operating point.
    # Deep copy already carries all res_* tables and dispatch values.
    net_base = copy.deepcopy(nn_solved)

    # Run base PF with results init (uses OPF voltages as warm start)
    for init in ("results", "dc", "flat"):
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

    line_ids = sorted(net_base.line.index.tolist())
    records = []
    peak_loading_pct_by_line = {int(lid): 0.0 for lid in line_ids}
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
            if len(nn.res_line):
                max_loading_pct = float(nn.res_line.loading_percent.max())
                worst_line = int(nn.res_line.loading_percent.idxmax())
                overloaded_line_ids = [
                    int(x)
                    for x in nn.res_line.loading_percent[
                        nn.res_line.loading_percent > 100
                    ].index
                ]
                for line_id, loading_pct in nn.res_line.loading_percent.items():
                    peak_loading_pct_by_line[int(line_id)] = max(
                        peak_loading_pct_by_line.get(int(line_id), 0.0),
                        float(loading_pct),
                    )
            else:
                max_loading_pct = float("nan")
                worst_line = None
                overloaded_line_ids = []
        else:
            max_loading_pct = float("nan")
            worst_line = None
            overloaded_line_ids = []

        records.append(
            {
                "contingency_line": int(lid),
                "pf_converged": converged,
                "n1_feasible": converged and n_overloads == 0,
                "n_overloads": n_overloads,
                "max_loading_percent": max_loading_pct,
                "worst_line": worst_line,
                "overloaded_line_ids": overloaded_line_ids,
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
    if return_peak_loadings:
        return records, peak_loading_pct_by_line
    return records


def _update_scopf_line_limits(
    current_limits_pct: Mapping[int, float],
    peak_loading_pct_by_line: Mapping[int, float],
    *,
    security_target_pct: float = 99.0,
    min_limit_pct: float = 40.0,
) -> tuple[dict[int, float], list[int]]:
    """Internal helper for module-local processing."""
    updated = {int(lid): float(pct) for lid, pct in current_limits_pct.items()}
    changed: list[int] = []
    for lid, peak_pct in peak_loading_pct_by_line.items():
        peak = float(peak_pct)
        if not math.isfinite(peak) or peak <= 100.0:
            continue
        current = float(updated.get(int(lid), security_target_pct))
        tightened = max(
            float(min_limit_pct), current * float(security_target_pct) / peak
        )
        if tightened < current - 1e-6:
            updated[int(lid)] = float(tightened)
            changed.append(int(lid))
    return updated, changed


def _total_generation_dispatch_mw(nn) -> float:
    """Internal helper for module-local processing."""
    total = 0.0
    for table_name in ("res_gen", "res_sgen", "res_ext_grid"):
        table = getattr(nn, table_name, None)
        if table is None or len(table) == 0 or "p_mw" not in table.columns:
            continue
        total += float(table["p_mw"].sum())
    return total


def _opf_constraint_summary(nn, label: str) -> dict:
    """Internal helper for module-local processing."""
    max_line_loading_pct = float("nan")
    min_line_loading_headroom_pct = float("nan")
    if hasattr(nn, "res_line") and nn.res_line is not None and len(nn.res_line):
        actual = np.asarray(nn.res_line.loading_percent.values, dtype=float)
        target = np.asarray(
            nn.line.loc[nn.res_line.index, "max_loading_percent"].values, dtype=float
        )
        max_line_loading_pct = float(np.max(actual))
        min_line_loading_headroom_pct = float(np.min(target - actual))

    max_trafo_loading_pct = float("nan")
    min_trafo_loading_headroom_pct = float("nan")
    if hasattr(nn, "res_trafo") and nn.res_trafo is not None and len(nn.res_trafo):
        actual = np.asarray(nn.res_trafo.loading_percent.values, dtype=float)
        target = np.asarray(
            nn.trafo.loc[nn.res_trafo.index, "max_loading_percent"].values, dtype=float
        )
        max_trafo_loading_pct = float(np.max(actual))
        min_trafo_loading_headroom_pct = float(np.min(target - actual))

    return {
        "label": label,
        "max_line_loading_pct": max_line_loading_pct,
        "min_line_loading_headroom_pct": min_line_loading_headroom_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
        "min_trafo_loading_headroom_pct": min_trafo_loading_headroom_pct,
    }


def _opf_line_limit_consistency_df(nn) -> "DataFrame":
    """Internal helper for module-local processing."""
    import pandas as pd

    from stability_radius.radii.common import estimate_line_limit_mva_with_flag

    if not hasattr(nn, "line") or nn.line is None or len(nn.line) == 0:
        return pd.DataFrame(
            columns=[
                "line_id",
                "max_loading_percent",
                "opf_nominal_limit_mva",
                "opf_limit_mva",
                "proxy_nominal_limit_mva",
                "proxy_limit_mva",
                "abs_diff_mva",
                "rel_diff_pct",
                "is_unconstrained_proxy",
                "limit_mismatch",
            ]
        ).set_index("line_id")

    rows: list[dict[str, object]] = []
    for lid in sorted(int(x) for x in nn.line.index):
        opf_nominal_limit_mva = _line_opf_nominal_limit_mva(nn, lid)
        opf_limit_mva = _line_opf_effective_limit_mva(nn, lid)
        if not math.isfinite(opf_limit_mva) or opf_limit_mva <= 0.0:
            continue

        max_loading_percent = _line_loading_limit_pct(nn, lid)
        row = nn.line.loc[lid]
        proxy_limit_mva, is_unconstrained = estimate_line_limit_mva_with_flag(nn, row)
        mult = max_loading_percent / 100.0
        proxy_nominal_limit_mva = (
            float(proxy_limit_mva) / mult
            if mult > 0.0 and math.isfinite(proxy_limit_mva)
            else float("nan")
        )
        abs_diff_mva = float(proxy_limit_mva) - float(opf_limit_mva)
        rel_diff_pct = (
            100.0 * abs_diff_mva / float(opf_limit_mva)
            if math.isfinite(opf_limit_mva) and abs(opf_limit_mva) > 1e-12
            else float("nan")
        )
        rows.append(
            {
                "line_id": lid,
                "max_loading_percent": float(max_loading_percent),
                "opf_nominal_limit_mva": float(opf_nominal_limit_mva),
                "opf_limit_mva": float(opf_limit_mva),
                "proxy_nominal_limit_mva": float(proxy_nominal_limit_mva),
                "proxy_limit_mva": float(proxy_limit_mva),
                "abs_diff_mva": float(abs_diff_mva),
                "rel_diff_pct": float(rel_diff_pct),
                "is_unconstrained_proxy": bool(is_unconstrained),
                "limit_mismatch": not np.isclose(
                    float(proxy_limit_mva),
                    float(opf_limit_mva),
                    rtol=1e-9,
                    atol=1e-6,
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "line_id",
                "max_loading_percent",
                "opf_nominal_limit_mva",
                "opf_limit_mva",
                "proxy_nominal_limit_mva",
                "proxy_limit_mva",
                "abs_diff_mva",
                "rel_diff_pct",
                "is_unconstrained_proxy",
                "limit_mismatch",
            ]
        ).set_index("line_id")
    return pd.DataFrame(rows).set_index("line_id").sort_index()


def _opf_line_limit_consistency_summary(nn, label: str) -> dict:
    """Internal helper for module-local processing."""
    df = _opf_line_limit_consistency_df(nn)
    if df.empty:
        return {"label": label, "n_lines_checked": 0}

    abs_diff = df["abs_diff_mva"].abs()
    rel_diff = df["rel_diff_pct"].abs()
    worst_line = int(abs_diff.idxmax()) if len(abs_diff) else None
    summary = {
        "label": label,
        "n_lines_checked": int(len(df)),
        "n_limit_mismatch": int(df["limit_mismatch"].sum()),
        "max_abs_limit_diff_mva": float(abs_diff.max()),
        "median_abs_limit_diff_mva": float(abs_diff.median()),
        "max_rel_limit_diff_pct": float(rel_diff.max())
        if len(rel_diff)
        else float("nan"),
        "worst_line_id": worst_line,
    }
    if summary["n_limit_mismatch"] > 0:
        logger.warning(
            "[%s] Proxy-vs-OPF line limit mismatch: %d/%d lines, max_abs_diff=%.6f MVA (line=%s).",
            label,
            summary["n_limit_mismatch"],
            summary["n_lines_checked"],
            summary["max_abs_limit_diff_mva"],
            worst_line,
        )
    else:
        logger.info(
            "[%s] Proxy-vs-OPF line limits are consistent across %d lines.",
            label,
            summary["n_lines_checked"],
        )
    return summary


def _solve_scopf(
    net_lossless,
    slack_bus: int,
    line_indices: list[int],
    input_path: str,
    n_iter: int,
):
    """Internal helper for module-local processing."""
    current_limits_pct = {int(lid): 99.0 for lid in line_indices}
    nn_scopf = None
    base_pf_scopf = None
    total_cost = float("nan")
    final_records: list[dict] = []

    for iteration in range(max(int(n_iter), 1)):
        logger.info("[scopf] Iteration %d/%d", iteration + 1, max(int(n_iter), 1))
        nn_scopf, base_pf_scopf, total_cost = _solve_cost_opf(
            net_lossless,
            line_indices,
            input_path=input_path,
            max_loading_percent=99.0,
            per_line_loading_limits_pct=current_limits_pct,
            label=f"scopf_iter{iteration + 1}",
        )
        final_records, peak_loading_pct_by_line = _ac_n1_screen(
            nn_scopf,
            f"scopf_iter{iteration + 1}",
            return_peak_loadings=True,
        )
        diverged = sum(1 for rec in final_records if not rec.get("pf_converged"))
        if diverged:
            logger.warning(
                "[scopf] %d contingencies diverged during screening; keeping current tightened limits.",
                diverged,
            )

        worst_peak = max(peak_loading_pct_by_line.values(), default=float("nan"))
        logger.info("[scopf] Worst post-contingency loading = %.2f%%", worst_peak)
        updated_limits_pct, changed_lines = _update_scopf_line_limits(
            current_limits_pct,
            peak_loading_pct_by_line,
            security_target_pct=99.0,
        )
        if not changed_lines:
            logger.info("[scopf] AC contingency screening is satisfied. Converged.")
            break

        current_limits_pct = updated_limits_pct
        logger.info(
            "[scopf] Tightened %d lines; new min pre-contingency limit = %.2f%%",
            len(changed_lines),
            min(current_limits_pct.values()),
        )

    if nn_scopf is None or base_pf_scopf is None:
        raise RuntimeError("[scopf] No feasible SCOPF operating point was produced.")

    scopf_results, scopf_h_vectors = _compute_radii(
        net_lossless,
        base_pf_scopf,
        slack_bus,
        label="scopf",
    )
    return (
        nn_scopf,
        base_pf_scopf,
        scopf_results,
        scopf_h_vectors,
        total_cost,
        final_records,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _radii_to_df(results: dict) -> "DataFrame":
    """Internal helper for module-local processing."""
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
    """Internal helper for module-local processing."""
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
    """Internal helper for module-local processing."""
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
    """Internal helper for module-local processing."""
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
    """Internal helper for module-local processing."""
    df.to_csv(path)
    logger.info("Saved %s: %s (%d rows)", label, path, len(df))


def _dc_n1_to_df(n1_results: dict) -> "DataFrame":
    """Internal helper for module-local processing."""
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
    """Internal helper for module-local processing."""
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
    cost_opf_eur_h,
    radius_opf_eur_h,
    cost_increase_pct,
    cost_gen_mw,
    radius_gen_mw,
    output_path,
):
    """Internal helper for module-local processing."""

    def _fmt(val, fmt=".4f"):
        """Internal helper for module-local processing."""
        if isinstance(val, float):
            return format(val, fmt)
        return str(val)

    lines = [
        "=" * 70,
        "N-1 STABILITY DEMO - COMPARISON SUMMARY",
        "=" * 70,
        "",
        f"  r_target (tightening parameter): {r_target:.4f} MW",
        f"  sigma_p = {sigma_p_mw} MW, sigma_q = {sigma_q_mvar} MVAr",
        "",
        "--- Generation Cost (true cost-minimising AC OPF) ---",
        f"  {'Metric':<35} {'Cost OPF':>12} {'Radius OPF':>12}",
        "  " + "-" * 60,
        f"  {'Total generation cost ($/h)':<35} {_fmt(cost_opf_eur_h, '.2f'):>12} {_fmt(radius_opf_eur_h, '.2f'):>12}",
        f"  {'Total generation dispatch (MW)':<35} {_fmt(cost_gen_mw, '.2f'):>12} {_fmt(radius_gen_mw, '.2f'):>12}",
        f"  {'Cost increase (%)':<35} {'baseline':>12} {_fmt(cost_increase_pct, '.3f'):>12}",
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
        ("n_lines", "Lines with N-1 radius computed"),
        ("n_already_n1_infeasible", "Lines already N-1 infeasible"),
        ("ac_n1_radius_min", "Min AC N-1 radius (MW, positive)"),
        ("ac_n1_radius_median", "Median AC N-1 radius (MW)"),
        ("ac_n1_radius_p10", "P10 AC N-1 radius (MW)"),
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
        ("sigma_radius_min", "Min sigma-radius"),
        ("sigma_radius_median", "Median sigma-radius"),
        ("sigma_radius_p10", "P10 sigma-radius"),
        ("max_overload_prob", "Max overload probability"),
        ("mean_overload_prob", "Mean overload probability"),
        ("n_prob_above_1pct", "Lines with P(overload) > 1%"),
        ("n_prob_above_5pct", "Lines with P(overload) > 5%"),
        ("max_cantelli_ub", "Max Cantelli upper bound"),
        ("pi_system", "System Performance Index"),
        ("pi_max", "Max line Performance Index"),
        ("min_headroom_mva", "Min headroom (MVA)"),
    ]:
        cv = _fmt(
            cost_sigma.get(key, "N/A"),
            ".6f" if "prob" in key or "cantelli" in key else ".4f",
        )
        rv = _fmt(
            radius_sigma.get(key, "N/A"),
            ".6f" if "prob" in key or "cantelli" in key else ".4f",
        )
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
    """Internal helper for module-local processing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def _get_radii(res):
        """Internal helper for module-local processing."""
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
    """Internal helper for module-local processing."""
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
    """Internal helper for module-local processing."""
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


def _apply_dispatch_from_solved(net_base, nn_solved) -> None:
    """Apply generator dispatch and voltage setpoints from a solved OPF network."""
    if hasattr(nn_solved, "res_gen") and nn_solved.res_gen is not None:
        common = net_base.gen.index.intersection(nn_solved.res_gen.index)
        net_base.gen.loc[common, "p_mw"] = nn_solved.res_gen.loc[common, "p_mw"].values
    if hasattr(nn_solved, "res_sgen") and nn_solved.res_sgen is not None:
        common = net_base.sgen.index.intersection(nn_solved.res_sgen.index)
        net_base.sgen.loc[common, "p_mw"] = nn_solved.res_sgen.loc[
            common, "p_mw"
        ].values
    if hasattr(nn_solved, "res_bus") and nn_solved.res_bus is not None:
        gen_bus_idx = net_base.gen.bus.unique()
        for bidx in gen_bus_idx:
            if bidx in nn_solved.res_bus.index:
                vm = float(nn_solved.res_bus.at[bidx, "vm_pu"])
                net_base.gen.loc[net_base.gen.bus == bidx, "vm_pu"] = vm
        if (
            hasattr(net_base, "ext_grid")
            and net_base.ext_grid is not None
            and len(net_base.ext_grid)
        ):
            for egidx in net_base.ext_grid.index:
                bus = int(net_base.ext_grid.at[egidx, "bus"])
                if bus in nn_solved.res_bus.index:
                    net_base.ext_grid.at[egidx, "vm_pu"] = float(
                        nn_solved.res_bus.at[bus, "vm_pu"]
                    )


def _compute_ac_n1_radii(
    net_lossless, nn_solved, slack_bus: int, line_indices: list[int], label: str
) -> dict:
    """Compute per-line AC N-1 stability radius.

    For each line l: AC N-1 radius = min over contingencies k != l of
    (AC L2 radius of line l when line k is disconnected at the existing dispatch).

    nn_solved: the solved pandapower network (from runopp) — used to apply
               the correct dispatch; base_pf is used for AC L2 at base case.

    Returns dict: line_id (int) -> min_n1_radius (float).
    """
    import pandapower as pp
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius

    logger.info(
        "[%s] Computing AC N-1 stability radii (%d contingencies)...",
        label,
        len(line_indices),
    )
    t0 = time.time()

    # Prepare base network at OPF operating point
    net_base = copy.deepcopy(net_lossless)
    _apply_dispatch_from_solved(net_base, nn_solved)

    # Run base PF with results warm start
    for init in ("results", "dc", "flat"):
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
    """Internal helper for module-local processing."""
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
        "ac_n1_radius_p10": float(np.percentile(positive, 10))
        if positive
        else float("nan"),
    }


def _plot_ac_n1_radius_cdf(cost_n1: dict, radius_n1: dict, output_path: Path) -> None:
    """Internal helper for module-local processing."""
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


def _append_metric_table(
    lines: list[str],
    *,
    title: str,
    regime_order: list[tuple[str, str]],
    summaries: Mapping[str, Mapping[str, object]],
    metrics: list[tuple[str, str, str]],
) -> None:
    """Internal helper for module-local processing."""

    def _fmt(value: object, fmt: str) -> str:
        """Internal helper for module-local processing."""
        if value is None:
            return "N/A"
        if isinstance(value, str):
            return value
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not math.isfinite(numeric):
            return "nan"
        if fmt == "d":
            return str(int(round(numeric)))
        return format(numeric, fmt)

    lines.extend(
        [
            "",
            f"--- {title} ---",
            f"  {'Metric':<35}" + "".join(f" {name:>12}" for _, name in regime_order),
            "  " + "-" * (35 + 13 * len(regime_order)),
        ]
    )
    for key, label, fmt in metrics:
        row = [f"  {label:<35}"]
        for regime_key, _ in regime_order:
            value = summaries.get(regime_key, {}).get(key, "N/A")
            row.append(f" {_fmt(value, fmt):>12}")
        lines.append("".join(row))


def _build_comparison_text(
    *,
    regime_order: list[tuple[str, str]],
    dispatch_summaries: Mapping[str, Mapping[str, object]],
    limit_consistency_summaries: Mapping[str, Mapping[str, object]],
    constraint_summaries: Mapping[str, Mapping[str, object]],
    radius_summaries: Mapping[str, Mapping[str, object]],
    ac_n1_radius_summaries: Mapping[str, Mapping[str, object]],
    sigma_summaries: Mapping[str, Mapping[str, object]],
    dc_n1_summaries: Mapping[str, Mapping[str, object]],
    screen_summaries: Mapping[str, Mapping[str, object]],
    verify: Mapping[str, object],
    r_target: float,
    sigma_p_mw: float,
    sigma_q_mvar: float,
) -> str:
    """Internal helper for module-local processing."""
    lines = [
        "=" * 70,
        "N-1 STABILITY DEMO - COMPARISON SUMMARY",
        "=" * 70,
        "",
        f"  r_target (tightening parameter): {r_target:.4f} MW",
        f"  sigma_p = {sigma_p_mw:.3f} MW, sigma_q = {sigma_q_mvar:.3f} MVAr",
    ]

    _append_metric_table(
        lines,
        title="Generation Cost",
        regime_order=regime_order,
        summaries=dispatch_summaries,
        metrics=[
            ("total_cost_eur_h", "Total generation cost ($/h)", ".2f"),
            ("generation_dispatch_mw", "Net generation dispatch (MW)", ".2f"),
            ("cost_increase_pct", "Cost increase vs Cost OPF (%)", ".3f"),
        ],
    )
    _append_metric_table(
        lines,
        title="Line Limit Consistency (proxy vs OPF)",
        regime_order=regime_order,
        summaries=limit_consistency_summaries,
        metrics=[
            ("n_lines_checked", "Lines checked", "d"),
            ("n_limit_mismatch", "Lines with mismatch", "d"),
            ("max_abs_limit_diff_mva", "Max abs limit diff (MVA)", ".6f"),
            ("max_rel_limit_diff_pct", "Max rel limit diff (%)", ".6f"),
        ],
    )
    _append_metric_table(
        lines,
        title="Post-PF Loading Diagnostics (current-based)",
        regime_order=regime_order,
        summaries=constraint_summaries,
        metrics=[
            ("max_line_loading_pct", "Max line loading (%)", ".2f"),
            ("min_line_loading_headroom_pct", "Min line loading headroom (%)", ".2f"),
            ("max_trafo_loading_pct", "Max trafo loading (%)", ".2f"),
            ("min_trafo_loading_headroom_pct", "Min trafo loading headroom (%)", ".2f"),
        ],
    )
    _append_metric_table(
        lines,
        title="AC L2 Stability Radius (constrained lines only)",
        regime_order=regime_order,
        summaries=radius_summaries,
        metrics=[
            ("n_constrained", "Constrained lines", "d"),
            ("radius_min", "Min radius", ".4f"),
            ("radius_median", "Median radius", ".4f"),
            ("radius_mean", "Mean radius", ".4f"),
            ("loading_ratio_mean", "Mean loading ratio", ".4f"),
            ("loading_ratio_max", "Max loading ratio", ".4f"),
        ],
    )
    _append_metric_table(
        lines,
        title="AC N-1 Stability Radius (min over contingencies)",
        regime_order=regime_order,
        summaries=ac_n1_radius_summaries,
        metrics=[
            ("n_lines", "Lines with N-1 radius", "d"),
            ("n_already_n1_infeasible", "Lines already N-1 infeasible", "d"),
            ("ac_n1_radius_min", "Min AC N-1 radius (MW)", ".4f"),
            ("ac_n1_radius_median", "Median AC N-1 radius (MW)", ".4f"),
            ("ac_n1_radius_p10", "P10 AC N-1 radius (MW)", ".4f"),
        ],
    )
    _append_metric_table(
        lines,
        title="AC Sigma-Radius and Proxy Headroom",
        regime_order=regime_order,
        summaries=sigma_summaries,
        metrics=[
            ("sigma_radius_min", "Min sigma-radius", ".4f"),
            ("sigma_radius_median", "Median sigma-radius", ".4f"),
            ("sigma_radius_p10", "P10 sigma-radius", ".4f"),
            ("max_overload_prob", "Max overload probability", ".6f"),
            ("mean_overload_prob", "Mean overload probability", ".6f"),
            ("n_prob_above_1pct", "Lines with P(overload) > 1%", "d"),
            ("n_prob_above_5pct", "Lines with P(overload) > 5%", "d"),
            ("max_cantelli_ub", "Max Cantelli upper bound", ".6f"),
            ("pi_system", "System performance index", ".4f"),
            ("pi_max", "Max line performance index", ".4f"),
            ("min_headroom_mva", "Min headroom vs MVA proxy", ".4f"),
        ],
    )
    _append_metric_table(
        lines,
        title="DC N-1 Effective Radius",
        regime_order=regime_order,
        summaries=dc_n1_summaries,
        metrics=[
            ("dc_n1_radius_min", "Min N-1 effective radius", ".4f"),
            ("dc_n1_radius_median", "Median N-1 effective radius", ".4f"),
        ],
    )
    _append_metric_table(
        lines,
        title="AC N-1 Screening",
        regime_order=regime_order,
        summaries=screen_summaries,
        metrics=[
            ("n1_pass", "N-1 passed", "d"),
            ("n1_fail", "N-1 failed (overloads)", "d"),
            ("n1_diverged", "N-1 diverged", "d"),
            ("n1_pass_rate_pct", "N-1 pass rate (%)", ".2f"),
            ("max_overloads_in_contingency", "Max overloads in any N-1", "d"),
        ],
    )

    lines += ["", "--- Worst-Case Perturbation Verification (Cost OPF) ---"]
    if verify.get("verified"):
        lines += [
            f"  Worst constrained line:    {verify.get('worst_line')}",
            f"  Stability radius:          {float(verify.get('radius', float('nan'))):.4f}",
            f"  PF converged after perturbation: {verify.get('pf_converged')}",
            f"  Target line loading:       {verify.get('target_line_loading_pct', 'N/A')}%",
            f"  Total overloaded lines:    {verify.get('total_overloaded_lines', 'N/A')}",
            "  => VERIFIED: stability radius h* perturbation triggers overload.",
        ]
    elif verify.get("reason"):
        lines += [f"  Not verified: {verify['reason']}"]
    else:
        loading = verify.get("target_line_loading_pct")
        lines += [
            f"  Worst constrained line:    {verify.get('worst_line')}",
            f"  Stability radius:          {float(verify.get('radius', float('nan'))):.4f}",
            f"  Target line loading after perturbation: "
            f"{f'{float(loading):.1f}%' if loading is not None else 'N/A'}",
            "  => NOTE: balanced-norm h* direction did not trigger target overload.",
        ]

    if any(
        float(
            limit_consistency_summaries.get(regime_key, {}).get(
                "n_limit_mismatch", float("nan")
            )
        )
        > 0.0
        for regime_key, _ in regime_order
    ):
        lines += [
            "",
            "Warning:",
            "  The stability-radius proxy branch limit does not match the OPF branch limit on at least one line.",
            "  Interpret headroom- and radius-based metrics with caution for that regime.",
        ]

    if any(
        float(sigma_summaries.get(regime_key, {}).get("min_headroom_mva", float("nan")))
        < 0.0
        or float(
            constraint_summaries.get(regime_key, {}).get(
                "min_line_loading_headroom_pct", float("nan")
            )
        )
        < 0.0
        for regime_key, _ in regime_order
    ):
        lines += [
            "",
            "Note:",
            "  `min_headroom_mva` is the stability-radius MVA proxy margin (`S_limit_proxy - |S|`).",
            "  Pandapower AC OPF in this demo uses apparent-power branch limits (`OPF_FLOW_LIM=0`).",
            "  The loading table above is a post-PF current-based diagnostic (`loading_percent`).",
            "  Negative proxy headroom or slightly negative loading headroom can therefore coexist with a converged AC OPF point.",
        ]

    lines += ["", "=" * 70, ""]
    return "\n".join(lines)


def _write_comparison(
    *,
    output_path: Path,
    regime_order: list[tuple[str, str]],
    dispatch_summaries: Mapping[str, Mapping[str, object]],
    limit_consistency_summaries: Mapping[str, Mapping[str, object]],
    constraint_summaries: Mapping[str, Mapping[str, object]],
    radius_summaries: Mapping[str, Mapping[str, object]],
    ac_n1_radius_summaries: Mapping[str, Mapping[str, object]],
    sigma_summaries: Mapping[str, Mapping[str, object]],
    dc_n1_summaries: Mapping[str, Mapping[str, object]],
    screen_summaries: Mapping[str, Mapping[str, object]],
    verify: Mapping[str, object],
    r_target: float,
    sigma_p_mw: float,
    sigma_q_mvar: float,
) -> None:
    """Internal helper for module-local processing."""
    text = _build_comparison_text(
        regime_order=regime_order,
        dispatch_summaries=dispatch_summaries,
        limit_consistency_summaries=limit_consistency_summaries,
        constraint_summaries=constraint_summaries,
        radius_summaries=radius_summaries,
        ac_n1_radius_summaries=ac_n1_radius_summaries,
        sigma_summaries=sigma_summaries,
        dc_n1_summaries=dc_n1_summaries,
        screen_summaries=screen_summaries,
        verify=verify,
        r_target=r_target,
        sigma_p_mw=sigma_p_mw,
        sigma_q_mvar=sigma_q_mvar,
    )
    print(text)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Summary saved: %s", output_path)


def _plot_multi_regime_radius_cdf(
    regime_results: Mapping[str, tuple[str, dict]],
    output_path: Path,
) -> None:
    """Internal helper for module-local processing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not available; skipping %s", output_path.name)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for color_idx, (regime_key, (display_name, results)) in enumerate(
        regime_results.items()
    ):
        radii = sorted(
            float(v["radius_ac_l2"])
            for v in results.values()
            if not v.get("is_unconstrained", False)
            and math.isfinite(v.get("radius_ac_l2", float("nan")))
        )
        if not radii:
            continue
        plotted = True
        ax.plot(
            radii,
            np.arange(1, len(radii) + 1) / len(radii),
            label=display_name,
            linewidth=2,
            color=f"C{color_idx}",
        )
    if not plotted:
        logger.warning("No AC L2 radius data available for %s", output_path.name)
        plt.close(fig)
        return

    ax.set_xlabel("AC L2 Stability Radius")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative Distribution of AC L2 Stability Radius")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_multi_regime_ac_n1_radius_cdf(
    regime_n1_radii: Mapping[str, tuple[str, dict[int, float]]],
    output_path: Path,
) -> None:
    """Internal helper for module-local processing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not available; skipping %s", output_path.name)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for color_idx, (_, (display_name, n1_radii)) in enumerate(regime_n1_radii.items()):
        radii = sorted(float(v) for v in n1_radii.values() if math.isfinite(v))
        if not radii:
            continue
        plotted = True
        ax.plot(
            radii,
            np.arange(1, len(radii) + 1) / len(radii),
            label=display_name,
            linewidth=2,
            color=f"C{color_idx}",
        )
    if not plotted:
        logger.warning("No AC N-1 radius data available for %s", output_path.name)
        plt.close(fig)
        return

    ax.set_xlabel("AC N-1 Stability Radius (MW)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of AC N-1 Stability Radius per Line")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_multi_regime_n1_overloads(
    regime_records: Mapping[str, tuple[str, list[dict]]],
    output_path: Path,
) -> None:
    """Internal helper for module-local processing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib is not available; skipping %s", output_path.name)
        return

    frames: list[tuple[str, str, "DataFrame"]] = []
    for regime_key, (display_name, records) in regime_records.items():
        if not records:
            continue
        frames.append(
            (
                regime_key,
                display_name,
                pd.DataFrame(records).set_index("contingency_line"),
            )
        )
    if not frames:
        logger.warning("No AC N-1 screening data available for %s", output_path.name)
        return

    common = sorted(set.intersection(*(set(frame.index) for _, _, frame in frames)))
    if not common:
        logger.warning("No common contingencies available for %s", output_path.name)
        return

    contingency_ids = list(common)
    overload_matrix = []
    loading_matrix = []
    for _, _, frame in frames:
        overload_matrix.append(
            frame.loc[common, "n_overloads"].clip(lower=0).to_numpy(dtype=float)
        )
        if "max_loading_percent" in frame.columns:
            loading_matrix.append(
                frame.loc[common, "max_loading_percent"].to_numpy(dtype=float)
            )
        else:
            loading_matrix.append(np.full(len(common), float("nan")))

    overload_arr = np.vstack(overload_matrix)
    loading_arr = np.vstack(loading_matrix)
    worst_score = np.nanmax(
        np.nan_to_num(overload_arr, nan=0.0), axis=0
    ) * 1000.0 + np.nanmax(
        np.nan_to_num(loading_arr, nan=0.0),
        axis=0,
    )
    top_k = min(10, len(contingency_ids))
    top_idx = np.argsort(worst_score)[::-1][:top_k]
    top_contingencies = [contingency_ids[int(i)] for i in top_idx]
    max_overload_count = float(np.nanmax(np.nan_to_num(overload_arr, nan=0.0)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_counts, ax_loading, ax_topk, ax_status = axes.flatten()

    for color_idx, (_, display_name, frame) in enumerate(frames):
        overload_values = (
            frame.loc[common, "n_overloads"].clip(lower=0).to_numpy(dtype=float)
        )
        sorted_overloads = np.sort(overload_values)[::-1]
        ax_counts.step(
            np.arange(1, len(sorted_overloads) + 1),
            sorted_overloads,
            where="mid",
            linewidth=2.0,
            color=f"C{color_idx}",
            label=display_name,
        )

        if "max_loading_percent" in frame.columns:
            loading_values = np.nan_to_num(
                frame.loc[common, "max_loading_percent"].to_numpy(dtype=float),
                nan=0.0,
            )
            sorted_loading = np.sort(loading_values)[::-1]
            ax_loading.step(
                np.arange(1, len(sorted_loading) + 1),
                sorted_loading,
                where="mid",
                linewidth=2.0,
                color=f"C{color_idx}",
                label=display_name,
            )

    ax_counts.set_title("Sorted Overload Count per Contingency")
    ax_counts.set_xlabel("Contingency rank")
    ax_counts.set_ylabel("Overloaded lines")
    ax_counts.grid(True, alpha=0.25)
    ax_counts.legend(fontsize=9)

    ax_loading.axhline(
        100.0, color="tab:red", linestyle="--", linewidth=1.2, label="100% loading"
    )
    ax_loading.set_title("Sorted Peak Loading per Contingency")
    ax_loading.set_xlabel("Contingency rank")
    ax_loading.set_ylabel("Peak loading (%)")
    ax_loading.grid(True, alpha=0.25)
    finite_loading = np.asarray(loading_arr[np.isfinite(loading_arr)], dtype=float)
    if finite_loading.size:
        lower = max(0.0, min(95.0, float(np.floor(finite_loading.min() - 5.0))))
        upper = max(105.0, float(np.ceil(finite_loading.max() + 5.0)))
        if upper > lower:
            ax_loading.set_ylim(lower, upper)
    ax_loading.legend(fontsize=9)

    width = 0.8 / len(frames)
    x = np.arange(len(top_contingencies))
    if max_overload_count <= 0.0:
        for idx, (_, display_name, frame) in enumerate(frames):
            values = np.nan_to_num(
                frame.loc[top_contingencies, "max_loading_percent"].to_numpy(
                    dtype=float
                ),
                nan=0.0,
            )
            offset = (idx - (len(frames) - 1) / 2.0) * width
            ax_topk.bar(
                x + offset,
                values,
                width,
                alpha=0.9,
                color=f"C{idx}",
                label=display_name,
            )
        ax_topk.axhline(100.0, color="tab:red", linestyle="--", linewidth=1.2)
        ax_topk.set_title("Top Contingencies by Peak Loading")
        ax_topk.set_ylabel("Peak loading (%)")
    else:
        for idx, (_, display_name, frame) in enumerate(frames):
            values = (
                frame.loc[top_contingencies, "n_overloads"]
                .clip(lower=0)
                .to_numpy(dtype=float)
            )
            offset = (idx - (len(frames) - 1) / 2.0) * width
            ax_topk.bar(
                x + offset,
                values,
                width,
                alpha=0.9,
                color=f"C{idx}",
                label=display_name,
            )
        ax_topk.set_title("Top Contingencies by Screening Severity")
        ax_topk.set_ylabel("Overloaded lines")
    ax_topk.set_xlabel("Outaged line")
    ax_topk.set_xticks(x)
    ax_topk.set_xticklabels(
        [str(v) for v in top_contingencies], rotation=45, ha="right", fontsize=8
    )
    ax_topk.grid(True, axis="y", alpha=0.25)
    ax_topk.legend(fontsize=9)

    y = np.arange(len(frames))
    pass_share = []
    fail_share = []
    diverged_share = []
    labels = []
    for _, display_name, frame in frames:
        total = len(common)
        passed = int(frame.loc[common, "n1_feasible"].fillna(False).astype(bool).sum())
        diverged = int(
            (~frame.loc[common, "pf_converged"].fillna(False).astype(bool)).sum()
        )
        failed = max(total - passed - diverged, 0)
        labels.append(display_name)
        pass_share.append(100.0 * passed / total if total else 0.0)
        fail_share.append(100.0 * failed / total if total else 0.0)
        diverged_share.append(100.0 * diverged / total if total else 0.0)

    ax_status.barh(y, pass_share, color="#2E8B57", alpha=0.9, label="Pass")
    ax_status.barh(
        y,
        fail_share,
        left=pass_share,
        color="#D95F02",
        alpha=0.9,
        label="Overload fail",
    )
    ax_status.barh(
        y,
        diverged_share,
        left=np.asarray(pass_share) + np.asarray(fail_share),
        color="#7570B3",
        alpha=0.9,
        label="Diverged",
    )
    ax_status.set_yticks(y)
    ax_status.set_yticklabels(labels)
    ax_status.set_xlim(0.0, 100.0)
    ax_status.set_xlabel("Contingency share (%)")
    ax_status.set_title("AC N-1 Screening Outcome Mix")
    ax_status.grid(True, axis="x", alpha=0.25)
    ax_status.legend(fontsize=9, loc="lower right")

    fig.suptitle("AC N-1 Screening Summary", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_cost_security_tradeoff(
    *,
    regime_order: list[tuple[str, str]],
    dispatch_summaries: Mapping[str, Mapping[str, object]],
    screen_summaries: Mapping[str, Mapping[str, object]],
    ac_n1_radius_summaries: Mapping[str, Mapping[str, object]],
    output_path: Path,
) -> None:
    """Internal helper for module-local processing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not available; skipping %s", output_path.name)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for color_idx, (regime_key, display_name) in enumerate(regime_order):
        cost_increase = float(
            dispatch_summaries.get(regime_key, {}).get(
                "cost_increase_pct", float("nan")
            )
        )
        pass_rate = float(
            screen_summaries.get(regime_key, {}).get("n1_pass_rate_pct", float("nan"))
        )
        min_n1_radius = float(
            ac_n1_radius_summaries.get(regime_key, {}).get(
                "ac_n1_radius_min", float("nan")
            )
        )
        axes[0].scatter(cost_increase, pass_rate, s=90, color=f"C{color_idx}")
        axes[0].annotate(
            display_name,
            (cost_increase, pass_rate),
            textcoords="offset points",
            xytext=(5, 5),
        )
        axes[1].scatter(cost_increase, min_n1_radius, s=90, color=f"C{color_idx}")
        axes[1].annotate(
            display_name,
            (cost_increase, min_n1_radius),
            textcoords="offset points",
            xytext=(5, 5),
        )

    axes[0].set_title("Cost Increase vs AC N-1 Pass Rate")
    axes[0].set_xlabel("Cost increase vs Cost OPF (%)")
    axes[0].set_ylabel("AC N-1 pass rate (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Cost Increase vs Min AC N-1 Radius")
    axes[1].set_xlabel("Cost increase vs Cost OPF (%)")
    axes[1].set_ylabel("Min AC N-1 radius (MW)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _parse_args() -> argparse.Namespace:
    """Internal helper for module-local processing."""
    parser = argparse.ArgumentParser(
        description="N-1 Stability Demo: Cost OPF vs Radius OPF vs SCOPF comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to MATPOWER .m file")
    parser.add_argument(
        "--output-dir",
        default="n1_demo",
        help="Artifact directory name; non-artifact paths are normalized under run_artifacts/n1_stability_demo/",
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
        "--scopf-iter",
        type=int,
        default=3,
        help="Number of screening-based SCOPF tightening iterations",
    )
    parser.add_argument(
        "--sigma-p",
        type=float,
        default=5.0,
        help="Per-bus P injection std dev (MW) for sigma-radius/probability",
    )
    parser.add_argument(
        "--sigma-q",
        type=float,
        default=2.0,
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
    """Run the command-line entry point."""
    args = _parse_args()
    out_dir = _resolve_output_dir(args.output_dir)
    _setup_logging(args.verbose, log_file=out_dir / "debug.log")
    logger.info("Output directory: %s", out_dir.resolve())

    regime_order = [
        ("cost_opf", "Cost OPF"),
        ("radius_opf", "Radius OPF"),
        ("scopf", "SCOPF"),
    ]

    # Phase 1: Load
    net_lossless, slack_bus, line_indices = _load_and_prepare(
        args.input, args.slack_bus
    )

    # Phase 2: True cost-minimising AC OPF (baseline)
    nn_cost, base_pf_cost, cost_opf_eur_h = _solve_cost_opf(
        net_lossless,
        line_indices,
        input_path=args.input,
        max_loading_percent=99.0,
        label="cost_opf",
    )
    cost_results, cost_h_vectors = _compute_radii(
        net_lossless, base_pf_cost, slack_bus, label="cost_opf"
    )

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

    neg_radius_lines = [
        k
        for k, v in cost_results.items()
        if not v.get("is_unconstrained", False)
        and float(v.get("radius_ac_l2", 0.0)) < 0
    ]
    if neg_radius_lines:
        logger.warning(
            "[cost_opf] %d lines have NEGATIVE radius (proxy limit exceeded): %s",
            len(neg_radius_lines),
            neg_radius_lines[:5],
        )

    r_target = float(args.r_target)
    if r_target <= 0.0:
        positive_radii = [
            v["radius_ac_l2"]
            for v in cost_results.values()
            if not v.get("is_unconstrained", False)
            and math.isfinite(v.get("radius_ac_l2", float("nan")))
            and v["radius_ac_l2"] > 0
        ]
        if positive_radii:
            r_target = 10.0 * float(min(positive_radii))
            logger.info("Auto r_target = 10 * min_positive_radius = %.4f MW", r_target)
        else:
            r_target = 5.0

    # Phase 3: Stability-radius-guided OPF
    nn_radius, base_pf_radius, radius_results, radius_h_vectors, radius_opf_eur_h = (
        _solve_radius_opf(
            net_lossless,
            slack_bus,
            line_indices,
            cost_results,
            r_target=r_target,
            n_iter=args.n_iter,
            input_path=args.input,
        )
    )
    if base_pf_radius is None:
        logger.warning(
            "Radius OPF produced no result; falling back to cost OPF base point."
        )
        nn_radius = nn_cost
        base_pf_radius = base_pf_cost
        radius_results = cost_results
        radius_h_vectors = cost_h_vectors
        radius_opf_eur_h = cost_opf_eur_h

    # Phase 4: Screening-based SCOPF
    (
        nn_scopf,
        base_pf_scopf,
        scopf_results,
        scopf_h_vectors,
        scopf_opf_eur_h,
        scopf_n1_records,
    ) = _solve_scopf(
        net_lossless,
        slack_bus,
        line_indices,
        input_path=args.input,
        n_iter=args.scopf_iter,
    )

    regimes: dict[str, dict[str, object]] = {
        "cost_opf": {
            "display_name": "Cost OPF",
            "nn": nn_cost,
            "base_pf": base_pf_cost,
            "results": cost_results,
            "h_vectors": cost_h_vectors,
            "total_cost_eur_h": cost_opf_eur_h,
            "n1_records": [],
            "dc_n1_results": {},
            "ac_n1_radii": {},
        },
        "radius_opf": {
            "display_name": "Radius OPF",
            "nn": nn_radius,
            "base_pf": base_pf_radius,
            "results": radius_results,
            "h_vectors": radius_h_vectors,
            "total_cost_eur_h": radius_opf_eur_h,
            "n1_records": [],
            "dc_n1_results": {},
            "ac_n1_radii": {},
        },
        "scopf": {
            "display_name": "SCOPF",
            "nn": nn_scopf,
            "base_pf": base_pf_scopf,
            "results": scopf_results,
            "h_vectors": scopf_h_vectors,
            "total_cost_eur_h": scopf_opf_eur_h,
            "n1_records": scopf_n1_records,
            "dc_n1_results": {},
            "ac_n1_radii": {},
        },
    }

    # Phase 5: DC N-1 radii
    if not args.skip_dc_n1:
        for regime_key, _ in regime_order:
            regime = regimes[regime_key]
            regime["dc_n1_results"] = _dc_n1_radii(
                net_lossless,
                regime["base_pf"],
                slack_bus,
                line_indices,
                regime_key,
            )
            if regime["dc_n1_results"]:
                _save_csv(
                    _dc_n1_to_df(regime["dc_n1_results"]),
                    out_dir / f"dc_n1_{regime_key}.csv",
                    f"DC N-1 ({regime_key})",
                )

    # Phase 6: AC N-1 stability radius
    if not args.skip_ac_n1_radius:
        import pandas as pd

        for regime_key, _ in regime_order:
            regime = regimes[regime_key]
            regime["ac_n1_radii"] = _compute_ac_n1_radii(
                net_lossless,
                regime["nn"],
                slack_bus,
                line_indices,
                regime_key,
            )
            if regime["ac_n1_radii"]:
                pd.DataFrame(
                    [
                        {"line_id": lid, "ac_n1_radius": value}
                        for lid, value in regime["ac_n1_radii"].items()
                    ]
                ).set_index("line_id").sort_index().to_csv(
                    out_dir / f"ac_n1_radii_{regime_key}.csv"
                )

    # Phase 7: AC N-1 screening
    if not args.skip_n1_screening:
        regimes["cost_opf"]["n1_records"] = _ac_n1_screen(nn_cost, "cost_opf")
        regimes["radius_opf"]["n1_records"] = _ac_n1_screen(nn_radius, "radius_opf")
    for regime_key, _ in regime_order:
        records = list(regimes[regime_key]["n1_records"])
        if records:
            _save_n1_csv(
                records,
                out_dir / f"n1_screening_{regime_key}.csv",
                f"AC N-1 {regime_key}",
            )

    # Save per-line radii
    for regime_key, _ in regime_order:
        _save_csv(
            _radii_to_df(regimes[regime_key]["results"]),
            out_dir / f"{regime_key}_radii.csv",
            f"{regime_key} radii",
        )
        limit_df = _opf_line_limit_consistency_df(regimes[regime_key]["nn"])
        if not limit_df.empty:
            _save_csv(
                limit_df,
                out_dir / f"opf_line_limit_consistency_{regime_key}.csv",
                f"{regime_key} limit consistency",
            )

    # Phase 8: AC sigma-radius and summaries
    sigma_p_mw = float(args.sigma_p)
    sigma_q_mvar = float(args.sigma_q)
    dispatch_summaries: dict[str, dict[str, object]] = {}
    limit_consistency_summaries: dict[str, dict[str, object]] = {}
    constraint_summaries: dict[str, dict[str, object]] = {}
    radius_summaries: dict[str, dict[str, object]] = {}
    ac_n1_radius_summaries: dict[str, dict[str, object]] = {}
    sigma_summaries: dict[str, dict[str, object]] = {}
    dc_n1_summaries: dict[str, dict[str, object]] = {}
    screen_summaries: dict[str, dict[str, object]] = {}

    baseline_cost = float(regimes["cost_opf"]["total_cost_eur_h"])
    for regime_key, _ in regime_order:
        regime = regimes[regime_key]
        total_cost = float(regime["total_cost_eur_h"])
        dispatch_summaries[regime_key] = {
            "total_cost_eur_h": total_cost,
            "generation_dispatch_mw": _total_generation_dispatch_mw(regime["nn"]),
            "cost_increase_pct": (
                100.0 * (total_cost - baseline_cost) / max(baseline_cost, 1.0)
                if math.isfinite(total_cost) and math.isfinite(baseline_cost)
                else float("nan")
            ),
        }
        limit_consistency_summaries[regime_key] = _opf_line_limit_consistency_summary(
            regime["nn"], regime_key
        )
        constraint_summaries[regime_key] = _opf_constraint_summary(
            regime["nn"], regime_key
        )
        radius_summaries[regime_key] = _summary_stats(regime["results"], regime_key)
        ac_n1_radius_summaries[regime_key] = _ac_n1_radius_summary(
            regime["ac_n1_radii"], regime_key
        )
        sigma_summaries[regime_key] = _compute_sigma_and_baselines(
            regime["results"],
            regime["h_vectors"],
            line_indices,
            sigma_p_mw=sigma_p_mw,
            sigma_q_mvar=sigma_q_mvar,
            label=regime_key,
        )
        dc_n1_summaries[regime_key] = _dc_n1_summary(
            regime["dc_n1_results"], regime_key
        )
        screen_summaries[regime_key] = _n1_summary(regime["n1_records"], regime_key)

    _write_comparison(
        output_path=out_dir / "comparison_summary.txt",
        regime_order=regime_order,
        dispatch_summaries=dispatch_summaries,
        limit_consistency_summaries=limit_consistency_summaries,
        constraint_summaries=constraint_summaries,
        radius_summaries=radius_summaries,
        ac_n1_radius_summaries=ac_n1_radius_summaries,
        sigma_summaries=sigma_summaries,
        dc_n1_summaries=dc_n1_summaries,
        screen_summaries=screen_summaries,
        verify=verify_result,
        r_target=r_target,
        sigma_p_mw=sigma_p_mw,
        sigma_q_mvar=sigma_q_mvar,
    )

    _plot_multi_regime_radius_cdf(
        {
            regime_key: (
                regimes[regime_key]["display_name"],
                regimes[regime_key]["results"],
            )
            for regime_key, _ in regime_order
        },
        out_dir / "plot_radius_cdf.png",
    )
    _plot_multi_regime_ac_n1_radius_cdf(
        {
            regime_key: (
                regimes[regime_key]["display_name"],
                regimes[regime_key]["ac_n1_radii"],
            )
            for regime_key, _ in regime_order
        },
        out_dir / "plot_ac_n1_radius_cdf.png",
    )
    _plot_multi_regime_n1_overloads(
        {
            regime_key: (
                regimes[regime_key]["display_name"],
                regimes[regime_key]["n1_records"],
            )
            for regime_key, _ in regime_order
        },
        out_dir / "plot_n1_overloads.png",
    )
    _plot_cost_security_tradeoff(
        regime_order=regime_order,
        dispatch_summaries=dispatch_summaries,
        screen_summaries=screen_summaries,
        ac_n1_radius_summaries=ac_n1_radius_summaries,
        output_path=out_dir / "plot_cost_security_tradeoff.png",
    )

    logger.info("Done. All outputs in: %s", out_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
