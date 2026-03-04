"""AC Feasible Power Flow (AC FPF) via pandapower.runopp().

This module solves an AC OPF **feasibility** problem: instead of minimizing
generation cost, it finds the closest feasible operating point to an initial
dispatch guess.

Objective
---------
    min  Sum_i (P_{g,i} - P_{g,i}^0)^2

    s.t. AC power flow equations         (equality)
         P_g^min <= P_g <= P_g^max       (generator active limits)
         Q_g^min <= Q_g <= Q_g^max       (generator reactive limits)
         V^min   <= V   <= V^max         (bus voltage limits)
         |S_ij|  <= S_ij^max             (line thermal limits)

The quadratic cost (P - P0)^2 = P^2 - 2*P0*P + P0^2 is implemented via
pandapower's polynomial cost interface:
    cp2 = 1,  cp1 = -2*P0,  cp0 = P0^2

Solver backend
--------------
pandapower.runopp() uses PYPOWER's interior-point solver (PIPS) by default.
No additional NLP solver (IPOPT, Pyomo) is required.

Return type
-----------
Returns ``PyPSAAPFResult`` — the same type used by the AC PF solver — so
downstream code (AC certificate, AC feasibility check, DC base point from
ACPF injections) works without modification.
"""

from __future__ import annotations

import copy
import logging
import math
import time as _time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from stability_radius.base_point.pandapower_tools import (
    apply_lossless_policy_to_pandapower_net,
    ensure_ext_grid_at_slack,
    resolve_slack_bus_id,
)
from stability_radius.base_point.pypsa_pf import PyPSAAPFResult
from stability_radius.config import DEFAULT_OPF, OPFConfig
from stability_radius.pp_helpers import is_in_service as _is_in_service

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ACFPFConfig:
    """Configuration for the AC Feasible Power Flow solver.

    Parameters
    ----------
    pg0_source : str
        Source for the initial dispatch guess Pg0:
        - ``"case"``: use dispatch from the input case (net.gen.p_mw)
        - ``"midpoint"``: use (min_p_mw + max_p_mw) / 2 for each generator
    vm_min_pu : float
        Minimum bus voltage magnitude (p.u.) for OPF constraint.
    vm_max_pu : float
        Maximum bus voltage magnitude (p.u.) for OPF constraint.
    max_iteration : int
        Maximum PDIPM interior-point iterations for pandapower.runopp().
    max_loading_percent : float
        Line loading limit (%) for OPF.  Set below 100 to compensate for
        PIPS solver tolerance so the solution satisfies the true 100% limit.
    pdipm_feastol : float
        Feasibility (equality) tolerance for PDIPM.
        0 means use OPF_VIOLATION default (5e-6).
    pdipm_gradtol : float
        Gradient tolerance for PDIPM.
    pdipm_comptol : float
        Complementary condition (inequality) tolerance for PDIPM.
    pdipm_costtol : float
        Optimality tolerance for PDIPM.
    opf_violation : float
        Constraint violation tolerance (also used as PDIPM_FEASTOL default).
    init : str
        Power flow initialization for runopp: "dc" or "flat".
    """

    pg0_source: str = "case"
    vm_min_pu: float = 0.9
    vm_max_pu: float = 1.1
    max_iteration: int = 300
    max_loading_percent: float = 99.0
    pdipm_feastol: float = 1e-4
    pdipm_gradtol: float = 1e-4
    pdipm_comptol: float = 1e-4
    pdipm_costtol: float = 1e-4
    opf_violation: float = 1e-4
    init: str = "dc"


def _determine_pg0(
    row: Any,
    *,
    pg0_source: str,
) -> float:
    """Determine the initial dispatch guess Pg0 for a single generator/sgen/ext_grid."""
    if pg0_source == "midpoint":
        p_min = float(row.get("min_p_mw", 0.0))
        p_max = float(row.get("max_p_mw", 0.0))
        if not math.isfinite(p_min):
            p_min = 0.0
        if not math.isfinite(p_max) or p_max <= 0.0:
            p_max = max(p_min, 0.0)
        return (p_min + p_max) / 2.0

    # pg0_source == "case": use current dispatch
    p = float(row.get("p_mw", 0.0))
    if not math.isfinite(p):
        p = 0.0
    return p


def _setup_gen_for_opp(
    nn: Any,
    *,
    pg0_source: str,
) -> dict[str, float]:
    """Set controllable flags, bounds, and costs on generators.

    Returns a dict of {gen_name: Pg0_mw} for metadata.
    """
    import pandapower as pp

    pg0_map: dict[str, float] = {}

    # ---- net.gen ----
    if hasattr(nn, "gen") and nn.gen is not None and len(nn.gen):
        if "controllable" not in nn.gen.columns:
            nn.gen["controllable"] = False
        for gid in [int(x) for x in sorted(nn.gen.index)]:
            row = nn.gen.loc[gid]
            if not _is_in_service(row):
                continue

            nn.gen.at[gid, "controllable"] = True

            p_min = float(row.get("min_p_mw", 0.0))
            p_max = float(row.get("max_p_mw", 0.0))
            if not math.isfinite(p_min):
                p_min = 0.0
            if not math.isfinite(p_max) or p_max <= 0.0:
                p_max = max(p_min + 1.0, 100.0)

            nn.gen.at[gid, "min_p_mw"] = p_min
            nn.gen.at[gid, "max_p_mw"] = p_max

            q_min = float(row.get("min_q_mvar", -999.0))
            q_max = float(row.get("max_q_mvar", 999.0))
            if not math.isfinite(q_min):
                q_min = -999.0
            if not math.isfinite(q_max):
                q_max = 999.0
            nn.gen.at[gid, "min_q_mvar"] = q_min
            nn.gen.at[gid, "max_q_mvar"] = q_max

            pg0 = _determine_pg0(row, pg0_source=pg0_source)
            name = f"gen_{gid}"
            pg0_map[name] = pg0

            pp.create_poly_cost(
                nn,
                gid,
                "gen",
                cp2_eur_per_mw2=1.0,
                cp1_eur_per_mw=-2.0 * pg0,
                cp0_eur=pg0 * pg0,
            )

    # ---- net.sgen ----
    if hasattr(nn, "sgen") and nn.sgen is not None and len(nn.sgen):
        if "controllable" not in nn.sgen.columns:
            nn.sgen["controllable"] = False
        for sid in [int(x) for x in sorted(nn.sgen.index)]:
            row = nn.sgen.loc[sid]
            if not _is_in_service(row):
                continue

            nn.sgen.at[sid, "controllable"] = True

            p_min = float(row.get("min_p_mw", 0.0))
            p_max = float(row.get("max_p_mw", 0.0))
            if not math.isfinite(p_min):
                p_min = 0.0
            if not math.isfinite(p_max) or p_max <= 0.0:
                p_max = max(p_min + 1.0, 100.0)

            nn.sgen.at[sid, "min_p_mw"] = p_min
            nn.sgen.at[sid, "max_p_mw"] = p_max

            q_min = float(row.get("min_q_mvar", -999.0))
            q_max = float(row.get("max_q_mvar", 999.0))
            if not math.isfinite(q_min):
                q_min = -999.0
            if not math.isfinite(q_max):
                q_max = 999.0
            nn.sgen.at[sid, "min_q_mvar"] = q_min
            nn.sgen.at[sid, "max_q_mvar"] = q_max

            pg0 = _determine_pg0(row, pg0_source=pg0_source)
            name = f"sgen_{sid}"
            pg0_map[name] = pg0

            pp.create_poly_cost(
                nn,
                sid,
                "sgen",
                cp2_eur_per_mw2=1.0,
                cp1_eur_per_mw=-2.0 * pg0,
                cp0_eur=pg0 * pg0,
            )

    # ---- net.ext_grid ----
    if hasattr(nn, "ext_grid") and nn.ext_grid is not None and len(nn.ext_grid):
        if "controllable" not in nn.ext_grid.columns:
            nn.ext_grid["controllable"] = False

        total_load = 0.0
        if hasattr(nn, "load") and nn.load is not None and len(nn.load):
            total_load = float(nn.load["p_mw"].sum())

        for eid in [int(x) for x in sorted(nn.ext_grid.index)]:
            row = nn.ext_grid.loc[eid]
            if not _is_in_service(row):
                continue

            nn.ext_grid.at[eid, "controllable"] = True

            # Wide P bounds for slack flexibility.
            p_bound = max(1000.0, 2.0 * total_load if total_load > 0 else 1000.0)
            nn.ext_grid.at[eid, "min_p_mw"] = -p_bound
            nn.ext_grid.at[eid, "max_p_mw"] = p_bound
            nn.ext_grid.at[eid, "min_q_mvar"] = -p_bound
            nn.ext_grid.at[eid, "max_q_mvar"] = p_bound

            # Ext_grid target: 0 (minimize slack usage).
            pg0 = 0.0
            name = f"ext_{eid}"
            pg0_map[name] = pg0

            pp.create_poly_cost(
                nn,
                eid,
                "ext_grid",
                cp2_eur_per_mw2=1.0,
                cp1_eur_per_mw=-2.0 * pg0,
                cp0_eur=pg0 * pg0,
            )

    return pg0_map


def _set_voltage_limits(nn: Any, *, vm_min_pu: float, vm_max_pu: float) -> None:
    """Set bus voltage limits for OPF."""
    if hasattr(nn, "bus") and nn.bus is not None and len(nn.bus):
        nn.bus["min_vm_pu"] = vm_min_pu
        nn.bus["max_vm_pu"] = vm_max_pu


def _set_line_thermal_limits(nn: Any) -> None:
    """Ensure line thermal limits are set for OPF.

    pandapower.runopp() enforces ``max_loading_percent`` on lines.
    If not already set, default to 100%.  Lines with zero or missing
    ``max_i_ka`` are given a large surrogate value so the OPF does not
    treat them as zero-capacity.
    """
    if hasattr(nn, "line") and nn.line is not None and len(nn.line):
        if "max_loading_percent" not in nn.line.columns:
            nn.line["max_loading_percent"] = 100.0
        else:
            nn.line["max_loading_percent"] = nn.line["max_loading_percent"].fillna(
                100.0
            )

        # Ensure max_i_ka is set (needed for loading calculation).
        if "max_i_ka" in nn.line.columns:
            mask = nn.line["max_i_ka"].isna() | (nn.line["max_i_ka"] <= 0)
            if mask.any():
                nn.line.loc[mask, "max_i_ka"] = 100.0  # large surrogate
                logger.debug(
                    "AC FPF: set surrogate max_i_ka=100 kA on %d unconstrained lines",
                    int(mask.sum()),
                )

    if hasattr(nn, "trafo") and nn.trafo is not None and len(nn.trafo):
        if "max_loading_percent" not in nn.trafo.columns:
            nn.trafo["max_loading_percent"] = 100.0
        else:
            nn.trafo["max_loading_percent"] = nn.trafo["max_loading_percent"].fillna(
                100.0
            )


def _clear_existing_costs(nn: Any) -> None:
    """Remove any existing cost functions from the network."""
    if hasattr(nn, "poly_cost") and nn.poly_cost is not None and len(nn.poly_cost):
        nn.poly_cost.drop(nn.poly_cost.index, inplace=True)
    if hasattr(nn, "pwl_cost") and nn.pwl_cost is not None and len(nn.pwl_cost):
        nn.pwl_cost.drop(nn.pwl_cost.index, inplace=True)


def solve_ac_fpf(
    *,
    net: Any,
    slack_bus: int,
    line_indices: Sequence[int],
    lossless: bool = False,
    fpf_cfg: ACFPFConfig | None = None,
    opf_cfg: OPFConfig | None = None,
) -> PyPSAAPFResult:
    """Solve AC Feasibility Power Flow via pandapower.runopp().

    Parameters
    ----------
    net : pandapower network
    slack_bus : slack bus id or position
    line_indices : list of line indices to monitor
    lossless : if True, apply lossless policy (r=0)
    fpf_cfg : AC FPF solver configuration
    opf_cfg : OPF config (for unconstrained_line_nom_mw)

    Returns
    -------
    PyPSAAPFResult
        Bus voltages, line flows, bus_p_mw, and solver metadata.
        ``status`` is ``"PP_OPP_OK"`` on success.
    """
    try:
        import pandapower as pp
    except ImportError as e:
        raise ImportError("pandapower is required for AC FPF solver.") from e

    cfg = fpf_cfg if fpf_cfg is not None else ACFPFConfig()
    _opf_cfg = opf_cfg if opf_cfg is not None else DEFAULT_OPF

    pg0_source = str(cfg.pg0_source).strip().lower()
    if pg0_source not in {"case", "midpoint"}:
        raise ValueError("fpf_cfg.pg0_source must be 'case' or 'midpoint'")

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    if not bus_ids:
        raise ValueError("Network has no buses.")

    slack_bus_id = resolve_slack_bus_id(net, int(slack_bus))
    ensure_ext_grid_at_slack(net, int(slack_bus_id))

    # ---- Prepare network copy ----
    nn = (
        apply_lossless_policy_to_pandapower_net(net)
        if bool(lossless)
        else copy.deepcopy(net)
    )

    n_buses = int(len(nn.bus)) if hasattr(nn, "bus") and nn.bus is not None else 0
    n_lines = int(len(nn.line)) if hasattr(nn, "line") and nn.line is not None else 0

    logger.info(
        "AC FPF: preparing OPP (buses=%d, lines=%d, lossless=%s, "
        "pg0_source=%s, vm_bounds=[%.2f, %.2f], max_iter=%d)",
        n_buses,
        n_lines,
        bool(lossless),
        pg0_source,
        cfg.vm_min_pu,
        cfg.vm_max_pu,
        cfg.max_iteration,
    )

    # ---- Setup constraints ----
    _clear_existing_costs(nn)
    _set_voltage_limits(nn, vm_min_pu=cfg.vm_min_pu, vm_max_pu=cfg.vm_max_pu)
    _set_line_thermal_limits(nn)

    # Tighten line loading to compensate for PIPS solver tolerance.
    # Without this, the solver may overshoot the true limit by ~0.5-1%.
    if cfg.max_loading_percent < 100.0:
        if hasattr(nn, "line") and nn.line is not None and len(nn.line):
            nn.line["max_loading_percent"] = cfg.max_loading_percent
        if hasattr(nn, "trafo") and nn.trafo is not None and len(nn.trafo):
            nn.trafo["max_loading_percent"] = cfg.max_loading_percent

    # ---- Setup generators with costs ----
    pg0_map = _setup_gen_for_opp(nn, pg0_source=pg0_source)
    logger.info(
        "AC FPF: configured %d generators/ext_grid with quadratic feasibility costs",
        len(pg0_map),
    )

    # ---- Solve with retry cascade ----
    pf_attempt: str = "primary"
    pf_repairs: list[str] = []

    init_method = cfg.init if cfg.init in ("dc", "flat") else "dc"
    runopp_kwargs: dict[str, Any] = dict(
        calculate_voltage_angles=True,
        init=init_method,
        OPF_FLOW_LIM=0,  # apparent power (MVA) limit on line flows
        RETURN_RAW_DER=0,
        PDIPM_MAX_IT=cfg.max_iteration,
        PDIPM_FEASTOL=cfg.pdipm_feastol,
        PDIPM_GRADTOL=cfg.pdipm_gradtol,
        PDIPM_COMPTOL=cfg.pdipm_comptol,
        PDIPM_COSTTOL=cfg.pdipm_costtol,
        OPF_VIOLATION=cfg.opf_violation,
    )
    runopp_kwargs["numba"] = True

    n_attempts = 5
    # Attempt 1: primary (configured bounds)
    try:
        logger.info(
            "AC FPF attempt 1/%d (primary): vm=[%.2f,%.2f], init=%s, "
            "max_iter=%d, feastol=%.1e",
            n_attempts,
            cfg.vm_min_pu,
            cfg.vm_max_pu,
            init_method,
            cfg.max_iteration,
            cfg.pdipm_feastol,
        )
        t0 = _time.perf_counter()
        pp.runopp(nn, **runopp_kwargs)
        logger.info(
            "AC FPF attempt 1/%d (primary) completed in %.2f sec",
            n_attempts,
            _time.perf_counter() - t0,
        )
    except Exception as e1:
        elapsed1 = _time.perf_counter() - t0
        logger.warning(
            "AC FPF attempt 1/%d (primary) FAILED after %.2f sec: %s",
            n_attempts,
            elapsed1,
            e1,
        )

        # Attempt 2: wider voltage bounds
        wider_min = min(cfg.vm_min_pu, 0.85)
        wider_max = max(cfg.vm_max_pu, 1.15)
        _set_voltage_limits(nn, vm_min_pu=wider_min, vm_max_pu=wider_max)
        pf_repairs.append(f"vm_bounds_relaxed_to_{wider_min:.2f}_{wider_max:.2f}")

        try:
            logger.info(
                "AC FPF attempt 2/%d (wider V): vm=[%.2f,%.2f]",
                n_attempts,
                wider_min,
                wider_max,
            )
            t1 = _time.perf_counter()
            pp.runopp(nn, **runopp_kwargs)
            logger.info(
                "AC FPF attempt 2/%d (wider V) completed in %.2f sec",
                n_attempts,
                _time.perf_counter() - t1,
            )
            pf_attempt = "relaxed_v"
        except Exception as e2:
            elapsed2 = _time.perf_counter() - t1
            logger.warning(
                "AC FPF attempt 2/%d (wider V) FAILED after %.2f sec: %s",
                n_attempts,
                elapsed2,
                e2,
            )

            # Attempt 3: very wide bounds + relaxed Q limits + flat init
            _set_voltage_limits(nn, vm_min_pu=0.80, vm_max_pu=1.20)
            if hasattr(nn, "gen") and nn.gen is not None and len(nn.gen):
                nn.gen["min_q_mvar"] = nn.gen["min_q_mvar"].clip(upper=-1e6)
                nn.gen["max_q_mvar"] = nn.gen["max_q_mvar"].clip(lower=1e6)
            pf_repairs.extend(["vm_bounds_relaxed_to_0.80_1.20", "q_limits_relaxed"])

            runopp_kwargs_flat = {**runopp_kwargs, "init": "flat"}
            try:
                logger.info(
                    "AC FPF attempt 3/%d (relaxed V+Q, flat): vm=[0.80,1.20]",
                    n_attempts,
                )
                t2 = _time.perf_counter()
                pp.runopp(nn, **runopp_kwargs_flat)
                logger.info(
                    "AC FPF attempt 3/%d (relaxed V+Q, flat) completed in %.2f sec",
                    n_attempts,
                    _time.perf_counter() - t2,
                )
                pf_attempt = "relaxed_all"
            except Exception as e3:
                elapsed3 = _time.perf_counter() - t2
                logger.warning(
                    "AC FPF attempt 3/%d (relaxed V+Q, flat) FAILED after %.2f sec: %s",
                    n_attempts,
                    elapsed3,
                    e3,
                )

                # Attempt 4: relax line loading to 100% + even wider tolerances
                if hasattr(nn, "line") and nn.line is not None and len(nn.line):
                    nn.line["max_loading_percent"] = 100.0
                if hasattr(nn, "trafo") and nn.trafo is not None and len(nn.trafo):
                    nn.trafo["max_loading_percent"] = 100.0
                pf_repairs.append("max_loading_100pct")

                runopp_kwargs_loose = {
                    **runopp_kwargs_flat,
                    "PDIPM_FEASTOL": 1e-2,
                    "PDIPM_GRADTOL": 1e-2,
                    "PDIPM_COMPTOL": 1e-2,
                    "PDIPM_COSTTOL": 1e-2,
                    "OPF_VIOLATION": 1e-2,
                    "PDIPM_MAX_IT": max(cfg.max_iteration, 500),
                }
                try:
                    logger.info(
                        "AC FPF attempt 4/%d (loading=100%%, loose tol, flat): "
                        "vm=[0.80,1.20]",
                        n_attempts,
                    )
                    t3 = _time.perf_counter()
                    pp.runopp(nn, **runopp_kwargs_loose)
                    logger.info(
                        "AC FPF attempt 4/%d (loading=100%%, loose tol) "
                        "completed in %.2f sec",
                        n_attempts,
                        _time.perf_counter() - t3,
                    )
                    pf_attempt = "relaxed_loading"
                    pf_repairs.append("loose_tolerances")
                except Exception as e4:
                    elapsed4 = _time.perf_counter() - t3
                    logger.warning(
                        "AC FPF attempt 4/%d (loading=100%%, loose tol) "
                        "FAILED after %.2f sec: %s",
                        n_attempts,
                        elapsed4,
                        e4,
                    )

                    # Attempt 5: MW flow limits instead of MVA
                    runopp_kwargs_mw = {
                        **runopp_kwargs_loose,
                        "OPF_FLOW_LIM": 1,
                    }
                    try:
                        logger.info(
                            "AC FPF attempt 5/%d (MW limits, loose tol, flat): "
                            "vm=[0.80,1.20]",
                            n_attempts,
                        )
                        t4 = _time.perf_counter()
                        pp.runopp(nn, **runopp_kwargs_mw)
                        logger.info(
                            "AC FPF attempt 5/%d (MW limits) completed in %.2f sec",
                            n_attempts,
                            _time.perf_counter() - t4,
                        )
                        pf_attempt = "relaxed_mw_limits"
                        pf_repairs.append("mw_flow_limits")
                    except Exception as e5:
                        elapsed5 = _time.perf_counter() - t4
                        logger.exception(
                            "AC FPF attempt 5/%d (MW limits) FAILED after %.2f sec: %s",
                            n_attempts,
                            elapsed5,
                            e5,
                        )
                        raise RuntimeError(
                            "pandapower.runopp failed "
                            f"(all {n_attempts} AC FPF attempts exhausted)."
                        ) from e5

    # ---- Check convergence ----
    converged = bool(getattr(nn, "OPF_converged", True))
    if not converged:
        raise RuntimeError("pandapower AC OPF did not converge (OPF_converged=False).")

    # ---- Extract results ----
    if not hasattr(nn, "res_bus") or nn.res_bus is None or len(nn.res_bus) == 0:
        raise RuntimeError("pandapower.runopp did not produce res_bus results.")
    if not hasattr(nn, "res_line") or nn.res_line is None or len(nn.res_line) == 0:
        raise RuntimeError("pandapower.runopp did not produce res_line results.")

    v_mag = np.asarray(
        [float(nn.res_bus.loc[bid, "vm_pu"]) for bid in bus_ids], dtype=float
    )
    va_deg = np.asarray(
        [float(nn.res_bus.loc[bid, "va_degree"]) for bid in bus_ids], dtype=float
    )
    v_ang = (va_deg * math.pi / 180.0).astype(float, copy=False)

    # Bus net active power injection.
    bus_p_mw_arr: np.ndarray | None = None
    if "p_mw" in nn.res_bus.columns:
        bus_p_mw_arr = np.asarray(
            [float(nn.res_bus.loc[bid, "p_mw"]) for bid in bus_ids], dtype=float
        )

    # Line flows.
    idx = [int(x) for x in line_indices]
    p0 = np.zeros(len(idx), dtype=float)
    q0 = np.zeros(len(idx), dtype=float)
    p1 = np.zeros(len(idx), dtype=float)
    q1 = np.zeros(len(idx), dtype=float)

    for pos, lid in enumerate(idx):
        if lid not in nn.line.index:
            raise ValueError(f"Requested line id {lid} is missing in net.line.")
        row = nn.line.loc[lid]
        if not _is_in_service(row):
            continue
        p0[pos] = float(nn.res_line.loc[lid, "p_from_mw"])
        q0[pos] = float(nn.res_line.loc[lid, "q_from_mvar"])
        p1[pos] = float(nn.res_line.loc[lid, "p_to_mw"])
        q1[pos] = float(nn.res_line.loc[lid, "q_to_mvar"])

    if np.max(np.abs(v_ang)) > 10.0:
        raise ValueError(
            "Unexpectedly large bus voltage angles from AC FPF. "
            "Expected radians after conversion; got max|angle| > 10."
        )

    # ---- Log solution summary ----
    logger.info(
        "AC FPF solved: status=PP_OPP_OK attempt=%s "
        "v_mag[min,med,max]=[%.4g,%.4g,%.4g] "
        "bus_p_sum=%.4g MW",
        pf_attempt,
        float(np.min(v_mag)),
        float(np.median(v_mag)),
        float(np.max(v_mag)),
        float(np.sum(bus_p_mw_arr)) if bus_p_mw_arr is not None else float("nan"),
    )

    return PyPSAAPFResult(
        bus_ids=tuple(bus_ids),
        v_mag_pu=v_mag,
        v_ang_rad=v_ang,
        line_ids=tuple(idx),
        line_p0_mw=p0,
        line_q0_mvar=q0,
        line_p1_mw=p1,
        line_q1_mvar=q1,
        status="PP_OPP_OK",
        pf_attempt=pf_attempt,
        pf_repairs=list(pf_repairs),
        bus_p_mw=bus_p_mw_arr,
    )
