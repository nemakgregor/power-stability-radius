from __future__ import annotations

"""
AC power flow (PF) base point generation.

This module can solve the base point using either:
- PyPSA PF (historical stack)
- pandapower.runpp (explicit robust alternative)

Key contract (downstream)
-------------------------
The downstream AC certificate needs:
- (V, theta) per bus
- per monitored line (P,Q) at both ends

Ordering is deterministic:
- buses: sorted(net.bus.index)
- monitored lines: provided `line_indices` or sorted(net.line.index)

Robustness knobs (explicit, deterministic)
------------------------------------------
- solver: "pypsa" | "pandapower"
- init:   "flat" | "dc" | "pp"

Correctness policy (no solver fallback)
---------------------------------------
init="pp" means: run pandapower.runpp first and use it as an explicit initial guess
(for PyPSA PF).

If you request solver="pypsa" and PyPSA PF fails, we FAIL FAST.
We do NOT silently return pandapower results, because that would:
- mislabel the PF backend in results metadata,
- break AC MC base-regime consistency checks,
- hide solver instability.

Lossless alignment
------------------
For the first "sound" AC certificate version, we keep PF and Jacobian aligned with the
project's lossless policy:
  lossless=True => r=0 for lines and transformer series r ignored.
"""

import copy
import logging
import math
import time as _time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from stability_radius.base_point.pandapower_tools import (
    apply_gen_dispatch_to_pandapower_net,
    apply_lossless_policy_to_pandapower_net,
    detect_q_limit_events,
    ensure_ext_grid_at_slack,
    resolve_slack_bus_id,
)
from stability_radius.config import DEFAULT_OPF, OPFConfig
from stability_radius.dc.dc_model import trafo_tap_ratio
from stability_radius.pp_helpers import bus_vn_kv as _bus_vn_kv
from stability_radius.pp_helpers import is_in_service as _is_in_service

logger = logging.getLogger(__name__)

_AC_CARRIER = "AC"
_X_OHM_EPS = 1e-12


@dataclass(frozen=True)
class PyPSAAPFResult:
    """
    Minimal AC PF result required by the AC linearization code.

    Units
    -----
    - v_mag_pu: p.u.
    - v_ang_rad: rad
    - line p/q: MW / MVAr
      Sign convention matches:
        - PyPSA: p0/q0, p1/q1 are powers leaving bus0/bus1 into the branch.
        - pandapower: res_line p_from/q_from, p_to/q_to are powers leaving from_bus/to_bus into the line.
    """

    bus_ids: tuple[int, ...]
    v_mag_pu: np.ndarray  # (n_bus,)
    v_ang_rad: np.ndarray  # (n_bus,)

    line_ids: tuple[int, ...]  # monitored pandapower net.line indices ordering
    line_p0_mw: np.ndarray  # (m_line,) flow at bus0 end (leaving bus0)
    line_q0_mvar: np.ndarray  # (m_line,)
    line_p1_mw: np.ndarray  # (m_line,) flow at bus1 end (leaving bus1)
    line_q1_mvar: np.ndarray  # (m_line,)

    status: str
    pf_attempt: str = "primary"  # "primary" | "alt_init" | "relaxed"
    pf_repairs: list[str] | None = None  # list of repair actions applied
    distributed_slack_requested: bool = False
    distributed_slack_used: bool = False
    bus_p_mw: np.ndarray | None = None  # (n_bus,) net P injection per bus from AC PF
    bus_q_mvar: np.ndarray | None = None  # (n_bus,) net Q injection per bus from AC PF
    q_limit_hit: bool = False
    q_limit_events: tuple[dict[str, Any], ...] = ()
    opp_gen_dispatch: dict[str, float] | None = None  # gen_id -> P_mw from OPP
    opp_vm_pu: dict[int, float] | None = None  # bus_id -> Vm from OPP


def _ensure_carrier_table(n: Any, carrier_name: str) -> None:
    """
    Ensure that `carrier_name` exists in `n.carriers`.

    Some PyPSA versions emit warnings when a component's `carrier` is set but the carrier
    is missing in `network.carriers`. We explicitly define the carrier to keep logs clean.
    """
    if not hasattr(n, "carriers"):
        return
    try:
        carriers = n.carriers
    except Exception:  # noqa: BLE001
        return
    try:
        if str(carrier_name) in carriers.index:
            return
    except Exception:  # noqa: BLE001
        return
    n.add("Carrier", str(carrier_name))


def _line_r_x_total_ohm_from_pp(
    line_row: Any, *, lossless: bool
) -> tuple[float, float]:
    """
    Convert pandapower line parameters to total (r,x) in Ohm.

    lossless=True enforces r=0 to keep PF closer to the project's DC assumptions.
    """
    x_ohm_per_km = float(line_row.get("x_ohm_per_km", np.nan))
    r_ohm_per_km = float(line_row.get("r_ohm_per_km", 0.0))
    length_km = float(line_row.get("length_km", np.nan))
    parallel = float(line_row.get("parallel", 1.0))

    if not math.isfinite(x_ohm_per_km):
        raise ValueError(f"Line: x_ohm_per_km must be finite; got {x_ohm_per_km!r}")
    if not math.isfinite(length_km):
        raise ValueError(f"Line: length_km must be finite; got {length_km!r}")
    if not math.isfinite(parallel) or parallel <= 0:
        raise ValueError(f"Line: parallel must be finite and >0; got {parallel!r}")

    x_ohm = float(x_ohm_per_km) * float(length_km) / float(parallel)
    if (not math.isfinite(x_ohm)) or abs(x_ohm) <= _X_OHM_EPS:
        raise ValueError(f"Line: invalid x_ohm={x_ohm!r} (must be finite and non-zero)")

    if bool(lossless):
        return 0.0, float(x_ohm)

    if not math.isfinite(r_ohm_per_km):
        raise ValueError(f"Line: r_ohm_per_km must be finite; got {r_ohm_per_km!r}")
    r_ohm = float(r_ohm_per_km) * float(length_km) / float(parallel)
    if not math.isfinite(r_ohm):
        raise ValueError(f"Line: invalid r_ohm={r_ohm!r}")
    return float(r_ohm), float(x_ohm)


def _sum_pq_by_bus(net: Any, table_name: str) -> dict[int, tuple[float, float]]:
    """
    Sum (p,q) per bus for a pandapower element table.

    Supports `net.load` and `net.shunt` (both use columns: bus, p_mw, q_mvar).
    """
    if not hasattr(net, table_name):
        return {}
    table = getattr(net, table_name)
    if table is None or len(table) == 0:
        return {}
    if "bus" not in table.columns:
        return {}

    out: dict[int, tuple[float, float]] = {}
    for _, row in table.iterrows():
        if not _is_in_service(row):
            continue
        bus = int(row["bus"])
        p = float(row.get("p_mw", 0.0))
        q = float(row.get("q_mvar", 0.0))
        if not math.isfinite(p):
            p = 0.0
        if not math.isfinite(q):
            q = 0.0
        p0, q0 = out.get(bus, (0.0, 0.0))
        out[bus] = (float(p0 + p), float(q0 + q))
    return out


def _trafo_series_rx_pu_from_pp_row(trafo_row: Any) -> tuple[float, float]:
    """
    Extract transformer series (r,x) in per unit from pandapower trafo row.

    Returns (r_pu, x_pu).
    """
    vk_percent = float(trafo_row.get("vk_percent", np.nan))
    if not math.isfinite(vk_percent):
        raise ValueError(
            f"Trafo {getattr(trafo_row, 'name', 'unknown')}: vk_percent invalid"
        )

    vkr_percent = float(trafo_row.get("vkr_percent", 0.0))
    if not math.isfinite(vkr_percent):
        vkr_percent = 0.0

    z_pu = float(vk_percent) / 100.0
    r_pu = float(vkr_percent) / 100.0
    x_pu2 = z_pu * z_pu - r_pu * r_pu
    x_pu = float(math.sqrt(max(x_pu2, 0.0)))
    if x_pu <= 0.0:
        raise ValueError(
            f"Trafo {getattr(trafo_row, 'name', 'unknown')}: derived x_pu must be >0"
        )
    return r_pu, x_pu


def _apply_distributed_slack_weights(nn: Any) -> None:
    """Set ``slack_weight`` on generators and ext_grid proportionally to headroom.

    Headroom for each generator is ``max_p_mw - p_mw`` (clamped to >= 0).
    For ``ext_grid`` entries (which lack explicit bounds in pandapower), a
    participation weight equal to the average headroom of normal generators
    is assigned so that the slack bus absorbs a moderate share of losses
    rather than all of them.

    This function modifies ``nn`` **in-place**.
    """
    gen_headroom_sum = 0.0
    gen_count_positive = 0

    if hasattr(nn, "gen") and nn.gen is not None and len(nn.gen):
        max_p = nn.gen["max_p_mw"].fillna(0.0)
        p_set = nn.gen["p_mw"].fillna(0.0)
        headroom = (max_p - p_set).clip(lower=0.0)
        nn.gen["slack_weight"] = headroom

        gen_headroom_sum = float(headroom.sum())
        gen_count_positive = int((headroom > 0).sum())

        logger.info(
            "Distributed slack: gen headroom sum=%.4f MW, participating gens=%d/%d",
            gen_headroom_sum,
            gen_count_positive,
            int(len(nn.gen)),
        )

    if hasattr(nn, "ext_grid") and nn.ext_grid is not None and len(nn.ext_grid):
        if gen_count_positive > 0 and gen_headroom_sum > 0:
            avg_headroom = gen_headroom_sum / gen_count_positive
        else:
            avg_headroom = 100.0  # reasonable default (MW)
        nn.ext_grid["slack_weight"] = avg_headroom
        logger.info(
            "Distributed slack: ext_grid slack_weight=%.4f MW (avg gen headroom)",
            avg_headroom,
        )


def _solve_ac_pf_with_pandapower(
    *,
    net: Any,
    slack_bus: int,
    line_indices: Sequence[int],
    gen_dispatch_mw_by_name: Mapping[str, float] | None,
    lossless: bool,
    init: str,
    distributed_slack: bool = False,
    trafo_model: str = "pi",
) -> PyPSAAPFResult:
    """Solve PF using pandapower.runpp and return PyPSAAPFResult.

    Parameters
    ----------
    distributed_slack:
        When True, distribute the active-power slack among generators
        proportionally to their headroom (P_max - P_set).  This avoids
        dumping all loss-compensation onto a single slack bus and keeps
        generator outputs within bounds.  Requires pandapower >= 2.10.
    trafo_model:
        Transformer equivalent circuit model: ``"pi"`` (recommended) or ``"t"``.
    """
    try:
        import pandapower as pp  # type: ignore
    except ImportError as e:
        raise ImportError("pandapower is required for solver='pandapower'.") from e

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    if not bus_ids:
        raise ValueError("Network has no buses.")

    slack_bus_id = resolve_slack_bus_id(net, int(slack_bus))
    ensure_ext_grid_at_slack(net, int(slack_bus_id))

    n_buses = int(len(bus_ids))
    nn = (
        apply_lossless_policy_to_pandapower_net(net)
        if bool(lossless)
        else copy.deepcopy(net)
    )
    apply_gen_dispatch_to_pandapower_net(nn, gen_dispatch_mw_by_name)

    init_eff = str(init).strip().lower()
    if init_eff not in {"flat", "dc", "pp"}:
        raise ValueError("init must be flat|dc|pp for pandapower solver as well.")
    # Allow "dc" init (DC power flow warmstart) — it provides a much better
    # starting point for Newton-Raphson on difficult networks (e.g. case57).
    if init_eff == "pp":
        init_eff = "flat"

    trafo_model_eff = str(trafo_model).strip().lower()
    if trafo_model_eff not in {"pi", "t"}:
        raise ValueError("trafo_model must be pi|t")

    logger.info(
        "Solving AC PF with pandapower.runpp: buses=%d lines=%d lossless=%s init=%s "
        "distributed_slack=%s trafo_model=%s",
        int(len(nn.bus)),
        int(len(nn.line)) if hasattr(nn, "line") and nn.line is not None else 0,
        bool(lossless),
        init_eff,
        bool(distributed_slack),
        trafo_model_eff,
    )

    max_iter = 300

    # Track which solver attempt succeeded for repair metadata.
    pf_attempt: str = "primary"  # "primary" | "alt_init" | "relaxed"
    pf_repairs: list[str] = []
    distributed_slack_requested = bool(distributed_slack)

    # Guard against pandapower realloc corruption with distributed_slack on
    # large networks.  Known bug: the distributed_slack Newton-Raphson path
    # triggers C-level memory corruption (realloc: invalid next size -> SIGABRT)
    # on networks with ~300+ buses, killing the entire process.  Python
    # exception handlers cannot catch this.
    _DISTRIBUTED_SLACK_MAX_BUSES = 300
    if bool(distributed_slack) and n_buses >= _DISTRIBUTED_SLACK_MAX_BUSES:
        logger.warning(
            "Auto-disabling distributed_slack for %d-bus network to prevent "
            "pandapower realloc corruption (threshold=%d buses). "
            "Loss compensation will be assigned to the slack bus only.",
            n_buses,
            _DISTRIBUTED_SLACK_MAX_BUSES,
        )
        distributed_slack = False
        pf_repairs.append("distributed_slack_auto_disabled_large_network")
    distributed_slack_used = bool(distributed_slack)

    # ---- Distributed slack: set participation weights based on headroom ----
    if bool(distributed_slack):
        _apply_distributed_slack_weights(nn)

    runpp_kwargs: dict[str, Any] = dict(
        calculate_voltage_angles=True,
        enforce_q_lims=True,
        init=str(init_eff),
        max_iteration=max_iter,
        trafo_model=str(trafo_model_eff),
    )
    if bool(distributed_slack):
        runpp_kwargs["distributed_slack"] = True

    try:
        logger.info(
            "pp.runpp attempt 1/3 (primary): init='%s' max_iter=%d distributed_slack=%s",
            init_eff,
            max_iter,
            bool(distributed_slack),
        )
        t_pf = _time.perf_counter()
        pp.runpp(nn, **runpp_kwargs)
        logger.info(
            "pp.runpp attempt 1/3 (primary) completed in %.2f sec",
            _time.perf_counter() - t_pf,
        )
    except Exception as e_first:
        elapsed_pf = _time.perf_counter() - t_pf
        logger.warning(
            "pp.runpp attempt 1/3 (primary) FAILED after %.2f sec with init='%s': %s",
            elapsed_pf,
            init_eff,
            e_first,
        )
        # Retry with the opposite initialisation strategy.
        alt_init = "dc" if init_eff == "flat" else "flat"
        logger.warning(
            "pandapower.runpp failed with init='%s', retrying with init='%s' (%d iterations)",
            init_eff,
            alt_init,
            max_iter,
        )
        runpp_kwargs["init"] = alt_init
        try:
            logger.info(
                "pp.runpp attempt 2/3 (alt_init): init='%s' max_iter=%d",
                alt_init,
                max_iter,
            )
            t_pf2 = _time.perf_counter()
            pp.runpp(nn, **runpp_kwargs)
            logger.info(
                "pp.runpp attempt 2/3 (alt_init) completed in %.2f sec",
                _time.perf_counter() - t_pf2,
            )
            pf_attempt = "alt_init"
            pf_repairs.append(f"init_changed_to_{alt_init}")
        except Exception as e_alt:
            elapsed_pf2 = _time.perf_counter() - t_pf2
            logger.warning(
                "pp.runpp attempt 2/3 (alt_init) FAILED after %.2f sec: %s",
                elapsed_pf2,
                e_alt,
            )
            # Final fallback: relax settings (no Q limits, no distributed slack).
            logger.warning(
                "pandapower.runpp failed with both init strategies; "
                "retrying with relaxed settings (enforce_q_lims=False, "
                "distributed_slack=False, init='flat', max_iteration=%d)",
                max_iter,
            )
            relaxed_kwargs: dict[str, Any] = dict(
                calculate_voltage_angles=True,
                enforce_q_lims=False,
                init="flat",
                max_iteration=max_iter,
                trafo_model=str(trafo_model_eff),
            )
            try:
                logger.info(
                    "pp.runpp attempt 3/3 (relaxed): init='flat' max_iter=%d "
                    "enforce_q_lims=False distributed_slack=False",
                    max_iter,
                )
                t_pf3 = _time.perf_counter()
                pp.runpp(nn, **relaxed_kwargs)
                logger.info(
                    "pp.runpp attempt 3/3 (relaxed) completed in %.2f sec",
                    _time.perf_counter() - t_pf3,
                )
                pf_attempt = "relaxed"
                distributed_slack_used = False
                pf_repairs.extend(
                    [
                        "enforce_q_lims_disabled",
                        "distributed_slack_disabled",
                        "init_flat",
                    ]
                )
            except Exception as e_relaxed:
                elapsed_pf3 = _time.perf_counter() - t_pf3
                logger.exception(
                    "pp.runpp attempt 3/3 (relaxed) FAILED after %.2f sec: %s",
                    elapsed_pf3,
                    e_relaxed,
                )
                raise RuntimeError(
                    "pandapower.runpp failed (all 3 attempts exhausted)."
                ) from e_relaxed

    converged = bool(getattr(nn, "converged", True))
    if not converged:
        raise RuntimeError("pandapower AC PF did not converge (net.converged=False).")

    if not hasattr(nn, "res_bus") or nn.res_bus is None or len(nn.res_bus) == 0:
        raise RuntimeError("pandapower did not produce res_bus results.")

    v_mag = np.asarray(
        [float(nn.res_bus.loc[bid, "vm_pu"]) for bid in bus_ids], dtype=float
    )
    va_deg = np.asarray(
        [float(nn.res_bus.loc[bid, "va_degree"]) for bid in bus_ids], dtype=float
    )
    v_ang = (va_deg * math.pi / 180.0).astype(float, copy=False)

    # Extract net active power injection per bus (gen - load, MW) for ACPF mode.
    bus_p_mw_arr: np.ndarray | None = None
    if "p_mw" in nn.res_bus.columns:
        bus_p_mw_arr = np.asarray(
            [float(nn.res_bus.loc[bid, "p_mw"]) for bid in bus_ids], dtype=float
        )

    # Extract net reactive power injection per bus (gen - load, MVAr).
    bus_q_mvar_arr: np.ndarray | None = None
    if "q_mvar" in nn.res_bus.columns:
        bus_q_mvar_arr = np.asarray(
            [float(nn.res_bus.loc[bid, "q_mvar"]) for bid in bus_ids], dtype=float
        )

    if not hasattr(nn, "res_line") or nn.res_line is None or len(nn.res_line) == 0:
        raise RuntimeError("pandapower did not produce res_line results.")

    q_limit_events = detect_q_limit_events(nn)
    if q_limit_events:
        logger.warning(
            "pandapower AC PF solved with %d generator reactive limit event(s); "
            "fixed-PV/PQ AC linearization is diagnostic-only for this active set.",
            int(len(q_limit_events)),
        )

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
            "Unexpectedly large bus voltage angles from pandapower PF. "
            "Expected radians after conversion; got max|angle| > 10."
        )

    status = "PP_PF_OK"
    logger.info(
        "pandapower AC PF done: status=%s v_mag_pu[min,median,max]=[%.6g,%.6g,%.6g]",
        status,
        float(np.min(v_mag)),
        float(np.median(v_mag)),
        float(np.max(v_mag)),
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
        status=status,
        pf_attempt=pf_attempt,
        pf_repairs=list(pf_repairs),
        distributed_slack_requested=bool(distributed_slack_requested),
        distributed_slack_used=bool(distributed_slack_used),
        bus_p_mw=bus_p_mw_arr,
        bus_q_mvar=bus_q_mvar_arr,
        q_limit_hit=bool(q_limit_events),
        q_limit_events=tuple(q_limit_events),
    )


def solve_ac_pf_base_point_from_pandapower(
    *,
    net: Any,
    slack_bus: int,
    line_indices: Sequence[int] | None = None,
    gen_dispatch_mw_by_name: Mapping[str, float] | None = None,
    opf_cfg: OPFConfig | None = None,
    lossless: bool = True,
    solver: str = "pypsa",
    init: str = "flat",
    dc_init_vm_pu: np.ndarray | None = None,
    dc_init_va_rad: np.ndarray | None = None,
    distributed_slack: bool = False,
    trafo_model: str = "pi",
) -> PyPSAAPFResult:
    """
    Solve AC PF and return base voltages + per-line P/Q flows.

    See module docstring for the detailed contract.
    """
    cfg = opf_cfg if opf_cfg is not None else DEFAULT_OPF

    solver_eff = str(solver).strip().lower()
    if solver_eff not in {"pypsa", "pandapower"}:
        raise ValueError("solver must be pypsa|pandapower")

    init_eff = str(init).strip().lower()
    if init_eff not in {"flat", "dc", "pp"}:
        raise ValueError("init must be flat|dc|pp")

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    if not bus_ids:
        raise ValueError("Network has no buses.")

    idx = (
        [int(x) for x in sorted(net.line.index)]
        if line_indices is None
        else [int(x) for x in line_indices]
    )

    if solver_eff == "pandapower":
        return _solve_ac_pf_with_pandapower(
            net=net,
            slack_bus=int(slack_bus),
            line_indices=idx,
            gen_dispatch_mw_by_name=gen_dispatch_mw_by_name,
            lossless=bool(lossless),
            init=init_eff,
            distributed_slack=bool(distributed_slack),
            trafo_model=str(trafo_model),
        )

    pp_init: PyPSAAPFResult | None = None
    if init_eff == "pp":
        pp_init = _solve_ac_pf_with_pandapower(
            net=net,
            slack_bus=int(slack_bus),
            line_indices=idx,
            gen_dispatch_mw_by_name=gen_dispatch_mw_by_name,
            lossless=bool(lossless),
            init="flat",
            distributed_slack=bool(distributed_slack),
            trafo_model=str(trafo_model),
        )
        logger.info(
            "AC PF init='pp': obtained pandapower base point for initial guess."
        )

    if init_eff == "dc":
        if dc_init_vm_pu is None or dc_init_va_rad is None:
            raise ValueError(
                "init='dc' requires dc_init_vm_pu and dc_init_va_rad (full bus arrays)."
            )

    try:
        import pandas as pd
        import pypsa
    except ImportError as e:
        raise ImportError("PyPSA (and pandas) is required for solver='pypsa'.") from e

    bus_pos = {bid: pos for pos, bid in enumerate(bus_ids)}
    if int(slack_bus) in bus_pos:
        slack_bus_id = int(slack_bus)
    elif 0 <= int(slack_bus) < len(bus_ids):
        slack_bus_id = int(bus_ids[int(slack_bus)])
    else:
        raise ValueError(
            f"slack_bus must be a valid bus id or position. Got {slack_bus!r}."
        )

    ensure_ext_grid_at_slack(net, int(slack_bus_id))

    n = pypsa.Network()
    n.set_snapshots(pd.Index([0]))
    _ensure_carrier_table(n, _AC_CARRIER)

    sn_mva = float(getattr(net, "sn_mva", np.nan))
    if not math.isfinite(sn_mva) or sn_mva <= 0.0:
        raise ValueError(f"pandapower net.sn_mva must be finite and >0; got {sn_mva!r}")
    n.sn_mva = float(sn_mva)

    for bid in bus_ids:
        vn_kv = _bus_vn_kv(net, bid)
        if not math.isfinite(vn_kv) or vn_kv <= 0.0:
            raise ValueError(f"Invalid vn_kv for bus {bid}: {vn_kv!r}")

        bus_kwargs: dict[str, Any] = {"v_nom": float(vn_kv)}
        if hasattr(n, "buses") and "carrier" in getattr(n, "buses").columns:
            bus_kwargs["carrier"] = _AC_CARRIER
        n.add("Bus", str(bid), **bus_kwargs)

    bus_id_set = set(bus_ids)

    load_pq: dict[int, tuple[float, float]] = {}
    for tbl in ("load", "shunt"):
        pq = _sum_pq_by_bus(net, tbl)
        for bus, (p, q) in pq.items():
            p0, q0 = load_pq.get(bus, (0.0, 0.0))
            load_pq[bus] = (float(p0 + p), float(q0 + q))

    for bus in sorted(load_pq.keys()):
        if bus not in bus_id_set:
            continue
        p, q = load_pq[bus]
        if abs(p) <= 0.0 and abs(q) <= 0.0:
            continue
        n.add("Load", f"load_{bus}", bus=str(bus), p_set=float(p), q_set=float(q))

    slack_gen_name: str | None = None

    total_load_p = float(sum(p for p, _ in load_pq.values()))
    if hasattr(net, "ext_grid") and net.ext_grid is not None and len(net.ext_grid):
        for eid in [int(x) for x in sorted(net.ext_grid.index)]:
            row = net.ext_grid.loc[eid]
            if not _is_in_service(row):
                continue
            bus = int(row.get("bus", -1))
            if bus not in bus_id_set:
                raise ValueError(
                    f"pandapower ext_grid {eid} refers to missing bus {bus}"
                )

            name = f"ext_{eid}"
            is_slack = bus == slack_bus_id and slack_gen_name is None
            control = "Slack" if is_slack else "PV"
            if is_slack:
                slack_gen_name = name

            p_nom = max(1.0, float(total_load_p))
            if (
                (not is_slack)
                and gen_dispatch_mw_by_name
                and name in gen_dispatch_mw_by_name
            ):
                p_set = float(gen_dispatch_mw_by_name[name])
            else:
                p_set = 0.0
            if not math.isfinite(p_set):
                p_set = 0.0

            v_set = float(row.get("vm_pu", 1.0)) if "vm_pu" in row else 1.0

            n.add(
                "Generator",
                name,
                bus=str(bus),
                p_nom=float(p_nom),
                p_set=float(p_set),
                q_set=0.0,
                control=str(control),
                v_set=float(v_set),
            )

    if hasattr(net, "gen") and net.gen is not None and len(net.gen):
        skipped: list[int] = []
        for gid in [int(x) for x in sorted(net.gen.index)]:
            row = net.gen.loc[gid]
            if not _is_in_service(row):
                continue
            bus = int(row.get("bus", -1))
            if bus not in bus_id_set:
                raise ValueError(f"pandapower gen {gid} refers to missing bus {bus}")

            p_max = float(row.get("max_p_mw", np.nan))
            if not math.isfinite(p_max) or p_max <= 0.0:
                skipped.append(int(gid))
                continue

            name = f"gen_{gid}"
            p_nom = float(p_max)

            if gen_dispatch_mw_by_name and name in gen_dispatch_mw_by_name:
                p_set = float(gen_dispatch_mw_by_name[name])
            else:
                p_set = float(row.get("p_mw", 0.0))
            if not math.isfinite(p_set):
                p_set = 0.0

            v_set = float(row.get("vm_pu", 1.0)) if "vm_pu" in row else 1.0

            gen_kwargs: dict[str, Any] = dict(
                bus=str(bus),
                p_nom=float(p_nom),
                p_set=float(p_set),
                q_set=0.0,
                control="PV",
                v_set=float(v_set),
            )

            if "min_q_mvar" in row and "max_q_mvar" in row:
                try:
                    qmin = float(row.get("min_q_mvar", float("nan")))
                    qmax = float(row.get("max_q_mvar", float("nan")))
                except (TypeError, ValueError):
                    qmin, qmax = float("nan"), float("nan")
                if math.isfinite(qmin):
                    gen_kwargs["q_min"] = float(qmin)
                if math.isfinite(qmax):
                    gen_kwargs["q_max"] = float(qmax)

            n.add("Generator", name, **gen_kwargs)

        if skipped:
            logger.warning(
                "AC PF: skipped %d pandapower gen(s) with non-positive/invalid max_p_mw. First: %s",
                int(len(skipped)),
                skipped[:20],
            )

    # ---- sgen entries (additional generators at same bus from MATPOWER) ----
    if hasattr(net, "sgen") and net.sgen is not None and len(net.sgen):
        skipped_sgen: list[int] = []
        for sid in [int(x) for x in sorted(net.sgen.index)]:
            row = net.sgen.loc[sid]
            if not _is_in_service(row):
                continue
            bus = int(row.get("bus", -1))
            if bus not in bus_id_set:
                raise ValueError(f"pandapower sgen {sid} refers to missing bus {bus}")

            p_max = float(row.get("max_p_mw", np.nan))
            if not math.isfinite(p_max) or p_max <= 0.0:
                skipped_sgen.append(int(sid))
                continue

            name = f"sgen_{sid}"
            p_nom = float(p_max)

            if gen_dispatch_mw_by_name and name in gen_dispatch_mw_by_name:
                p_set = float(gen_dispatch_mw_by_name[name])
            else:
                p_set = float(row.get("p_mw", 0.0))
            if not math.isfinite(p_set):
                p_set = 0.0

            v_set = float(row.get("vm_pu", 1.0)) if "vm_pu" in row else 1.0

            sgen_kwargs: dict[str, Any] = dict(
                bus=str(bus),
                p_nom=float(p_nom),
                p_set=float(p_set),
                q_set=0.0,
                control="PV",
                v_set=float(v_set),
            )

            if "min_q_mvar" in row and "max_q_mvar" in row:
                try:
                    qmin = float(row.get("min_q_mvar", float("nan")))
                    qmax = float(row.get("max_q_mvar", float("nan")))
                except (TypeError, ValueError):
                    qmin, qmax = float("nan"), float("nan")
                if math.isfinite(qmin):
                    sgen_kwargs["q_min"] = float(qmin)
                if math.isfinite(qmax):
                    sgen_kwargs["q_max"] = float(qmax)

            n.add("Generator", name, **sgen_kwargs)

        if skipped_sgen:
            logger.warning(
                "AC PF: skipped %d pandapower sgen(s) with non-positive/invalid max_p_mw. First: %s",
                int(len(skipped_sgen)),
                skipped_sgen[:20],
            )

    if slack_gen_name is None:
        logger.error(
            "AC PF requires an ext_grid at the requested slack_bus=%s. "
            "No in-service ext_grid found at that bus.",
            str(slack_bus_id),
        )
        raise RuntimeError(
            "AC PF base point failed: no in-service pandapower ext_grid at the requested slack bus."
        )

    if not hasattr(net, "line") or net.line is None or len(net.line) == 0:
        raise ValueError(
            "Network has no net.line entries; cannot solve AC PF for lines."
        )

    unconstrained_nom = float(cfg.unconstrained_line_nom_mw)
    if not math.isfinite(unconstrained_nom) or unconstrained_nom <= 0:
        unconstrained_nom = 1.0e6

    in_service_flags: dict[int, bool] = {}
    for lid in idx:
        row = net.line.loc[int(lid)]
        in_service = bool(_is_in_service(row))
        in_service_flags[int(lid)] = in_service
        if not in_service:
            continue

        fb = int(row.get("from_bus", -1))
        tb = int(row.get("to_bus", -1))
        if fb not in bus_id_set or tb not in bus_id_set:
            raise ValueError(f"Line {lid} refers to missing buses {fb}->{tb}")

        r_ohm, x_ohm = _line_r_x_total_ohm_from_pp(row, lossless=bool(lossless))
        n.add(
            "Line",
            f"line_{int(lid)}",
            bus0=str(fb),
            bus1=str(tb),
            r=float(r_ohm),
            x=float(x_ohm),
            s_nom=float(unconstrained_nom),
        )

    n_phase_shifters = 0
    added_trafos = 0
    if hasattr(net, "trafo") and net.trafo is not None and len(net.trafo):
        for tid in [int(x) for x in sorted(net.trafo.index)]:
            row = net.trafo.loc[tid]
            if not _is_in_service(row):
                continue

            hv = int(row.get("hv_bus", -1))
            lv = int(row.get("lv_bus", -1))
            if hv not in bus_id_set or lv not in bus_id_set:
                raise ValueError(f"Trafo {tid} refers to missing buses {hv}->{lv}")

            sn_trafo = float(row.get("sn_mva", np.nan))
            if not math.isfinite(sn_trafo) or sn_trafo <= 0.0:
                raise ValueError(
                    f"Trafo {tid}: sn_mva must be finite and >0; got {sn_trafo!r}"
                )

            shift_deg = float(row.get("shift_degree", 0.0))
            if not math.isfinite(shift_deg):
                raise ValueError(
                    f"Trafo {tid}: shift_degree must be finite; got {shift_deg!r}"
                )
            if abs(float(shift_deg)) > 1e-9:
                n_phase_shifters += 1

            tap = float(trafo_tap_ratio(row))
            if not math.isfinite(tap) or tap <= 0.0:
                raise ValueError(f"Trafo {tid}: invalid tap_ratio={tap!r}")

            r_pu, x_pu = _trafo_series_rx_pu_from_pp_row(row)

            n.add(
                "Transformer",
                f"trafo_{tid}",
                bus0=str(hv),
                bus1=str(lv),
                model="pi",
                s_nom=float(sn_trafo),
                r=0.0 if bool(lossless) else float(r_pu),
                x=float(x_pu),
                tap_ratio=float(tap),
                tap_side=0,
                phase_shift=float(shift_deg),
            )
            added_trafos += 1

    if n_phase_shifters > 0:
        logger.info(
            "AC PF model: enabled phase shifting transformers: count=%d",
            int(n_phase_shifters),
        )

    if hasattr(net, "impedance") and net.impedance is not None and len(net.impedance):
        sn_system = float(getattr(net, "sn_mva", np.nan))
        if not math.isfinite(sn_system) or sn_system <= 0:
            raise ValueError("pandapower net.sn_mva must be finite and positive.")

        for iid in [int(x) for x in sorted(net.impedance.index)]:
            row = net.impedance.loc[iid]
            if not _is_in_service(row):
                continue
            fb = int(row.get("from_bus", -1))
            tb = int(row.get("to_bus", -1))
            if fb not in bus_id_set or tb not in bus_id_set:
                raise ValueError(f"Impedance {iid} refers to missing buses {fb}->{tb}")

            x_pu = float(row.get("xft_pu", np.nan))
            if not math.isfinite(x_pu) or abs(float(x_pu)) <= 1e-12:
                raise ValueError(f"Impedance {iid}: invalid xft_pu={x_pu!r}")

            vn_kv = _bus_vn_kv(net, fb)
            if not math.isfinite(vn_kv) or vn_kv <= 0:
                raise ValueError(f"Impedance {iid}: invalid vn_kv for from_bus={fb}")

            z_base_ohm = (float(vn_kv) * float(vn_kv)) / float(sn_system)
            x_ohm = float(x_pu) * float(z_base_ohm)

            n.add(
                "Line",
                f"impedance_{iid}",
                bus0=str(fb),
                bus1=str(tb),
                r=0.0,
                x=float(x_ohm),
                s_nom=float(unconstrained_nom),
            )

    snap = n.snapshots[0]
    try:
        n.buses_t["v_mag_pu"] = pd.DataFrame(
            1.0, index=n.snapshots, columns=n.buses.index, dtype=float
        )
        n.buses_t["v_ang"] = pd.DataFrame(
            0.0, index=n.snapshots, columns=n.buses.index, dtype=float
        )

        if init_eff == "dc":
            vm0 = np.asarray(dc_init_vm_pu, dtype=float).reshape(-1)
            va0 = np.asarray(dc_init_va_rad, dtype=float).reshape(-1)
            if vm0.shape != (len(bus_ids),) or va0.shape != (len(bus_ids),):
                raise ValueError("dc_init_vm_pu/dc_init_va_rad shape mismatch.")
            for bid, vm, va in zip(bus_ids, vm0.tolist(), va0.tolist()):
                n.buses_t.v_mag_pu.loc[snap, str(bid)] = float(vm)
                n.buses_t.v_ang.loc[snap, str(bid)] = float(va)

        if init_eff == "pp" and pp_init is not None:
            for bid, vm, va in zip(
                bus_ids, pp_init.v_mag_pu.tolist(), pp_init.v_ang_rad.tolist()
            ):
                n.buses_t.v_mag_pu.loc[snap, str(bid)] = float(vm)
                n.buses_t.v_ang.loc[snap, str(bid)] = float(va)

    except Exception:  # noqa: BLE001
        logger.debug(
            "Failed to set explicit PF initial guess in PyPSA (best-effort).",
            exc_info=True,
        )

    logger.info(
        "Solving PyPSA AC PF: buses=%d loads=%d generators=%d monitored_lines(in_service)=%d "
        "trafos=%d lossless=%s init=%s slack_gen=%s",
        int(len(n.buses.index)),
        int(len(getattr(n, "loads", []).index)) if hasattr(n, "loads") else 0,
        int(len(n.generators.index)),
        int(sum(bool(v) for v in in_service_flags.values())),
        int(added_trafos),
        bool(lossless),
        str(init_eff),
        str(slack_gen_name),
    )

    if not hasattr(n, "pf"):
        raise RuntimeError("Unsupported PyPSA version: Network.pf() is required.")

    try:
        try:
            n.pf()
        except TypeError:
            n.pf(n.snapshots)
    except Exception as e:
        logger.exception("PyPSA AC PF failed: %s", e)
        raise RuntimeError(
            "PyPSA AC PF failed. "
            "If you need a more robust PF backend, set solver='pandapower' "
            "(or set compute.ac.pf_solver=pandapower in YAML)."
        ) from e

    converged = bool(getattr(n, "pf_converged", True))
    if not converged:
        raise RuntimeError("PyPSA AC PF did not converge (pf_converged=False).")

    if (
        not hasattr(n, "buses_t")
        or not hasattr(n.buses_t, "v_mag_pu")
        or not hasattr(n.buses_t, "v_ang")
    ):
        raise RuntimeError("PyPSA AC PF did not produce bus voltage results.")

    bus_names = [str(b) for b in bus_ids]
    v_mag = np.asarray(
        [float(n.buses_t.v_mag_pu.loc[snap, bn]) for bn in bus_names], dtype=float
    )
    v_ang = np.asarray(
        [float(n.buses_t.v_ang.loc[snap, bn]) for bn in bus_names], dtype=float
    )

    if not hasattr(n, "lines_t") or not all(
        hasattr(n.lines_t, a) for a in ("p0", "q0", "p1", "q1")
    ):
        raise RuntimeError(
            "PyPSA AC PF did not produce line flow results (lines_t.p0/q0/p1/q1)."
        )

    p0 = np.zeros(len(idx), dtype=float)
    q0 = np.zeros(len(idx), dtype=float)
    p1 = np.zeros(len(idx), dtype=float)
    q1 = np.zeros(len(idx), dtype=float)

    for pos, lid in enumerate(idx):
        if not bool(in_service_flags.get(int(lid), True)):
            continue
        name = f"line_{int(lid)}"
        p0[pos] = float(n.lines_t.p0.loc[snap, name])
        q0[pos] = float(n.lines_t.q0.loc[snap, name])
        p1[pos] = float(n.lines_t.p1.loc[snap, name])
        q1[pos] = float(n.lines_t.q1.loc[snap, name])

    if np.max(np.abs(v_ang)) > 10.0:
        raise ValueError(
            "Unexpectedly large bus voltage angles from PyPSA PF. "
            "Expected radians; got max|angle| > 10. Check PyPSA version/units."
        )

    status = "PYPSA_PF_OK"
    logger.info(
        "PyPSA AC PF done: status=%s v_mag_pu[min,median,max]=[%.6g,%.6g,%.6g]",
        status,
        float(np.min(v_mag)),
        float(np.median(v_mag)),
        float(np.max(v_mag)),
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
        status=status,
    )
