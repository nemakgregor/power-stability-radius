from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from stability_radius.config import DEFAULT_OPF, OPFConfig
from stability_radius.dc.dc_model import trafo_tap_ratio, trafo_x_total_ohm

logger = logging.getLogger(__name__)

_AC_CARRIER = "AC"

# NOTE:
# ext_grid is a "slack-like" unlimited source used to keep the OPF feasible when real generators
# are missing/insufficient after MATPOWER->pandapower conversion.
#
# It must be expensive (so it is used only if needed), but NOT astronomically expensive:
# huge costs (e.g. 1e6) hurt LP scaling in HiGHS and can lead to small primal feasibility drift,
# which then shows up as OPF->DCOperator flow mismatches at MW level.
_EXT_GRID_MARGINAL_COST_BASE = 1_000.0

_X_OHM_EPS = 1e-12
_SHIFT_DEG_EPS = 1e-9


@dataclass(frozen=True)
class PyPSAOPFResult:
    """
    Minimal OPF result required by the rest of this project.

    Notes
    -----
    - bus_injections_mw is aligned with bus_ids ordering, and should be balanced
      (sum ~= 0) up to solver tolerances.
    - line_flows_mw is aligned with the provided pandapower line_indices ordering.

    Units
    -----
    - P / injections / flows: MW
    - bus v_nom: kV
    - line r/x: Ohm (PyPSA expects Ohm; it converts to p.u. internally)
    """

    line_flows_mw: np.ndarray  # aligned with provided pandapower line_indices ordering
    bus_ids: tuple[int, ...]  # sorted pandapower net.bus.index
    bus_injections_mw: np.ndarray  # aligned with bus_ids
    status: str
    objective: float


def _is_in_service(row: Any) -> bool:
    """Return pandapower element in_service flag."""
    return bool(row.get("in_service", True))


def _bus_vn_kv(net: Any, bus_id: int) -> float:
    """Return bus nominal voltage vn_kv if available."""
    if bus_id in net.bus.index and "vn_kv" in net.bus.columns:
        return float(net.bus.loc[bus_id, "vn_kv"])
    return float("nan")


def _z_base_ohm(*, vn_kv: float, sn_mva: float) -> float:
    """Z_base in Ohm: Z = V_kV^2 / S_MVA."""
    v = float(vn_kv)
    s = float(sn_mva)
    if not math.isfinite(v) or v <= 0:
        return 0.0
    if not math.isfinite(s) or s <= 0:
        return 0.0
    return (v * v) / s


def _line_r_x_ohm_from_pp(
    net: Any, line_row: Any, *, strict_units: bool
) -> tuple[float, float]:
    """
    Convert pandapower line parameters (Ohm/km) to total r,x in Ohm for PyPSA.

    Important (PyPSA unit contract)
    -------------------------------
    PyPSA `Line` expects:
      - r in Ohm
      - x in Ohm

    Strict units
    ------------
    If strict_units=True, require x_ohm > 0. If False, allow negative x_ohm
    (but still reject near-zero |x_ohm| which would be singular in DC).
    """
    fb = int(line_row.get("from_bus", -1))
    vn_kv = _bus_vn_kv(net, fb)
    if not math.isfinite(vn_kv) or vn_kv <= 0:
        raise ValueError(f"Invalid vn_kv for from_bus={fb} (vn_kv={vn_kv})")

    x_ohm_per_km = float(line_row.get("x_ohm_per_km", 0.0))
    length_km = float(line_row.get("length_km", 0.0))
    parallel = float(line_row.get("parallel", 1.0))
    if not math.isfinite(parallel) or parallel <= 0:
        parallel = 1.0

    x_ohm = x_ohm_per_km * length_km / parallel
    if not math.isfinite(x_ohm) or abs(x_ohm) <= _X_OHM_EPS:
        raise ValueError(f"Invalid series reactance for line: x_ohm={x_ohm}")

    if bool(strict_units) and float(x_ohm) <= 0.0:
        raise ValueError(
            "strict_units: line series reactance must be >0 (Ohm). "
            f"Got x_ohm={x_ohm} for from_bus={fb}."
        )

    # Lossless DC policy: resistances are ignored.
    r_ohm = 0.0
    return float(r_ohm), float(x_ohm)


def _impedance_x_ohm_from_pp(net: Any, imp_row: Any, *, strict_units: bool) -> float:
    """
    Convert pandapower impedance element to series reactance in Ohm.

    pandapower impedance stores xft_pu (per unit on system base).
    For PyPSA Line we need x in Ohm:
        x_ohm = x_pu * (V_kV^2 / S_MVA)

    Strict units
    ------------
    If strict_units=True, require x_ohm > 0 (equivalently x_pu > 0).
    """
    sn_system = float(getattr(net, "sn_mva", np.nan))
    if not math.isfinite(sn_system) or sn_system <= 0:
        raise ValueError("pandapower net.sn_mva must be finite and positive for OPF")

    x_pu = float(imp_row.get("xft_pu", np.nan))
    if not math.isfinite(x_pu) or abs(x_pu) <= _X_OHM_EPS:
        raise ValueError(f"Invalid impedance xft_pu={x_pu}")

    if bool(strict_units) and float(x_pu) <= 0.0:
        raise ValueError(
            f"strict_units: impedance xft_pu must be >0 (per unit). Got xft_pu={x_pu}."
        )

    fb = int(imp_row.get("from_bus", -1))
    vn_kv = _bus_vn_kv(net, fb)
    if not math.isfinite(vn_kv) or vn_kv <= 0:
        raise ValueError(f"Invalid vn_kv for impedance from_bus={fb} (vn_kv={vn_kv})")

    z_base = _z_base_ohm(vn_kv=vn_kv, sn_mva=sn_system)
    if not math.isfinite(z_base) or z_base <= 0:
        raise ValueError("Invalid z_base computed from vn_kv and sn_mva")

    x_ohm = float(x_pu * z_base)
    if not math.isfinite(x_ohm) or abs(x_ohm) <= _X_OHM_EPS:
        raise ValueError(f"Invalid impedance x_ohm={x_ohm}")

    if bool(strict_units) and float(x_ohm) <= 0.0:
        raise ValueError(
            "strict_units: impedance series reactance must be >0 (Ohm). "
            f"Got x_ohm={x_ohm}."
        )

    return float(x_ohm)


def _sum_p_by_bus(net: Any, table_name: str, *, p_col: str) -> dict[int, float]:
    """Sum active power per bus for a pandapower element table."""
    if not hasattr(net, table_name):
        return {}
    table = getattr(net, table_name)
    if table is None or len(table) == 0:
        return {}
    if "bus" not in table.columns:
        return {}

    out: dict[int, float] = {}
    for _, row in table.iterrows():
        if not _is_in_service(row):
            continue
        bus = int(row["bus"])
        p = float(row.get(p_col, 0.0))
        if not math.isfinite(p):
            continue
        out[bus] = out.get(bus, 0.0) + p
    return out


def _pp_gen_p_bounds_to_pypsa(
    *, gid: int, p_min_mw: float, p_max_mw: float
) -> tuple[float, float] | None:
    """
    Convert pandapower generator active power bounds into PyPSA parameters.

    Returns
    -------
    (p_nom, p_min_pu) or None
        - Returns None if the generator has no active power capability for this model
          (p_max_mw <= 0). This is common for synchronous condensers / reactive-only devices
          in MATPOWER/PGLib converted networks.
        - Raises ValueError for non-finite bounds or inconsistent bounds (p_min_mw > p_max_mw).

    Notes
    -----
    - This project uses p_max_pu=1.0 for included generators, thus p_nom == p_max_mw.
    - We intentionally do NOT try to "guess" missing/invalid bounds (no heuristics).
    """
    p_min = float(p_min_mw)
    p_max = float(p_max_mw)

    if not math.isfinite(p_min):
        raise ValueError(f"Invalid min_p_mw for pandapower gen {gid}: {p_min}")
    if not math.isfinite(p_max):
        raise ValueError(f"Invalid max_p_mw for pandapower gen {gid}: {p_max}")

    # Allow converted networks to contain in-service "generators" with zero
    # active capability (e.g., synchronous condensers). For DC OPF base-point we skip them.
    if p_max <= 0.0:
        return None

    if p_min > p_max:
        raise ValueError(
            f"Inconsistent active power bounds for pandapower gen {gid}: "
            f"min_p_mw={p_min} > max_p_mw={p_max}"
        )

    p_nom = float(p_max)
    p_min_pu = float(p_min / p_nom) if p_nom > 0 else 0.0
    return p_nom, p_min_pu


def _ensure_carrier_table(n: Any, carrier_name: str) -> None:
    """
    Ensure that `carrier_name` exists in `n.carriers`.

    Why this exists
    ---------------
    Some PyPSA versions emit warnings when a component's `carrier` is set but the carrier
    is missing in `network.carriers`. We explicitly define the carrier to avoid ambiguous
    AC/DC network interpretation and keep logs clean/deterministic.
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


def solve_dc_opf_base_flows_from_pandapower(
    *,
    net: Any,
    line_indices: Sequence[int],
    line_limits_mw: np.ndarray,
    opf_cfg: OPFConfig | None = None,
    strict_units: bool = True,
    allow_phase_shift: bool = False,
) -> PyPSAOPFResult:
    """
    Solve a single-snapshot DC OPF using PyPSA + HiGHS and return base line flows.

    Unit contract (important)
    -------------------------
    - bus vn_kv / PyPSA Bus.v_nom: kV
    - branch r/x / PyPSA Line.r/x: Ohm
    - thermal limits:
        * input `line_limits_mw` is treated as **MVA assumed MW** (PF=1 under DC).
        * PyPSA expects s_nom in MVA; for DC (P-only) PF=1 implies MVA==MW.

    strict_units
    ------------
    - If True: require x_ohm > 0 for lines/impedances and net.sn_mva > 0.
    - If False: allow negative x_ohm (but still reject |x| ~ 0).

    Phase shifters (shift_degree)
    -----------------------------
    The OPF model and the project's DCOperator do not model phase shifting transformers.
    Therefore:
    - allow_phase_shift=False (default): raise on any shift_degree != 0.
    - allow_phase_shift=True: ignore the shift and log a WARNING.
    """
    cfg = opf_cfg if opf_cfg is not None else DEFAULT_OPF
    strict = bool(strict_units)
    allow_shift = bool(allow_phase_shift)

    try:
        import pandas as pd
        import pypsa
    except ImportError as e:
        raise ImportError(
            "PyPSA (and pandas) is required for OPF base-point generation."
        ) from e

    idx = [int(x) for x in line_indices]
    limits = np.asarray(line_limits_mw, dtype=float).reshape(-1)
    if limits.shape != (len(idx),):
        raise ValueError(
            f"line_limits_mw must have shape ({len(idx)},), got {limits.shape}"
        )

    solver_name = str(cfg.highs.solver_name)
    if solver_name.lower() != "highs":
        raise ValueError(
            f"Project policy violation: solver must be 'highs', got {solver_name!r}"
        )

    unconstrained_nom = float(cfg.unconstrained_line_nom_mw)
    if not math.isfinite(unconstrained_nom) or unconstrained_nom <= 0:
        raise ValueError(
            "opf_cfg.unconstrained_line_nom_mw must be finite and >0 "
            f"(got {cfg.unconstrained_line_nom_mw!r})"
        )

    logger.debug(
        "Building PyPSA network for lossless DC OPF. Policy: r=0.0, x in Ohm for PyPSA Line. "
        "strict_units=%s allow_phase_shift=%s",
        bool(strict),
        bool(allow_shift),
    )

    n = pypsa.Network()
    n.set_snapshots(pd.Index([0]))

    # Explicitly define AC carrier to avoid PyPSA warnings and accidental DC network semantics.
    _ensure_carrier_table(n, _AC_CARRIER)

    sn_mva = float(getattr(net, "sn_mva", np.nan))
    if strict and (not math.isfinite(sn_mva) or sn_mva <= 0.0):
        raise ValueError(
            f"strict_units: pandapower net.sn_mva must be finite and >0; got {sn_mva!r}"
        )
    if math.isfinite(sn_mva) and sn_mva > 0:
        n.sn_mva = sn_mva

    # buses (stable ordering)
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    for b in bus_ids:
        vn_kv = _bus_vn_kv(net, b)
        if not math.isfinite(vn_kv) or vn_kv <= 0:
            raise ValueError(f"Invalid vn_kv for bus {b}: {vn_kv}")

        bus_kwargs: dict[str, Any] = {"v_nom": float(vn_kv)}
        # Bus has carrier field in modern PyPSA. Keep it guarded for compatibility.
        if hasattr(n, "buses") and "carrier" in getattr(n, "buses").columns:
            bus_kwargs["carrier"] = _AC_CARRIER

        n.add("Bus", str(b), **bus_kwargs)

    # loads: aggregate per bus (pandapower load + shunt p_mw)
    load_by_bus = _sum_p_by_bus(net, "load", p_col="p_mw")
    shunt_p_by_bus = _sum_p_by_bus(net, "shunt", p_col="p_mw")
    if shunt_p_by_bus:
        for bus, p in shunt_p_by_bus.items():
            load_by_bus[bus] = load_by_bus.get(bus, 0.0) + float(p)
        logger.debug(
            "Included pandapower shunt p_mw into OPF loads: buses=%d, total_shunt_p_mw=%.6g",
            int(len(shunt_p_by_bus)),
            float(sum(shunt_p_by_bus.values())),
        )

    total_load = float(sum(load_by_bus.values()))
    for b in sorted(load_by_bus.keys()):
        p = float(load_by_bus[b])
        if abs(p) <= 0.0:
            continue
        n.add("Load", f"load_{b}", bus=str(b), p_set=float(p))

    # generators from pandapower gen/ext_grid
    gen_rank = 0
    skipped_nonpositive_pmax: list[int] = []

    if hasattr(net, "gen") and net.gen is not None and len(net.gen):
        bus_id_set = set(bus_ids)
        for gid in [int(x) for x in sorted(net.gen.index)]:
            row = net.gen.loc[gid]
            if not _is_in_service(row):
                continue

            bus = int(row.get("bus", -1))
            if bus not in bus_id_set:
                raise ValueError(f"pandapower gen {gid} refers to missing bus {bus}")

            try:
                p_min = float(row.get("min_p_mw", 0.0))
                p_max = float(row.get("max_p_mw", np.nan))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Failed to parse min_p_mw/max_p_mw for pandapower gen {gid}."
                ) from e

            bounds = _pp_gen_p_bounds_to_pypsa(gid=gid, p_min_mw=p_min, p_max_mw=p_max)
            if bounds is None:
                skipped_nonpositive_pmax.append(int(gid))
                logger.debug(
                    "Skipping pandapower gen %d (bus=%d): non-positive max_p_mw=%.6g (min_p_mw=%.6g).",
                    int(gid),
                    int(bus),
                    float(p_max),
                    float(p_min),
                )
                continue

            p_nom, p_min_pu = bounds
            gen_rank += 1
            n.add(
                "Generator",
                f"gen_{gid}",
                bus=str(bus),
                p_nom=float(p_nom),
                p_min_pu=float(p_min_pu),
                p_max_pu=1.0,
                marginal_cost=float(gen_rank),
            )

    if skipped_nonpositive_pmax:
        logger.warning(
            "Skipped %d in-service pandapower gen(s) with non-positive max_p_mw (reactive-only / zero-capacity). "
            "First ids: %s",
            int(len(skipped_nonpositive_pmax)),
            skipped_nonpositive_pmax[:20],
        )

    if hasattr(net, "ext_grid") and net.ext_grid is not None and len(net.ext_grid):
        p_nom_ext = max(float(total_load), 1.0)
        bus_id_set = set(bus_ids)
        for eid in [int(x) for x in sorted(net.ext_grid.index)]:
            row = net.ext_grid.loc[eid]
            if not _is_in_service(row):
                continue
            bus = int(row.get("bus", -1))
            if bus not in bus_id_set:
                raise ValueError(
                    f"pandapower ext_grid {eid} refers to missing bus {bus}"
                )

            gen_rank += 1
            mc = float(_EXT_GRID_MARGINAL_COST_BASE + gen_rank)
            logger.debug(
                "Adding ext_grid ext_%d: bus=%d, p_nom=%.6g MW, marginal_cost=%.6g",
                int(eid),
                int(bus),
                float(p_nom_ext),
                float(mc),
            )
            n.add(
                "Generator",
                f"ext_{eid}",
                bus=str(bus),
                p_nom=float(p_nom_ext),
                p_min_pu=0.0,
                p_max_pu=1.0,
                marginal_cost=float(mc),
            )

    if len(n.generators.index) == 0:
        raise RuntimeError(
            "No generators found in pandapower net (gen/ext_grid). Cannot solve OPF."
        )

    # monitored lines (net.line) with provided limits
    in_service_flags: dict[int, bool] = {}
    bus_id_set = set(bus_ids)
    for pos, lid in enumerate(idx):
        row = net.line.loc[lid]
        in_service = bool(_is_in_service(row))
        in_service_flags[int(lid)] = in_service

        if not in_service:
            continue

        fb = int(row.get("from_bus", -1))
        tb = int(row.get("to_bus", -1))
        if fb not in bus_id_set or tb not in bus_id_set:
            raise ValueError(f"Line {lid} refers to missing buses {fb}->{tb}")

        r_ohm, x_ohm = _line_r_x_ohm_from_pp(net, row, strict_units=strict)

        s_nom = float(limits[pos])
        if not math.isfinite(s_nom) or math.isinf(s_nom):
            s_nom = unconstrained_nom
        if s_nom < 0:
            raise ValueError(f"Negative line limit for line {lid}: {limits[pos]}")

        n.add(
            "Line",
            f"line_{lid}",
            bus0=str(fb),
            bus1=str(tb),
            r=float(r_ohm),
            x=float(x_ohm),
            s_nom=float(s_nom),
        )

    # add trafos/impedances as additional lines into the DC model (unconstrained)
    if hasattr(net, "trafo") and net.trafo is not None and len(net.trafo):
        for tid in [int(x) for x in sorted(net.trafo.index)]:
            row = net.trafo.loc[tid]
            if not _is_in_service(row):
                continue
            hv = int(row.get("hv_bus", -1))
            lv = int(row.get("lv_bus", -1))
            if hv not in bus_id_set or lv not in bus_id_set:
                raise ValueError(f"Trafo {tid} refers to missing buses {hv}->{lv}")

            # Phase shift check: by default reject because our surrogate ignores it.
            shift_deg = float(row.get("shift_degree", 0.0))
            if math.isfinite(shift_deg) and abs(float(shift_deg)) > _SHIFT_DEG_EPS:
                if not allow_shift:
                    raise ValueError(
                        f"Trafo {tid}: non-zero shift_degree={shift_deg} deg is not supported. "
                        "Set allow_phase_shift=True only if you explicitly accept ignoring phase shifters."
                    )
                logger.warning(
                    "Trafo %d has non-zero shift_degree=%.6g deg; phase shift is ignored (allow_phase_shift=1).",
                    int(tid),
                    float(shift_deg),
                )

            x_ohm = float(trafo_x_total_ohm(net, row, strict_units=strict))
            if not math.isfinite(x_ohm) or abs(x_ohm) <= _X_OHM_EPS:
                raise ValueError(f"Invalid trafo x_ohm={x_ohm} for trafo {tid}")
            if strict and float(x_ohm) <= 0.0:
                raise ValueError(
                    f"strict_units: trafo series reactance must be >0 (Ohm). Got x_ohm={x_ohm} for trafo {tid}."
                )

            # Apply MATPOWER-style DC tap modeling: x_eff = x * tap  (=> b_eff = b / tap)
            tap = float(trafo_tap_ratio(row))
            x_ohm_eff = float(x_ohm * tap)

            if tap != 1.0:
                logger.debug(
                    "Trafo %d: applying DC tap ratio in OPF model: tap=%.6g, x_ohm=%.6g, x_ohm_eff=%.6g",
                    int(tid),
                    float(tap),
                    float(x_ohm),
                    float(x_ohm_eff),
                )

            n.add(
                "Line",
                f"trafo_{tid}",
                bus0=str(hv),
                bus1=str(lv),
                r=0.0,
                x=float(x_ohm_eff),
                s_nom=unconstrained_nom,
            )

    if hasattr(net, "impedance") and net.impedance is not None and len(net.impedance):
        for iid in [int(x) for x in sorted(net.impedance.index)]:
            row = net.impedance.loc[iid]
            if not _is_in_service(row):
                continue
            fb = int(row.get("from_bus", -1))
            tb = int(row.get("to_bus", -1))
            if fb not in bus_id_set or tb not in bus_id_set:
                raise ValueError(f"Impedance {iid} refers to missing buses {fb}->{tb}")

            x_ohm = _impedance_x_ohm_from_pp(net, row, strict_units=strict)

            n.add(
                "Line",
                f"impedance_{iid}",
                bus0=str(fb),
                bus1=str(tb),
                r=0.0,
                x=float(x_ohm),
                s_nom=unconstrained_nom,
            )

    logger.info(
        "Solving PyPSA DC OPF (HiGHS): buses=%d, loads=%d, generators=%d, lines(monitored,in_service)=%d",
        int(len(n.buses.index)),
        int(len(n.loads.index)),
        int(len(n.generators.index)),
        int(sum(bool(v) for v in in_service_flags.values())),
    )

    if not hasattr(n, "optimize"):
        raise RuntimeError(
            "Unsupported PyPSA version: Network.optimize() is required by this project."
        )

    try:
        res = n.optimize(
            solver_name=solver_name, solver_options=cfg.highs.solver_options()
        )
    except Exception as e:
        logger.exception("PyPSA OPF failed: %s", e)
        raise RuntimeError("PyPSA OPF failed.") from e

    objective = float(getattr(n, "objective", float("nan")))
    status = str(getattr(res, "status", "ok")) if res is not None else "ok"

    if not hasattr(n, "lines_t") or not hasattr(n.lines_t, "p0"):
        raise RuntimeError(
            "PyPSA did not produce line flow results (lines_t.p0 missing)."
        )

    if not hasattr(n, "generators_t") or not hasattr(n.generators_t, "p"):
        raise RuntimeError(
            "PyPSA did not produce generator dispatch results (generators_t.p missing)."
        )

    snap = n.snapshots[0]

    # line flows for monitored pandapower net.line entries only (aligned with idx)
    flows: list[float] = []
    for lid in idx:
        if not bool(in_service_flags.get(int(lid), True)):
            flows.append(0.0)
            continue

        name = f"line_{lid}"
        v = float(n.lines_t.p0.loc[snap, name])
        flows.append(v)

    # bus injections for OPF->DC consistency checks
    bus_names = [str(b) for b in bus_ids]

    gen_p = n.generators_t.p.loc[snap, :]
    gen_bus = n.generators.bus
    gen_by_bus = gen_p.groupby(gen_bus).sum()

    if len(n.loads.index) > 0:
        load_by_bus = n.loads.p_set.groupby(n.loads.bus).sum()
    else:
        load_by_bus = gen_by_bus.iloc[0:0].copy()

    inj_by_bus = gen_by_bus.reindex(bus_names, fill_value=0.0) - load_by_bus.reindex(
        bus_names, fill_value=0.0
    )
    bus_inj = np.asarray(
        [float(inj_by_bus.get(str(b), 0.0)) for b in bus_ids], dtype=float
    )

    inj_sum = float(np.sum(bus_inj))
    if abs(inj_sum) > 1e-6:
        logger.warning(
            "PyPSA OPF produced non-zero total injection sum=%.6g MW (should be ~0). "
            "This may indicate solver tolerances or model inconsistency.",
            inj_sum,
        )
    else:
        logger.debug("OPF bus injection balance check: sum=%.6g MW", inj_sum)

    out = PyPSAOPFResult(
        line_flows_mw=np.asarray(flows, dtype=float),
        bus_ids=tuple(bus_ids),
        bus_injections_mw=bus_inj,
        status=str(status),
        objective=float(objective),
    )

    logger.info(
        "PyPSA OPF done: status=%s, objective=%s",
        out.status,
        f"{out.objective:.6g}" if math.isfinite(out.objective) else "n/a",
    )
    return out
