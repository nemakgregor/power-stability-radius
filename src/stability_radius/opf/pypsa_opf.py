from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from stability_radius.config import DEFAULT_OPF
from stability_radius.dc.dc_model import trafo_x_total_ohm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PyPSAOPFResult:
    """Minimal OPF result required by the rest of this project."""

    line_flows_mw: np.ndarray  # aligned with provided pandapower line_indices ordering
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


def _line_r_x_pu_from_pp(net: Any, line_row: Any) -> tuple[float, float]:
    """
    Convert pandapower line parameters (Ohm/km) to per-unit total r,x for PyPSA.
    """
    sn_mva = float(getattr(net, "sn_mva", np.nan))
    if not math.isfinite(sn_mva) or sn_mva <= 0:
        raise ValueError("pandapower net.sn_mva must be finite and positive for OPF")

    fb = int(line_row.get("from_bus", -1))
    vn_kv = _bus_vn_kv(net, fb)
    if not math.isfinite(vn_kv) or vn_kv <= 0:
        raise ValueError(f"Invalid vn_kv for from_bus={fb} (vn_kv={vn_kv})")

    r_ohm_per_km = float(line_row.get("r_ohm_per_km", 0.0))
    x_ohm_per_km = float(line_row.get("x_ohm_per_km", 0.0))
    length_km = float(line_row.get("length_km", 0.0))
    parallel = float(line_row.get("parallel", 1.0))
    if not math.isfinite(parallel) or parallel <= 0:
        parallel = 1.0

    r_ohm = r_ohm_per_km * length_km / parallel
    x_ohm = x_ohm_per_km * length_km / parallel

    z_base = _z_base_ohm(vn_kv=vn_kv, sn_mva=sn_mva)
    if not math.isfinite(z_base) or z_base <= 0:
        raise ValueError("Invalid z_base computed from vn_kv and sn_mva")

    r_pu = r_ohm / z_base
    x_pu = x_ohm / z_base
    if not math.isfinite(x_pu) or abs(x_pu) <= 1e-12:
        raise ValueError(f"Invalid per-unit reactance for line: x_pu={x_pu}")
    return float(r_pu), float(x_pu)


def _trafo_x_pu_from_pp(net: Any, trafo_row: Any) -> float:
    """
    Approximate transformer series reactance in per unit on system base.
    """
    sn_system = float(getattr(net, "sn_mva", np.nan))
    if not math.isfinite(sn_system) or sn_system <= 0:
        raise ValueError("pandapower net.sn_mva must be finite and positive for OPF")

    hv_bus = int(trafo_row.get("hv_bus", -1))
    vn_hv_kv = float(trafo_row.get("vn_hv_kv", np.nan))
    if not math.isfinite(vn_hv_kv) or vn_hv_kv <= 0:
        vn_hv_kv = _bus_vn_kv(net, hv_bus)

    if not math.isfinite(vn_hv_kv) or vn_hv_kv <= 0:
        raise ValueError(
            f"Invalid hv vn_kv for trafo hv_bus={hv_bus} (vn_kv={vn_hv_kv})"
        )

    x_ohm = float(trafo_x_total_ohm(net, trafo_row))
    if not math.isfinite(x_ohm) or abs(x_ohm) <= 1e-12:
        raise ValueError(f"Invalid trafo x_ohm={x_ohm}")

    z_base_system = _z_base_ohm(vn_kv=vn_hv_kv, sn_mva=sn_system)
    if not math.isfinite(z_base_system) or z_base_system <= 0:
        raise ValueError("Invalid system z_base for trafo conversion")
    x_pu = float(x_ohm / z_base_system)
    if not math.isfinite(x_pu) or abs(x_pu) <= 1e-12:
        raise ValueError(f"Invalid per-unit reactance for trafo: x_pu={x_pu}")
    return x_pu


def _impedance_x_pu_from_pp(net: Any, imp_row: Any) -> float:
    """Convert pandapower impedance element to x_pu on system base."""
    x_pu = float(imp_row.get("xft_pu", np.nan))
    if not math.isfinite(x_pu) or abs(x_pu) <= 1e-12:
        raise ValueError(f"Invalid impedance xft_pu={x_pu}")
    return float(x_pu)


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

    # Key fix: allow converted networks to contain in-service "generators" with zero
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


def solve_dc_opf_base_flows_from_pandapower(
    *,
    net: Any,
    line_indices: Sequence[int],
    line_limits_mw: np.ndarray,
) -> PyPSAOPFResult:
    """
    Solve a single-snapshot DC OPF using PyPSA + HiGHS and return base line flows.

    Project policy
    --------------
    - Solver is enforced globally: HiGHS (see stability_radius.config.DEFAULT_OPF).
    - No legacy API fallbacks (requires modern PyPSA with Network.optimize()).
    """
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

    solver_name = str(DEFAULT_OPF.highs.solver_name)
    if solver_name.lower() != "highs":
        raise ValueError(
            f"Project policy violation: solver must be 'highs', got {solver_name!r}"
        )

    unconstrained_nom = float(DEFAULT_OPF.unconstrained_line_nom_mw)
    if not math.isfinite(unconstrained_nom) or unconstrained_nom <= 0:
        raise ValueError(
            "DEFAULT_OPF.unconstrained_line_nom_mw must be finite and >0 "
            f"(got {DEFAULT_OPF.unconstrained_line_nom_mw!r})"
        )

    n = pypsa.Network()
    n.set_snapshots(pd.Index([0]))

    sn_mva = float(getattr(net, "sn_mva", np.nan))
    if math.isfinite(sn_mva) and sn_mva > 0:
        n.sn_mva = sn_mva

    # buses (stable ordering)
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    for b in bus_ids:
        vn_kv = _bus_vn_kv(net, b)
        if not math.isfinite(vn_kv) or vn_kv <= 0:
            raise ValueError(f"Invalid vn_kv for bus {b}: {vn_kv}")
        n.add("Bus", str(b), v_nom=float(vn_kv))

    # loads: aggregate per bus
    load_by_bus = _sum_p_by_bus(net, "load", p_col="p_mw")
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
            n.add(
                "Generator",
                f"ext_{eid}",
                bus=str(bus),
                p_nom=float(p_nom_ext),
                p_min_pu=0.0,
                p_max_pu=1.0,
                marginal_cost=float(1_000_000 + gen_rank),
            )

    if len(n.generators.index) == 0:
        raise RuntimeError(
            "No generators found in pandapower net (gen/ext_grid). Cannot solve OPF."
        )

    # monitored lines (net.line) with provided limits
    in_service_flags: dict[int, bool] = {}
    for pos, lid in enumerate(idx):
        row = net.line.loc[lid]
        in_service = bool(_is_in_service(row))
        in_service_flags[int(lid)] = in_service

        if not in_service:
            continue

        fb = int(row.get("from_bus", -1))
        tb = int(row.get("to_bus", -1))
        if fb not in set(bus_ids) or tb not in set(bus_ids):
            raise ValueError(f"Line {lid} refers to missing buses {fb}->{tb}")

        r_pu, x_pu = _line_r_x_pu_from_pp(net, row)

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
            r=float(r_pu),
            x=float(x_pu),
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
            if hv not in set(bus_ids) or lv not in set(bus_ids):
                raise ValueError(f"Trafo {tid} refers to missing buses {hv}->{lv}")

            x_pu = _trafo_x_pu_from_pp(net, row)
            n.add(
                "Line",
                f"trafo_{tid}",
                bus0=str(hv),
                bus1=str(lv),
                r=0.0,
                x=float(x_pu),
                s_nom=unconstrained_nom,
            )

    if hasattr(net, "impedance") and net.impedance is not None and len(net.impedance):
        for iid in [int(x) for x in sorted(net.impedance.index)]:
            row = net.impedance.loc[iid]
            if not _is_in_service(row):
                continue
            fb = int(row.get("from_bus", -1))
            tb = int(row.get("to_bus", -1))
            if fb not in set(bus_ids) or tb not in set(bus_ids):
                raise ValueError(f"Impedance {iid} refers to missing buses {fb}->{tb}")

            x_pu = _impedance_x_pu_from_pp(net, row)
            n.add(
                "Line",
                f"impedance_{iid}",
                bus0=str(fb),
                bus1=str(tb),
                r=0.0,
                x=float(x_pu),
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
            solver_name=solver_name, solver_options=DEFAULT_OPF.highs.solver_options()
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

    snap = n.snapshots[0]
    flows: list[float] = []
    for lid in idx:
        if not bool(in_service_flags.get(int(lid), True)):
            flows.append(0.0)
            continue

        name = f"line_{lid}"
        v = float(n.lines_t.p0.loc[snap, name])
        flows.append(v)

    out = PyPSAOPFResult(
        line_flows_mw=np.asarray(flows, dtype=float),
        status=str(status),
        objective=float(objective),
    )

    logger.info(
        "PyPSA OPF done: status=%s, objective=%s",
        out.status,
        f"{out.objective:.6g}" if math.isfinite(out.objective) else "n/a",
    )
    return out
