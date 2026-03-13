from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from stability_radius.config import DEFAULT_OPF, OPFConfig
from stability_radius.dc.dc_model import trafo_tap_ratio

logger = logging.getLogger(__name__)

_AC_CARRIER = "AC"
_X_OHM_EPS = 1e-12


@dataclass(frozen=True)
class PyPSAOPFResult:
    """
    Minimal OPF result required by the rest of this project.

    Units
    -----
    - P / injections / flows: MW
    - line r/x: Ohm (PyPSA expects Ohm)
    """

    line_flows_mw: np.ndarray
    bus_ids: tuple[int, ...]
    bus_injections_mw: np.ndarray
    status: str
    objective: float
    gen_dispatch_mw_by_name: tuple[tuple[str, float], ...] = ()


def _is_in_service(row: Any) -> bool:
    return bool(row.get("in_service", True))


def _bus_vn_kv(net: Any, bus_id: int) -> float:
    if bus_id in net.bus.index and "vn_kv" in net.bus.columns:
        return float(net.bus.loc[bus_id, "vn_kv"])
    return float("nan")


def _line_r_x_ohm_from_pp(
    net: Any, line_row: Any, *, line_id: int
) -> tuple[float, float]:
    fb = int(line_row.get("from_bus", -1))
    tb = int(line_row.get("to_bus", -1))

    vn_kv = _bus_vn_kv(net, fb)
    if not math.isfinite(vn_kv) or vn_kv <= 0:
        raise ValueError(
            f"Line {int(line_id)}: invalid vn_kv for from_bus={fb} (vn_kv={vn_kv!r})"
        )

    x_ohm_per_km = float(line_row.get("x_ohm_per_km", np.nan))
    length_km = float(line_row.get("length_km", np.nan))
    parallel = float(line_row.get("parallel", 1.0))

    if not math.isfinite(x_ohm_per_km):
        raise ValueError(
            f"Line {int(line_id)}: x_ohm_per_km must be finite; got {x_ohm_per_km!r}"
        )
    if not math.isfinite(length_km):
        raise ValueError(
            f"Line {int(line_id)}: length_km must be finite; got {length_km!r}"
        )
    if not math.isfinite(parallel) or parallel <= 0.0:
        raise ValueError(
            f"Line {int(line_id)}: parallel must be finite and >0; got {parallel!r}"
        )

    x_ohm = x_ohm_per_km * length_km / parallel
    if not math.isfinite(x_ohm) or abs(float(x_ohm)) <= _X_OHM_EPS:
        raise ValueError(
            "Invalid series reactance for line. "
            f"line_id={int(line_id)} from_bus={fb} to_bus={tb} "
            f"=> x_ohm={x_ohm!r}"
        )

    # Lossless DC policy
    return 0.0, float(x_ohm)


def _sum_p_by_bus(net: Any, table_name: str, *, p_col: str) -> dict[int, float]:
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
    p_min = float(p_min_mw)
    p_max = float(p_max_mw)

    if not math.isfinite(p_min):
        raise ValueError(f"Invalid min_p_mw for pandapower gen {gid}: {p_min}")
    if not math.isfinite(p_max):
        raise ValueError(f"Invalid max_p_mw for pandapower gen {gid}: {p_max}")

    if p_max <= 0.0:
        return None
    if p_min > p_max:
        raise ValueError(
            f"Inconsistent bounds for pandapower gen {gid}: min_p_mw={p_min} > max_p_mw={p_max}"
        )

    p_nom = float(p_max)
    p_min_pu = float(p_min / p_nom) if p_nom > 0 else 0.0
    return p_nom, p_min_pu


def _ensure_carrier_table(n: Any, carrier_name: str) -> None:
    if not hasattr(n, "carriers"):
        return
    try:
        carriers = n.carriers
        if str(carrier_name) in carriers.index:
            return
    except Exception:  # noqa: BLE001
        return
    n.add("Carrier", str(carrier_name))


def _trafo_series_rx_pu_from_pp_row(trafo_row: Any) -> tuple[float, float]:
    vk_percent = float(trafo_row.get("vk_percent", np.nan))
    if not math.isfinite(vk_percent):
        raise ValueError("Trafo: vk_percent invalid")

    vkr_percent = float(trafo_row.get("vkr_percent", 0.0))
    if not math.isfinite(vkr_percent):
        vkr_percent = 0.0

    z_pu = float(vk_percent) / 100.0
    r_pu = float(vkr_percent) / 100.0
    x_pu2 = z_pu * z_pu - r_pu * r_pu
    x_pu = float(math.sqrt(max(x_pu2, 0.0)))
    if x_pu <= 0.0:
        raise ValueError("Trafo: derived x_pu must be >0")
    return r_pu, x_pu


def solve_dc_opf_base_flows_from_pandapower(
    *,
    net: Any,
    line_indices: Sequence[int],
    line_limits_mw: np.ndarray,
    opf_cfg: OPFConfig | None = None,
) -> PyPSAOPFResult:
    """
    Solve a single-snapshot DC OPF using PyPSA + HiGHS and return base line flows.
    """
    cfg = opf_cfg if opf_cfg is not None else DEFAULT_OPF

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
        raise ValueError("opf_cfg.unconstrained_line_nom_mw must be finite and >0")

    ext_grid_cost_base = float(getattr(cfg, "ext_grid_marginal_cost_base", 1000.0))
    if not math.isfinite(ext_grid_cost_base) or ext_grid_cost_base <= 0:
        raise ValueError("opf_cfg.ext_grid_marginal_cost_base must be finite and >0")

    n = pypsa.Network()
    n.set_snapshots(pd.Index([0]))
    _ensure_carrier_table(n, _AC_CARRIER)

    sn_mva = float(getattr(net, "sn_mva", np.nan))
    if not math.isfinite(sn_mva) or sn_mva <= 0.0:
        raise ValueError("pandapower net.sn_mva must be finite and >0")
    n.sn_mva = float(sn_mva)

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    for b in bus_ids:
        vn_kv = _bus_vn_kv(net, b)
        if not math.isfinite(vn_kv) or vn_kv <= 0:
            raise ValueError(f"Invalid vn_kv for bus {b}: {vn_kv}")
        bus_kwargs: dict[str, Any] = {"v_nom": float(vn_kv)}
        if hasattr(n, "buses") and "carrier" in getattr(n, "buses").columns:
            bus_kwargs["carrier"] = _AC_CARRIER
        n.add("Bus", str(b), **bus_kwargs)

    load_by_bus = _sum_p_by_bus(net, "load", p_col="p_mw")
    shunt_p_by_bus = _sum_p_by_bus(net, "shunt", p_col="p_mw")
    for bus, p in shunt_p_by_bus.items():
        load_by_bus[bus] = load_by_bus.get(bus, 0.0) + float(p)

    total_load = float(sum(load_by_bus.values()))
    for b in sorted(load_by_bus.keys()):
        p = float(load_by_bus[b])
        if abs(p) <= 0.0:
            continue
        n.add("Load", f"load_{b}", bus=str(b), p_set=float(p))

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

            p_min = float(row.get("min_p_mw", 0.0))
            p_max = float(row.get("max_p_mw", np.nan))
            bounds = _pp_gen_p_bounds_to_pypsa(gid=gid, p_min_mw=p_min, p_max_mw=p_max)
            if bounds is None:
                skipped_nonpositive_pmax.append(int(gid))
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
            "Skipped %d in-service pandapower gen(s) with non-positive max_p_mw. First ids: %s",
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
            mc = float(ext_grid_cost_base + gen_rank)
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

        r_ohm, x_ohm = _line_r_x_ohm_from_pp(net, row, line_id=int(lid))

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
                raise ValueError(f"Trafo {tid}: sn_mva must be finite and >0")

            shift_deg = float(row.get("shift_degree", 0.0))
            if not math.isfinite(shift_deg):
                raise ValueError(f"Trafo {tid}: shift_degree must be finite")

            tap = float(trafo_tap_ratio(row))
            if not math.isfinite(tap) or tap <= 0.0:
                raise ValueError(f"Trafo {tid}: invalid tap_ratio={tap!r}")

            s_nom = float(unconstrained_nom)
            scale = float(s_nom / sn_trafo)

            _r_pu, x_pu = _trafo_series_rx_pu_from_pp_row(row)
            x_pu_scaled = float(x_pu * scale)

            n.add(
                "Transformer",
                f"trafo_{tid}",
                bus0=str(hv),
                bus1=str(lv),
                model="t",
                s_nom=float(s_nom),
                r=0.0,
                x=float(x_pu_scaled),
                tap_ratio=float(tap),
                tap_side=0,
                phase_shift=float(shift_deg),
            )
            added_trafos += 1

    logger.info(
        "Solving PyPSA DC OPF (HiGHS): buses=%d, loads=%d, generators=%d, lines(in_service)=%d, trafos=%d",
        int(len(n.buses.index)),
        int(len(n.loads.index)),
        int(len(n.generators.index)),
        int(sum(bool(v) for v in in_service_flags.values())),
        int(added_trafos),
    )

    if not hasattr(n, "optimize"):
        raise RuntimeError(
            "Unsupported PyPSA version: Network.optimize() is required by this project."
        )

    res = n.optimize(solver_name=solver_name, solver_options=cfg.highs.solver_options())

    objective = float(getattr(n, "objective", float("nan")))
    status = str(getattr(res, "status", "ok")) if res is not None else "ok"

    snap = n.snapshots[0]

    flows: list[float] = []
    for lid in idx:
        if not bool(in_service_flags.get(int(lid), True)):
            flows.append(0.0)
            continue
        flows.append(float(n.lines_t.p0.loc[snap, f"line_{lid}"]))

    bus_names = [str(b) for b in bus_ids]
    gen_p = n.generators_t.p.loc[snap, :]
    gen_bus = n.generators.bus
    gen_by_bus = gen_p.groupby(gen_bus).sum()

    load_by_bus2 = (
        n.loads.p_set.groupby(n.loads.bus).sum()
        if len(n.loads.index) > 0
        else gen_by_bus.iloc[0:0].copy()
    )

    inj_by_bus = gen_by_bus.reindex(bus_names, fill_value=0.0) - load_by_bus2.reindex(
        bus_names, fill_value=0.0
    )
    bus_inj = np.asarray(
        [float(inj_by_bus.get(str(b), 0.0)) for b in bus_ids], dtype=float
    )

    gen_dispatch_pairs = tuple(
        (str(name), float(gen_p.loc[name])) for name in sorted(gen_p.index)
    )

    out = PyPSAOPFResult(
        line_flows_mw=np.asarray(flows, dtype=float),
        bus_ids=tuple(bus_ids),
        bus_injections_mw=bus_inj,
        status=str(status),
        objective=float(objective),
        gen_dispatch_mw_by_name=gen_dispatch_pairs,
    )

    logger.info(
        "PyPSA OPF done: status=%s, objective=%s, gens=%d",
        out.status,
        f"{out.objective:.6g}" if math.isfinite(out.objective) else "n/a",
        int(len(out.gen_dispatch_mw_by_name)),
    )
    return out
