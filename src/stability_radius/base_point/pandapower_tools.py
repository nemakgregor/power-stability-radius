from __future__ import annotations

"""
Shared helpers for working with pandapower networks in a deterministic way.

Design principles
-----------------
- Deterministic behavior: explicit policies and stable ordering.
- No heavy dependencies at import time: functions operate on an already created `net`.
- Explicit error messages: if a requirement is not met, we raise immediately.

Notes about "lossless"
----------------------
The AC certificate implementation assumes a lossless series model (r=0).
For correctness, AC Monte Carlo should use the same policy as the certificate.
"""

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def apply_lossless_policy_to_pandapower_net(net: Any) -> Any:
    """
    Return a deep-copied pandapower net with a deterministic series-only policy applied.

    Policy
    ------
    - net.line.r_ohm_per_km  = 0.0   (lossless lines)
    - net.line.c_nf_per_km   = 0.0   (no shunt charging — series-only model)
    - net.line.g_us_per_km   = 0.0   (no shunt conductance)
    - net.trafo.vkr_percent  = 0.0   (lossless transformers)
    - net.trafo.i0_percent   = 0.0   (no magnetizing shunt branch)
    - net.trafo.pfe_kw       = 0.0   (no iron-loss shunt branch)
    - net.impedance.rft_pu   = 0.0   (lossless impedances)
    - net.impedance.rtf_pu   = 0.0   (lossless impedances, reverse direction)
    - net.shunt.in_service   = False  (disable bus shunt devices)
    - net.ward.in_service    = False  (disable ward equivalents)
    - net.xward.in_service   = False  (disable extended ward equivalents)

    This aligns AC PF / AC MC with the certificate's internal linearization,
    which uses a series-only Ybus model (no shunt elements).  Without removing
    shunt elements, the verification PF includes voltage-dependent admittances
    that the Jacobian does not model, causing systematic line-flow prediction errors.
    """
    nn = copy.deepcopy(net)

    if hasattr(nn, "line") and nn.line is not None and len(nn.line):
        if "r_ohm_per_km" in nn.line.columns:
            nn.line.loc[:, "r_ohm_per_km"] = 0.0
        if "c_nf_per_km" in nn.line.columns:
            nn.line.loc[:, "c_nf_per_km"] = 0.0
        if "g_us_per_km" in nn.line.columns:
            nn.line.loc[:, "g_us_per_km"] = 0.0

    if hasattr(nn, "trafo") and nn.trafo is not None and len(nn.trafo):
        # vkr: series resistance; i0/pfe: magnetizing / iron-loss shunt branch.
        # The series-only Jacobian models none of these, so all must be zeroed
        # to keep the verification PF consistent with the linearization.
        for col in ("vkr_percent", "i0_percent", "pfe_kw"):
            if col in nn.trafo.columns:
                nn.trafo.loc[:, col] = 0.0

    if hasattr(nn, "impedance") and nn.impedance is not None and len(nn.impedance):
        for col in ("rft_pu", "rtf_pu"):
            if col in nn.impedance.columns:
                nn.impedance.loc[:, col] = 0.0

    # Fail fast on element types the series-only Jacobian cannot represent
    # and the policy cannot neutralize.  Silently keeping them would create
    # exactly the kind of systematic PF-vs-Jacobian mismatch this policy
    # exists to prevent.
    for tbl in ("trafo3w", "tcsc", "svc", "ssc", "dcline"):
        df = getattr(nn, tbl, None)
        if df is not None and len(df):
            n_active = (
                int(df["in_service"].sum()) if "in_service" in df.columns else len(df)
            )
            if n_active > 0:
                raise ValueError(
                    f"Lossless policy: net contains {n_active} in-service "
                    f"'{tbl}' element(s), which the series-only AC certificate "
                    "model cannot represent. Remove or convert them first."
                )
    sw = getattr(nn, "switch", None)
    if sw is not None and len(sw) and "z_ohm" in sw.columns:
        closed = sw["closed"] if "closed" in sw.columns else True
        n_z = int(((sw["z_ohm"].astype(float).abs() > 0.0) & closed).sum())
        if n_z > 0:
            raise ValueError(
                f"Lossless policy: net contains {n_z} closed switch(es) with "
                "z_ohm != 0, which the series-only AC certificate model "
                "cannot represent."
            )

    # Disable shunt devices — the series-only Jacobian does not model them.
    for tbl in ("shunt", "ward", "xward"):
        df = getattr(nn, tbl, None)
        if df is not None and len(df) and "in_service" in df.columns:
            nn_shunts = int(df["in_service"].sum())
            if nn_shunts > 0:
                df.loc[:, "in_service"] = False
                logger.info(
                    "Lossless policy: disabled %d %s element(s) (series-only model).",
                    nn_shunts,
                    tbl,
                )

    return nn


def detect_q_limit_events(net: Any, *, tol_mvar: float = 1e-6) -> list[dict[str, Any]]:
    """Return generator-like elements whose solved Q output is at a Q limit.

    The reduced AC certificate assumes a fixed PV/PQ active set.  A solved
    generator at its reactive limit is a useful diagnostic that pandapower may
    have changed the effective active set under ``enforce_q_lims=True``.
    """
    events: list[dict[str, Any]] = []

    specs = (
        ("gen", "res_gen", "q_mvar", "min_q_mvar", "max_q_mvar"),
        ("ext_grid", "res_ext_grid", "q_mvar", "min_q_mvar", "max_q_mvar"),
    )
    for table_name, result_name, q_col, qmin_col, qmax_col in specs:
        table = getattr(net, table_name, None)
        result = getattr(net, result_name, None)
        if table is None or result is None or len(table) == 0 or len(result) == 0:
            continue
        for idx in table.index:
            if idx not in result.index:
                continue
            row = table.loc[idx]
            if not bool(row.get("in_service", True)):
                continue
            if q_col not in result.columns:
                continue
            q = float(result.loc[idx, q_col])
            qmin = float(row.get(qmin_col, np.nan))
            qmax = float(row.get(qmax_col, np.nan))
            if not np.isfinite(q):
                continue

            at_min = bool(np.isfinite(qmin) and q <= qmin + float(tol_mvar))
            at_max = bool(np.isfinite(qmax) and q >= qmax - float(tol_mvar))
            if not at_min and not at_max:
                continue

            events.append(
                {
                    "element": str(table_name),
                    "element_index": int(idx),
                    "bus": int(row.get("bus", -1)),
                    "q_mvar": float(q),
                    "q_min_mvar": float(qmin) if np.isfinite(qmin) else None,
                    "q_max_mvar": float(qmax) if np.isfinite(qmax) else None,
                    "at_min": bool(at_min),
                    "at_max": bool(at_max),
                }
            )

    return events


def resolve_slack_bus_id(net: Any, slack_bus: int) -> int:
    """
    Resolve slack bus identifier.

    Parameters
    ----------
    slack_bus:
        Either:
        - actual pandapower bus id (must be present in net.bus.index), or
        - position in sorted(net.bus.index) ordering, or
        - -1 to auto-detect from net.ext_grid.

    The resolved bus is validated against ``net.ext_grid``: if there is
    exactly one in-service ext_grid and the resolved bus doesn't match,
    the ext_grid bus is used instead (with a warning). If there are
    multiple in-service ext_grids and auto-detection is requested
    (``slack_bus=-1``), the smallest ext_grid bus id is used.
    """
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    bus_pos = {bid: pos for pos, bid in enumerate(bus_ids)}

    ext_grid_buses = _get_ext_grid_buses(net)

    if int(slack_bus) == -1:
        if len(ext_grid_buses) == 1:
            sb = ext_grid_buses[0]
            logger.info("Auto-detected slack bus from ext_grid: bus %d", sb)
            return int(sb)
        if len(ext_grid_buses) > 1:
            sb = ext_grid_buses[0]
            logger.warning(
                "Multiple ext_grid buses %s; using smallest bus id: bus %d",
                ext_grid_buses,
                sb,
            )
            return int(sb)
        raise ValueError("slack_bus=-1 (auto-detect) but no in-service ext_grid found.")

    if int(slack_bus) in bus_pos:
        resolved = int(slack_bus)
    elif 0 <= int(slack_bus) < len(bus_ids):
        resolved = int(bus_ids[int(slack_bus)])
    else:
        raise ValueError(f"slack_bus must be bus id or position. Got {slack_bus!r}.")

    # Validate against ext_grid
    if len(ext_grid_buses) == 1 and resolved != ext_grid_buses[0]:
        logger.warning(
            "Resolved slack_bus=%d (from input %d) does NOT match the "
            "ext_grid bus %d. Using ext_grid bus %d instead. "
            "Set slack_bus=%d in config to suppress this warning.",
            resolved,
            int(slack_bus),
            ext_grid_buses[0],
            ext_grid_buses[0],
            ext_grid_buses[0],
        )
        return int(ext_grid_buses[0])

    return int(resolved)


def _get_ext_grid_buses(net: Any) -> list[int]:
    """Return sorted unique in-service ext_grid bus IDs."""
    ext_buses: list[int] = []
    if hasattr(net, "ext_grid") and net.ext_grid is not None and len(net.ext_grid):
        for eid in net.ext_grid.index:
            row = net.ext_grid.loc[eid]
            in_service = True
            try:
                in_service = bool(row.get("in_service", True))
            except Exception:
                pass
            if in_service:
                ext_buses.append(int(row["bus"]))
    return sorted(set(ext_buses))


def ensure_ext_grid_at_slack(net: Any, slack_bus_id: int) -> None:
    """Ensure pandapower net has an in-service ext_grid at the requested slack bus.

    Auto-creates one if missing (e.g. RTE MATPOWER files without a type-3 bus).
    """
    import pandapower as pp

    has_ext_grid = (
        hasattr(net, "ext_grid") and net.ext_grid is not None and len(net.ext_grid)
    )
    if has_ext_grid:
        for _, row in net.ext_grid.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            if int(row.get("bus", -1)) == int(slack_bus_id):
                return  # already present

    logger.warning(
        "No in-service ext_grid at slack bus %d; creating one automatically.",
        int(slack_bus_id),
    )
    pp.create_ext_grid(net, bus=int(slack_bus_id), vm_pu=1.0, va_degree=0.0)


def apply_gen_dispatch_to_pandapower_net(
    net: Any,
    gen_dispatch_mw_by_name: Mapping[str, float] | Sequence[Sequence[Any]] | None,
) -> None:
    """
    Apply active power dispatch to pandapower net in-place.

    Supported keys (project convention)
    -----------------------------------
    - "gen_<pp_gen_idx>" -> net.gen.at[idx, "p_mw"]
    - "sgen_<pp_sgen_idx>" -> net.sgen.at[idx, "p_mw"]

    Notes
    -----
    - "ext_<idx>" is ignored (ext_grid is slack-like; P is endogenous in PF).
    - Only P is applied (Q is left to PF / controls).
    - This function is deterministic and silent on unknown keys.
    """
    if gen_dispatch_mw_by_name is None:
        return

    # Normalize inputs into (name -> value) mapping.
    mapping: dict[str, float] = {}
    if isinstance(gen_dispatch_mw_by_name, Mapping):
        for k, v in gen_dispatch_mw_by_name.items():
            try:
                mapping[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    else:
        # e.g. JSON list of pairs: [["gen_0", 10.0], ["ext_0", 0.0], ...]
        for item in gen_dispatch_mw_by_name:
            if not isinstance(item, Sequence) or len(item) != 2:
                continue
            k, v = item[0], item[1]
            try:
                mapping[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

    if not mapping:
        return

    applied = 0

    # Apply dispatch to net.gen entries.
    if hasattr(net, "gen") and net.gen is not None and len(net.gen):
        for name, p in mapping.items():
            if not name.startswith("gen_"):
                continue
            try:
                gid = int(name.split("_", 1)[1])
            except Exception:  # noqa: BLE001
                continue
            if gid not in net.gen.index:
                continue
            if not np.isfinite(p):
                continue
            net.gen.at[gid, "p_mw"] = float(p)
            applied += 1

    # Apply dispatch to net.sgen entries (additional generators from MATPOWER).
    if hasattr(net, "sgen") and net.sgen is not None and len(net.sgen):
        for name, p in mapping.items():
            if not name.startswith("sgen_"):
                continue
            try:
                sid = int(name.split("_", 1)[1])
            except Exception:  # noqa: BLE001
                continue
            if sid not in net.sgen.index:
                continue
            if not np.isfinite(p):
                continue
            net.sgen.at[sid, "p_mw"] = float(p)
            applied += 1

    logger.debug(
        "Applied generator dispatch to pandapower net: applied=%d (gen+sgen)",
        int(applied),
    )


def apply_opp_result_to_pandapower_net(
    net: Any,
    *,
    opp_gen_dispatch: Mapping[str, float] | None,
    opp_vm_pu: Mapping[int, float] | None,
) -> None:
    """Apply OPP (AC OPF) result to *net* in-place for ``runpp`` reproducibility.

    Sets generator active power (``gen.p_mw``, ``sgen.p_mw``) and PV bus
    voltage setpoints (``gen.vm_pu``, ``ext_grid.vm_pu``) so that a
    subsequent ``pp.runpp()`` reproduces the OPP operating point.
    """
    if opp_gen_dispatch:
        apply_gen_dispatch_to_pandapower_net(net, opp_gen_dispatch)

    if opp_vm_pu:
        # Set gen voltage setpoints.
        if hasattr(net, "gen") and net.gen is not None and len(net.gen):
            for gid in net.gen.index:
                gen_bus = int(net.gen.at[gid, "bus"])
                if gen_bus in opp_vm_pu:
                    net.gen.at[gid, "vm_pu"] = float(opp_vm_pu[gen_bus])

        # Set ext_grid voltage setpoints.
        if hasattr(net, "ext_grid") and net.ext_grid is not None and len(net.ext_grid):
            for eid in net.ext_grid.index:
                eg_bus = int(net.ext_grid.at[eid, "bus"])
                if eg_bus in opp_vm_pu:
                    net.ext_grid.at[eid, "vm_pu"] = float(opp_vm_pu[eg_bus])

    logger.debug("Applied OPP result to pandapower net (gen dispatch + vm setpoints).")
