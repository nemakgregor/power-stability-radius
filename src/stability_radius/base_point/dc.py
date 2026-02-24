from __future__ import annotations

import logging
from typing import Any

import numpy as np

from stability_radius.config import DEFAULT_OPF, OPFConfig
from stability_radius.dc.dc_model import DCOperator, build_dc_operator
from stability_radius.radii.common import (
    LineBaseQuantities,
    estimate_line_limit_mva_with_flag,
    get_line_base_quantities,
)

from .types import BasePointDC

logger = logging.getLogger(__name__)


def _resolve_slack_bus_id(*, bus_ids: list[int], slack_bus: int) -> int:
    """
    Resolve slack bus as:
    - exact bus id if present, else
    - positional index in sorted(bus_ids).
    """
    bus_pos = {int(bid): pos for pos, bid in enumerate(bus_ids)}
    if int(slack_bus) in bus_pos:
        return int(slack_bus)
    if 0 <= int(slack_bus) < len(bus_ids):
        return int(bus_ids[int(slack_bus)])
    raise ValueError(
        f"slack_bus must be a valid bus id or position. Got {slack_bus!r}."
    )


def _is_in_service(row: Any) -> bool:
    return bool(row.get("in_service", True))


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
        if not np.isfinite(p):
            continue
        out[bus] = out.get(bus, 0.0) + float(p)
    return out


def case_bus_injections_mw(
    *, net: Any, slack_bus: int
) -> tuple[list[int], int, np.ndarray]:
    """
    Deterministically build a balanced bus injection vector from the case (no OPF).

    Returns
    -------
    (bus_ids_sorted, slack_bus_id, injections_mw)

    Notes
    -----
    Slack bus injection is adjusted to enforce exact balance.
    """
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    if not bus_ids:
        raise ValueError("Network has no buses.")

    slack_bus_id = _resolve_slack_bus_id(bus_ids=bus_ids, slack_bus=int(slack_bus))

    load = _sum_p_by_bus(net, "load", p_col="p_mw")
    shunt = _sum_p_by_bus(net, "shunt", p_col="p_mw")
    gen = _sum_p_by_bus(net, "gen", p_col="p_mw")
    sgen = _sum_p_by_bus(net, "sgen", p_col="p_mw")

    bus_pos = {int(b): i for i, b in enumerate(bus_ids)}
    p = np.zeros(len(bus_ids), dtype=float)
    for b in bus_ids:
        i = int(bus_pos[int(b)])
        p_load = float(load.get(int(b), 0.0) + shunt.get(int(b), 0.0))
        p_gen = float(gen.get(int(b), 0.0) + sgen.get(int(b), 0.0))
        p[i] = p_gen - p_load

    slack_pos = int(bus_pos[int(slack_bus_id)])
    imbalance = float(np.sum(p))
    p[slack_pos] -= imbalance

    logger.debug(
        "Case injections built: sum_before=%.6g MW, sum_after=%.6g MW, slack_bus_id=%d",
        float(imbalance),
        float(np.sum(p)),
        int(slack_bus_id),
    )
    return bus_ids, int(slack_bus_id), p


def build_dc_base_point_case(
    *,
    net: Any,
    slack_bus: int,
    dc_op: DCOperator | None = None,
    limit_factor: float = 1.0,
) -> tuple[BasePointDC, LineBaseQuantities, DCOperator]:
    """
    Build DC base point from case injections (no OPF), using a DCOperator reconstruction.

    Returns
    -------
    (bp_dc, base_quantities, dc_operator)
    """
    if float(limit_factor) <= 0.0:
        raise ValueError("limit_factor must be positive.")

    bus_ids, slack_bus_id, injections = case_bus_injections_mw(
        net=net, slack_bus=int(slack_bus)
    )
    op = (
        dc_op if dc_op is not None else build_dc_operator(net, slack_bus=int(slack_bus))
    )

    # Line limits (MW assumed from MVA under PF=1 convention)
    line_ids = [int(x) for x in sorted(net.line.index)]
    limits = np.empty(len(line_ids), dtype=float)
    is_unconstrained = np.zeros(len(line_ids), dtype=bool)

    for pos, lid in enumerate(line_ids):
        lim, is_uc = estimate_line_limit_mva_with_flag(net, net.line.loc[lid])
        limits[pos] = float(lim) * float(limit_factor)
        is_unconstrained[pos] = bool(is_uc)

    flows = np.asarray(
        op.flows_from_bus_injections_mw(injections), dtype=float
    ).reshape(-1)
    if flows.shape != (len(line_ids),):
        raise ValueError("DCOperator returned unexpected flow shape.")

    p0_abs = np.abs(flows)
    margins = np.maximum(limits - p0_abs, 0.0)

    base = LineBaseQuantities(
        line_indices=line_ids,
        flow0_mw=flows,
        p0_abs_mw=p0_abs,
        limit_mva_assumed_mw=limits,
        margin_mw=margins,
        is_unconstrained=is_unconstrained,
        opf_status="case",
        opf_objective=float("nan"),
        bus_ids=bus_ids,
        bus_injections_mw=injections,
        opf_limits_mw=None,
        opf_gen_dispatch_mw_by_name=None,
    )

    bp = BasePointDC(
        source="case",
        slack_bus=int(slack_bus),
        bus_ids=tuple(bus_ids),
        bus_injections_mw=injections,
        line_ids=tuple(line_ids),
        line_flows_mw=flows,
        line_limits_mw=limits,
        status="case",
        objective=float("nan"),
        gen_dispatch_mw_by_name=(),
    )
    return bp, base, op


def build_dc_base_point_dc_opf(
    *,
    net: Any,
    slack_bus: int,
    opf_cfg: OPFConfig | None = None,
    limit_factor: float = 1.0,
) -> tuple[BasePointDC, LineBaseQuantities]:
    """
    Build DC base point using DC OPF (PyPSA+HiGHS) as dispatch source.

    Notes
    -----
    This is a *dispatch choice*; AC certificate is still computed around AC PF.
    """
    cfg = opf_cfg if opf_cfg is not None else DEFAULT_OPF
    base = get_line_base_quantities(net, limit_factor=float(limit_factor), opf_cfg=cfg)

    if base.bus_ids is None or base.bus_injections_mw is None:
        raise ValueError(
            "OPF base quantities must include bus_ids and bus_injections_mw."
        )

    bp = BasePointDC(
        source="dc_opf",
        slack_bus=int(slack_bus),
        bus_ids=tuple(int(x) for x in base.bus_ids),
        bus_injections_mw=np.asarray(base.bus_injections_mw, dtype=float),
        line_ids=tuple(int(x) for x in base.line_indices),
        line_flows_mw=np.asarray(base.flow0_mw, dtype=float),
        line_limits_mw=np.asarray(base.limit_mva_assumed_mw, dtype=float),
        status=str(base.opf_status or "ok"),
        objective=float(base.opf_objective)
        if base.opf_objective is not None
        else float("nan"),
        gen_dispatch_mw_by_name=tuple(base.opf_gen_dispatch_mw_by_name or ()),
    )
    return bp, base
