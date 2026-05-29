from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np

from stability_radius.base_point.pandapower_opp import (
    ACFPFConfig,
    solve_ac_fpf,
)
from stability_radius.base_point.pypsa_pf import (
    PyPSAAPFResult,
    solve_ac_pf_base_point_from_pandapower,
)
from stability_radius.radii.common import estimate_line_limit_mva

from .types import BasePointAC

logger = logging.getLogger(__name__)


def solve_ac_pf_base_point(
    *,
    net: Any,
    slack_bus: int,
    pf_solver: str,
    pf_init: str,
    lossless: bool,
    gen_dispatch_mw_by_name: Mapping[str, float] | Sequence[Sequence[Any]] | None,
    line_indices: Sequence[int] | None = None,
    distributed_slack: bool = False,
    trafo_model: str = "pi",
) -> tuple[BasePointAC, PyPSAAPFResult]:
    """
    Solve AC PF base point and return both:
    - BasePointAC (JSON-friendly, stores Vm/Va for reproducibility)
    - PyPSAAPFResult (used by AC certificate computation code)

    Parameters
    ----------
    distributed_slack:
        Distribute the slack (loss compensation) among generators
        proportionally to their headroom.  Passed to pandapower backend.
    trafo_model:
        Transformer model: ``"pi"`` or ``"t"``.
    """
    # The public AC certificate path currently fail-fasts on lossless=false.
    # Keep this lower-level wrapper explicit because it is also used by
    # experiments that solve PF base points before certificate construction.

    solver_eff = str(pf_solver).strip().lower()
    if solver_eff not in {"pandapower", "pypsa"}:
        raise ValueError("pf_solver must be pandapower|pypsa")

    init_eff = str(pf_init).strip().lower()
    if init_eff not in {"flat", "dc", "pp"}:
        raise ValueError("pf_init must be flat|dc|pp")

    line_ids = (
        [int(x) for x in sorted(net.line.index)]
        if line_indices is None
        else [int(x) for x in line_indices]
    )
    if not line_ids:
        raise ValueError("Network has no lines (net.line empty).")

    logger.info(
        "AC base point: solve PF (solver=%s, init=%s, lossless=%s, lines=%d)",
        solver_eff,
        init_eff,
        bool(lossless),
        int(len(line_ids)),
    )

    base_pf = solve_ac_pf_base_point_from_pandapower(
        net=net,
        slack_bus=int(slack_bus),
        line_indices=line_ids,
        gen_dispatch_mw_by_name=gen_dispatch_mw_by_name
        if gen_dispatch_mw_by_name is not None
        else {},
        lossless=bool(lossless),
        solver=str(solver_eff),
        init=str(init_eff),
        distributed_slack=bool(distributed_slack),
        trafo_model=str(trafo_model),
    )

    # Limits per line (MVA) for reproducibility / verification checks
    s_limit_mva = np.empty(len(line_ids), dtype=float)
    for pos, lid in enumerate(line_ids):
        s_limit_mva[pos] = float(estimate_line_limit_mva(net, net.line.loc[lid]))

    bp = BasePointAC(
        pf_solver=str(solver_eff),
        pf_init=str(init_eff),
        lossless=bool(lossless),
        slack_bus=int(slack_bus),
        bus_ids=tuple(int(x) for x in base_pf.bus_ids),
        vm_pu=np.asarray(base_pf.v_mag_pu, dtype=float),
        va_rad=np.asarray(base_pf.v_ang_rad, dtype=float),
        line_ids=tuple(int(x) for x in base_pf.line_ids),
        p_from_mw=np.asarray(base_pf.line_p0_mw, dtype=float),
        q_from_mvar=np.asarray(base_pf.line_q0_mvar, dtype=float),
        p_to_mw=np.asarray(base_pf.line_p1_mw, dtype=float),
        q_to_mvar=np.asarray(base_pf.line_q1_mvar, dtype=float),
        s_limit_mva=s_limit_mva,
        status=str(base_pf.status),
        gen_dispatch_mw_by_name=tuple(
            (str(k), float(v)) for k, v in (gen_dispatch_mw_by_name or {}).items()
        )
        if isinstance(gen_dispatch_mw_by_name, Mapping)
        else (),
        pf_attempt=str(getattr(base_pf, "pf_attempt", "primary")),
        pf_repairs=tuple(getattr(base_pf, "pf_repairs", None) or ()),
        distributed_slack_requested=bool(
            getattr(base_pf, "distributed_slack_requested", False)
        ),
        distributed_slack_used=bool(getattr(base_pf, "distributed_slack_used", False)),
        q_limit_hit=bool(getattr(base_pf, "q_limit_hit", False)),
        q_limit_events=tuple(getattr(base_pf, "q_limit_events", ()) or ()),
        bus_p_mw=np.asarray(base_pf.bus_p_mw, dtype=float)
        if base_pf.bus_p_mw is not None
        else None,
    )

    return bp, base_pf


def solve_ac_fpf_base_point(
    *,
    net: Any,
    slack_bus: int,
    lossless: bool,
    fpf_cfg: ACFPFConfig | None = None,
    opf_cfg: "OPFConfig | None" = None,
    line_indices: Sequence[int] | None = None,
) -> tuple[BasePointAC, PyPSAAPFResult]:
    """Solve AC Feasible Power Flow and return BasePointAC + PyPSAAPFResult.

    This wraps ``solve_ac_fpf()`` (pandapower.runopp) and packages
    the result into the same ``BasePointAC`` type used by the AC PF path,
    so downstream code (AC certificate, AC feasibility check) works
    without modification.

    Parameters
    ----------
    net : pandapower network
    slack_bus : slack bus id or position
    lossless : if True, apply lossless policy (r=0)
    fpf_cfg : AC FPF solver configuration (optional)
    opf_cfg : OPF config for unconstrained_line_nom_mw (optional)
    line_indices : list of line indices to monitor (optional, default=all)
    """
    line_ids = (
        [int(x) for x in sorted(net.line.index)]
        if line_indices is None
        else [int(x) for x in line_indices]
    )
    if not line_ids:
        raise ValueError("Network has no lines (net.line empty).")

    logger.info(
        "AC FPF base point: solve OPP (lossless=%s, lines=%d)",
        bool(lossless),
        int(len(line_ids)),
    )

    base_pf = solve_ac_fpf(
        net=net,
        slack_bus=int(slack_bus),
        line_indices=line_ids,
        lossless=bool(lossless),
        fpf_cfg=fpf_cfg,
        opf_cfg=opf_cfg,
    )

    # Limits per line (MVA) for reproducibility / verification checks.
    s_limit_mva = np.empty(len(line_ids), dtype=float)
    for pos, lid in enumerate(line_ids):
        s_limit_mva[pos] = float(estimate_line_limit_mva(net, net.line.loc[lid]))

    # Capture OPP gen dispatch for reproducibility (MC needs it to
    # reconstruct the same base point from the raw case file).
    opp_dispatch = getattr(base_pf, "opp_gen_dispatch", None) or {}
    gen_dispatch_tuple = tuple((str(k), float(v)) for k, v in opp_dispatch.items())

    bp = BasePointAC(
        pf_solver="pandapower_opp",
        pf_init="n/a",
        lossless=bool(lossless),
        slack_bus=int(slack_bus),
        bus_ids=tuple(int(x) for x in base_pf.bus_ids),
        vm_pu=np.asarray(base_pf.v_mag_pu, dtype=float),
        va_rad=np.asarray(base_pf.v_ang_rad, dtype=float),
        line_ids=tuple(int(x) for x in base_pf.line_ids),
        p_from_mw=np.asarray(base_pf.line_p0_mw, dtype=float),
        q_from_mvar=np.asarray(base_pf.line_q0_mvar, dtype=float),
        p_to_mw=np.asarray(base_pf.line_p1_mw, dtype=float),
        q_to_mvar=np.asarray(base_pf.line_q1_mvar, dtype=float),
        s_limit_mva=s_limit_mva,
        status=str(base_pf.status),
        gen_dispatch_mw_by_name=gen_dispatch_tuple,
        pf_attempt=str(getattr(base_pf, "pf_attempt", "primary")),
        pf_repairs=tuple(getattr(base_pf, "pf_repairs", None) or ()),
        distributed_slack_requested=bool(
            getattr(base_pf, "distributed_slack_requested", False)
        ),
        distributed_slack_used=bool(getattr(base_pf, "distributed_slack_used", False)),
        q_limit_hit=bool(getattr(base_pf, "q_limit_hit", False)),
        q_limit_events=tuple(getattr(base_pf, "q_limit_events", ()) or ()),
        bus_p_mw=np.asarray(base_pf.bus_p_mw, dtype=float)
        if base_pf.bus_p_mw is not None
        else None,
    )

    return bp, base_pf
