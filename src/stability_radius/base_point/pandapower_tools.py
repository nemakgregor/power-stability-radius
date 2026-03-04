from __future__ import annotations

"""
Shared helpers for working with pandapower networks in a deterministic way.

Design principles
-----------------
- Deterministic behavior: no implicit fallbacks, stable ordering.
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
    Return a deep-copied pandapower net with a deterministic lossless policy applied.

    Policy
    ------
    - net.line.r_ohm_per_km = 0.0
    - net.trafo.vkr_percent = 0.0
    - net.impedance.rft_pu = 0.0

    This aligns AC PF / AC MC with the certificate's internal linearization.
    """
    nn = copy.deepcopy(net)

    if hasattr(nn, "line") and nn.line is not None and len(nn.line):
        if "r_ohm_per_km" in nn.line.columns:
            nn.line.loc[:, "r_ohm_per_km"] = 0.0

    if hasattr(nn, "trafo") and nn.trafo is not None and len(nn.trafo):
        if "vkr_percent" in nn.trafo.columns:
            nn.trafo.loc[:, "vkr_percent"] = 0.0

    if hasattr(nn, "impedance") and nn.impedance is not None and len(nn.impedance):
        if "rft_pu" in nn.impedance.columns:
            nn.impedance.loc[:, "rft_pu"] = 0.0

    return nn


def resolve_slack_bus_id(net: Any, slack_bus: int) -> int:
    """
    Resolve slack bus identifier.

    Parameters
    ----------
    slack_bus:
        Either:
        - actual pandapower bus id (must be present in net.bus.index), or
        - position in sorted(net.bus.index) ordering.
    """
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    bus_pos = {bid: pos for pos, bid in enumerate(bus_ids)}
    if int(slack_bus) in bus_pos:
        return int(slack_bus)
    if 0 <= int(slack_bus) < len(bus_ids):
        return int(bus_ids[int(slack_bus)])
    raise ValueError(f"slack_bus must be bus id or position. Got {slack_bus!r}.")


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
