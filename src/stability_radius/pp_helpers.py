from __future__ import annotations

"""
Small canonical helpers for pandapower-like tables.

Why this module exists
----------------------
Several submodules historically re-implemented the same tiny helper utilities:

- is_in_service(row)
- bus_vn_kv(net, bus_id)
- resolve_slack_pos(bus_ids, slack_bus)

For AC Stability Radius workflow correctness, we want:
- consistent slack interpretation (bus id OR positional index),
- consistent treatment of in_service flags across element tables,
- no silent divergence across DC/AC stacks.

Design notes
------------
- This module is intentionally dependency-light (no pandapower import).
- It operates on "pandapower-like" objects: `net.bus` is expected to be a pandas-like table.
"""

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


def is_in_service(row: Any, *, default: bool = True) -> bool:
    """
    Return True if an element row is considered in service.

    Parameters
    ----------
    row:
        A pandas Series-like row or a dict-like object supporting `.get()`.
    default:
        Used when the flag is missing or not readable.

    Returns
    -------
    bool
        True iff row["in_service"] (or row.get("in_service")) is truthy.
    """
    try:
        v = row.get("in_service", default)
    except Exception:  # noqa: BLE001
        try:
            v = row["in_service"]
        except Exception:  # noqa: BLE001
            v = default
    return bool(v)


def bus_vn_kv(net: Any, bus_id: int) -> float:
    """
    Return bus nominal voltage (vn_kv) if available, else NaN.

    This helper is intentionally conservative: it does not try to infer voltages.
    """
    try:
        bus_tbl = getattr(net, "bus", None)
        if bus_tbl is None or len(bus_tbl) == 0:
            return float("nan")
        if bus_id not in bus_tbl.index:
            return float("nan")
        if "vn_kv" not in bus_tbl.columns:
            return float("nan")
        return float(bus_tbl.loc[int(bus_id), "vn_kv"])
    except Exception:  # noqa: BLE001
        return float("nan")


def resolve_slack_pos(bus_ids: Sequence[int], slack_bus: int) -> int:
    """
    Resolve slack bus position in a deterministic bus ordering.

    Parameters
    ----------
    bus_ids:
        Stable bus ordering (typically sorted net.bus.index).
    slack_bus:
        Either:
        - an actual bus id present in bus_ids, or
        - a positional index in [0, len(bus_ids)-1].

    Returns
    -------
    int
        Slack bus position in the provided `bus_ids` ordering.

    Raises
    ------
    ValueError
        If slack_bus is neither a valid bus id nor a valid position.
    """
    ids = [int(x) for x in bus_ids]
    pos_by_id = {bid: pos for pos, bid in enumerate(ids)}

    sb = int(slack_bus)
    if sb in pos_by_id:
        logger.debug("Resolved slack_bus=%d as bus id -> pos=%d", sb, pos_by_id[sb])
        return int(pos_by_id[sb])

    if 0 <= sb < len(ids):
        logger.debug("Resolved slack_bus=%d as positional index.", sb)
        return int(sb)

    raise ValueError(
        f"slack_bus must be a valid bus id or position. Got {slack_bus!r}; "
        f"valid positions: [0, {len(ids) - 1}]"
    )
