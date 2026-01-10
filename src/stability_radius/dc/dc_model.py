from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def build_dc_matrices(net, slack_bus: int = 0) -> Tuple[np.ndarray, object]:
    """
    Build a simple DC PTDF-like sensitivity matrix `H_full` for pandapower networks.

    The method is version-stable (no reliance on pandapower internal PTDF helpers).
    It builds a susceptance Laplacian from `net.line` only:

        Bbus = A^T * diag(b) * A
        H_red = diag(b) * A_red * Bred^{-1}
        H_full has a zero slack column, and H_red columns for non-slack buses.

    Parameters
    ----------
    net:
        pandapower network object.
    slack_bus:
        Slack bus identifier. If it matches a `net.bus.index` value, it is treated
        as a bus index. Otherwise, it is treated as a positional index (0..n-1)
        in the current bus ordering (backward-compatible behavior).

    Returns
    -------
    (H_full, net):
        H_full is an (m_lines x n_buses) numpy array.
    """
    bus_index = list(net.bus.index)
    n_bus = len(bus_index)
    if n_bus == 0:
        raise ValueError("Network has no buses.")

    bus_pos = {bus_id: pos for pos, bus_id in enumerate(bus_index)}

    # Accept either bus id or positional slack index
    if slack_bus in bus_pos:
        slack_pos = bus_pos[slack_bus]
    elif 0 <= int(slack_bus) < n_bus:
        slack_pos = int(slack_bus)
    else:
        raise ValueError(
            f"slack_bus must be a valid bus index or position. Got {slack_bus!r}; "
            f"valid positions: [0, {n_bus - 1}], valid bus indices: {bus_index[:10]}..."
        )

    line_index = list(net.line.index)
    m_line = len(line_index)
    if m_line == 0:
        raise ValueError("Network has no lines; cannot build DC matrices.")

    A = np.zeros((m_line, n_bus), dtype=float)  # incidence (from=+1, to=-1)
    b = np.zeros(m_line, dtype=float)  # branch susceptance proxy (1/ohm)

    for row_pos, (_, row) in enumerate(net.line.loc[line_index].iterrows()):
        if "in_service" in row and not bool(row["in_service"]):
            continue

        fb = row["from_bus"]
        tb = row["to_bus"]
        if fb not in bus_pos or tb not in bus_pos:
            continue

        f = bus_pos[fb]
        t = bus_pos[tb]

        x_ohm_per_km = float(row.get("x_ohm_per_km", 0.0))
        length_km = float(row.get("length_km", 0.0))
        x_total = x_ohm_per_km * length_km

        # Guard against invalid or zero reactance.
        if not np.isfinite(x_total) or abs(x_total) < 1e-12:
            logger.warning(
                "Line %s has invalid x_total=%s; setting b=0.", row.name, x_total
            )
            continue

        b_i = 1.0 / x_total
        b[row_pos] = b_i
        A[row_pos, f] = 1.0
        A[row_pos, t] = -1.0

    # Bbus = A^T diag(b) A (Laplacian-like)
    weighted_A = b[:, None] * A
    Bbus = A.T @ weighted_A

    mask = np.ones(n_bus, dtype=bool)
    mask[slack_pos] = False
    Bred = Bbus[np.ix_(mask, mask)]
    A_red = A[:, mask]

    # n_bus == 1 case
    if Bred.size == 0:
        H_full = np.zeros((m_line, n_bus), dtype=float)
        return H_full, net

    try:
        # Avoid forming an explicit inverse:
        # (b*A_red) @ inv(Bred) == solve(Bred.T, (b*A_red).T).T
        RHS = (b[:, None] * A_red).T  # (n-1, m)
        H_red = np.linalg.solve(Bred.T, RHS).T  # (m, n-1)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            "Reduced B matrix is singular. The network may be disconnected or have invalid line reactances."
        ) from e

    H_full = np.zeros((m_line, n_bus), dtype=float)
    H_full[:, mask] = H_red
    # Slack column remains zeros.

    logger.debug(
        "Built DC matrices: n_bus=%d, m_line=%d, H_full=%s", n_bus, m_line, H_full.shape
    )
    return H_full, net
