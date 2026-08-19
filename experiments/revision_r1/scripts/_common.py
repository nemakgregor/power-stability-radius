"""Shared helpers for the revision-r1 experiment scripts."""

from __future__ import annotations

import copy
import json
import logging
import resource
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import pandapower as pp  # noqa: E402

from stability_radius.base_point.pandapower_tools import (  # noqa: E402
    apply_lossless_policy_to_pandapower_net,
)
from stability_radius.base_point.pypsa_pf import (  # noqa: E402
    solve_ac_pf_base_point_from_pandapower,
)
from stability_radius.geometry.balanced import (  # noqa: E402
    make_ac_block_specs,
    worst_case_l2_direction,
)
from stability_radius.parsers.matpower import load_network  # noqa: E402
from stability_radius.radii.ac_l2 import compute_ac_l2_radius  # noqa: E402
from stability_radius.workflows import expand_h_reduced_to_full  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "input"

# Replay PF settings: MUST match the base-point solver convention
# (solve_ac_pf_base_point_from_pandapower uses enforce_q_lims=True), otherwise
# the replayed operating point is not the certified one.
PF_KW = dict(
    calculate_voltage_angles=True,
    enforce_q_lims=True,
    trafo_model="pi",
    tolerance_mva=1e-9,
    max_iteration=100,
)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_case(case_file: str):
    """Load + lossless policy; returns (net, ext_grid slack bus id)."""
    net = apply_lossless_policy_to_pandapower_net(
        load_network(str(DATA_DIR / case_file))
    )
    slack = int(net.ext_grid.bus.iloc[0])
    return net, slack


def certificate_with_h(net, slack):
    """PF base point + AC L2 certificate with h-vectors; returns everything."""
    base_pf = solve_ac_pf_base_point_from_pandapower(
        net=net, slack_bus=slack, solver="pandapower", init="dc", lossless=True
    )
    ac = compute_ac_l2_radius(
        net, base_pf=base_pf, slack_bus=slack, lossless=True, return_h_vectors=True
    )
    hv = ac.pop("_h_vectors")
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)
    h_from = expand_h_reduced_to_full(
        hv["h_from"],
        n_bus=n_bus,
        slack_pos=int(hv["slack_pos"]),
        pq_mask=hv.get("pq_mask"),
    )
    h_to = expand_h_reduced_to_full(
        hv["h_to"],
        n_bus=n_bus,
        slack_pos=int(hv["slack_pos"]),
        pq_mask=hv.get("pq_mask"),
    )
    return base_pf, ac, hv, h_from, h_to, bus_ids


def balanced_direction(h_full: np.ndarray, n_bus: int, pq_mask) -> np.ndarray:
    """Unit worst-case direction in the balanced [P; Q(PQ-only)] subspace."""
    q_bus_indices = np.flatnonzero(np.asarray(pq_mask, dtype=bool))
    blocks = make_ac_block_specs(int(n_bus), balance=True, q_bus_indices=q_bus_indices)
    return worst_case_l2_direction(np.asarray(h_full, dtype=float).reshape(-1), blocks)


def apply_du_and_solve(net, bus_ids, du, init="results"):
    """Apply du=[dP;dQ] as sgens on a deep copy, run PF; returns solved net or None."""
    n_bus = len(bus_ids)
    nn = copy.deepcopy(net)
    for pos, bid in enumerate(bus_ids):
        pp.create_sgen(nn, int(bid), p_mw=float(du[pos]), q_mvar=float(du[n_bus + pos]))
    try:
        pp.runpp(nn, init=init, **PF_KW)
    except Exception:
        return None
    return nn if bool(getattr(nn, "converged", True)) else None


def end_s_mva(nn, lid: int, end: str) -> float:
    r = nn.res_line.loc[int(lid)]
    if end == "from":
        return float(np.hypot(r.p_from_mw, r.q_from_mvar))
    return float(np.hypot(r.p_to_mw, r.q_to_mvar))


def save_json(name: str, obj) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=float)
    print(f"saved {path}")
    return path
