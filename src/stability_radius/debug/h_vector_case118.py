"""Diagnostic script: test h-vector FD prediction on case118.

Reproduces the experiment's operating point and tests whether h^T·du
matches the actual ΔS from pandapower PF for simple single-bus perturbations.
"""
from __future__ import annotations

import copy
import logging
import math

import numpy as np

from stability_radius.base_point.pandapower_opp import solve_ac_fpf
from stability_radius.base_point.pandapower_tools import (
    apply_lossless_policy_to_pandapower_net,
    apply_opp_result_to_pandapower_net,
    ensure_ext_grid_at_slack,
    resolve_slack_bus_id,
)
from stability_radius.radii.ac_l2 import compute_ac_l2_radius
from stability_radius.utils import create_module_output_dir, setup_output_dir_logging
from stability_radius.workflows import _expand_h_reduced_to_full

logger = logging.getLogger(__name__)


def main() -> int:
    artifact_dir = create_module_output_dir(module_name="debug_h_vector_case118")
    setup_output_dir_logging(artifact_dir)
    logger.info("Artifact directory: %s", str(artifact_dir))

    try:
        import pandapower as pp
    except ImportError:
        logger.error("pandapower not available")
        return 1

    # Load case118
    from pandapower.networks import case118

    net_raw = case118()
    slack_bus = 0  # use bus 0 as slack for simplicity

    bus_ids = [int(x) for x in sorted(net_raw.bus.index)]
    n_bus = len(bus_ids)
    slack_bus_id = resolve_slack_bus_id(net_raw, slack_bus)
    slack_pos = bus_ids.index(slack_bus_id)
    logger.info("case118: %d buses, slack_bus=%d (pos=%d)", n_bus, slack_bus_id, slack_pos)

    # Apply lossless policy
    net = apply_lossless_policy_to_pandapower_net(net_raw)

    # Run base PF
    pp.runpp(
        net,
        calculate_voltage_angles=True,
        enforce_q_lims=True,
        init="flat",
        max_iteration=100,
        tolerance_mva=1e-8,
    )
    assert net.converged, "Base PF did not converge"

    # Extract base-point Vm, Va
    vm = np.array([float(net.res_bus.loc[b, "vm_pu"]) for b in bus_ids], dtype=float)
    va_deg = np.array([float(net.res_bus.loc[b, "va_degree"]) for b in bus_ids], dtype=float)
    va = va_deg * math.pi / 180.0

    line_ids = [int(x) for x in sorted(net.line.index)]
    n_lines = len(line_ids)

    # Create a base_pf-like object
    from stability_radius.base_point.pypsa_pf import PyPSAAPFResult

    p0 = np.array([float(net.res_line.loc[lid, "p_from_mw"]) for lid in line_ids])
    q0 = np.array([float(net.res_line.loc[lid, "q_from_mvar"]) for lid in line_ids])
    p1 = np.array([float(net.res_line.loc[lid, "p_to_mw"]) for lid in line_ids])
    q1 = np.array([float(net.res_line.loc[lid, "q_to_mvar"]) for lid in line_ids])

    base_pf = PyPSAAPFResult(
        bus_ids=tuple(bus_ids),
        v_mag_pu=vm,
        v_ang_rad=va,
        line_ids=tuple(line_ids),
        line_p0_mw=p0,
        line_q0_mvar=q0,
        line_p1_mw=p1,
        line_q1_mvar=q1,
        status="PF_OK",
    )

    # Compute h-vectors
    ac_results = compute_ac_l2_radius(
        net,
        base_pf=base_pf,
        slack_bus=slack_bus_id,
        lossless=True,
        return_h_vectors=True,
    )

    h_vecs_raw = ac_results.pop("_h_vectors")
    h_from_full = _expand_h_reduced_to_full(
        h_vecs_raw["h_from"], n_bus=n_bus, slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )
    h_to_full = _expand_h_reduced_to_full(
        h_vecs_raw["h_to"], n_bus=n_bus, slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )

    # Base |S|
    s0_from = np.sqrt(p0 ** 2 + q0 ** 2)
    s0_to = np.sqrt(p1 ** 2 + q1 ** 2)

    # --- FD test: perturb a SINGLE non-slack bus ---
    eps = 0.01  # MW
    test_bus_pos = slack_pos + 1 if slack_pos < n_bus - 1 else slack_pos - 1
    test_bus_id = bus_ids[test_bus_pos]
    logger.info("Testing single-bus P perturbation: bus %d (pos %d), eps=%.4f MW",
                test_bus_id, test_bus_pos, eps)

    delta_u = np.zeros(2 * n_bus, dtype=float)
    delta_u[test_bus_pos] = eps  # +eps MW at test bus

    # Perturbed PF
    net_pert = copy.deepcopy(net)
    pp.create_sgen(
        net_pert,
        bus=test_bus_id,
        p_mw=eps,
        q_mvar=0.0,
        in_service=True,
    )
    pp.runpp(
        net_pert,
        calculate_voltage_angles=True,
        enforce_q_lims=True,
        init="flat",
        max_iteration=100,
        tolerance_mva=1e-8,
    )
    assert net_pert.converged, "Perturbed PF did not converge"

    # Compare for first 10 lines
    logger.info("\n%-6s %-6s %15s %15s %15s %10s", "Line", "End", "dS_pred", "dS_actual", "h_norm", "rel_err")
    for pos in range(min(10, n_lines)):
        lid = line_ids[pos]
        # From end
        s1_from = math.sqrt(
            float(net_pert.res_line.loc[lid, "p_from_mw"]) ** 2
            + float(net_pert.res_line.loc[lid, "q_from_mvar"]) ** 2
        )
        ds_from_actual = s1_from - s0_from[pos]
        ds_from_pred = float(h_from_full[pos, :] @ delta_u)

        # To end
        s1_to = math.sqrt(
            float(net_pert.res_line.loc[lid, "p_to_mw"]) ** 2
            + float(net_pert.res_line.loc[lid, "q_to_mvar"]) ** 2
        )
        ds_to_actual = s1_to - s0_to[pos]
        ds_to_pred = float(h_to_full[pos, :] @ delta_u)

        # Relative errors
        for end, ds_pred, ds_actual in [("from", ds_from_pred, ds_from_actual),
                                         ("to", ds_to_pred, ds_to_actual)]:
            h_vec = h_from_full[pos, :] if end == "from" else h_to_full[pos, :]
            h_norm = float(np.linalg.norm(h_vec))
            if abs(ds_actual) > 1e-12:
                rel_err = abs(ds_pred - ds_actual) / abs(ds_actual)
                logger.info("%-6d %-6s %15.6e %15.6e %15.6e %10.4f",
                           lid, end, ds_pred, ds_actual, h_norm, rel_err)
            else:
                logger.info("%-6d %-6s %15.6e %15.6e %15.6e %10s",
                           lid, end, ds_pred, ds_actual, h_norm, "skip")

    # --- Second FD test: balanced perturbation ---
    logger.info("\n--- Balanced perturbation test ---")
    bus_a_pos = 2  # pick two non-slack buses
    bus_b_pos = 5
    if bus_a_pos == slack_pos:
        bus_a_pos += 1
    if bus_b_pos == slack_pos:
        bus_b_pos += 1

    delta_u2 = np.zeros(2 * n_bus, dtype=float)
    delta_u2[bus_a_pos] = +eps
    delta_u2[bus_b_pos] = -eps

    net_pert2 = copy.deepcopy(net)
    pp.create_sgen(net_pert2, bus=bus_ids[bus_a_pos], p_mw=+eps, q_mvar=0.0, in_service=True)
    pp.create_sgen(net_pert2, bus=bus_ids[bus_b_pos], p_mw=-eps, q_mvar=0.0, in_service=True)
    pp.runpp(
        net_pert2,
        calculate_voltage_angles=True,
        enforce_q_lims=True,
        init="flat",
        max_iteration=100,
        tolerance_mva=1e-8,
    )
    assert net_pert2.converged, "Balanced perturbed PF did not converge"

    logger.info("%-6s %-6s %15s %15s %15s %10s", "Line", "End", "dS_pred", "dS_actual", "h_norm", "rel_err")
    n_checked = 0
    n_pass = 0
    for pos in range(n_lines):
        lid = line_ids[pos]
        s1_from = math.sqrt(
            float(net_pert2.res_line.loc[lid, "p_from_mw"]) ** 2
            + float(net_pert2.res_line.loc[lid, "q_from_mvar"]) ** 2
        )
        s1_to = math.sqrt(
            float(net_pert2.res_line.loc[lid, "p_to_mw"]) ** 2
            + float(net_pert2.res_line.loc[lid, "q_to_mvar"]) ** 2
        )
        ds_from_actual = s1_from - s0_from[pos]
        ds_to_actual = s1_to - s0_to[pos]
        ds_from_pred = float(h_from_full[pos, :] @ delta_u2)
        ds_to_pred = float(h_to_full[pos, :] @ delta_u2)

        for end, ds_pred, ds_actual in [("from", ds_from_pred, ds_from_actual),
                                         ("to", ds_to_pred, ds_to_actual)]:
            if abs(ds_actual) > 1e-8:
                rel_err = abs(ds_pred - ds_actual) / abs(ds_actual)
                n_checked += 1
                if rel_err < 0.05:
                    n_pass += 1
                if pos < 10 or rel_err > 0.5:
                    logger.info("%-6d %-6s %15.6e %15.6e %15.6e %10.4f%s",
                               lid, end, ds_pred, ds_actual,
                               float(np.linalg.norm(h_from_full[pos, :] if end == "from" else h_to_full[pos, :])),
                               rel_err, " FAIL" if rel_err > 0.05 else "")

    logger.info(
        "\nBalanced perturbation: %d / %d line-ends passed (<5%% rel_err)",
        n_pass,
        n_checked,
    )
    return 0
