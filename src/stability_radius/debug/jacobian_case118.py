"""Diagnostic: verify J * dx = du directly on case118 via finite differences."""
from __future__ import annotations

import copy
import logging
import math

import numpy as np

from stability_radius.ac.ac_model import build_ac_operator
from stability_radius.base_point.pandapower_tools import (
    apply_lossless_policy_to_pandapower_net,
    resolve_slack_bus_id,
)
from stability_radius.utils import create_module_output_dir, setup_output_dir_logging

logger = logging.getLogger(__name__)


def main() -> int:
    artifact_dir = create_module_output_dir(module_name="debug_jacobian_case118")
    setup_output_dir_logging(artifact_dir)
    logger.info("Artifact directory: %s", str(artifact_dir))

    try:
        import pandapower as pp
    except ImportError:
        logger.error("pandapower not available")
        return 1

    from pandapower.networks import case118

    net_raw = case118()
    slack_bus = 0

    bus_ids = [int(x) for x in sorted(net_raw.bus.index)]
    n_bus = len(bus_ids)
    slack_bus_id = resolve_slack_bus_id(net_raw, slack_bus)
    slack_pos = bus_ids.index(slack_bus_id)

    net = apply_lossless_policy_to_pandapower_net(net_raw)

    pp.runpp(
        net,
        calculate_voltage_angles=True,
        enforce_q_lims=True,
        init="flat",
        max_iteration=100,
        tolerance_mva=1e-8,
    )
    assert net.converged

    vm = np.array([float(net.res_bus.loc[b, "vm_pu"]) for b in bus_ids], dtype=float)
    va_deg = np.array([float(net.res_bus.loc[b, "va_degree"]) for b in bus_ids], dtype=float)
    va = va_deg * math.pi / 180.0

    line_ids = [int(x) for x in sorted(net.line.index)]

    op = build_ac_operator(
        net=net,
        slack_bus=slack_bus_id,
        vm_pu=vm,
        va_rad=va,
        line_indices=line_ids,
        lossless=True,
    )

    n_red = op.n_red
    n_pq = op.n_pq
    n_vars = op.n_vars

    logger.info("n_bus=%d, n_red(theta)=%d, n_pq=%d, n_vars=%d, J_shape=%s",
                n_bus, n_red, n_pq, n_vars, op.J.shape)

    # ---- Test 1: Jacobian prediction for bus injection ----
    # Perturb injection at a PQ bus (via sgen) and verify that
    # J^{-1} du = dx matches the actual voltage change from PF.

    # Find a PQ bus (non-slack, non-PV)
    test_bus_pos = None
    for bp in range(n_bus):
        if bp != slack_pos and op.pq_mask[bp]:
            test_bus_pos = bp
            break
    assert test_bus_pos is not None, "No PQ bus found"

    test_bus_id = bus_ids[test_bus_pos]
    test_theta_pos = op.theta_red_pos[test_bus_pos]

    eps = 0.01  # MW
    logger.info("Test bus: %d (pos=%d, theta_red_pos=%d), eps=%.4f MW",
                test_bus_id, test_bus_pos, test_theta_pos, eps)

    # du vector in reduced space: +eps MW at test bus, P block
    du_red = np.zeros(n_vars, dtype=float)
    du_red[test_theta_pos] = eps  # P injection (in theta/P block)

    # Predict dx = J^{-1} du
    dx_pred = op.J_lu.solve(du_red)
    dtheta_pred = dx_pred[:n_red]       # theta changes for all non-slack
    dV_pq_pred = dx_pred[n_red:]        # V changes for PQ buses only

    # Actual: perturb and run PF
    net_pert = copy.deepcopy(net)
    pp.create_sgen(net_pert, bus=test_bus_id, p_mw=eps, q_mvar=0.0, in_service=True)
    pp.runpp(
        net_pert,
        calculate_voltage_angles=True,
        enforce_q_lims=False,
        init="flat",
        max_iteration=100,
        tolerance_mva=1e-8,
    )
    assert net_pert.converged

    vm_pert = np.array([float(net_pert.res_bus.loc[b, "vm_pu"]) for b in bus_ids], dtype=float)
    va_pert_deg = np.array([float(net_pert.res_bus.loc[b, "va_degree"]) for b in bus_ids], dtype=float)
    va_pert = va_pert_deg * math.pi / 180.0

    # Actual dx (theta: exclude slack; V: PQ buses only)
    dtheta_actual = np.delete(va_pert - va, slack_pos)
    dV_actual_full = vm_pert - vm
    pq_indices = np.where(op.pq_mask)[0]
    dV_pq_actual = dV_actual_full[pq_indices]

    # Compare theta
    logger.info("\n--- Jacobian theta prediction (first 10 non-slack buses) ---")
    logger.info("%-6s %15s %15s", "RedPos", "dθ_pred", "dθ_actual")
    for ri in range(min(10, n_red)):
        logger.info("%-6d %15.6e %15.6e", ri, dtheta_pred[ri], dtheta_actual[ri])

    # Compare V (PQ buses only)
    logger.info("\n--- Jacobian V prediction (first 10 PQ buses) ---")
    logger.info("%-6s %15s %15s", "PQPos", "dV_pred", "dV_actual")
    for ri in range(min(10, n_pq)):
        logger.info("%-6d %15.6e %15.6e", ri, dV_pq_pred[ri], dV_pq_actual[ri])

    # Overall error
    err_theta = np.linalg.norm(dtheta_pred - dtheta_actual) / max(np.linalg.norm(dtheta_actual), 1e-15)
    err_V = np.linalg.norm(dV_pq_pred - dV_pq_actual) / max(np.linalg.norm(dV_pq_actual), 1e-15)
    logger.info("\nRelative error: theta=%.6e, V=%.6e", err_theta, err_V)

    # ---- Test 2: Direct h-vector check ----
    # Compute h = J^{-T} b for line 0, from-end
    # b = d|S|/d[theta, V] for line 0, from-end
    line_pos = 0
    lid = line_ids[line_pos]
    fb_pos = int(op.from_bus_pos[line_pos])
    tb_pos = int(op.to_bus_pos[line_pos])

    y = complex(op.y_series_pu[line_pos])
    g = float(np.real(y))
    b_im = float(np.imag(y))

    Vi = float(op.vm_pu[fb_pos])
    Vk = float(op.vm_pu[tb_pos])
    theta = float(op.va_rad[fb_pos] - op.va_rad[tb_pos])
    s_val = math.sin(theta)
    c_val = math.cos(theta)

    A = g * c_val + b_im * s_val
    Btmp = g * s_val - b_im * c_val

    # per-unit derivatives
    dP_dti_pu = Vi * Vk * Btmp
    dP_dtk_pu = -dP_dti_pu
    dQ_dti_pu = -Vi * Vk * A
    dQ_dtk_pu = -dQ_dti_pu

    dP_dVi_pu = 2.0 * g * Vi - Vk * A
    dP_dVk_pu = -Vi * A
    dQ_dVi_pu = -2.0 * b_im * Vi - Vk * Btmp
    dQ_dVk_pu = -Vi * Btmp

    scale = float(op.sn_mva)

    # Compute P, Q flow at this line end (from end)
    p0_line = float(net.res_line.loc[lid, "p_from_mw"])
    q0_line = float(net.res_line.loc[lid, "q_from_mvar"])
    s0_line = math.sqrt(p0_line**2 + q0_line**2)
    logger.info("\nLine %d from-end: P=%.4f MW, Q=%.4f MVAr, |S|=%.4f MVA", lid, p0_line, q0_line, s0_line)

    wP = p0_line / s0_line
    wQ = q0_line / s0_line

    # Build RHS
    rhs = np.zeros(n_vars, dtype=float)
    ri_fb_theta = int(op.theta_red_pos[fb_pos])
    ri_tb_theta = int(op.theta_red_pos[tb_pos])
    ri_fb_v = int(op.v_red_pos[fb_pos])
    ri_tb_v = int(op.v_red_pos[tb_pos])

    if ri_fb_theta >= 0:
        rhs[ri_fb_theta] += wP * scale * dP_dti_pu + wQ * scale * dQ_dti_pu
    if ri_tb_theta >= 0:
        rhs[ri_tb_theta] += wP * scale * dP_dtk_pu + wQ * scale * dQ_dtk_pu
    if ri_fb_v >= 0:
        rhs[n_red + ri_fb_v] += wP * scale * dP_dVi_pu + wQ * scale * dQ_dVi_pu
    if ri_tb_v >= 0:
        rhs[n_red + ri_tb_v] += wP * scale * dP_dVk_pu + wQ * scale * dQ_dVk_pu

    logger.info("RHS nonzero entries: %d (out of %d)", np.count_nonzero(rhs), len(rhs))
    logger.info("RHS ||b|| = %.6e", np.linalg.norm(rhs))

    # Solve h = J^{-T} b
    h = op.solve_J_transpose(rhs)
    h_P = h[:n_red]
    h_Q = h[n_red:]
    logger.info("h-vector: ||h_P|| = %.6e, ||h_Q|| = %.6e, ||h|| = %.6e",
                np.linalg.norm(h_P), np.linalg.norm(h_Q), np.linalg.norm(h))

    # Check prediction for the same single-bus perturbation
    # h^T du_red = h[test_theta_pos] * eps (P-block entry for the test bus)
    ds_pred = h[test_theta_pos] * eps

    # Actual ds
    s1_from = math.sqrt(
        float(net_pert.res_line.loc[lid, "p_from_mw"]) ** 2
        + float(net_pert.res_line.loc[lid, "q_from_mvar"]) ** 2
    )
    ds_actual = s1_from - s0_line

    logger.info("\nLine %d from-end FD check:", lid)
    logger.info("  h[test_theta_pos=%d] = %.6e", test_theta_pos, h[test_theta_pos])
    logger.info("  ds_pred = h * eps = %.6e", ds_pred)
    logger.info("  ds_actual = %.6e", ds_actual)
    if abs(ds_actual) > 1e-15:
        logger.info("  rel_err = %.4f", abs(ds_pred - ds_actual) / abs(ds_actual))

    # Check: does h^T correlate with dx prediction?
    # We know dx = J^{-1} du, and ds = (d|S|/dx)^T dx = b^T dx = b^T J^{-1} du
    # Also ds = h^T du = (J^{-T} b)^T du = b^T J^{-1} du  (same!)
    # So h^T du = b^T dx_pred
    ds_via_b = float(rhs @ dx_pred)
    logger.info("  ds_via_b = b^T * dx = %.6e (should match ds_pred)", ds_via_b)
    return 0
