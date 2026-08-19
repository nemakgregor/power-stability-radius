from __future__ import annotations

"""
Validation of the operator-norm radius for zero-flow (nondifferentiable) ends.

Network: symmetric diamond.  The tie line between the two symmetric buses
carries exactly zero flow, so |S0| = 0 and the scalar gradient of |S| does
not exist.  The 2D first-order model is exact at S0 = 0:

    |S(du)| = || [dP(du); dQ(du)] ||_2  + O(||du||^2)

so the certified bound  |S| <= sigma_max * ||du||  must hold, and along the
top singular direction it must be tight.
"""

import copy

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")


def _make_symmetric_diamond() -> tuple[object, int, int]:
    """Slack bus 0 feeds buses 1 and 2 through identical lines; a tie line
    1-2 carries zero flow by symmetry.  Returns (net, slack, tie_line_id)."""
    net = pp.create_empty_network(sn_mva=100.0)
    b0 = int(pp.create_bus(net, vn_kv=110.0))
    b1 = int(pp.create_bus(net, vn_kv=110.0))
    b2 = int(pp.create_bus(net, vn_kv=110.0))

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    # identical loads -> zero tie flow by symmetry
    pp.create_load(net, b1, p_mw=20.0, q_mvar=4.0)
    pp.create_load(net, b2, p_mw=20.0, q_mvar=4.0)

    common = dict(
        length_km=1.0,
        r_ohm_per_km=0.0,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1, x_ohm_per_km=0.10, **common
    )
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b2, x_ohm_per_km=0.10, **common
    )
    tie = int(
        pp.create_line_from_parameters(
            net, from_bus=b1, to_bus=b2, x_ohm_per_km=0.12, **common
        )
    )
    net.line.loc[:, "rateA"] = 100.0
    return net, b0, tie


def test_operator_norm_radius_finite_and_fd_tight() -> None:
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius

    net, slack, tie = _make_symmetric_diamond()
    bus_ids = [int(x) for x in sorted(net.bus.index)]

    base_pf = solve_ac_pf_base_point_from_pandapower(
        net=net, slack_bus=slack, solver="pandapower", init="flat", lossless=True
    )
    res = compute_ac_l2_radius(net, base_pf=base_pf, slack_bus=slack, lossless=True)
    row = res[f"line_{tie}"]

    # zero-flow end detected and certified via operator norm
    assert float(row["ac_s0_from_mva"]) < 1e-6
    assert row["nondifferentiable_apparent_power"] is True
    assert row["constraint_status_ac_l2"] == "ok_finite_operator_norm"
    r_cert = float(row["certificate_radius_ac_l2"])
    sigma = float(row["||h||2"])
    assert np.isfinite(r_cert) and r_cert > 0.0
    assert sigma > 0.0

    # FD tightness along the antisymmetric direction (the worst direction
    # for the tie line by symmetry): du = (0, +d, -d) on P.
    d = 0.5  # MW
    du_norm = np.sqrt(2.0) * d  # balanced L2 norm of the perturbation

    nn = copy.deepcopy(net)
    pp.create_sgen(nn, bus_ids[1], p_mw=+d, q_mvar=0.0)
    pp.create_sgen(nn, bus_ids[2], p_mw=-d, q_mvar=0.0)
    pp.runpp(
        nn,
        calculate_voltage_angles=True,
        enforce_q_lims=False,
        init="flat",
        tolerance_mva=1e-10,
        max_iteration=50,
    )
    s_tie = float(
        np.hypot(nn.res_line.loc[tie, "p_from_mw"], nn.res_line.loc[tie, "q_from_mvar"])
    )

    # certified bound holds ...
    assert s_tie <= sigma * du_norm * (1.0 + 1e-2) + 1e-9
    # ... and is tight along this direction (within 5%: symmetry makes the
    # antisymmetric P direction the top singular direction here).
    assert s_tie >= 0.95 * sigma * du_norm, (
        f"operator-norm bound too loose: |S|={s_tie:.6g}, "
        f"sigma*||du||={sigma * du_norm:.6g}"
    )
