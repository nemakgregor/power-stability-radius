from __future__ import annotations

"""
Regression test: slack-bus resolution consistency (operator vs h-expansion).

Historical bug: ``build_ac_operator`` resolved a positional ``slack_bus=0``
to bus position 0, while the h-vector expansion resolved it through
``resolve_slack_bus_id`` to the ext_grid bus.  On any network whose ext_grid
is NOT the first sorted bus, every ``h_P`` entry before the ext_grid position
was shifted by one bus, corrupting sigma radii, worst-case directions, and
finite-difference diagnostics (e.g. pglib case118: ext_grid bus 69).

This test builds a net whose ext_grid is deliberately NOT the first bus and
checks the full production path end-to-end against nonlinear finite
differences.
"""

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")

_TOL_REL = 0.01


def _make_net_extgrid_not_first() -> tuple[object, int]:
    """5-bus lossless ring; ext_grid on the 4th sorted bus (id 40)."""
    net = pp.create_empty_network(sn_mva=100.0)

    # Deliberately non-contiguous ids so positional and id-based slack
    # resolution disagree: sorted ids = [10, 20, 30, 40, 50].
    ids = [10, 20, 30, 40, 50]
    for bid in ids:
        pp.create_bus(net, vn_kv=110.0, index=bid)

    ext_bus = 40  # position 3 in sorted order, NOT position 0
    pp.create_ext_grid(net, ext_bus, vm_pu=1.0)

    pp.create_load(net, 10, p_mw=25.0, q_mvar=6.0)
    pp.create_load(net, 20, p_mw=15.0, q_mvar=4.0)
    pp.create_load(net, 50, p_mw=20.0, q_mvar=5.0)
    pp.create_gen(net, 30, p_mw=25.0, vm_pu=1.0)

    common = dict(
        length_km=1.0,
        r_ohm_per_km=0.0,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )
    edges = [
        (10, 20, 0.10),
        (20, 30, 0.12),
        (30, 40, 0.11),
        (40, 50, 0.13),
        (50, 10, 0.09),
    ]
    for fb, tb, x in edges:
        pp.create_line_from_parameters(
            net, from_bus=fb, to_bus=tb, x_ohm_per_km=x, **common
        )
    net.line.loc[:, "rateA"] = 1000.0
    return net, ext_bus


def test_h_vectors_correct_when_extgrid_not_first_bus() -> None:
    """Positional slack_bus=0 + ext_grid elsewhere must still give correct h."""
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius
    from stability_radius.workflows import expand_h_reduced_to_full

    net, ext_bus = _make_net_extgrid_not_first()
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)

    # The historical failure mode: config passes positional slack 0.
    base_pf = solve_ac_pf_base_point_from_pandapower(
        net=net, slack_bus=0, solver="pandapower", init="flat", lossless=True
    )
    ac = compute_ac_l2_radius(
        net, base_pf=base_pf, slack_bus=0, lossless=True, return_h_vectors=True
    )
    hv = ac.pop("_h_vectors")

    # 1) The operator must have eliminated the ext_grid bus, and the returned
    #    slack_pos must match it.
    assert int(hv["slack_bus_id"]) == ext_bus
    assert int(hv["slack_pos"]) == bus_ids.index(ext_bus)

    h_from = expand_h_reduced_to_full(
        hv["h_from"],
        n_bus=n_bus,
        slack_pos=int(hv["slack_pos"]),
        pq_mask=hv.get("pq_mask"),
    )

    # 2) End-to-end FD check on every line's from-end (P perturbations).
    import copy

    line_ids = [int(x) for x in sorted(net.line.index)]
    eps = 0.01  # MW

    base = copy.deepcopy(net)
    pp.runpp(
        base,
        calculate_voltage_angles=True,
        enforce_q_lims=False,
        init="flat",
        tolerance_mva=1e-10,
        max_iteration=50,
    )
    s0 = np.hypot(base.res_line.p_from_mw.values, base.res_line.q_from_mvar.values)

    for li, lid in enumerate(line_ids):
        h_p = h_from[li][:n_bus]
        # balanced pair perturbation on two non-slack buses
        pair = [b for b in bus_ids if b != ext_bus][:2]
        du = np.zeros(n_bus)
        du[bus_ids.index(pair[0])] = +eps
        du[bus_ids.index(pair[1])] = -eps

        nn = copy.deepcopy(base)
        for bid, val in zip(pair, (eps, -eps)):
            pp.create_sgen(nn, bid, p_mw=float(val), q_mvar=0.0)
        pp.runpp(
            nn,
            calculate_voltage_angles=True,
            enforce_q_lims=False,
            init="results",
            tolerance_mva=1e-10,
            max_iteration=50,
        )
        s1 = float(
            np.hypot(
                nn.res_line.loc[lid, "p_from_mw"], nn.res_line.loc[lid, "q_from_mvar"]
            )
        )

        d_actual = s1 - float(s0[li])
        d_pred = float(np.dot(h_p, du))
        denom = max(abs(d_actual), 1e-6)
        assert abs(d_pred - d_actual) / denom < _TOL_REL, (
            f"line {lid}: predicted {d_pred:.6g}, actual {d_actual:.6g} "
            f"(slack regression?)"
        )


def test_adjoint_residual_reported_and_small() -> None:
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius

    net, _ = _make_net_extgrid_not_first()
    base_pf = solve_ac_pf_base_point_from_pandapower(
        net=net, slack_bus=0, solver="pandapower", init="flat", lossless=True
    )
    ac = compute_ac_l2_radius(
        net, base_pf=base_pf, slack_bus=0, lossless=True, return_h_vectors=True
    )
    hv = ac.pop("_h_vectors")
    assert "adjoint_residual_max" in hv
    assert float(hv["adjoint_residual_max"]) < 1e-10
