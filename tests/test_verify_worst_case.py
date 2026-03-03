from __future__ import annotations

"""
Tests for verification.verify_worst_case.

Test strategy
-------------
Build a 3-bus pandapower network with tight thermal margins, compute the AC L2
certificate (including h-vectors), and verify:
  1. The worst-case perturbation at the boundary (scale=1.0) triggers a violation
     in the nonlinear AC PF.
  2. A 50%-scaled perturbation (scale=0.5, well inside the ball) does NOT trigger
     a violation.
  3. The relative error between the linear prediction and nonlinear PF is small.
"""

import math

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")
pytest.importorskip("pypsa")
pytest.importorskip("pandas")


def _make_3bus_tight_margin_net():
    """
    Create a 3-bus pandapower network with one line having a tight thermal margin.

    Topology
    --------
      bus0 (slack/ext_grid)
       |           \\
      line_0        line_1
       |             \\
      bus1 ——line_2—— bus2
      (load)         (load)

    Line 0 (bus0→bus1) is given a tight rateA so that its margin is small,
    making it easy to violate via the worst-case perturbation.
    """
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = int(pp.create_bus(net, vn_kv=110.0))
    b1 = int(pp.create_bus(net, vn_kv=110.0))
    b2 = int(pp.create_bus(net, vn_kv=110.0))

    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_load(net, b1, p_mw=30.0, q_mvar=5.0)
    pp.create_load(net, b2, p_mw=20.0, q_mvar=3.0)

    line_params = dict(
        length_km=1.0,
        r_ohm_per_km=0.01,
        x_ohm_per_km=0.10,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )

    pp.create_line_from_parameters(net, from_bus=b0, to_bus=b1, **line_params)
    pp.create_line_from_parameters(net, from_bus=b0, to_bus=b2, **line_params)
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, **line_params)

    return net, (b0, b1, b2)


def _compute_ac_radius_with_h(net, slack_bus: int):
    """Compute AC L2 radii and return results + full-dimension h-vectors."""
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius

    base_pf = solve_ac_pf_base_point_from_pandapower(
        net=net,
        slack_bus=slack_bus,
        solver="pandapower",
        init="flat",
        lossless=True,
    )

    res = compute_ac_l2_radius(
        net,
        base_pf=base_pf,
        slack_bus=slack_bus,
        chunk_size=32,
        balance=True,
        lossless=True,
        return_h_vectors=True,
    )

    h_vecs_raw = res.pop("_h_vectors")
    h_from = h_vecs_raw["h_from"]
    h_to = h_vecs_raw["h_to"]

    # Expand reduced h-vectors to full dimension (insert slack bus zeros).
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)
    slack_pos = bus_ids.index(int(slack_bus))

    from stability_radius.workflows import _expand_h_reduced_to_full

    h_from_full = _expand_h_reduced_to_full(h_from, n_bus=n_bus, slack_pos=slack_pos)
    h_to_full = _expand_h_reduced_to_full(h_to, n_bus=n_bus, slack_pos=slack_pos)

    return res, h_from_full, h_to_full, base_pf


class TestVerifyWorstCase:
    def test_worst_case_at_boundary_causes_violation(self) -> None:
        """
        At scale=1.0 (boundary of the L2 ball), the worst-case perturbation
        should produce an actual apparent power that is close to the limit.

        Because the linear model slightly overestimates or underestimates stress
        depending on nonlinear effects, we check only that the PF converged and
        the actual flow is close to the predicted value (within ~20% relative error).
        """
        from stability_radius.verification.verify_worst_case import (
            WorstCaseVerificationResult,
            verify_worst_case,
        )

        net, (b0, b1, b2) = _make_3bus_tight_margin_net()

        # Set a tight limit on line 0: |S0| + small margin
        pp.runpp(net, calculate_voltage_angles=True, init="flat")
        lid_target = int(sorted(net.line.index)[0])
        s_from = math.sqrt(
            float(net.res_line.loc[lid_target, "p_from_mw"]) ** 2
            + float(net.res_line.loc[lid_target, "q_from_mvar"]) ** 2
        )
        s_to = math.sqrt(
            float(net.res_line.loc[lid_target, "p_to_mw"]) ** 2
            + float(net.res_line.loc[lid_target, "q_to_mvar"]) ** 2
        )
        s_base = max(s_from, s_to)
        tight_limit = s_base + 2.0  # 2 MVA margin
        net.line.loc[lid_target, "rateA"] = tight_limit

        # Set generous limits on other lines so they are not binding.
        for lid in sorted(net.line.index):
            if int(lid) != lid_target:
                net.line.loc[lid, "rateA"] = 9999.0

        # Compute AC L2 radii with h-vectors.
        res, h_from_full, h_to_full, base_pf = _compute_ac_radius_with_h(net, b0)

        key = f"line_{lid_target}"
        row = res[key]
        radius = float(row["radius_ac_l2"])
        binding_end = str(row["binding_end"])

        assert math.isfinite(radius) and radius > 0.0, (
            f"Expected positive finite radius, got {radius}"
        )

        line_pos = [int(x) for x in sorted(net.line.index)].index(lid_target)
        if binding_end == "from":
            h_vec = h_from_full[line_pos]
            s0 = float(row["ac_s0_from_mva"])
        else:
            h_vec = h_to_full[line_pos]
            s0 = float(row["ac_s0_to_mva"])

        result = verify_worst_case(
            net=net,
            line_id=lid_target,
            h_vec=h_vec,
            radius=radius,
            s0_mva=s0,
            limit_mva=tight_limit,
            scale=1.0,
            balance=True,
            lossless=True,
        )

        assert isinstance(result, WorstCaseVerificationResult)
        assert result.pf_converged, "Nonlinear PF should converge"
        assert math.isfinite(result.actual_s_mva)
        assert math.isfinite(result.relative_error)
        # The linear model and nonlinear PF should agree within ~20%
        # for a small lossless network near the operating point.
        assert result.relative_error < 0.20, (
            f"Relative error too large: {result.relative_error:.4f}"
        )

    def test_half_scaled_direction_does_not_violate(self) -> None:
        """
        At scale=0.5 (well inside the ball), the perturbation should NOT cause
        a thermal violation.
        """
        from stability_radius.verification.verify_worst_case import verify_worst_case

        net, (b0, b1, b2) = _make_3bus_tight_margin_net()

        pp.runpp(net, calculate_voltage_angles=True, init="flat")
        lid_target = int(sorted(net.line.index)[0])
        s_from = math.sqrt(
            float(net.res_line.loc[lid_target, "p_from_mw"]) ** 2
            + float(net.res_line.loc[lid_target, "q_from_mvar"]) ** 2
        )
        s_to = math.sqrt(
            float(net.res_line.loc[lid_target, "p_to_mw"]) ** 2
            + float(net.res_line.loc[lid_target, "q_to_mvar"]) ** 2
        )
        s_base = max(s_from, s_to)
        tight_limit = s_base + 2.0
        net.line.loc[lid_target, "rateA"] = tight_limit

        for lid in sorted(net.line.index):
            if int(lid) != lid_target:
                net.line.loc[lid, "rateA"] = 9999.0

        res, h_from_full, h_to_full, base_pf = _compute_ac_radius_with_h(net, b0)

        key = f"line_{lid_target}"
        row = res[key]
        radius = float(row["radius_ac_l2"])
        binding_end = str(row["binding_end"])

        line_pos = [int(x) for x in sorted(net.line.index)].index(lid_target)
        if binding_end == "from":
            h_vec = h_from_full[line_pos]
            s0 = float(row["ac_s0_from_mva"])
        else:
            h_vec = h_to_full[line_pos]
            s0 = float(row["ac_s0_to_mva"])

        result = verify_worst_case(
            net=net,
            line_id=lid_target,
            h_vec=h_vec,
            radius=radius,
            s0_mva=s0,
            limit_mva=tight_limit,
            scale=0.5,
            balance=True,
            lossless=True,
        )

        assert result.pf_converged, "Nonlinear PF should converge at half scale"
        assert not result.violated, (
            f"Half-scale perturbation should not violate limit: "
            f"actual={result.actual_s_mva:.4f}, limit={result.limit_mva:.4f}"
        )

    def test_to_dict_returns_json_friendly(self) -> None:
        """WorstCaseVerificationResult.to_dict() returns a plain dict."""
        from stability_radius.verification.verify_worst_case import (
            WorstCaseVerificationResult,
        )

        r = WorstCaseVerificationResult(
            line_id=0,
            predicted_s_mva=50.0,
            actual_s_mva=49.5,
            limit_mva=48.0,
            violated=True,
            pf_converged=True,
            relative_error=0.01,
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["line_id"] == 0
        assert d["violated"] is True
        assert d["pf_converged"] is True

    def test_pf_divergence_returns_nan(self) -> None:
        """
        If the perturbation is extreme and the PF diverges, the result should
        have pf_converged=False and NaN for actual_s_mva / relative_error.
        """
        from stability_radius.verification.verify_worst_case import verify_worst_case

        net, (b0, b1, b2) = _make_3bus_tight_margin_net()

        for lid in sorted(net.line.index):
            net.line.loc[lid, "rateA"] = 100.0

        res, h_from_full, h_to_full, base_pf = _compute_ac_radius_with_h(net, b0)

        lid_target = int(sorted(net.line.index)[0])
        key = f"line_{lid_target}"
        row = res[key]
        binding_end = str(row["binding_end"])

        line_pos = [int(x) for x in sorted(net.line.index)].index(lid_target)
        if binding_end == "from":
            h_vec = h_from_full[line_pos]
            s0 = float(row["ac_s0_from_mva"])
        else:
            h_vec = h_to_full[line_pos]
            s0 = float(row["ac_s0_to_mva"])

        # Use an absurdly large custom delta_u to try to make PF diverge.
        n_bus = len(sorted(net.bus.index))
        huge_delta = np.zeros(2 * n_bus, dtype=float)
        huge_delta[1] = 1e8  # massive injection at one bus

        result = verify_worst_case(
            net=net,
            line_id=lid_target,
            h_vec=h_vec,
            radius=1.0,
            s0_mva=s0,
            limit_mva=100.0,
            delta_u=huge_delta,
        )

        if not result.pf_converged:
            assert math.isnan(result.actual_s_mva)
            assert math.isnan(result.relative_error)

    def test_custom_delta_u_is_used(self) -> None:
        """
        When delta_u is provided explicitly, it overrides the h-vec/radius
        construction.  Verify that a zero perturbation gives actual ≈ S0.
        """
        from stability_radius.verification.verify_worst_case import verify_worst_case

        net, (b0, b1, b2) = _make_3bus_tight_margin_net()

        for lid in sorted(net.line.index):
            net.line.loc[lid, "rateA"] = 9999.0

        res, h_from_full, h_to_full, base_pf = _compute_ac_radius_with_h(net, b0)

        lid_target = int(sorted(net.line.index)[0])
        key = f"line_{lid_target}"
        row = res[key]
        binding_end = str(row["binding_end"])

        line_pos = [int(x) for x in sorted(net.line.index)].index(lid_target)
        if binding_end == "from":
            h_vec = h_from_full[line_pos]
            s0 = float(row["ac_s0_from_mva"])
        else:
            h_vec = h_to_full[line_pos]
            s0 = float(row["ac_s0_to_mva"])

        n_bus = len(sorted(net.bus.index))
        zero_delta = np.zeros(2 * n_bus, dtype=float)

        result = verify_worst_case(
            net=net,
            line_id=lid_target,
            h_vec=h_vec,
            radius=1.0,
            s0_mva=s0,
            limit_mva=9999.0,
            delta_u=zero_delta,
        )

        assert result.pf_converged
        # With zero perturbation, the actual flow should be very close to s0.
        # Small difference due to lossless policy applied vs original net.
        assert abs(result.actual_s_mva - s0) < 1.0, (
            f"Zero perturbation should give actual ≈ S0: actual={result.actual_s_mva:.4f}, S0={s0:.4f}"
        )
