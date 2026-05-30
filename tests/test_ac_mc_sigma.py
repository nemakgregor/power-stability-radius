from __future__ import annotations

"""
Tests for verification.ac_monte_carlo_sigma.

Test strategy
-------------
Load the IEEE 14-bus case (pandapower built-in ``case14``), set tight but
feasible thermal limits, compute the AC sigma-radius certificate, and
verify that the per-bus covariance Monte Carlo respects the certificate
boundary.

Key assertion
-------------
At ``0.9 * r_σ`` (well inside the certified ball), the
``soundness_inside_sigma_ball`` must equal 1.0 — i.e., every MC sample
whose ``‖Σ^{-1/2} Δu‖₂ ≤ 0.9 * r_σ`` must have no thermal violations.
"""

import math

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pn = pytest.importorskip("pandapower.networks")
pytest.importorskip("scipy")
pytest.importorskip("pypsa")
pytest.importorskip("pandas")


def _make_case14_tight_limits():
    """
    Create a case14 pandapower network with finite tight thermal limits.

    Sets ``rateA`` on every line to ``max(|S_from|, |S_to|) + margin`` so
    that the base point is feasible but the margins are small enough for a
    meaningful sigma-radius certificate.

    Returns
    -------
    net : pandapower network
    slack_bus : int
    """
    net = pn.case14()
    pp.runpp(net, calculate_voltage_angles=True, init="flat")

    margin_mva = 10.0  # 10 MVA margin above base flow

    for lid in sorted(net.line.index):
        p_from = float(net.res_line.loc[lid, "p_from_mw"])
        q_from = float(net.res_line.loc[lid, "q_from_mvar"])
        p_to = float(net.res_line.loc[lid, "p_to_mw"])
        q_to = float(net.res_line.loc[lid, "q_to_mvar"])

        s_from = math.sqrt(p_from * p_from + q_from * q_from)
        s_to = math.sqrt(p_to * p_to + q_to * q_to)
        s_base = max(s_from, s_to)

        net.line.loc[lid, "rateA"] = s_base + margin_mva

    slack_bus = int(net.ext_grid.bus.iloc[0])
    return net, slack_bus


def _compute_ac_radius_and_sigma(net, slack_bus: int, sigma_val: float):
    """
    Compute AC L2 radii, h-vectors, and sigma-radius for *net*.

    Uses uniform per-bus sigma (``sigma_val`` MW for P, same for Q) so that
    the test is self-contained.

    Returns
    -------
    r_sigma_min : float
        Minimum finite sigma-radius across all lines.
    sigma_p : (n_bus,) array
    sigma_q : (n_bus,) array
    """
    from stability_radius.base_point.pypsa_pf import (
        solve_ac_pf_base_point_from_pandapower,
    )
    from stability_radius.radii.ac_l2 import compute_ac_l2_radius
    from stability_radius.radii.ac_sigma_radius import compute_ac_sigma_radius
    from stability_radius.workflows import expand_h_reduced_to_full

    # Solve AC PF base point
    base_pf = solve_ac_pf_base_point_from_pandapower(
        net=net,
        slack_bus=slack_bus,
        solver="pandapower",
        init="flat",
        lossless=True,
    )

    # Compute AC L2 radii with h-vectors
    res = compute_ac_l2_radius(
        net,
        base_pf=base_pf,
        slack_bus=slack_bus,
        chunk_size=64,
        balance=True,
        lossless=True,
        return_h_vectors=True,
    )

    h_vecs_raw = res.pop("_h_vectors")
    h_from = h_vecs_raw["h_from"]
    h_to = h_vecs_raw["h_to"]

    # Expand to full dimension (insert slack zeros)
    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)
    slack_pos = bus_ids.index(int(slack_bus))

    h_from_full = expand_h_reduced_to_full(
        h_from,
        n_bus=n_bus,
        slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )
    h_to_full = expand_h_reduced_to_full(
        h_to,
        n_bus=n_bus,
        slack_pos=slack_pos,
        pq_mask=h_vecs_raw.get("pq_mask"),
    )

    # Select binding h-vector per line
    line_ids = [int(x) for x in sorted(net.line.index)]
    m = len(line_ids)

    h_binding = np.zeros((m, 2 * n_bus), dtype=float)
    s0_binding = np.zeros(m, dtype=float)
    s_limit = np.zeros(m, dtype=float)

    for pos, lid in enumerate(line_ids):
        key = f"line_{lid}"
        row = res[key]
        binding_end = str(row["binding_end"])

        if binding_end == "from":
            h_binding[pos, :] = h_from_full[pos]
            s0_binding[pos] = float(row["ac_s0_from_mva"])
        else:
            h_binding[pos, :] = h_to_full[pos]
            s0_binding[pos] = float(row["ac_s0_to_mva"])

        s_limit[pos] = float(row["ac_s_limit_mva"])

    # Build uniform sigma arrays
    sigma_p = np.full(n_bus, float(sigma_val), dtype=float)
    sigma_q = np.full(n_bus, float(sigma_val), dtype=float)

    # Compute sigma-radius
    sigma_results = compute_ac_sigma_radius(
        h_vectors=h_binding,
        s_limit_mva=s_limit,
        s0_mva=s0_binding,
        sigma_p_mw=sigma_p,
        sigma_q_mvar=sigma_q,
        line_ids=line_ids,
        balance=True,
    )

    # Find minimum finite sigma-radius
    r_sigmas = []
    for lid in line_ids:
        key = f"line_{lid}"
        r = float(sigma_results[key]["radius_ac_sigma"])
        if math.isfinite(r) and r > 0:
            r_sigmas.append(r)

    assert len(r_sigmas) > 0, "No finite positive sigma radius found"
    r_sigma_min = min(r_sigmas)

    return r_sigma_min, sigma_p, sigma_q


class TestACSigmaMonteCarlo:
    def test_soundness_inside_sigma_ball_case14(self) -> None:
        """
        On case14 with 500 samples and r_check = 0.9 * r_σ, every sample
        inside the sigma ball must have no thermal violations.
        """
        from stability_radius.verification.ac_monte_carlo_sigma import (
            ACSigmaMCResult,
            run_ac_monte_carlo_sigma,
        )

        net, slack_bus = _make_case14_tight_limits()

        # Use moderate sigma (1 MW / 1 MVAr per bus)
        sigma_val = 1.0
        r_sigma_min, sigma_p, sigma_q = _compute_ac_radius_and_sigma(
            net, slack_bus, sigma_val
        )

        assert math.isfinite(r_sigma_min) and r_sigma_min > 0, (
            f"Expected positive finite r_sigma_min, got {r_sigma_min}"
        )

        r_check = 0.9 * r_sigma_min

        result = run_ac_monte_carlo_sigma(
            net=net,
            sigma_p_mw=sigma_p,
            sigma_q_mvar=sigma_q,
            r_sigma=r_check,
            n_samples=500,
            seed=42,
            feas_tol_mva=1e-3,
            lossless=True,
        )

        assert isinstance(result, ACSigmaMCResult)
        assert result.n_samples == 500
        assert result.n_pf_failures >= 0

        # The central assertion: soundness must be perfect inside the
        # 0.9 * r_σ ball.
        assert math.isfinite(result.soundness_inside_sigma_ball), (
            "Expected finite soundness (some samples should land inside the "
            "sigma ball), got NaN — possible issue with r_sigma being too small"
        )
        assert result.soundness_inside_sigma_ball == 1.0, (
            f"Certificate violation inside 0.9*r_sigma ball: "
            f"soundness={result.soundness_inside_sigma_ball:.6f}, "
            f"n_violations={result.n_violations}"
        )

    def test_empirical_overload_probability_keys(self) -> None:
        """
        The empirical_overload_probability dict must have one entry per line,
        with probabilities in [0, 1].
        """
        from stability_radius.verification.ac_monte_carlo_sigma import (
            run_ac_monte_carlo_sigma,
        )

        net, slack_bus = _make_case14_tight_limits()
        n_bus = len(sorted(net.bus.index))
        sigma_p = np.full(n_bus, 1.0, dtype=float)
        sigma_q = np.full(n_bus, 1.0, dtype=float)

        result = run_ac_monte_carlo_sigma(
            net=net,
            sigma_p_mw=sigma_p,
            sigma_q_mvar=sigma_q,
            r_sigma=1.0,
            n_samples=20,
            seed=123,
            lossless=True,
        )

        line_ids = sorted(net.line.index)
        assert len(result.empirical_overload_probability) == len(line_ids)

        for lid in line_ids:
            key = f"line_{lid}"
            assert key in result.empirical_overload_probability
            prob = result.empirical_overload_probability[key]
            assert 0.0 <= prob <= 1.0, f"Invalid probability for {key}: {prob}"

    def test_pf_failures_do_not_count_as_all_line_overloads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PF failures are system bad samples, not per-line thermal overloads."""
        from stability_radius.verification.ac_monte_carlo_sigma import (
            run_ac_monte_carlo_sigma,
        )

        net, _ = _make_case14_tight_limits()
        n_bus = len(sorted(net.bus.index))
        sigma_p = np.full(n_bus, 1.0, dtype=float)
        sigma_q = np.full(n_bus, 1.0, dtype=float)

        calls = {"n": 0}

        def fake_runpp(net_obj, *args, **kwargs):  # noqa: ANN001, ARG001
            calls["n"] += 1
            net_obj.converged = calls["n"] == 1

        monkeypatch.setattr(pp, "runpp", fake_runpp)

        result = run_ac_monte_carlo_sigma(
            net=net,
            sigma_p_mw=sigma_p,
            sigma_q_mvar=sigma_q,
            r_sigma=1.0,
            n_samples=4,
            seed=7,
            lossless=True,
        )

        assert result.n_pf_failures == 4
        assert result.n_violations == 4
        assert result.pf_failure_probability == 1.0
        assert result.bad_sample_probability == 1.0
        assert all(v == 0.0 for v in result.empirical_overload_probability.values())
        assert result.empirical_overload_probability == (
            result.empirical_overload_probability_conditional_on_pf_converged
        )

    def test_input_validation(self) -> None:
        """Sigma arrays and r_sigma must satisfy positivity constraints."""
        from stability_radius.verification.ac_monte_carlo_sigma import (
            run_ac_monte_carlo_sigma,
        )

        net, _ = _make_case14_tight_limits()
        n_bus = len(sorted(net.bus.index))

        good_sigma = np.full(n_bus, 1.0, dtype=float)
        bad_sigma = np.full(n_bus, -1.0, dtype=float)

        with pytest.raises(ValueError, match="sigma_p_mw"):
            run_ac_monte_carlo_sigma(
                net=net,
                sigma_p_mw=bad_sigma,
                sigma_q_mvar=good_sigma,
                r_sigma=1.0,
                n_samples=10,
            )

        with pytest.raises(ValueError, match="r_sigma"):
            run_ac_monte_carlo_sigma(
                net=net,
                sigma_p_mw=good_sigma,
                sigma_q_mvar=good_sigma,
                r_sigma=-1.0,
                n_samples=10,
            )
