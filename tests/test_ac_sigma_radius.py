from __future__ import annotations

import math

import numpy as np
import pytest

from stability_radius.radii.ac_sigma_radius import (
    compute_ac_sigma_radius,
    overload_probability_one_sided_limit,
    overload_probability_two_sided_signed,
)


def test_ac_sigma_radius_matches_margin_over_sigma_flow_synthetic_3bus() -> None:
    """
    Synthetic 3-bus test with a hand-constructed h-vector.

    Contract under test:
      r_sigma = (c - |S0|) / sigma_flow

    With balance=True the σ²-weighted projection is applied:
      hP_adj = hP - sum(σ²·hP)/sum(σ²)
      hQ_adj = hQ - sum(σ²·hQ)/sum(σ²)
    which ensures the worst-case perturbation satisfies 1ᵀΔP = 0, 1ᵀΔQ = 0.
    """
    n_bus = 3
    n_lines = 1

    # h = [hP; hQ], shape (n_lines, 2*n_bus)
    hP = np.array([1.0, -1.0, 0.0], dtype=float)
    hQ = np.array([0.5, 0.0, -0.5], dtype=float)
    h = np.concatenate([hP, hQ])[None, :]

    sigma_p = np.array([1.0, 2.0, 3.0], dtype=float)  # MW
    sigma_q = np.array([4.0, 5.0, 6.0], dtype=float)  # MVAr

    s0 = np.array([90.0], dtype=float)  # MVA (binding-end base magnitude)
    c = np.array([100.0], dtype=float)  # MVA
    margin = float(c[0] - s0[0])
    assert margin == pytest.approx(10.0)

    # Apply σ²-weighted balanced projection (same as compute_ac_sigma_radius)
    sigp2 = sigma_p * sigma_p
    sigq2 = sigma_q * sigma_q
    hP_adj = hP - np.sum(sigp2 * hP) / np.sum(sigp2)
    hQ_adj = hQ - np.sum(sigq2 * hQ) / np.sum(sigq2)

    expected_sigma_flow = math.sqrt(
        np.sum((sigma_p * hP_adj) ** 2) + np.sum((sigma_q * hQ_adj) ** 2)
    )

    res = compute_ac_sigma_radius(
        h_vectors=h,
        s_limit_mva=c,
        s0_mva=s0,
        sigma_p_mw=sigma_p,
        sigma_q_mvar=sigma_q,
        balance=True,
    )

    row = res["line_0"]
    assert float(row["sigma_flow_mva"]) == pytest.approx(
        expected_sigma_flow, rel=0.0, abs=1e-12
    )

    expected_r = margin / expected_sigma_flow
    assert float(row["radius_ac_sigma"]) == pytest.approx(
        expected_r, rel=0.0, abs=1e-12
    )

    # Worst-case point should hit the limit in the linearized model.
    assert float(row["worst_case_s_predicted_mva"]) == pytest.approx(
        float(c[0]), rel=0.0, abs=1e-10
    )

    dp = np.asarray(row["worst_case_dp_mw"], dtype=float)
    dq = np.asarray(row["worst_case_dq_mvar"], dtype=float)
    assert dp.shape == (n_bus,)
    assert dq.shape == (n_bus,)

    # Component-wise formula check (eq. [5]) with projected h.
    expected_dp = expected_r * (sigma_p * sigma_p * hP_adj) / expected_sigma_flow
    expected_dq = expected_r * (sigma_q * sigma_q * hQ_adj) / expected_sigma_flow
    assert np.allclose(dp, expected_dp, atol=1e-12, rtol=0.0)
    assert np.allclose(dq, expected_dq, atol=1e-12, rtol=0.0)

    # Balance constraint: worst-case perturbation must satisfy 1ᵀΔP = 0, 1ᵀΔQ = 0
    assert abs(np.sum(dp)) < 1e-10, f"sum(dp) = {np.sum(dp)}, not zero"
    assert abs(np.sum(dq)) < 1e-10, f"sum(dq) = {np.sum(dq)}, not zero"

    prob = float(row["overload_probability_ac"])
    assert 0.0 <= prob <= 1.0
    assert row["constraint_status_ac_sigma"] == "ok_finite"
    assert float(row["certificate_radius_ac_sigma"]) == pytest.approx(
        expected_r, rel=0.0, abs=1e-12
    )


def test_ac_overload_probability_is_one_sided_for_apparent_power() -> None:
    y0 = 90.0
    limit = 100.0
    sigma = 50.0

    one_sided = overload_probability_one_sided_limit(y0=y0, limit=limit, sigma=sigma)
    two_sided = overload_probability_two_sided_signed(
        flow0=y0, limit=limit, sigma=sigma
    )

    expected = 0.5 * math.erfc(((limit - y0) / sigma) / math.sqrt(2.0))
    assert one_sided == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert two_sided > one_sided


def test_ac_sigma_negative_margin_exports_nonnegative_certificate_radius() -> None:
    h = np.array([[1.0, -1.0, 0.5, -0.5]], dtype=float)
    sigma = np.array([1.0, 1.0], dtype=float)

    res = compute_ac_sigma_radius(
        h_vectors=h,
        s_limit_mva=np.array([80.0]),
        s0_mva=np.array([90.0]),
        sigma_p_mw=sigma,
        sigma_q_mvar=sigma,
        balance=True,
    )

    row = res["line_0"]
    assert float(row["radius_ac_sigma"]) < 0.0  # signed diagnostic field
    assert row["constraint_status_ac_sigma"] == "base_infeasible"
    assert float(row["certificate_radius_ac_sigma"]) == 0.0
    assert float(row["signed_distance_ac_sigma"]) < 0.0


def test_ac_sigma_pq_mask_excludes_pv_and_slack_q_coordinates() -> None:
    hP = np.zeros(4, dtype=float)
    hQ = np.array([1000.0, 1.0, -1.0, 1000.0], dtype=float)
    h = np.concatenate([hP, hQ])[None, :]
    sigma = np.ones(4, dtype=float)
    pq_mask = np.array([False, True, True, False])

    res = compute_ac_sigma_radius(
        h_vectors=h,
        s_limit_mva=np.array([100.0]),
        s0_mva=np.array([90.0]),
        sigma_p_mw=sigma,
        sigma_q_mvar=sigma,
        balance=True,
        pq_mask=pq_mask,
    )

    row = res["line_0"]
    assert float(row["sigma_flow_mva"]) == pytest.approx(math.sqrt(2.0))
    dq = np.asarray(row["worst_case_dq_mvar"], dtype=float)
    assert dq[0] == pytest.approx(0.0)
    assert dq[3] == pytest.approx(0.0)
    assert float(np.sum(dq[pq_mask])) == pytest.approx(0.0, abs=1e-12)
