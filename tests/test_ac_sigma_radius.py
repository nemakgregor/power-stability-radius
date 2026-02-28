from __future__ import annotations

import math

import numpy as np
import pytest

from stability_radius.radii.ac_sigma_radius import compute_ac_sigma_radius


def test_ac_sigma_radius_matches_margin_over_sigma_flow_synthetic_3bus() -> None:
    """
    Synthetic 3-bus test with a hand-constructed h-vector.

    Contract under test:
      r_sigma = (c - |S0|) / sigma_flow

    We choose h blocks already balanced (sum=0), so balance projection should not change results.
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

    expected_sigma_flow = math.sqrt(
        (sigma_p[0] * hP[0]) ** 2
        + (sigma_p[1] * hP[1]) ** 2
        + (sigma_p[2] * hP[2]) ** 2
        + (sigma_q[0] * hQ[0]) ** 2
        + (sigma_q[1] * hQ[1]) ** 2
        + (sigma_q[2] * hQ[2]) ** 2
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

    # Component-wise formula check (eq. [5]).
    expected_dp = expected_r * (sigma_p * sigma_p * hP) / expected_sigma_flow
    expected_dq = expected_r * (sigma_q * sigma_q * hQ) / expected_sigma_flow
    assert np.allclose(dp, expected_dp, atol=1e-12, rtol=0.0)
    assert np.allclose(dq, expected_dq, atol=1e-12, rtol=0.0)

    prob = float(row["overload_probability_ac"])
    assert 0.0 <= prob <= 1.0
