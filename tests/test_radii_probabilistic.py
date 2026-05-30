from __future__ import annotations

import math

import numpy as np
import pytest


def test_sigma_radius_and_overload_probability_nonzero_baseflow():
    from stability_radius.radii.probabilistic import (
        overload_probability_symmetric_limit,
        sigma_radius,
    )

    flow0 = 3.0
    limit = 10.0
    sigma = 2.0
    margin = limit - abs(flow0)

    r = sigma_radius(margin, sigma)
    assert r == pytest.approx(3.5)

    # Compare to the same closed-form computed manually using erfc:
    # Q(x) = 0.5 * erfc(x/sqrt(2))
    def Q(x: float) -> float:
        return 0.5 * math.erfc(x / math.sqrt(2.0))

    expected = Q((limit - abs(flow0)) / sigma) + Q((limit + abs(flow0)) / sigma)
    prob = overload_probability_symmetric_limit(flow0=flow0, limit=limit, sigma=sigma)
    assert prob == pytest.approx(expected)


def test_overload_probability_sigma_zero_edge_cases():
    from stability_radius.radii.probabilistic import (
        overload_probability_symmetric_limit,
    )

    assert overload_probability_symmetric_limit(flow0=5.0, limit=10.0, sigma=0.0) == 0.0
    assert (
        overload_probability_symmetric_limit(flow0=11.0, limit=10.0, sigma=0.0) == 1.0
    )


def test_sigma_radius_zero_sigma_preserves_infeasible_signed_distance():
    from stability_radius.radii.probabilistic import sigma_radius

    assert sigma_radius(-1.0, 0.0) == float("-inf")
    assert sigma_radius(0.0, 0.0) == float("inf")
    assert math.isnan(sigma_radius(1.0, float("nan")))


def test_flow_stddev_validates_covariance_inputs():
    from stability_radius.radii.probabilistic import flow_stddev

    g = np.array([1.0, -1.0])
    assert flow_stddev(g, np.array([4.0, 9.0])) == pytest.approx(math.sqrt(13.0))

    with pytest.raises(ValueError, match="finite and non-negative"):
        flow_stddev(g, np.array([1.0, -1.0]))

    with pytest.raises(ValueError, match="Sigma entries must be finite"):
        flow_stddev(g, np.array([[1.0, float("nan")], [float("nan"), 1.0]]))

    with pytest.raises(ValueError, match="Sigma must be symmetric"):
        flow_stddev(g, np.array([[1.0, 0.5], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="positive semidefinite"):
        flow_stddev(g, np.array([[1.0, 0.0], [0.0, -1.0]]))

    with pytest.raises(ValueError, match="g must contain only finite values"):
        flow_stddev(np.array([1.0, float("nan")]), np.ones(2))
