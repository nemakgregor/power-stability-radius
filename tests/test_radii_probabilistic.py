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
