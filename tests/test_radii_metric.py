from __future__ import annotations

import numpy as np
import pytest


def test_metric_radius_closed_form_diagonal_M():
    from stability_radius.radii.metric import metric_radius

    margin = 10.0
    g = np.array([1.0, 2.0])
    M = np.array([[2.0, 0.0], [0.0, 8.0]])

    # g^T M^{-1} g = 1^2/2 + 2^2/8 = 0.5 + 0.5 = 1.0
    # radius = 10 / sqrt(1) = 10
    r = metric_radius(margin, g, M)
    assert r == pytest.approx(10.0)


def test_metric_radius_rejects_non_spd():
    from stability_radius.radii.metric import metric_radius

    margin = 1.0
    g = np.array([1.0, 0.0])
    # Not SPD (eigenvalues: 1 and -1)
    M = np.array([[1.0, 0.0], [0.0, -1.0]])

    with pytest.raises(ValueError):
        metric_radius(margin, g, M)
