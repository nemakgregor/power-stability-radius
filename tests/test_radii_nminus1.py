from __future__ import annotations

import warnings

import numpy as np
import pytest


def test_lodf_parallel_lines_and_effective_nminus1_radius():
    """
    Synthetic 2-bus, 2-parallel-line system.

    A 1 MW transfer from bus0->bus1 splits equally:
      flow on each line = 0.5
    So PTDF matrix for line endpoint transfers is:
      [[0.5, 0.5],
       [0.5, 0.5]]

    Outage of one line makes remaining line carry full flow (LODF=1).
    """
    from stability_radius.radii.nminus1 import (
        effective_nminus1_l2_radii,
        lodf_from_ptdf,
        ptdf_for_line_transfers,
    )

    # Choose H so that H @ [1,-1] = [0.5,0.5]
    H = np.array([[0.25, -0.25], [0.25, -0.25]], dtype=float)
    E = np.array([[1.0, -1.0], [1.0, -1.0]], dtype=float)

    ptdf = ptdf_for_line_transfers(H, E)
    assert ptdf == pytest.approx(np.array([[0.5, 0.5], [0.5, 0.5]]))

    lodf_res = lodf_from_ptdf(ptdf, islanding="raise")
    lodf = lodf_res.lodf
    assert lodf == pytest.approx(np.array([[-1.0, 1.0], [1.0, -1.0]]))

    base_flows = np.array([0.5, 0.5], dtype=float)
    limits = np.array([1.0, 1.0], dtype=float)
    G = H

    # Ensure no RuntimeWarning (divide by zero) is emitted
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        best_r, argmin = effective_nminus1_l2_radii(
            base_flows=base_flows,
            limits=limits,
            G=G,
            lodf=lodf,
            update_sensitivities=True,
        )

    assert not any(isinstance(w.message, RuntimeWarning) for w in captured), captured

    # If either line is outaged, the other hits its limit: margin=0 => radius=0
    assert best_r == pytest.approx(np.array([0.0, 0.0]))
    assert set(argmin.tolist()) <= {0, 1}
    assert argmin[0] != 0
    assert argmin[1] != 1
