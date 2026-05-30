from __future__ import annotations

import math

import numpy as np
import pytest

from stability_radius.geometry.balanced import (
    BlockSpec,
    dual_norm_l2_balanced,
    dual_norm_l2_balanced_from_block_vectors,
    dual_norm_l2_balanced_rows,
    make_ac_block_specs,
    project_dual_balanced,
    project_dual_balanced_rows,
    worst_case_l2_direction,
)


def test_project_dual_balanced_two_ac_blocks() -> None:
    h = np.array([3.0, 1.0, 2.0, 0.0])

    out = project_dual_balanced(h, make_ac_block_specs(2))

    assert np.allclose(out, np.array([1.0, -1.0, 1.0, -1.0]))
    assert float(np.sum(out[:2])) == pytest.approx(0.0)
    assert float(np.sum(out[2:])) == pytest.approx(0.0)


def test_weighted_projection_uses_sigma_squared_mean() -> None:
    h = np.array([1.0, -1.0, 0.0])
    weights = np.array([1.0, 4.0, 9.0])
    block = BlockSpec(name="P", indices=np.arange(3), weights=weights)

    out = project_dual_balanced(h, (block,))
    expected_mu = float(np.sum(weights * h) / np.sum(weights))

    assert np.allclose(out, h - expected_mu)
    assert float(np.sum(weights * out)) == pytest.approx(0.0, abs=1e-12)


def test_project_dual_balanced_rows_matches_row_loop() -> None:
    H = np.array(
        [
            [3.0, 1.0, 2.0, 0.0],
            [0.0, 2.0, 4.0, 6.0],
        ]
    )
    specs = make_ac_block_specs(2)

    rowwise = project_dual_balanced_rows(H, specs)
    expected = np.vstack([project_dual_balanced(row, specs) for row in H])

    assert np.allclose(rowwise, expected)
    assert np.allclose(
        dual_norm_l2_balanced_rows(H, specs), np.linalg.norm(expected, axis=1)
    )


def test_worst_case_l2_direction_is_projected_unit_vector() -> None:
    h = np.array([3.0, 1.0, 2.0, 0.0])

    direction = worst_case_l2_direction(h, make_ac_block_specs(2))

    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert float(np.sum(direction[:2])) == pytest.approx(0.0)
    assert float(np.sum(direction[2:])) == pytest.approx(0.0)
    assert dual_norm_l2_balanced(h, make_ac_block_specs(2)) == pytest.approx(2.0)


def test_reduced_block_norm_accounts_for_implicit_slack_zero() -> None:
    p_red = np.array([1.0, 2.0])
    q_red = np.array([4.0, 6.0])

    got = dual_norm_l2_balanced_from_block_vectors(
        (p_red, q_red),
        total_sizes=(3, 2),
    )

    p_full = np.array([0.0, 1.0, 2.0])
    p_proj = p_full - np.mean(p_full)
    q_proj = q_red - np.mean(q_red)
    expected = math.sqrt(float(np.dot(p_proj, p_proj) + np.dot(q_proj, q_proj)))

    assert got == pytest.approx(expected)


def test_make_ac_block_specs_can_restrict_q_to_pq_buses() -> None:
    h = np.array([1.0, 2.0, 3.0, 0.0, 4.0, 0.0])

    out = project_dual_balanced(h, make_ac_block_specs(3, q_bus_indices=np.array([1])))

    assert np.allclose(out[:3], np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(out[3:], np.zeros(3))
