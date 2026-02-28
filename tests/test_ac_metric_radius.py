from __future__ import annotations

import math

import numpy as np
import pytest

from stability_radius.radii.ac_metric_radius import compute_ac_metric_radius


class TestACMetricRadiusDiagonalM:
    """Verify closed-form formula with diagonal M."""

    def test_closed_form_single_line(self) -> None:
        """
        Hand-calculated example:
          h = [1, -1, 0.5, -0.5]  (2 buses, hP=[1,-1], hQ=[0.5,-0.5])
          M = diag([2, 8, 4, 16])  (diagonal SPD)
          margin = 10 MVA

        h is already balanced (sum(hP)=0, sum(hQ)=0), so balance should not
        change the result.

        h^T M^{-1} h = 1/2 + 1/8 + 0.25/4 + 0.25/16
                      = 0.5 + 0.125 + 0.0625 + 0.015625
                      = 0.703125
        denom = sqrt(0.703125)
        radius = 10 / sqrt(0.703125)
        """
        n_bus = 2
        h = np.array([[1.0, -1.0, 0.5, -0.5]])
        M_diag = np.array([2.0, 8.0, 4.0, 16.0])
        c = np.array([100.0])
        s0 = np.array([90.0])
        margin = 10.0

        expected_htMinvh = 1.0 / 2.0 + 1.0 / 8.0 + 0.25 / 4.0 + 0.25 / 16.0
        expected_denom = math.sqrt(expected_htMinvh)
        expected_radius = margin / expected_denom

        res = compute_ac_metric_radius(
            h_vectors=h,
            s_limit_mva=c,
            s0_mva=s0,
            M=M_diag,
            balance=True,
        )
        row = res["line_0"]
        assert float(row["metric_denom"]) == pytest.approx(expected_denom, abs=1e-12)
        assert float(row["margin_mva"]) == pytest.approx(margin, abs=1e-12)
        assert float(row["radius_ac_metric"]) == pytest.approx(
            expected_radius, abs=1e-12
        )


class TestACMetricRadiusDenseM:
    """Verify dense (Cholesky) path."""

    def test_dense_diagonal_matches_diagonal_path(self) -> None:
        """Dense M that is actually diagonal must give the same result as the diagonal path."""
        h = np.array([[3.0, -1.0, 2.0, -2.0]])
        M_diag = np.array([1.0, 4.0, 9.0, 16.0])
        M_dense = np.diag(M_diag)
        c = np.array([50.0])
        s0 = np.array([40.0])

        res_diag = compute_ac_metric_radius(
            h_vectors=h,
            s_limit_mva=c,
            s0_mva=s0,
            M=M_diag,
            balance=False,
        )
        res_dense = compute_ac_metric_radius(
            h_vectors=h,
            s_limit_mva=c,
            s0_mva=s0,
            M=M_dense,
            balance=False,
        )

        assert float(res_diag["line_0"]["metric_denom"]) == pytest.approx(
            float(res_dense["line_0"]["metric_denom"]), abs=1e-12
        )
        assert float(res_diag["line_0"]["radius_ac_metric"]) == pytest.approx(
            float(res_dense["line_0"]["radius_ac_metric"]), abs=1e-12
        )

    def test_dense_non_diagonal(self) -> None:
        """Dense non-diagonal SPD matrix — verify against manual Cholesky."""
        # M = [[4, 2], [2, 3]]  (SPD, eigenvalues ~1.17 and ~5.83)
        # h = [1, 1] (trivial 1-bus case, d=2)
        # M^{-1} = [[3, -2], [-2, 4]] / 8
        # h^T M^{-1} h = (3 - 2 - 2 + 4) / 8 = 3/8
        # denom = sqrt(3/8)
        h = np.array([[1.0, 1.0]])
        M_dense = np.array([[4.0, 2.0], [2.0, 3.0]])
        c = np.array([20.0])
        s0 = np.array([12.0])
        margin = 8.0

        expected_htMinvh = 3.0 / 8.0
        expected_denom = math.sqrt(expected_htMinvh)
        expected_radius = margin / expected_denom

        res = compute_ac_metric_radius(
            h_vectors=h,
            s_limit_mva=c,
            s0_mva=s0,
            M=M_dense,
            balance=False,  # 1 bus, balance would zero-out each block
        )
        row = res["line_0"]
        assert float(row["metric_denom"]) == pytest.approx(expected_denom, abs=1e-12)
        assert float(row["radius_ac_metric"]) == pytest.approx(
            expected_radius, abs=1e-12
        )


class TestACMetricRadiusIdentityMatchesL2:
    """When M = I, metric radius must equal the plain L2 radius (balance=False)."""

    def test_identity_gives_l2_norm(self) -> None:
        n_bus = 3
        d = 2 * n_bus
        rng = np.random.RandomState(42)
        n_lines = 5
        h = rng.randn(n_lines, d)
        c = np.full(n_lines, 100.0)
        s0 = rng.uniform(50.0, 90.0, size=n_lines)

        M_identity = np.eye(d)

        res = compute_ac_metric_radius(
            h_vectors=h,
            s_limit_mva=c,
            s0_mva=s0,
            M=M_identity,
            balance=False,
        )

        for i in range(n_lines):
            row = res[f"line_{i}"]
            expected_denom = float(np.linalg.norm(h[i, :]))
            expected_radius = (c[i] - s0[i]) / expected_denom

            assert float(row["metric_denom"]) == pytest.approx(
                expected_denom, abs=1e-10
            )
            assert float(row["radius_ac_metric"]) == pytest.approx(
                expected_radius, abs=1e-10
            )

    def test_identity_diagonal_gives_l2_norm(self) -> None:
        """Diagonal M = ones(d) should also match L2 norm."""
        n_bus = 3
        d = 2 * n_bus
        rng = np.random.RandomState(99)
        n_lines = 4
        h = rng.randn(n_lines, d)
        c = np.full(n_lines, 80.0)
        s0 = rng.uniform(30.0, 70.0, size=n_lines)

        M_ones = np.ones(d)

        res = compute_ac_metric_radius(
            h_vectors=h,
            s_limit_mva=c,
            s0_mva=s0,
            M=M_ones,
            balance=False,
        )

        for i in range(n_lines):
            row = res[f"line_{i}"]
            expected_denom = float(np.linalg.norm(h[i, :]))
            assert float(row["metric_denom"]) == pytest.approx(
                expected_denom, abs=1e-10
            )


class TestACMetricRadiusValidation:
    """Input validation edge cases."""

    def test_rejects_non_spd_dense(self) -> None:
        h = np.array([[1.0, 0.0]])
        M_bad = np.array([[1.0, 0.0], [0.0, -1.0]])
        c = np.array([10.0])
        s0 = np.array([5.0])

        with pytest.raises(ValueError, match="positive definite"):
            compute_ac_metric_radius(
                h_vectors=h,
                s_limit_mva=c,
                s0_mva=s0,
                M=M_bad,
                balance=False,
            )

    def test_rejects_non_positive_diagonal(self) -> None:
        h = np.array([[1.0, 0.0]])
        M_bad = np.array([1.0, -1.0])
        c = np.array([10.0])
        s0 = np.array([5.0])

        with pytest.raises(ValueError, match="strictly positive"):
            compute_ac_metric_radius(
                h_vectors=h,
                s_limit_mva=c,
                s0_mva=s0,
                M=M_bad,
                balance=False,
            )

    def test_rejects_shape_mismatch(self) -> None:
        h = np.array([[1.0, 2.0, 3.0, 4.0]])  # d=4
        M_wrong = np.array([1.0, 2.0, 3.0])  # d=3, mismatch
        c = np.array([10.0])
        s0 = np.array([5.0])

        with pytest.raises(ValueError):
            compute_ac_metric_radius(
                h_vectors=h,
                s_limit_mva=c,
                s0_mva=s0,
                M=M_wrong,
                balance=False,
            )

    def test_zero_denom_gives_inf(self) -> None:
        """When h is zero, denom is zero and radius should be +inf (margin > 0)."""
        h = np.array([[0.0, 0.0, 0.0, 0.0]])
        M = np.ones(4)
        c = np.array([100.0])
        s0 = np.array([50.0])

        res = compute_ac_metric_radius(
            h_vectors=h,
            s_limit_mva=c,
            s0_mva=s0,
            M=M,
            balance=False,
        )
        assert res["line_0"]["radius_ac_metric"] == float("inf")


class TestACMetricRadiusBalance:
    """Verify balance projection affects the result correctly."""

    def test_balanced_projects_h(self) -> None:
        """
        With unbalanced h (sum(hP) != 0), balance=True should subtract the mean
        from each block, changing the denominator.

        h = [3, 1, 2, 0]  → n_bus=2
        hP = [3, 1], mean=2 → projected hP = [1, -1]
        hQ = [2, 0], mean=1 → projected hQ = [1, -1]

        M = I (diagonal ones)
        Unbalanced denom = sqrt(9+1+4+0) = sqrt(14)
        Balanced denom   = sqrt(1+1+1+1) = 2
        """
        h = np.array([[3.0, 1.0, 2.0, 0.0]])
        M = np.ones(4)
        c = np.array([50.0])
        s0 = np.array([40.0])

        res_unbal = compute_ac_metric_radius(
            h_vectors=h,
            s_limit_mva=c,
            s0_mva=s0,
            M=M,
            balance=False,
        )
        res_bal = compute_ac_metric_radius(
            h_vectors=h,
            s_limit_mva=c,
            s0_mva=s0,
            M=M,
            balance=True,
        )

        assert float(res_unbal["line_0"]["metric_denom"]) == pytest.approx(
            math.sqrt(14.0), abs=1e-12
        )
        assert float(res_bal["line_0"]["metric_denom"]) == pytest.approx(2.0, abs=1e-12)
