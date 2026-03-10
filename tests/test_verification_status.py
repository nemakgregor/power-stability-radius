"""Tests for stability_radius.verification.status — report-friendly summary labels."""

from __future__ import annotations

from stability_radius.verification.status import (
    BASE_INFEASIBLE_SUMMARY,
    CERT_UNSOUND,
    MC_INCONCLUSIVE,
    NOT_COMPUTED,
    OK,
    TRIVIAL_RADIUS,
    summarize_status,
)
from stability_radius.verification.types import (
    BASE_INFEASIBLE,
    BASE_OK,
    BASE_UNKNOWN,
    PROB_OK,
    PROB_UNKNOWN,
    RADIUS_INVALID,
    RADIUS_OK,
    RADIUS_UNKNOWN,
    RADIUS_ZERO_BINDING,
    SOUND_FAIL,
    SOUND_PASS,
    SOUND_SKIPPED_TRIVIAL_RADIUS,
    BasePointCheck,
    OverallCheck,
    ProbabilisticCheck,
    RadiusCheck,
    SoundnessCheck,
    VerificationInputs,
    VerificationResult,
)


def _make_vr(
    *,
    base_status: str = BASE_OK,
    radius_status: str = RADIUS_OK,
    r_star: float = 1.0,
    soundness_status: str = SOUND_PASS,
) -> VerificationResult:
    """Build a minimal VerificationResult for testing."""
    return VerificationResult(
        schema_version=2,
        inputs=VerificationInputs(
            case_id="test",
            results_path="test.json",
            input_case_path="test.m",
            slack_bus=0,
            n_bus=3,
            n_line=3,
            dim_balance=2,
            n_samples=100,
            seed=42,
            chunk_size=64,
            sigma_mw=1.0,
        ),
        base_point=BasePointCheck(
            status=base_status, violated_lines=0, max_violation_mw=0.0
        ),
        radius=RadiusCheck(
            status=radius_status,
            r_star=r_star,
            argmin_line_pos=0,
            argmin_line_idx=0,
            min_margin_mw=1.0,
            argmin_margin_mw=1.0,
            argmin_norm_g=1.0,
        ),
        soundness=SoundnessCheck(
            status=soundness_status,
            n_ball_samples=100,
            violation_samples=0,
            max_violation_mw=0.0,
            max_violation_line_idx=-1,
            tol_mw=1e-6,
        ),
        probabilistic=ProbabilisticCheck(
            status=PROB_OK,
            p_safe_gaussian_percent=99.0,
            p_safe_gaussian_ci95_low_percent=98.0,
            p_safe_gaussian_ci95_high_percent=100.0,
            p_ball_analytic_percent=50.0,
            p_ball_mc_percent=50.0,
            p_ball_mc_ci95_low_percent=45.0,
            p_ball_mc_ci95_high_percent=55.0,
            eta_safe_given_in_ball_percent=100.0,
            eta_ci95_low_percent=100.0,
            eta_ci95_high_percent=100.0,
            rho=1.0,
        ),
        overall=OverallCheck(status="OK"),
    )


class TestSummarizeStatus:
    """Contract: each branch maps to exactly one summary label."""

    def test_none_returns_not_computed(self):
        assert summarize_status(None) == NOT_COMPUTED

    def test_ok_when_base_ok_radius_ok_sound_pass(self):
        vr = _make_vr()
        assert summarize_status(vr) == OK

    def test_base_infeasible(self):
        vr = _make_vr(base_status=BASE_INFEASIBLE)
        assert summarize_status(vr) == BASE_INFEASIBLE_SUMMARY

    def test_cert_unsound(self):
        vr = _make_vr(soundness_status=SOUND_FAIL)
        assert summarize_status(vr) == CERT_UNSOUND

    def test_trivial_radius_when_binding(self):
        vr = _make_vr(radius_status=RADIUS_ZERO_BINDING, r_star=0.0)
        assert summarize_status(vr) == TRIVIAL_RADIUS

    def test_mc_inconclusive_for_unknown_base(self):
        vr = _make_vr(base_status=BASE_UNKNOWN)
        assert summarize_status(vr) == MC_INCONCLUSIVE

    def test_mc_inconclusive_for_unknown_radius(self):
        vr = _make_vr(radius_status=RADIUS_UNKNOWN)
        assert summarize_status(vr) == MC_INCONCLUSIVE

    def test_mc_inconclusive_for_skipped_soundness(self):
        vr = _make_vr(soundness_status=SOUND_SKIPPED_TRIVIAL_RADIUS)
        assert summarize_status(vr) == MC_INCONCLUSIVE

    def test_base_infeasible_takes_precedence_over_sound_fail(self):
        """BASE_INFEASIBLE is checked first in the branch order."""
        vr = _make_vr(base_status=BASE_INFEASIBLE, soundness_status=SOUND_FAIL)
        assert summarize_status(vr) == BASE_INFEASIBLE_SUMMARY
