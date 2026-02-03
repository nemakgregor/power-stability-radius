from __future__ import annotations

from stability_radius.verification.generate_report import _case_card_md
from stability_radius.verification.types import (
    BASE_OK,
    OVERALL_FAIL,
    OVERALL_OK,
    PROB_OK,
    RADIUS_OK,
    SOUND_FAIL,
    SOUND_PASS,
    BasePointCheck,
    OverallCheck,
    ProbabilisticCheck,
    RadiusCheck,
    SoundnessCheck,
    VerificationInputs,
    VerificationResult,
    overall_from_components,
)


def _make_minimal_vr(
    *, soundness_status: str, radius_status: str
) -> VerificationResult:
    inputs = VerificationInputs(
        case_id="caseX",
        results_path="/abs/results.json",
        input_case_path="/abs/case.m",
        slack_bus=0,
        n_bus=3,
        n_line=2,
        dim_balance=2,
        n_samples=100,
        seed=0,
        chunk_size=10,
        sigma_mw=1.0,
    )

    base = BasePointCheck(status=BASE_OK, violated_lines=0, max_violation_mw=0.0)

    radius = RadiusCheck(
        status=radius_status,
        r_star=1.0,
        argmin_line_pos=0,
        argmin_line_idx=0,
        min_margin_mw=1.0,
        argmin_margin_mw=1.0,
        argmin_norm_g=1.0,
    )

    sound = SoundnessCheck(
        status=soundness_status,
        n_ball_samples=10,
        violation_samples=0 if soundness_status == SOUND_PASS else 1,
        max_violation_mw=0.0,
        max_violation_line_idx=-1,
        tol_mw=1e-6,
    )

    prob = ProbabilisticCheck(
        status=PROB_OK,
        p_safe_gaussian_percent=100.0,
        p_safe_gaussian_ci95_low_percent=99.0,
        p_safe_gaussian_ci95_high_percent=100.0,
        p_ball_analytic_percent=0.0,
        p_ball_mc_percent=0.0,
        p_ball_mc_ci95_low_percent=0.0,
        p_ball_mc_ci95_high_percent=0.0,
        eta_safe_given_in_ball_percent=100.0,
        eta_ci95_low_percent=99.0,
        eta_ci95_high_percent=100.0,
        rho=0.1,
    )

    overall = overall_from_components(
        base_status=base.status,
        radius_status=radius.status,
        soundness_status=sound.status,
        probabilistic_status=prob.status,
    )

    return VerificationResult(
        schema_version=1,
        inputs=inputs,
        base_point=base,
        radius=radius,
        soundness=sound,
        probabilistic=prob,
        comparisons={},
        overall=overall,
    )


def test_overall_fail_only_on_sound_fail_or_radius_invalid() -> None:
    vr_ok = _make_minimal_vr(soundness_status=SOUND_PASS, radius_status=RADIUS_OK)
    assert vr_ok.overall.status == OVERALL_OK

    vr_sound_fail = _make_minimal_vr(
        soundness_status=SOUND_FAIL, radius_status=RADIUS_OK
    )
    assert vr_sound_fail.overall.status == OVERALL_FAIL

    # Radius invalid => FAIL (even if soundness says PASS)
    vr_radius_invalid = _make_minimal_vr(
        soundness_status=SOUND_PASS, radius_status="RADIUS_INVALID"
    )
    assert vr_radius_invalid.overall.status == OVERALL_FAIL


def test_case_card_contains_required_sections() -> None:
    vr_ok = _make_minimal_vr(soundness_status=SOUND_PASS, radius_status=RADIUS_OK)

    md = _case_card_md(
        case="caseX",
        results_status="ok",
        vr=vr_ok,
        comparisons={},
        time_sec=1.0,
    )

    for section in (
        "### Inputs",
        "### Base point",
        "### Radius",
        "### Soundness",
        "### Probabilistic",
    ):
        assert section in md
