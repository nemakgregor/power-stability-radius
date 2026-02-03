from __future__ import annotations

from stability_radius.verification.types import (
    BASE_INFEASIBLE,
    BASE_OK,
    RADIUS_OK,
    RADIUS_ZERO_BINDING,
    SOUND_FAIL,
    SOUND_PASS,
    SOUND_SKIPPED_TRIVIAL_RADIUS,
    BasePointCheck,
    RadiusCheck,
    SoundnessCheck,
)
from stability_radius.verification.verify_certificate import (
    SOUNDNESS_TRIVIAL_TRUE,
    SOUNDNESS_UNSOUND,
    SOUNDNESS_UNKNOWN,
    USEFULNESS_NA,
    USEFULNESS_NONZERO_RADIUS,
    USEFULNESS_ZERO_RADIUS,
    interpret_certificate_components,
)


def test_interpretation_trivial_radius_is_trivial_true_and_zero_radius() -> None:
    base = BasePointCheck(status=BASE_OK, violated_lines=0, max_violation_mw=0.0)
    radius = RadiusCheck(
        status=RADIUS_ZERO_BINDING,
        r_star=0.0,
        argmin_line_pos=0,
        argmin_line_idx=10,
        min_margin_mw=0.0,
        argmin_margin_mw=0.0,
        argmin_norm_g=1.0,
    )
    sound = SoundnessCheck(
        status=SOUND_SKIPPED_TRIVIAL_RADIUS,
        n_ball_samples=100,
        violation_samples=0,
        max_violation_mw=float("-inf"),
        max_violation_line_idx=-1,
        tol_mw=1e-6,
    )

    interp = interpret_certificate_components(base=base, radius=radius, soundness=sound)
    assert interp.soundness == SOUNDNESS_TRIVIAL_TRUE
    assert interp.usefulness == USEFULNESS_ZERO_RADIUS


def test_interpretation_unsound_is_unsound() -> None:
    base = BasePointCheck(status=BASE_OK, violated_lines=0, max_violation_mw=0.0)
    radius = RadiusCheck(
        status=RADIUS_OK,
        r_star=1.0,
        argmin_line_pos=0,
        argmin_line_idx=10,
        min_margin_mw=1.0,
        argmin_margin_mw=1.0,
        argmin_norm_g=1.0,
    )
    sound = SoundnessCheck(
        status=SOUND_FAIL,
        n_ball_samples=100,
        violation_samples=1,
        max_violation_mw=0.1,
        max_violation_line_idx=10,
        tol_mw=1e-6,
    )

    interp = interpret_certificate_components(base=base, radius=radius, soundness=sound)
    assert interp.soundness == SOUNDNESS_UNSOUND
    assert interp.usefulness == USEFULNESS_NONZERO_RADIUS


def test_interpretation_base_infeasible_is_unknown_and_na() -> None:
    base = BasePointCheck(
        status=BASE_INFEASIBLE, violated_lines=1, max_violation_mw=1.0
    )
    radius = RadiusCheck(
        status=RADIUS_OK,
        r_star=1.0,
        argmin_line_pos=0,
        argmin_line_idx=10,
        min_margin_mw=1.0,
        argmin_margin_mw=1.0,
        argmin_norm_g=1.0,
    )
    sound = SoundnessCheck(
        status=SOUND_PASS,
        n_ball_samples=100,
        violation_samples=0,
        max_violation_mw=0.0,
        max_violation_line_idx=-1,
        tol_mw=1e-6,
    )

    interp = interpret_certificate_components(base=base, radius=radius, soundness=sound)
    assert interp.soundness == SOUNDNESS_UNKNOWN
    assert interp.usefulness == USEFULNESS_NA
