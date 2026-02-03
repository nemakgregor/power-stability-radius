from __future__ import annotations

"""
Certificate interpretation helpers (soundness vs usefulness).

Why this module exists
----------------------
The project already computes:
- a *soundness* check (hard correctness) via MC in the certified L2 ball
- probabilistic safety metrics under a chosen disturbance distribution

However, a common edge case is r* = 0:
- the certificate is **trivially true** (there are no non-zero perturbations in a 0-radius ball)
- but it is **non-informative** / useless in practice

To avoid conflating these two notions, this module provides an explicit, typed
interpretation layer that can be used by:
- reports (human-facing)
- logs / CI gating (machine-facing)
- tests (TDD-style semantics validation)

This interpretation is intentionally conservative:
- it does NOT introduce any heuristics (e.g., "small rho => useless")
- it only classifies the logically crisp r*=0 case as "zero_radius"
"""

import logging
import math
from dataclasses import dataclass

from .types import (
    BASE_OK,
    BasePointCheck,
    RadiusCheck,
    SoundnessCheck,
    VerificationResult,
    SOUND_FAIL,
    SOUND_PASS,
)

logger = logging.getLogger(__name__)

# Public, stable labels (strings are JSON-friendly and easy to display).
SOUNDNESS_SOUND = "sound"
SOUNDNESS_UNSOUND = "unsound"
SOUNDNESS_TRIVIAL_TRUE = "trivial_true"
SOUNDNESS_UNKNOWN = "unknown"

USEFULNESS_NONZERO_RADIUS = "nonzero_radius"
USEFULNESS_ZERO_RADIUS = "zero_radius"
USEFULNESS_NA = "n/a"


@dataclass(frozen=True)
class CertificateInterpretation:
    """
    High-level certificate interpretation.

    Fields
    ------
    soundness:
        One of: sound | unsound | trivial_true | unknown
    usefulness:
        One of: nonzero_radius | zero_radius | n/a
    notes:
        Deterministic extra tags (diagnostics) to help debugging/reporting.
        Must not be used as a machine-readable "status bag".
    """

    soundness: str
    usefulness: str
    notes: tuple[str, ...] = ()


def interpret_certificate_components(
    *,
    base: BasePointCheck,
    radius: RadiusCheck,
    soundness: SoundnessCheck,
) -> CertificateInterpretation:
    """
    Interpret certificate semantics from component checks.

    Design contract
    ---------------
    - If base point is infeasible/unknown -> soundness=unknown, usefulness=n/a
    - If r* is not finite or r* < 0 -> soundness=unknown, usefulness=n/a
    - If r* == 0 and base is OK -> soundness=trivial_true, usefulness=zero_radius
    - If r* > 0:
        * SOUND_FAIL -> unsound
        * SOUND_PASS -> sound
        * otherwise -> unknown (e.g., skipped due to no samples)
      usefulness=nonzero_radius (no further heuristics)

    Notes
    -----
    This function deliberately does not attempt to infer "usefulness" from probability
    mass (rho / p_ball), because that would require application-specific thresholds.
    """
    notes: list[str] = []

    base_status = str(base.status)
    if base_status != BASE_OK:
        notes.append(f"base={base_status}")
        return CertificateInterpretation(
            soundness=SOUNDNESS_UNKNOWN,
            usefulness=USEFULNESS_NA,
            notes=tuple(notes),
        )

    r_star = float(radius.r_star)
    if not math.isfinite(r_star):
        notes.append("r_star_not_finite")
        return CertificateInterpretation(
            soundness=SOUNDNESS_UNKNOWN,
            usefulness=USEFULNESS_NA,
            notes=tuple(notes),
        )
    if r_star < 0.0:
        notes.append("r_star_negative")
        return CertificateInterpretation(
            soundness=SOUNDNESS_UNKNOWN,
            usefulness=USEFULNESS_NA,
            notes=tuple(notes),
        )

    if r_star == 0.0:
        # Logically: sound within the 0-radius ball, but useless.
        return CertificateInterpretation(
            soundness=SOUNDNESS_TRIVIAL_TRUE,
            usefulness=USEFULNESS_ZERO_RADIUS,
            notes=tuple(notes),
        )

    # r_star > 0
    sound_status = str(soundness.status)
    if sound_status == SOUND_FAIL:
        return CertificateInterpretation(
            soundness=SOUNDNESS_UNSOUND,
            usefulness=USEFULNESS_NONZERO_RADIUS,
            notes=tuple(notes),
        )
    if sound_status == SOUND_PASS:
        return CertificateInterpretation(
            soundness=SOUNDNESS_SOUND,
            usefulness=USEFULNESS_NONZERO_RADIUS,
            notes=tuple(notes),
        )

    notes.append(f"soundness={sound_status}")
    return CertificateInterpretation(
        soundness=SOUNDNESS_UNKNOWN,
        usefulness=USEFULNESS_NONZERO_RADIUS,
        notes=tuple(notes),
    )


def interpret_certificate(vr: VerificationResult | None) -> CertificateInterpretation:
    """
    Convenience wrapper: interpret semantics from a full VerificationResult.

    Parameters
    ----------
    vr:
        VerificationResult or None.

    Returns
    -------
    CertificateInterpretation
        Soundness/usefulness labels.
    """
    if vr is None:
        return CertificateInterpretation(
            soundness=SOUNDNESS_UNKNOWN,
            usefulness=USEFULNESS_NA,
            notes=("not_computed",),
        )

    return interpret_certificate_components(
        base=vr.base_point, radius=vr.radius, soundness=vr.soundness
    )
