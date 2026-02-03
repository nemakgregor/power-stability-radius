from __future__ import annotations

"""
High-level verification status semantics (report-friendly).

Why this module exists
----------------------
The project already has detailed component statuses in `stability_radius.verification.types`
(BasePointCheck / RadiusCheck / SoundnessCheck / ProbabilisticCheck).

For reporting and paper-style narratives, it is helpful to provide a *single* summary label
that is:
- scientifically honest (trivial radius is not a failure),
- stable across refactors,
- easy to scan in tables.

These summary statuses intentionally collapse detail and should NOT be used for
fine-grained debugging (use VerificationResult.* fields for that).
"""

import logging

from .types import (
    BASE_INFEASIBLE,
    BASE_OK,
    RADIUS_OK,
    RADIUS_ZERO_BINDING,
    SOUND_FAIL,
    SOUND_PASS,
    VerificationResult,
)

logger = logging.getLogger(__name__)

OK = "OK"
TRIVIAL_RADIUS = "TRIVIAL_RADIUS"
BASE_INFEASIBLE_SUMMARY = "BASE_INFEASIBLE"
CERT_UNSOUND = "CERT_UNSOUND"
MC_INCONCLUSIVE = "MC_INCONCLUSIVE"
NOT_COMPUTED = "NOT_COMPUTED"


def summarize_status(vr: VerificationResult | None) -> str:
    """
    Map a detailed VerificationResult into a single summary status.

    Mapping contract
    ----------------
    - OK:
        base feasible, r*>0, and soundness check passes.
    - TRIVIAL_RADIUS:
        base feasible, but r*=0 due to a binding constraint (certificate is vacuous).
    - BASE_INFEASIBLE:
        base violates stored limits (hard failure of the input/base point).
    - CERT_UNSOUND:
        a counterexample was found inside the certified ball.
    - MC_INCONCLUSIVE:
        anything else (skipped checks, invalid radii, missing artifacts, etc).
    """
    if vr is None:
        return NOT_COMPUTED

    if str(vr.base_point.status) == BASE_INFEASIBLE:
        return BASE_INFEASIBLE_SUMMARY

    if str(vr.soundness.status) == SOUND_FAIL:
        return CERT_UNSOUND

    if (
        str(vr.base_point.status) == BASE_OK
        and str(vr.radius.status) == RADIUS_ZERO_BINDING
        and float(vr.radius.r_star) == 0.0
    ):
        return TRIVIAL_RADIUS

    if (
        str(vr.base_point.status) == BASE_OK
        and str(vr.radius.status) == RADIUS_OK
        and str(vr.soundness.status) == SOUND_PASS
    ):
        return OK

    return MC_INCONCLUSIVE
