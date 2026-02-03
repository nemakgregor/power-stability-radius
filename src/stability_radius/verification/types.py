from __future__ import annotations

"""
Typed verification results and status semantics.

Goals
-----
- No "status string concatenation" like: config_mismatch_generated_n1_match_undefined
- Component-level statuses + explicit reasons list
- Deterministic, JSON-friendly structures (dataclasses)

Overall status semantics
------------------------
- FAIL only if:
    * soundness == SOUND_FAIL, or
    * radius == RADIUS_INVALID
- WARN for:
    * base infeasible / soundness skipped / trivial radius / degenerate probability mass, etc.
- OK only if:
    BASE_OK and RADIUS_OK and SOUND_PASS
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

# ---------- Base point statuses ----------
BASE_OK = "BASE_OK"
BASE_INFEASIBLE = "BASE_INFEASIBLE"
BASE_UNKNOWN = "BASE_UNKNOWN"

# ---------- Radius computation statuses ----------
RADIUS_OK = "RADIUS_OK"
RADIUS_ZERO_BINDING = "RADIUS_ZERO_BINDING"
RADIUS_ZERO_BAD_LIMITS = "RADIUS_ZERO_BAD_LIMITS"
RADIUS_INVALID = "RADIUS_INVALID"
RADIUS_UNKNOWN = "RADIUS_UNKNOWN"

# ---------- Soundness statuses ----------
SOUND_PASS = "SOUND_PASS"
SOUND_FAIL = "SOUND_FAIL"
SOUND_SKIPPED_BASE_INFEASIBLE = "SOUND_SKIPPED_BASE_INFEASIBLE"
SOUND_SKIPPED_TRIVIAL_RADIUS = "SOUND_SKIPPED_TRIVIAL_RADIUS"
SOUND_SKIPPED_INVALID_RADIUS = "SOUND_SKIPPED_INVALID_RADIUS"
SOUND_SKIPPED_NO_SAMPLES = "SOUND_SKIPPED_NO_SAMPLES"

# ---------- Probabilistic statuses (soft, not pass/fail) ----------
PROB_OK = "PROB_OK"
PROB_DEGENERATE_DIMENSION = "PROB_DEGENERATE_DIMENSION"
PROB_MC_UNSTABLE = "PROB_MC_UNSTABLE"
PROB_UNKNOWN = "PROB_UNKNOWN"

# ---------- Overall statuses ----------
OVERALL_OK = "OK"
OVERALL_WARN = "WARN"
OVERALL_FAIL = "FAIL"


@dataclass(frozen=True)
class VerificationInputs:
    """Inputs and metadata that define a verification run."""

    case_id: str
    results_path: str
    input_case_path: str

    slack_bus: int
    n_bus: int
    n_line: int
    dim_balance: int

    n_samples: int
    seed: int
    chunk_size: int

    sigma_mw: float


@dataclass(frozen=True)
class BasePointCheck:
    """Feasibility of the base point w.r.t. the stored limits."""

    status: str
    violated_lines: int
    max_violation_mw: float


@dataclass(frozen=True)
class RadiusCheck:
    """Quality and summary of the L2 radius certificate."""

    status: str
    r_star: float
    argmin_line_pos: int
    argmin_line_idx: int

    min_margin_mw: float
    argmin_margin_mw: float
    argmin_norm_g: float


@dataclass(frozen=True)
class SoundnessCheck:
    """Soundness check inside the certified ball (hard correctness signal)."""

    status: str
    n_ball_samples: int
    violation_samples: int
    max_violation_mw: float
    max_violation_line_idx: int
    tol_mw: float


@dataclass(frozen=True)
class ProbabilisticCheck:
    """Probabilistic metrics under the chosen Δp distribution (soft indicators)."""

    status: str

    # Gaussian safety probability
    p_safe_gaussian_percent: float
    p_safe_gaussian_ci95_low_percent: float
    p_safe_gaussian_ci95_high_percent: float

    # Ball mass under the same Gaussian
    p_ball_analytic_percent: float
    p_ball_mc_percent: float
    p_ball_mc_ci95_low_percent: float
    p_ball_mc_ci95_high_percent: float

    # Conditional tightness (should be ~100% if certificate is correct and checks match definitions)
    eta_safe_given_in_ball_percent: float
    eta_ci95_low_percent: float
    eta_ci95_high_percent: float

    # Dimensionless normalized radius
    rho: float


@dataclass(frozen=True)
class OverallCheck:
    """Aggregated status with explicit reasons list (no concatenation)."""

    status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VerificationResult:
    """
    Full per-case verification result.

    JSON serialization
    ------------------
    Use `.to_dict()` which is stable and JSON-friendly.
    """

    schema_version: int
    inputs: VerificationInputs
    base_point: BasePointCheck
    radius: RadiusCheck
    soundness: SoundnessCheck
    probabilistic: ProbabilisticCheck
    comparisons: dict[str, Any] = field(default_factory=dict)
    overall: OverallCheck = field(
        default_factory=lambda: OverallCheck(status=OVERALL_WARN)
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses to a JSON-friendly dict."""
        return asdict(self)


def overall_from_components(
    *,
    base_status: str,
    radius_status: str,
    soundness_status: str,
    probabilistic_status: str,
    extra_reasons: Sequence[str] = (),
) -> OverallCheck:
    """
    Compute OVERALL status from component statuses (explicit contract).

    Contract
    --------
    - FAIL only if SOUND_FAIL or RADIUS_INVALID.
    - OK only if BASE_OK and RADIUS_OK and SOUND_PASS.
    - Otherwise WARN.
    """
    reasons: list[str] = []

    if radius_status == RADIUS_INVALID:
        reasons.append("radius_invalid")
    if soundness_status == SOUND_FAIL:
        reasons.append("soundness_fail")

    if base_status != BASE_OK:
        reasons.append(f"base={base_status}")
    if radius_status != RADIUS_OK:
        reasons.append(f"radius={radius_status}")
    if soundness_status != SOUND_PASS:
        reasons.append(f"soundness={soundness_status}")
    if probabilistic_status != PROB_OK:
        reasons.append(f"prob={probabilistic_status}")

    for r in extra_reasons:
        s = str(r).strip()
        if s:
            reasons.append(s)

    if radius_status == RADIUS_INVALID or soundness_status == SOUND_FAIL:
        return OverallCheck(status=OVERALL_FAIL, reasons=tuple(reasons))

    if (
        base_status == BASE_OK
        and radius_status == RADIUS_OK
        and soundness_status == SOUND_PASS
    ):
        # Probabilistic status may still be WARN-like, but by contract OK requires PROB_OK as well.
        if probabilistic_status == PROB_OK and not extra_reasons:
            return OverallCheck(status=OVERALL_OK, reasons=tuple())
        return OverallCheck(status=OVERALL_WARN, reasons=tuple(reasons))

    return OverallCheck(status=OVERALL_WARN, reasons=tuple(reasons))
