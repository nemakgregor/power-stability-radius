from __future__ import annotations

"""
stability_radius.verification

This subpackage contains *offline verification* utilities used by the unified CLI:
- Monte Carlo verification of the L2 certificate
- Multi-case Markdown report generation

Important packaging note
------------------------
The repository uses a `src/` layout. Previously, verification code lived under the
repository root (`verification/`), which is NOT importable when running:

  python src/power_stability_radius.py ...

because Python puts `src/` (not repo root) on sys.path. This caused:

  ModuleNotFoundError: No module named 'verification'

The implementation is now part of the installed `stability_radius` package to keep
imports stable.
"""

from .types import (  # noqa: F401
    BasePointCheck,
    OverallCheck,
    ProbabilisticCheck,
    RadiusCheck,
    SoundnessCheck,
    VerificationInputs,
    VerificationResult,
    overall_from_components,
)
