from __future__ import annotations

"""
Base-point computation API.

This package provides *single-source-of-truth* builders for base regimes used by:
- compute (certificate/radius generation)
- monte-carlo/report (verification)

Design principles
-----------------
- Deterministic ordering (sorted bus_ids / line_ids).
- No hidden defaults: all important regime choices are explicit in config/CLI.
- Reuse: compute and verification must share the same base-point logic.

Public dataclasses
------------------
- BasePointDC
- BasePointAC
"""

from .ac import solve_ac_pf_base_point
from .dc import (
    build_dc_base_point_case,
    build_dc_base_point_dc_opf,
    build_dc_base_point_from_acpf,
)
from .types import BasePointAC, BasePointDC

__all__ = [
    "BasePointDC",
    "BasePointAC",
    "build_dc_base_point_case",
    "build_dc_base_point_dc_opf",
    "build_dc_base_point_from_acpf",
    "solve_ac_pf_base_point",
]
