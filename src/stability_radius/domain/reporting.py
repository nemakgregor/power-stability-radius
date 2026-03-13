"""Domain types for report generation and verification workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportCaseSpec:
    """Immutable specification of one report case."""

    case_id: str
    input_case_path: Path
    results_path: Path
    known_critical_pairs: tuple[tuple[int, int], ...] = ()
