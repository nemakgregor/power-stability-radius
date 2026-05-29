"""Application helpers for preparing report-generation inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from stability_radius.domain import ReportCaseSpec


def _is_seq_not_str(value: Any) -> bool:
    """Internal helper for module-local processing."""
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def parse_report_cases_from_cfg(
    *,
    cfg_loaded: Any,
    results_dir_abs: Path,
    base_dir: Path,
    select_value: Callable[[Any, str, Any], Any],
) -> list[ReportCaseSpec]:
    """Resolve `report.cases` from config into explicit typed case specs."""
    raw = select_value(cfg_loaded, "report.cases", None)
    if raw is None or not _is_seq_not_str(raw) or len(raw) == 0:
        raise ValueError(
            "Missing required config key `report.cases` (must be a non-empty list)."
        )

    out: list[ReportCaseSpec] = []
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"report.cases[{i}] must be a mapping/dict, got {type(item)}"
            )

        case_id = str(item.get("id", "")).strip()
        input_path = str(item.get("input", "")).strip()
        results_name = str(item.get("results", "")).strip()
        if not case_id or not input_path or not results_name:
            raise ValueError(f"report.cases[{i}] must have id/input/results.")

        rp = Path(results_name).expanduser()
        rp_abs = rp if rp.is_absolute() else (results_dir_abs / rp).resolve()

        ip = Path(input_path).expanduser()
        ip_abs = ip if ip.is_absolute() else (base_dir / ip).resolve()

        known = item.get("known_critical_pairs", None)
        known_pairs: list[tuple[int, int]] = []
        if known is not None:
            if not _is_seq_not_str(known):
                raise ValueError(
                    f"report.cases[{i}].known_critical_pairs must be a list."
                )
            for j, pair in enumerate(known):
                if not _is_seq_not_str(pair) or len(pair) != 2:
                    raise ValueError(
                        f"report.cases[{i}].known_critical_pairs[{j}] must be a 2-element pair."
                    )
                known_pairs.append((int(pair[0]), int(pair[1])))

        out.append(
            ReportCaseSpec(
                case_id=case_id,
                input_case_path=ip_abs,
                results_path=rp_abs,
                known_critical_pairs=tuple(known_pairs),
            )
        )

    return out
