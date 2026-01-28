from __future__ import annotations

"""
Generate verification/report.md from stored per-case results under verification/results/.

Workflow (deterministic policy)
-------------------------------
- Base point is solved ONLY via OPF: PyPSA DC OPF + HiGHS
- Radii are computed around the OPF base point
- Monte Carlo coverage is evaluated for each case
- An aggregated Markdown report is generated

Input data policy
-----------------
- If an expected input `.m` case file is missing and its filename matches a supported
  dataset (MATPOWER case<N>.m/ieee<N>.m or PGLib-OPF pglib_opf_*.m), it is downloaded
  deterministically (stable URL candidate ordering).

No PF/AC-OPF modes.
"""

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Fix src-layout imports when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stability_radius.config import DEFAULT_LOGGING, DEFAULT_MC, LoggingConfig
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.common import line_key
from stability_radius.utils import log_stage, setup_logging
from stability_radius.utils.download import ensure_case_file
from verification.monte_carlo import estimate_coverage_percent

logger = logging.getLogger("stability_radius.verification.generate_report")


@dataclass(frozen=True)
class TopMatchStats:
    """Top-10 match/recall stats with explicit counts (for audit)."""

    top_k: int
    known_k: int
    common_k: int
    match_percent_common_over_10: float
    recall_percent_common_over_known: float


@dataclass(frozen=True)
class NMinus1MatchStats:
    """N-1 criticality match stats with explicit counts (for audit)."""

    status: str
    match_percent: float
    median_radius_nminus1: float
    median_p0_mw: float
    selected_lines: int
    evaluated_lines: int
    hits: int


def _load_results(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj)}")
    return obj


def _get_meta(results: Dict[str, Any]) -> Dict[str, Any]:
    meta = results.get("__meta__")
    return meta if isinstance(meta, dict) else {}


def _get_meta_time_sec(results: Dict[str, Any]) -> float:
    try:
        return float(_get_meta(results).get("compute_time_sec", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _get_meta_input_path(results: Dict[str, Any], fallback: str) -> str:
    meta = _get_meta(results)
    p = meta.get("input_path")
    return str(p) if isinstance(p, str) and p.strip() else str(fallback)


def _get_meta_slack_bus(results: Dict[str, Any], fallback: int = 0) -> int:
    meta = _get_meta(results)
    try:
        return int(meta.get("slack_bus", fallback))
    except (TypeError, ValueError):
        return int(fallback)


def _sorted_line_indices(net) -> List[int]:
    return [int(x) for x in sorted(net.line.index)]


def _append_status(status: str, suffix: str) -> str:
    if not status or status == "ok":
        return str(suffix)
    return f"{status}_{suffix}"


def _fmt_float_or_na(x: float, *, digits: int = 6) -> str:
    if not math.isfinite(float(x)):
        return "n/a"
    return f"{float(x):.{digits}g}"


def _fmt_percent_or_na(x: float, *, decimals: int = 3) -> str:
    if not math.isfinite(float(x)):
        return "n/a"
    return f"{float(x):.{decimals}f}%"


def _fmt_ok(v: bool) -> str:
    return "PASS" if v else "FAIL"


def _validate_results_for_monte_carlo(results: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate that results.json contains fields required for Monte Carlo.
    """
    required = ("flow0_mw", "p_limit_mw_est", "radius_l2")

    meta = _get_meta(results)
    dm = str(meta.get("dispatch_mode", "")).strip().lower()
    if dm != "opf_pypsa":
        return False, f"dispatch_mode={dm!r} is not supported (expected 'opf_pypsa')"

    solver = str(meta.get("opf_solver", "")).strip().lower()
    if solver != "highs":
        return False, f"opf_solver={solver!r} is not supported (expected 'highs')"

    line_keys = [
        k for k in results.keys() if isinstance(k, str) and k.startswith("line_")
    ]
    if not line_keys:
        return False, "no line_* entries"

    missing_examples: list[str] = []
    nan_limit_examples: list[str] = []
    finite_r = 0

    for k in sorted(line_keys):
        row = results.get(k)
        if not isinstance(row, dict):
            missing_examples.append(f"{k}: not a dict")
            if len(missing_examples) >= 5:
                break
            continue

        for f in required:
            if f not in row:
                missing_examples.append(f"{k}: missing {f}")
                break

        try:
            c = float(row.get("p_limit_mw_est", float("nan")))
        except (TypeError, ValueError):
            c = float("nan")
        if math.isnan(c):
            nan_limit_examples.append(k)

        try:
            r = float(row.get("radius_l2", float("nan")))
        except (TypeError, ValueError):
            r = float("nan")
        if math.isfinite(r):
            finite_r += 1

    if missing_examples:
        return False, "missing required fields: " + "; ".join(missing_examples[:5])
    if nan_limit_examples:
        return (
            False,
            "p_limit_mw_est contains NaN (first 10): "
            + ", ".join(nan_limit_examples[:10]),
        )
    if finite_r <= 0:
        return False, "no finite radius_l2 values (cannot define MC scaling)"

    return True, "ok"


def _results_config_matches(
    *,
    results: Dict[str, Any],
    expected_dc_mode: str,
    expected_slack_bus: int,
    expected_compute_nminus1: bool,
) -> tuple[bool, str]:
    meta = _get_meta(results)

    dm = str(meta.get("dispatch_mode", "")).strip().lower()
    if dm != "opf_pypsa":
        return False, f"dispatch_mode={dm!r} != 'opf_pypsa'"

    solver = str(meta.get("opf_solver", "")).strip().lower()
    if solver != "highs":
        return False, f"opf_solver={solver!r} != 'highs'"

    dc_mode = str(meta.get("dc_mode", "")).strip().lower()
    if dc_mode != str(expected_dc_mode).strip().lower():
        return False, f"dc_mode={dc_mode!r} != {str(expected_dc_mode)!r}"

    try:
        sb = int(meta.get("slack_bus", expected_slack_bus))
    except (TypeError, ValueError):
        sb = expected_slack_bus
    if sb != int(expected_slack_bus):
        return False, f"slack_bus={sb} != {int(expected_slack_bus)}"

    n1 = bool(meta.get("nminus1_computed", False))
    if n1 != bool(expected_compute_nminus1):
        return False, f"nminus1_computed={n1} != {bool(expected_compute_nminus1)}"

    return True, "ok"


def _top_k_risky_lines(
    *,
    results: Dict[str, Any],
    net,
    k: int = 10,
    radius_field: str = "radius_l2",
) -> List[Dict[str, Any]]:
    items: List[Tuple[float, int]] = []
    for lid in _sorted_line_indices(net):
        row = results.get(line_key(lid))
        if not isinstance(row, dict):
            continue
        try:
            r = float(row.get(radius_field, float("inf")))
        except (TypeError, ValueError):
            r = float("inf")
        if math.isfinite(r):
            items.append((r, lid))

    items.sort(key=lambda t: (t[0], t[1]))
    top = items[:k]

    out: List[Dict[str, Any]] = []
    for r, lid in top:
        lrow = net.line.loc[lid]
        out.append(
            {
                "line_idx": int(lid),
                "from_bus": int(lrow["from_bus"]),
                "to_bus": int(lrow["to_bus"]),
                radius_field: float(r),
            }
        )
    return out


def _match_stats_top10(
    *,
    top10: Sequence[Dict[str, Any]],
    known_pairs: Sequence[Tuple[int, int]],
) -> TopMatchStats:
    top_k = int(len(top10))
    known = {frozenset((int(a), int(b))) for a, b in known_pairs}
    known_k = int(len(known))

    if top_k <= 0:
        return TopMatchStats(
            top_k=0,
            known_k=known_k,
            common_k=0,
            match_percent_common_over_10=float("nan"),
            recall_percent_common_over_known=float("nan"),
        )

    top_pairs = {frozenset((int(x["from_bus"]), int(x["to_bus"]))) for x in top10}
    common = int(len(known.intersection(top_pairs)))

    match_top10 = 100.0 * float(common) / 10.0
    recall = 100.0 * float(common) / float(known_k) if known_k else float("nan")

    return TopMatchStats(
        top_k=top_k,
        known_k=known_k,
        common_k=common,
        match_percent_common_over_10=float(match_top10),
        recall_percent_common_over_known=float(recall),
    )


def _nminus1_critical_match_stats(*, results: Dict[str, Any], net) -> NMinus1MatchStats:
    line_indices = _sorted_line_indices(net)
    m = len(line_indices)

    p0 = np.full(m, np.nan, dtype=float)
    r_n1 = np.full(m, np.nan, dtype=float)
    worst = np.full(m, -1, dtype=int)

    for pos, lid in enumerate(line_indices):
        row = results.get(line_key(lid))
        if not isinstance(row, dict):
            continue
        try:
            p0[pos] = float(row.get("p0_mw", float("nan")))
        except (TypeError, ValueError):
            p0[pos] = float("nan")
        try:
            r_n1[pos] = float(row.get("radius_nminus1", float("nan")))
        except (TypeError, ValueError):
            r_n1[pos] = float("nan")
        try:
            worst[pos] = int(row.get("worst_contingency", -1))
        except (TypeError, ValueError):
            worst[pos] = -1

    finite_r = r_n1[np.isfinite(r_n1)]
    if finite_r.size == 0:
        return NMinus1MatchStats(
            status="no_finite_radius_nminus1",
            match_percent=float("nan"),
            median_radius_nminus1=float("nan"),
            median_p0_mw=float("nan"),
            selected_lines=0,
            evaluated_lines=0,
            hits=0,
        )

    median_r = float(np.median(finite_r))
    selected = np.where(np.isfinite(r_n1) & (r_n1 < median_r) & (worst >= 0))[0]
    selected_count = int(selected.size)
    if selected_count == 0:
        return NMinus1MatchStats(
            status="no_selected_lines",
            match_percent=float("nan"),
            median_radius_nminus1=float(median_r),
            median_p0_mw=float("nan"),
            selected_lines=0,
            evaluated_lines=0,
            hits=0,
        )

    finite_p0 = p0[np.isfinite(p0)]
    if finite_p0.size == 0:
        return NMinus1MatchStats(
            status="no_finite_p0",
            match_percent=float("nan"),
            median_radius_nminus1=float(median_r),
            median_p0_mw=float("nan"),
            selected_lines=selected_count,
            evaluated_lines=0,
            hits=0,
        )
    median_p0 = float(np.median(finite_p0))

    hits = 0
    total = 0
    for pos in selected.tolist():
        w = int(worst[pos])
        if not (0 <= w < m):
            continue
        if not math.isfinite(float(p0[w])):
            continue
        total += 1
        if float(p0[w]) > median_p0:
            hits += 1

    if total == 0:
        return NMinus1MatchStats(
            status="no_valid_worst_contingencies",
            match_percent=float("nan"),
            median_radius_nminus1=float(median_r),
            median_p0_mw=float(median_p0),
            selected_lines=selected_count,
            evaluated_lines=0,
            hits=0,
        )

    match = 100.0 * float(hits) / float(total)
    return NMinus1MatchStats(
        status="ok",
        match_percent=float(match),
        median_radius_nminus1=float(median_r),
        median_p0_mw=float(median_p0),
        selected_lines=selected_count,
        evaluated_lines=int(total),
        hits=int(hits),
    )


def _case_section_md(
    *,
    case: str,
    status: str,
    mc: Optional[Dict[str, Any]],
    top_risky: Optional[Sequence[Dict[str, Any]]],
    top_match: Optional[TopMatchStats],
    n1_match: Optional[NMinus1MatchStats],
    time_sec: float,
    known_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> str:
    lines: List[str] = []
    lines.append(f"## {case}")
    lines.append("")
    lines.append(f"- status: **{status}**")

    if mc is None:
        lines.append("- coverage % (MC): n/a")
    else:
        cov = float(mc.get("coverage_percent", float("nan")))
        denom = int(mc.get("total_feasible_in_box", 0))
        numer = int(mc.get("feasible_in_ball", 0))
        n_samples = int(mc.get("n_samples", 0))
        mc_status = str(mc.get("status", "")) or "unknown"

        if math.isfinite(cov):
            lines.append(
                f"- coverage % (MC): **{cov:.3f}%** "
                f"(= {numer} / {denom}; feasible_in_box={denom}/{n_samples}; mc_status={mc_status})"
            )
        else:
            lines.append(
                "- coverage % (MC): n/a "
                f"(mc_status={mc_status}; feasible_in_box={denom}/{n_samples})"
            )

        if "base_point_feasible" in mc:
            base_feasible = bool(mc.get("base_point_feasible", False))
            violated = int(mc.get("base_point_violated_lines", 0))
            vmax = float(mc.get("base_point_max_violation_mw", float("nan")))
            lines.append(
                "- base feasibility (w.r.t. stored limits): "
                + (
                    "**feasible**"
                    if base_feasible
                    else f"**infeasible** (violated={violated}, max_violation={_fmt_float_or_na(vmax)} MW)"
                )
            )

    if top_match is None:
        lines.append("- top risky match %: n/a")
    else:
        match = top_match.match_percent_common_over_10
        recall = top_match.recall_percent_common_over_known
        if math.isfinite(match):
            lines.append(
                f"- top risky match % (common/10): **{_fmt_float_or_na(match, digits=6)}%** "
                f"(common={top_match.common_k}, top_k={top_match.top_k}, known_k={top_match.known_k})"
            )
        else:
            lines.append("- top risky match %: n/a")
        if math.isfinite(recall):
            lines.append(
                f"- top risky recall % (common/known): **{_fmt_float_or_na(recall, digits=6)}%** "
                f"(= {top_match.common_k} / {top_match.known_k})"
            )

    if n1_match is None:
        lines.append("- N-1 critical match %: n/a")
    else:
        if math.isfinite(float(n1_match.match_percent)):
            lines.append(
                f"- N-1 critical match %: **{n1_match.match_percent:.3f}%** "
                f"(= {n1_match.hits} / {n1_match.evaluated_lines}; "
                f"selected={n1_match.selected_lines}; status={n1_match.status})"
            )
        else:
            lines.append(
                f"- N-1 critical match %: n/a (status={n1_match.status}; selected={n1_match.selected_lines})"
            )

    lines.append(
        f"- time sec (demo): **{time_sec:.3f}**"
        if math.isfinite(time_sec)
        else "- time sec (demo): n/a"
    )
    lines.append("")

    if top_risky:
        lines.append("### Top-10 risky lines (min radius_l2)")
        lines.append("")
        lines.append("| rank | line_idx | from_bus | to_bus | radius_l2 |")
        lines.append("|---:|---:|---:|---:|---:|")
        for i, row in enumerate(top_risky, start=1):
            lines.append(
                f"| {i} | {row['line_idx']} | {row['from_bus']} | {row['to_bus']} | {row['radius_l2']:.6g} |"
            )
        lines.append("")

    if math.isfinite(time_sec):
        lines.append("### Scalability check")
        lines.append("")
        lines.append(f"- Criterion: time < 10 sec => **{_fmt_ok(time_sec < 10.0)}**")
        lines.append("")

    if "1354_pegase" in case:
        lines.append("### Literature comparison (Nguyen 2018)")
        mc_cov = (
            float("nan")
            if mc is None
            else float(mc.get("coverage_percent", float("nan")))
        )
        mc_status = "n/a" if mc is None else str(mc.get("status", ""))
        cov_s = _fmt_percent_or_na(mc_cov, decimals=3)
        if cov_s == "n/a":
            lines.append(
                "Nguyen (2018) отмечает, что полито́пные convex inner approximations покрывают существенные доли true region "
                "порядка 50–90% на 1354 buses (arXiv:1708.06845v3). "
                f"Здесь coverage: n/a (MC status: {mc_status})."
            )
            lines.append(
                "Критерий адекватности по задаче: >70%. Условие не может быть проверено (coverage = n/a)."
            )
        else:
            lines.append(
                "Nguyen (2018) отмечает, что полито́пные convex inner approximations покрывают существенные доли true region "
                "порядка 50–90% на 1354 buses (arXiv:1708.06845v3). "
                f"Здесь получено coverage ≈ {cov_s}."
            )
            lines.append(
                "Критерий адекватности по задаче: >70%. "
                + (
                    "Условие выполнено."
                    if float(mc_cov) > 70.0
                    else "Условие не выполнено."
                )
            )
        lines.append("")

    if "9241_pegase" in case:
        lines.append("### Literature comparison (Nguyen 2018, Lee 2019)")
        mc_cov = (
            float("nan")
            if mc is None
            else float(mc.get("coverage_percent", float("nan")))
        )
        mc_status = "n/a" if mc is None else str(mc.get("status", ""))
        cov_s = _fmt_percent_or_na(mc_cov, decimals=3)
        if cov_s == "n/a":
            lines.append(
                "Nguyen (2018) на large-scale тестах также демонстрирует substantial fractions для внутренних аппроксимаций. "
                "Lee (2019, IEEE TPS) указывает, что convex quadratic restriction может быть достаточно большой для практической эксплуатации. "
                f"Здесь coverage: n/a (MC status: {mc_status})."
            )
            lines.append(
                "Критерий адекватности по задаче: >70%. Условие не может быть проверено (coverage = n/a)."
            )
        else:
            lines.append(
                "Nguyen (2018) на large-scale тестах также демонстрирует substantial fractions для внутренних аппроксимаций. "
                "Lee (2019, IEEE TPS) указывает, что convex quadratic restriction может быть достаточно большой для практической эксплуатации. "
                f"Здесь получено coverage ≈ {cov_s}."
            )
            lines.append(
                "Критерий адекватности по задаче: >70%. "
                + (
                    "Условие выполнено."
                    if float(mc_cov) > 70.0
                    else "Условие не выполнено."
                )
            )
        lines.append("")

    return "\n".join(lines)


def generate_report_text(
    *,
    results_dir: Path,
    n_samples: int,
    seed: int,
    chunk_size: int,
    generate_missing_results: bool,
    demo_dc_mode: str,
    demo_slack_bus: int,
    demo_compute_nminus1: bool,
    mc_box_radius_quantile: float,
) -> str:
    """
    Generate verification report markdown as a string.
    """
    if not (0.0 <= float(mc_box_radius_quantile) <= 1.0):
        raise ValueError("mc_box_radius_quantile must be within [0, 1].")

    demo_dc_mode_eff = str(demo_dc_mode).strip().lower()
    if demo_dc_mode_eff not in ("materialize", "operator"):
        raise ValueError("demo_dc_mode must be materialize|operator")

    cases = [
        ("case30", "data/input/pglib_opf_case30_ieee.m", results_dir / "case30.json"),
        (
            "case118",
            "data/input/pglib_opf_case118_ieee.m",
            results_dir / "case118.json",
        ),
        (
            "case300",
            "data/input/pglib_opf_case300_ieee.m",
            results_dir / "case300.json",
        ),
        (
            "case1354_pegase",
            "data/input/pglib_opf_case1354_pegase.m",
            results_dir / "case1354_pegase.json",
        ),
        (
            "case9241_pegase",
            "data/input/pglib_opf_case9241_pegase.m",
            results_dir / "case9241_pegase.json",
        ),
    ]

    known_case30 = [(1, 2), (2, 4), (4, 6)]
    known_case118 = [(38, 65), (30, 38)]

    table_rows: List[Tuple[str, str, float, float, float, float]] = []
    sections: List[str] = []

    out: List[str] = []
    out.append("# Verification report")
    out.append("")
    out.append("## Notes on workflow")
    out.append("")
    out.append("- Base point is produced by **opf_pypsa** (PyPSA DC OPF + HiGHS).")
    out.append(
        "- Input cases are auto-downloaded deterministically when missing (supported filenames only)."
    )
    out.append("- No PF/AC-OPF modes are supported.")
    out.append("")
    out.append("---")
    out.append("")

    for case, input_path_fallback, results_path in cases:
        rp = Path(results_path)
        status = "ok"
        results: Dict[str, Any] | None = None

        with log_stage(logger, f"{case}: Prepare results.json (base point)"):
            if rp.exists():
                try:
                    results = _load_results(rp)
                except Exception:
                    logger.exception("Failed to parse results: %s", str(rp))
                    results = None
                    status = _append_status(status, "invalid_results_json")

            if results is not None:
                ok_cfg, msg_cfg = _results_config_matches(
                    results=results,
                    expected_dc_mode=str(demo_dc_mode_eff),
                    expected_slack_bus=int(demo_slack_bus),
                    expected_compute_nminus1=bool(demo_compute_nminus1),
                )
                if not ok_cfg:
                    logger.info(
                        "%s: results.json config mismatch (%s). Will regenerate.",
                        case,
                        msg_cfg,
                    )
                    results = None
                    status = _append_status(status, "config_mismatch")

            if results is None:
                if not generate_missing_results:
                    status = _append_status(status, "missing_or_invalid_results")
                else:
                    from power_stability_radius import compute_results_for_case

                    ip = Path(input_path_fallback)
                    try:
                        ensured_ip = Path(ensure_case_file(str(ip)))
                        ip = ensured_ip
                    except Exception:
                        logger.exception(
                            "Failed to ensure/download input case file for %s: %s",
                            case,
                            str(ip),
                        )
                        status = _append_status(status, "missing_input")
                    else:
                        results = compute_results_for_case(
                            input_path=str(ip),
                            slack_bus=int(demo_slack_bus),
                            dc_mode=str(demo_dc_mode_eff),
                            dc_chunk_size=256,
                            dc_dtype=np.float64,
                            margin_factor=1.0,
                            inj_std_mw=1.0,
                            compute_nminus1=bool(demo_compute_nminus1),
                            nminus1_update_sensitivities=True,
                            nminus1_islanding="skip",
                        )
                        rp.parent.mkdir(parents=True, exist_ok=True)
                        rp.write_text(
                            json.dumps(results, indent=4, ensure_ascii=False) + "\n",
                            encoding="utf-8",
                        )
                        status = _append_status(status, "generated")
                        logger.info("Generated results for %s: %s", case, str(rp))

        if results is None:
            table_rows.append(
                (case, status, float("nan"), float("nan"), float("nan"), float("nan"))
            )
            sections.append(
                _case_section_md(
                    case=case,
                    status=status,
                    mc=None,
                    top_risky=None,
                    top_match=None,
                    n1_match=None,
                    time_sec=float("nan"),
                )
            )
            continue

        ok_mc, msg_mc = _validate_results_for_monte_carlo(results)
        if not ok_mc:
            logger.warning("Results for %s are invalid for MC: %s", case, msg_mc)
            status = _append_status(status, "invalid_for_mc")

        time_sec = _get_meta_time_sec(results)
        input_path = _get_meta_input_path(results, input_path_fallback)
        slack_bus = _get_meta_slack_bus(results, fallback=int(demo_slack_bus))

        ip = Path(input_path)
        if not ip.exists():
            # Prefer canonical path for this case to keep the repo layout stable.
            canonical = Path(input_path_fallback)
            try:
                ip = Path(ensure_case_file(str(canonical)))
                status = _append_status(status, "downloaded_input")
            except Exception:
                logger.exception(
                    "Missing input case file for %s and download failed. meta_input=%s, canonical=%s",
                    case,
                    str(Path(input_path)),
                    str(canonical),
                )
                status = _append_status(status, "missing_input")
                table_rows.append(
                    (case, status, float("nan"), float("nan"), float("nan"), time_sec)
                )
                sections.append(
                    _case_section_md(
                        case=case,
                        status=status,
                        mc=None,
                        top_risky=None,
                        top_match=None,
                        n1_match=None,
                        time_sec=time_sec,
                    )
                )
                continue

        with log_stage(logger, f"{case}: Load network (for report metadata)"):
            try:
                net = load_network(ip)
            except Exception:
                logger.exception("Failed to load network: %s", str(ip))
                status = _append_status(status, "load_network_failed")
                table_rows.append(
                    (case, status, float("nan"), float("nan"), float("nan"), time_sec)
                )
                sections.append(
                    _case_section_md(
                        case=case,
                        status=status,
                        mc=None,
                        top_risky=None,
                        top_match=None,
                        n1_match=None,
                        time_sec=time_sec,
                    )
                )
                continue

        with log_stage(logger, f"{case}: Monte Carlo coverage"):
            mc_stats = estimate_coverage_percent(
                results_path=rp,
                input_case_path=ip,
                slack_bus=int(slack_bus),
                n_samples=int(n_samples),
                seed=int(seed),
                chunk_size=int(chunk_size),
                box_radius_quantile=float(mc_box_radius_quantile),
            )
        coverage = float(mc_stats.get("coverage_percent", float("nan")))
        if not math.isfinite(coverage):
            status = _append_status(status, str(mc_stats.get("status", "mc_undefined")))

        top10 = None
        top_stats: Optional[TopMatchStats] = None
        known_pairs: Optional[Sequence[Tuple[int, int]]] = None

        try:
            top10 = _top_k_risky_lines(
                results=results, net=net, k=10, radius_field="radius_l2"
            )
        except Exception:
            logger.exception("Top-10 extraction failed for %s", case)
            top10 = None

        if case == "case30":
            known_pairs = known_case30
        elif case == "case118":
            known_pairs = known_case118

        if known_pairs is not None and top10 is not None:
            top_stats = _match_stats_top10(top10=top10, known_pairs=known_pairs)

        n1_stats: Optional[NMinus1MatchStats]
        try:
            n1_stats = _nminus1_critical_match_stats(results=results, net=net)
            if n1_stats.status != "ok":
                status = _append_status(status, "n1_match_undefined")
        except Exception:
            logger.exception("N-1 critical match failed for %s", case)
            n1_stats = None
            status = _append_status(status, "n1_match_failed")

        sections.append(
            _case_section_md(
                case=case,
                status=status,
                mc=mc_stats,
                top_risky=top10 if case in {"case30", "case118"} else None,
                top_match=top_stats,
                n1_match=n1_stats,
                time_sec=time_sec,
                known_pairs=known_pairs,
            )
        )

        top_match_val = (
            float(top_stats.match_percent_common_over_10)
            if top_stats is not None
            else float("nan")
        )
        n1_match_val = (
            float(n1_stats.match_percent) if n1_stats is not None else float("nan")
        )
        table_rows.append(
            (case, status, coverage, top_match_val, n1_match_val, time_sec)
        )

    out.append(
        "Таблица (критерии usability по задаче: coverage > 70%, match > 70%, time < 10 sec):"
    )
    out.append("")
    out.append(
        "| case | status | coverage % | top risky match % | N-1 critical match % | time sec |"
    )
    out.append("|---|---|---:|---:|---:|---:|")
    for case, status, cov, topm, n1m, t in table_rows:
        cov_s = f"{cov:.3f}" if math.isfinite(cov) else "n/a"
        top_s = f"{topm:.3f}" if math.isfinite(topm) else "n/a"
        n1_s = f"{n1m:.3f}" if math.isfinite(n1m) else "n/a"
        t_s = f"{t:.3f}" if math.isfinite(t) else "n/a"
        out.append(f"| {case} | {status} | {cov_s} | {top_s} | {n1_s} | {t_s} |")
    out.append("")
    out.append("---")
    out.append("")
    out.extend(sections)

    return "\n".join(out) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate verification report (Markdown)."
    )
    parser.add_argument("--log-level", default=DEFAULT_LOGGING.level_console, type=str)
    parser.add_argument("--n-samples", default=DEFAULT_MC.n_samples, type=int)
    parser.add_argument("--seed", default=DEFAULT_MC.seed, type=int)
    parser.add_argument("--chunk-size", default=DEFAULT_MC.chunk_size, type=int)
    parser.add_argument(
        "--box-radius-quantile", default=DEFAULT_MC.box_radius_quantile, type=float
    )

    parser.add_argument("--results-dir", default="verification/results", type=str)
    parser.add_argument("--out", default="verification/report.md", type=str)
    parser.add_argument("--runs-dir", default=DEFAULT_LOGGING.runs_dir, type=str)

    parser.add_argument("--generate-missing-results", default=1, type=int)

    parser.add_argument(
        "--dc-mode",
        default="operator",
        type=str,
        choices=("materialize", "operator"),
        help="DC model mode used for auto-generated results.",
    )
    parser.add_argument("--slack-bus", default=0, type=int)
    parser.add_argument("--compute-nminus1", default=0, type=int)

    args = parser.parse_args(list(argv) if argv is not None else None)

    run_dir = Path(
        setup_logging(
            LoggingConfig(
                runs_dir=str(args.runs_dir),
                level_console=str(args.log_level),
                level_file="DEBUG",
            )
        )
    )
    logger.info("Report run directory: %s", str(run_dir))

    report_text = generate_report_text(
        results_dir=Path(str(args.results_dir)),
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
        generate_missing_results=bool(args.generate_missing_results),
        demo_dc_mode=str(args.dc_mode),
        demo_slack_bus=int(args.slack_bus),
        demo_compute_nminus1=bool(args.compute_nminus1),
        mc_box_radius_quantile=float(args.box_radius_quantile),
    )

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    logger.info("Wrote report: %s", str(out_path))

    run_copy = run_dir / "verification_report.md"
    run_copy.write_text(report_text, encoding="utf-8")
    logger.info("Also wrote report copy to: %s", str(run_copy))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
