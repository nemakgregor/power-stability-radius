"""
Generate verification/report.md from stored per-case results under verification/results/.

This script computes:
- Monte Carlo coverage%
- Top risky lines (top-10 by smallest radius_l2) and match with known congested corridors (case30/case118)
- N-1 criticality match: whether worst_contingency corresponds to high base flow

Behavior:
- Always includes all configured cases.
- Can auto-generate missing verification/results/*.json (and auto-download inputs) using the same
  computation pipeline as the 'demo' command.

Report goal
-----------
The report is intended to be *verifiable*: all derived metrics include the underlying counts /
denominators (where applicable) so you can audit computations.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Fix src-layout imports when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stability_radius.parsers.matpower import load_network
from stability_radius.radii.common import line_key
from stability_radius.utils import setup_logging
from verification.monte_carlo import estimate_coverage_percent

logger = logging.getLogger("verification.generate_report")


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
    except Exception:
        return float("nan")


def _get_meta_input_path(results: Dict[str, Any], fallback: str) -> str:
    meta = _get_meta(results)
    p = meta.get("input_path")
    return str(p) if isinstance(p, str) and p.strip() else str(fallback)


def _get_meta_slack_bus(results: Dict[str, Any], fallback: int = 0) -> int:
    meta = _get_meta(results)
    try:
        return int(meta.get("slack_bus", fallback))
    except Exception:
        return int(fallback)


def _sorted_line_indices(net) -> List[int]:
    return [int(x) for x in sorted(net.line.index)]


def _append_status(status: str, suffix: str) -> str:
    """Append suffix to status in a stable, readable way."""
    if not status or status == "ok":
        return str(suffix)
    return f"{status}_{suffix}"


def _fmt_float_or_na(x: float, *, digits: int = 6) -> str:
    if not math.isfinite(float(x)):
        return "n/a"
    # Use general format; digits is significant digits here.
    return f"{float(x):.{digits}g}"


def _fmt_percent_or_na(x: float, *, decimals: int = 3) -> str:
    if not math.isfinite(float(x)):
        return "n/a"
    return f"{float(x):.{decimals}f}%"


def _fmt_ok(v: bool) -> str:
    return "PASS" if v else "FAIL"


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
        except Exception:
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
    """
    Returns a TopMatchStats with explicit counts.

    Definitions (as used in the report):
      - match% (common/10) = 100 * common / 10
      - recall% (common/known) = 100 * common / len(known)
    """
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
    """
    Produce N-1 criticality match stats, including the intermediate counts.

    Metric (as requested in the original task):
      1) Select lines with radius_nminus1 < median(radius_nminus1) (finite only)
      2) For each selected line m, check if its worst contingency w is a "high base flow" line:
           p0_mw[w] > median(p0_mw)
      3) match% = 100 * hits / evaluated

    Important edge cases:
    - If there are no finite radius_nminus1 values -> match is n/a with status.
    - If no selected lines / no valid worst contingencies -> match is n/a with status.
    """
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
        except Exception:
            p0[pos] = float("nan")
        try:
            r_n1[pos] = float(row.get("radius_nminus1", float("nan")))
        except Exception:
            r_n1[pos] = float("nan")
        try:
            worst[pos] = int(row.get("worst_contingency", -1))
        except Exception:
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

    # ---------- MC summary line ----------
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

    # ---------- Top risky summary ----------
    if top_match is None:
        lines.append("- top risky match %: n/a")
    else:
        match = top_match.match_percent_common_over_10
        recall = top_match.recall_percent_common_over_known
        lines.append(
            f"- top risky match % (common/10): **{_fmt_float_or_na(match, digits=6)}%** "
            f"(common={top_match.common_k}, top_k={top_match.top_k}, known_k={top_match.known_k})"
            if math.isfinite(match)
            else "- top risky match %: n/a"
        )
        if math.isfinite(recall):
            lines.append(
                f"- top risky recall % (common/known): **{_fmt_float_or_na(recall, digits=6)}%** "
                f"(= {top_match.common_k} / {top_match.known_k})"
            )

    # ---------- N-1 summary ----------
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

    # ---------- time ----------
    lines.append(
        f"- time sec (demo): **{time_sec:.3f}**"
        if math.isfinite(time_sec)
        else "- time sec (demo): n/a"
    )
    lines.append("")

    # ---------- MC details ----------
    if mc is not None:
        lines.append("### Monte Carlo coverage (details)")
        lines.append("")
        mc_status = str(mc.get("status", "")) or "unknown"
        cov = float(mc.get("coverage_percent", float("nan")))
        ci_lo = float(mc.get("coverage_ci95_low_percent", float("nan")))
        ci_hi = float(mc.get("coverage_ci95_high_percent", float("nan")))
        n_samples = int(mc.get("n_samples", 0))
        seed = int(mc.get("seed", 0))
        chunk_size = int(mc.get("chunk_size", 0))
        box_lo = float(mc.get("box_lo", float("nan")))
        box_hi = float(mc.get("box_hi", float("nan")))
        box_half_width = float(mc.get("box_half_width", float("nan")))
        min_r = float(mc.get("min_r", float("nan")))
        max_r = float(mc.get("max_r", float("nan")))
        n_bus = int(mc.get("n_bus", 0))
        denom = int(mc.get("total_feasible_in_box", 0))
        numer = int(mc.get("feasible_in_ball", 0))
        feasible_rate = float(mc.get("feasible_rate_in_box_percent", float("nan")))

        lines.append(f"- status: **{mc_status}**")
        lines.append(f"- n_samples={n_samples}, seed={seed}, chunk_size={chunk_size}")
        if n_bus > 0:
            lines.append(
                "- sampling box: "
                f"[box_lo, box_hi] = [{_fmt_float_or_na(box_lo)}, {_fmt_float_or_na(box_hi)}] "
                f"(computed as ±2*max_r/sqrt(n_bus); max_r={_fmt_float_or_na(max_r)}, n_bus={n_bus}, half_width={_fmt_float_or_na(box_half_width)})"
            )
        else:
            lines.append(
                "- sampling box: "
                f"[box_lo, box_hi] = [{_fmt_float_or_na(box_lo)}, {_fmt_float_or_na(box_hi)}] "
                f"(computed as ±2*max_r/sqrt(n_bus); max_r={_fmt_float_or_na(max_r)})"
            )

        lines.append(f"- min_r (guaranteed L2 ball radius) = {_fmt_float_or_na(min_r)}")
        lines.append(
            f"- total_feasible_in_box = {denom} / {n_samples} = {_fmt_float_or_na(feasible_rate, digits=6)}%"
        )
        lines.append(f"- feasible_in_ball = {numer}")
        lines.append(
            "- coverage = feasible_in_ball / total_feasible_in_box * 100 = "
            + (
                f"{cov:.6f}%"
                if math.isfinite(cov)
                else "n/a (denominator is 0, i.e. no feasible samples in box)"
            )
        )
        if math.isfinite(cov) and math.isfinite(ci_lo) and math.isfinite(ci_hi):
            lines.append(f"- 95% CI (Wald, diagnostic): [{ci_lo:.3f}%, {ci_hi:.3f}%]")
        lines.append("")

    # ---------- Known corridors + top match details ----------
    if known_pairs is not None:
        lines.append("### Known congested corridors (literature/manual)")
        lines.append("")
        lines.append(f"- known pairs: {', '.join(f'{a}-{b}' for a, b in known_pairs)}")
        if top_match is not None and math.isfinite(
            top_match.match_percent_common_over_10
        ):
            lines.append(
                f"- common in top-10: **{top_match.common_k}**; "
                f"match% (common/10) = **{top_match.match_percent_common_over_10:.3f}%**; "
                f"recall% (common/known) = **{top_match.recall_percent_common_over_known:.3f}%**"
            )
            lines.append(
                "- Calculation: "
                f"match = 100 * common / 10 = 100 * {top_match.common_k} / 10; "
                f"recall = 100 * common / known = 100 * {top_match.common_k} / {top_match.known_k}."
            )
            lines.append(
                "- Примечание: при малом числе известных пар метрика (common/10) имеет низкий максимум "
                f"(например, для {len(known_pairs)} известных линий максимум = {100.0 * len(known_pairs) / 10.0:.1f}%)."
            )
        lines.append("")

    # ---------- Top risky table ----------
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

    # ---------- Adequacy checks ----------
    if case in {"case30", "case118"} and top_match is not None:
        recall = top_match.recall_percent_common_over_known
        if math.isfinite(recall):
            lines.append("### Adequacy check (top risky)")
            lines.append("")
            lines.append(
                f"- Criterion (interpretable): recall% > 70% => **{_fmt_ok(recall > 70.0)}**"
            )
            lines.append("")

    if case in {"case30", "case118", "case1354_pegase"} and n1_match is not None:
        if math.isfinite(n1_match.match_percent):
            lines.append("### Adequacy check (N-1 criticality)")
            lines.append("")
            lines.append(
                f"- Criterion: N-1 critical match % > 80% => **{_fmt_ok(n1_match.match_percent > 80.0)}**"
            )
            lines.append("")
        else:
            lines.append("### Adequacy check (N-1 criticality)")
            lines.append("")
            lines.append(
                "- Criterion: N-1 critical match % > 80% => **n/a** (metric undefined)"
            )
            lines.append("")

    # ---------- Literature comparisons (now robust to n/a coverage) ----------
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
                "Критерий адекватности по задаче: >70%. "
                "Условие не может быть проверено (coverage = n/a)."
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
                "Критерий адекватности по задаче: >70%. "
                "Условие не может быть проверено (coverage = n/a)."
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

    if math.isfinite(time_sec):
        lines.append("### Scalability check")
        lines.append("")
        lines.append(f"- Criterion: time < 10 sec => **{_fmt_ok(time_sec < 10.0)}**")
        lines.append("")

    return "\n".join(lines)


def generate_report_text(
    *,
    results_dir: Path,
    n_samples: int,
    seed: int,
    chunk_size: int,
    generate_missing_results: bool,
    demo_pf_mode: str,
    demo_dc_mode: str,
    demo_slack_bus: int,
) -> str:
    """
    Generate verification report markdown as a string.

    Parameters
    ----------
    results_dir:
        Directory containing per-case results JSON files.
    n_samples, seed, chunk_size:
        Monte Carlo parameters.
    generate_missing_results:
        If True, missing case results are computed and written into results_dir.
    demo_pf_mode, demo_dc_mode, demo_slack_bus:
        Settings used when generating missing results.

    Returns
    -------
    str
        Markdown report text.
    """
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
    out.append("## How to reproduce (quick)")
    out.append("")
    out.append(
        "Generate report (auto-generates missing verification/results/*.json by default):"
    )
    out.append("")
    out.append("```bash")
    out.append("poetry run python src/power_stability_radius.py report \\")
    out.append("  --results-dir verification/results --out verification/report.md")
    out.append("```")
    out.append("")
    out.append("To disable auto-generation (only aggregate existing JSONs):")
    out.append("")
    out.append("```bash")
    out.append("poetry run python src/power_stability_radius.py report \\")
    out.append("  --generate-missing-results 0 \\")
    out.append("  --results-dir verification/results --out verification/report.md")
    out.append("```")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## Notes on metrics")
    out.append("")
    out.append(
        "- **MC coverage** is reported as `coverage = 100 * feasible_in_ball / total_feasible_in_box`."
    )
    out.append(
        "- If `total_feasible_in_box = 0`, then coverage is **n/a** with `mc_status=no_feasible_samples` (not a crash)."
    )
    out.append(
        "- Sampling box uses per-coordinate half-width `2*max_r/sqrt(n_bus)` (not `2*max_r`), "
        "because `max_r` is an L2 radius and the old scaling makes feasibility probability collapse in high dimensions."
    )
    out.append("")
    out.append("---")
    out.append("")

    for case, input_path_fallback, results_path in cases:
        rp = Path(results_path)
        status = "ok"
        results: Dict[str, Any] | None = None

        if not rp.exists() and generate_missing_results:
            logger.info("Missing results for %s. Generating: %s", case, str(rp))
            try:
                from power_stability_radius import compute_results_for_case

                results = compute_results_for_case(
                    input_path=str(input_path_fallback),
                    slack_bus=int(demo_slack_bus),
                    pf_mode=str(demo_pf_mode),
                    dc_mode=str(demo_dc_mode),
                    dc_chunk_size=256,
                    dc_dtype=np.float64,
                    margin_factor=1.0,
                    inj_std_mw=1.0,
                    nminus1_update_sensitivities=True,
                    nminus1_islanding="skip",
                )
                rp.parent.mkdir(parents=True, exist_ok=True)
                rp.write_text(
                    json.dumps(results, indent=4, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                status = "generated"
                logger.info("Generated results for %s: %s", case, str(rp))
            except Exception:
                logger.exception("Failed to generate results for %s", case)
                status = "generation_failed"

        if results is None:
            if not rp.exists():
                logger.error("Missing results: %s", str(rp))
                status = "missing_results" if status == "ok" else status
                table_rows.append(
                    (
                        case,
                        status,
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )
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

            try:
                results = _load_results(rp)
            except Exception:
                logger.exception("Failed to parse results: %s", str(rp))
                status = _append_status(status, "invalid_results_json")
                table_rows.append(
                    (
                        case,
                        status,
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )
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

        time_sec = _get_meta_time_sec(results)
        input_path = _get_meta_input_path(results, input_path_fallback)
        slack_bus = _get_meta_slack_bus(results, fallback=int(demo_slack_bus))

        ip = Path(input_path)
        if not ip.exists():
            # Try to download using the same logic as demo.
            try:
                from power_stability_radius import (
                    _ensure_input_case_file,
                )  # internal helper

                downloaded = _ensure_input_case_file(str(input_path))
                ip = Path(downloaded)
                logger.info("Downloaded missing input for %s: %s", case, str(ip))
            except Exception:
                logger.exception(
                    "Missing input case file and download failed: %s", str(ip)
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

        # ---------- Monte Carlo coverage ----------
        mc_stats: Optional[Dict[str, Any]] = None
        coverage = float("nan")
        try:
            mc_stats = estimate_coverage_percent(
                results_path=rp,
                input_case_path=ip,
                slack_bus=int(slack_bus),
                n_samples=int(n_samples),
                seed=int(seed),
                chunk_size=int(chunk_size),
            )
            coverage = float(mc_stats.get("coverage_percent", float("nan")))

            if not math.isfinite(coverage):
                status = _append_status(
                    status, str(mc_stats.get("status", "mc_undefined"))
                )
                # Not an error: the report already contains mc_status + counts.
                logger.debug(
                    "Monte Carlo coverage undefined for %s (mc_status=%s).",
                    case,
                    str(mc_stats.get("status", "mc_undefined")),
                )
        except Exception:
            logger.exception("Monte Carlo coverage failed for %s", case)
            coverage = float("nan")
            mc_stats = None
            status = _append_status(status, "mc_failed")

        # ---------- Risky lines and matching ----------
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

        # ---------- N-1 match (with stats) ----------
        n1_stats: Optional[NMinus1MatchStats]
        try:
            n1_stats = _nminus1_critical_match_stats(results=results, net=net)
            if n1_stats.status != "ok":
                status = _append_status(status, "n1_match_undefined")
        except Exception:
            logger.exception("N-1 critical match failed for %s", case)
            n1_stats = None
            status = _append_status(status, "n1_match_failed")

        # ---------- Section rendering ----------
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

        # ---------- Summary table row ----------
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
    parser.add_argument("--log-level", default="INFO", type=str)
    parser.add_argument("--n-samples", default=50_000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--chunk-size", default=256, type=int)

    parser.add_argument(
        "--results-dir",
        default="verification/results",
        type=str,
        help="Directory containing per-case results JSON files (default: verification/results).",
    )
    parser.add_argument(
        "--out",
        default="verification/report.md",
        type=str,
        help="Output report path (default: verification/report.md).",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        type=str,
        help="Directory where per-run folders and run.log are created.",
    )

    parser.add_argument(
        "--generate-missing-results",
        default=1,
        type=int,
        help="1: compute missing results JSONs automatically, 0: do not.",
    )
    parser.add_argument("--demo-pf-mode", default="dc", type=str, choices=("ac", "dc"))
    parser.add_argument(
        "--demo-dc-mode",
        default="auto",
        type=str,
        choices=("auto", "materialize", "operator"),
    )
    parser.add_argument("--demo-slack-bus", default=0, type=int)

    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(runs_dir=str(args.runs_dir)),
        settings=SimpleNamespace(
            logging=SimpleNamespace(
                level_console=str(args.log_level), level_file="DEBUG"
            )
        ),
    )
    run_dir = Path(setup_logging(cfg))
    logger.info("Report run directory: %s", str(run_dir))

    results_dir = Path(str(args.results_dir))
    out_path = Path(str(args.out))

    report_text = generate_report_text(
        results_dir=results_dir,
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
        generate_missing_results=bool(args.generate_missing_results),
        demo_pf_mode=str(args.demo_pf_mode),
        demo_dc_mode=str(args.demo_dc_mode),
        demo_slack_bus=int(args.demo_slack_bus),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    logger.info("Wrote report: %s", str(out_path))

    run_copy = run_dir / "verification_report.md"
    run_copy.write_text(report_text, encoding="utf-8")
    logger.info("Also wrote report copy to: %s", str(run_copy))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
