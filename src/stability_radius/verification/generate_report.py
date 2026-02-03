from __future__ import annotations

"""
Multi-case verification report generator (Markdown).

Report philosophy
-----------------
The report explicitly separates:
1) Soundness of the DC L2 certificate (hard correctness check)
2) Probabilistic metrics under a chosen Δp distribution (soft indicators)
3) Heuristic comparisons (optional, not used as correctness signals)
4) Literature notes (only with an explicit disclaimer about non-comparability)

This avoids mixing:
- "certificate is correct" vs
- "certificate covers meaningful probability mass" vs
- "this matches some external benchmark"
"""

import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from stability_radius.config import DEFAULT_MC, DEFAULT_OPF, OPFConfig
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.common import line_key
from stability_radius.utils import log_stage
from stability_radius.utils.download import ensure_case_file
from stability_radius.workflows import compute_results_for_case

from .monte_carlo import run_monte_carlo_verification
from .status import summarize_status
from .types import VerificationResult
from .verify_certificate import interpret_certificate

logger = logging.getLogger("stability_radius.verification.generate_report")

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class TopMatchStats:
    """Top-K match/recall stats with explicit counts (for audit)."""

    top_k: int
    known_k: int
    common_k: int
    match_percent_common_over_k: float
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


def _sorted_line_indices(net) -> List[int]:
    return [int(x) for x in sorted(net.line.index)]


def _fmt_float_or_na(x: float, *, digits: int = 6) -> str:
    if not math.isfinite(float(x)):
        return "n/a"
    return f"{float(x):.{digits}g}"


def _fmt_percent_or_na(x: float, *, decimals: int = 3) -> str:
    if not math.isfinite(float(x)):
        return "n/a"
    return f"{float(x):.{decimals}f}%"


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


def _match_stats_topk(
    *,
    top: Sequence[Dict[str, Any]],
    known_pairs: Sequence[Tuple[int, int]],
    k: int,
) -> TopMatchStats:
    top_k = int(len(top))
    known = {frozenset((int(a), int(b))) for a, b in known_pairs}
    known_k = int(len(known))

    if top_k <= 0:
        return TopMatchStats(
            top_k=0,
            known_k=known_k,
            common_k=0,
            match_percent_common_over_k=float("nan"),
            recall_percent_common_over_known=float("nan"),
        )

    top_pairs = {frozenset((int(x["from_bus"]), int(x["to_bus"]))) for x in top}
    common = int(len(known.intersection(top_pairs)))

    match = 100.0 * float(common) / float(k) if k > 0 else float("nan")
    recall = 100.0 * float(common) / float(known_k) if known_k else float("nan")

    return TopMatchStats(
        top_k=top_k,
        known_k=known_k,
        common_k=common,
        match_percent_common_over_k=float(match),
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


def _case_card_md(
    *,
    case: str,
    results_status: str,
    vr: Optional[VerificationResult],
    comparisons: dict[str, Any],
    time_sec: float,
) -> str:
    lines: List[str] = []
    lines.append(f"## {case}")
    lines.append("")
    lines.append(f"- results status: **{results_status}**")

    if vr is None:
        lines.append("- verification: n/a")
        lines.append("")
        return "\n".join(lines)

    interp = interpret_certificate(vr)

    lines.append(f"- overall: **{vr.overall.status}**")
    if vr.overall.reasons:
        lines.append(f"- reasons: `{list(vr.overall.reasons)}`")

    # Single summary status (report-friendly).
    lines.append(f"- summary: **{summarize_status(vr)}**")

    lines.append("")
    lines.append("### Inputs")
    lines.append("")
    lines.append(f"- slack_bus: {vr.inputs.slack_bus}")
    lines.append(f"- n_bus: {vr.inputs.n_bus}")
    lines.append(f"- n_line: {vr.inputs.n_line}")
    lines.append(f"- d (balanced dim): {vr.inputs.dim_balance}")
    lines.append(f"- sigma_mw: {_fmt_float_or_na(vr.inputs.sigma_mw)}")
    lines.append(f"- n_samples: {vr.inputs.n_samples}")
    lines.append(f"- seed: {vr.inputs.seed}")
    lines.append("")

    lines.append("### Base point")
    lines.append("")
    lines.append(f"- status: **{vr.base_point.status}**")
    lines.append(f"- violated_lines: {vr.base_point.violated_lines}")
    lines.append(
        f"- max_violation_mw: {_fmt_float_or_na(vr.base_point.max_violation_mw)}"
    )
    lines.append("")

    lines.append("### Radius (certificate)")
    lines.append("")
    lines.append(f"- status: **{vr.radius.status}**")
    lines.append(f"- r*: {_fmt_float_or_na(vr.radius.r_star)}")
    lines.append(f"- certificate_soundness: **{interp.soundness.upper()}**")
    lines.append(f"- certificate_usefulness: **{interp.usefulness.upper()}**")

    if float(vr.radius.r_star) == 0.0 and vr.base_point.status == "BASE_OK":
        lines.append(
            "- note: **TRIVIAL_TRUE** (r*=0 ⇒ certificate is correct but non-informative)"
        )
    lines.append(f"- argmin_line_idx: {vr.radius.argmin_line_idx}")
    lines.append(f"- argmin_margin_mw: {_fmt_float_or_na(vr.radius.argmin_margin_mw)}")
    lines.append(f"- argmin_norm_g: {_fmt_float_or_na(vr.radius.argmin_norm_g)}")
    lines.append("")

    lines.append("### Soundness (hard check)")
    lines.append("")
    lines.append(f"- status: **{vr.soundness.status}**")
    lines.append(f"- n_ball_samples: {vr.soundness.n_ball_samples}")
    lines.append(f"- violation_samples: {vr.soundness.violation_samples}")
    lines.append(
        f"- max_violation_mw: {_fmt_float_or_na(vr.soundness.max_violation_mw)}"
    )
    lines.append(f"- max_violation_line_idx: {vr.soundness.max_violation_line_idx}")
    lines.append("")

    lines.append("### Probabilistic metrics (soft)")
    lines.append("")
    lines.append(f"- status: **{vr.probabilistic.status}**")
    lines.append(
        "- p_safe (Gaussian, MC): "
        f"**{_fmt_percent_or_na(vr.probabilistic.p_safe_gaussian_percent)}** "
        f"(CI95=[{_fmt_percent_or_na(vr.probabilistic.p_safe_gaussian_ci95_low_percent)}, "
        f"{_fmt_percent_or_na(vr.probabilistic.p_safe_gaussian_ci95_high_percent)}])"
    )
    lines.append(
        "- p_ball (analytic): "
        f"**{_fmt_percent_or_na(vr.probabilistic.p_ball_analytic_percent)}** "
        "(= P(||Δp||₂ ≤ r*) under isotropic Gaussian in the balanced subspace, d=n_bus-1)"
    )
    lines.append(
        "- p_ball (MC): "
        f"{_fmt_percent_or_na(vr.probabilistic.p_ball_mc_percent)} "
        f"(CI95=[{_fmt_percent_or_na(vr.probabilistic.p_ball_mc_ci95_low_percent)}, "
        f"{_fmt_percent_or_na(vr.probabilistic.p_ball_mc_ci95_high_percent)}])"
    )
    lines.append(
        "- eta = P(safe | in ball): "
        f"{_fmt_percent_or_na(vr.probabilistic.eta_safe_given_in_ball_percent)} "
        f"(CI95=[{_fmt_percent_or_na(vr.probabilistic.eta_ci95_low_percent)}, "
        f"{_fmt_percent_or_na(vr.probabilistic.eta_ci95_high_percent)}])"
    )
    lines.append(
        f"- rho = r*/(sigma*sqrt(d)): {_fmt_float_or_na(vr.probabilistic.rho)}"
    )
    lines.append("")

    lines.append("### Comparisons (heuristics)")
    lines.append("")
    if comparisons:
        for k, v in comparisons.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- n/a")
    lines.append("")

    lines.append("### Limitations")
    lines.append("")
    lines.append(
        "- Эти метрики **не являются** прямой оценкой объёма/coverage области допустимости AC/OPF. "
        "Это: (i) проверка soundness DC-сертификата и (ii) вероятность безопасности "
        "под выбранным распределением Δp."
    )
    lines.append(
        "- Literature note: **not directly comparable** to convex-feasibility coverage metrics in prior work."
    )
    lines.append("")

    if math.isfinite(float(time_sec)):
        lines.append(f"- demo time_sec: **{time_sec:.3f}**")
    else:
        lines.append("- demo time_sec: n/a")
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
    demo_dc_chunk_size: int = 256,
    demo_dc_dtype: np.dtype = np.float64,
    demo_margin_factor: float = 1.0,
    demo_inj_std_mw: float = 1.0,
    demo_nminus1_update_sensitivities: bool = True,
    demo_nminus1_islanding: str = "skip",
    demo_opf_cfg: OPFConfig | None = None,
    demo_opf_dc_flow_consistency_tol_mw: float = 1e-3,
    demo_opf_bus_balance_tol_mw: float = 1e-6,
    mc_box_radius_quantile: float = DEFAULT_MC.box_radius_quantile,  # legacy (ignored)
    mc_box_feas_tol_mw: float = DEFAULT_MC.box_feas_tol_mw,
    mc_cert_tol_mw: float = DEFAULT_MC.cert_tol_mw,
    mc_cert_max_samples: int = DEFAULT_MC.cert_max_samples,
    strict_units: bool = True,
    allow_phase_shift: bool = False,
) -> str:
    """
    Generate verification report markdown as a string.
    """
    _ = float(mc_box_radius_quantile)

    demo_dc_mode_eff = str(demo_dc_mode).strip().lower()
    if demo_dc_mode_eff not in ("materialize", "operator"):
        raise ValueError("demo_dc_mode must be materialize|operator")

    opf_cfg = demo_opf_cfg if demo_opf_cfg is not None else DEFAULT_OPF

    results_dir_abs = Path(results_dir).resolve()

    # Deterministic case list.
    cases = [
        (
            "case30",
            _REPO_ROOT / "data/input/pglib_opf_case30_ieee.m",
            results_dir_abs / "case30.json",
        ),
        (
            "case118",
            _REPO_ROOT / "data/input/pglib_opf_case118_ieee.m",
            results_dir_abs / "case118.json",
        ),
        # (
        #     "case300",
        #     _REPO_ROOT / "data/input/pglib_opf_case300_ieee.m",
        #     results_dir_abs / "case300.json",
        # ),
        # (
        #     "case1354_pegase",
        #     _REPO_ROOT / "data/input/pglib_opf_case1354_pegase.m",
        #     results_dir_abs / "case1354_pegase.json",
        # ),
        # (
        #     "case9241_pegase",
        #     _REPO_ROOT / "data/input/pglib_opf_case9241_pegase.m",
        #     results_dir_abs / "case9241_pegase.json",
        # ),
    ]

    known_case30 = [(1, 2), (2, 4), (4, 6)]
    known_case118 = [(38, 65), (30, 38)]

    rows: List[
        Tuple[
            str,  # case
            str,  # overall
            str,  # base
            str,  # radius
            str,  # cert_soundness (interpreted)
            str,  # usefulness (interpreted)
            float,  # r*
            float,  # rho
            float,  # p_safe (MC) %
            float,  # p_ball (analytic) %
            float,  # time sec
        ]
    ] = []
    sections: List[str] = []

    out: List[str] = []
    out.append("# Verification report")
    out.append("")
    out.append("## Experimental setup")
    out.append("")
    out.append("### DC L2 certificate (contract)")
    out.append("")
    out.append(
        "- Linear DC model in the balanced injection subspace (dimension d = n_bus − 1): Δf = H Δp,  with 1^TΔp=0."
    )
    out.append("- Per-line margin: m_i = c_i − |f0_i|.")
    out.append(
        "- Per-line radius: r_i = m_i / ||g_i||₂, where g_i is the sensitivity row in balanced coordinates."
    )
    out.append("- Global certificate: r* = min_i r_i.")
    out.append(
        "- Soundness claim (DC only): if base point feasible and ||Δp||₂ ≤ r*, then all line limits are satisfied."
    )
    out.append("")
    out.append("### What we verify")
    out.append("")
    out.append(
        "- **Soundness**: uniform samples inside the certified L2 ball in the balanced subspace (hard check)."
    )
    out.append(
        "- **Probabilistic metrics** (soft): Δp is Gaussian in the balanced subspace (slack-invariant):"
    )
    out.append(
        "  - Sample z ~ N(0, σ² I_n), then project: Δp = z − mean(z)·1  (equivalently N(0, σ² I_d) in an orthonormal basis)."
    )
    out.append("  - p_safe = P(all lines safe) (MC + CI95)")
    out.append("  - p_ball = P(||Δp||₂ ≤ r*) (analytic chi-square CDF + MC + CI95)")
    out.append("  - eta = P(safe | in ball) (MC + CI95)")
    out.append("  - rho = r*/(σ√d) (dimensionless)")
    out.append("")
    out.append("### Literature notes")
    out.append("")
    out.append(
        "Любые сравнения с литературой про **coverage of feasibility region** "
        "не являются прямыми сравнениями: здесь другая метрика и другой домен/распределение. "
        "В отчёте это помечено как **not directly comparable**."
    )
    out.append("")
    out.append("---")
    out.append("")

    for case, input_path_fallback, results_path in cases:
        rp = Path(results_path)
        status_reasons: list[str] = []
        results: Dict[str, Any] | None = None

        with log_stage(logger, f"{case}: Prepare results.json"):
            if rp.exists():
                try:
                    results = _load_results(rp)
                except Exception as e:  # noqa: BLE001
                    logger.exception("Failed to parse results: %s", str(rp))
                    results = None
                    status_reasons.append(f"invalid_results_json:{type(e).__name__}")

            if results is not None:
                ok_cfg, msg_cfg = _results_config_matches(
                    results=results,
                    expected_dc_mode=str(demo_dc_mode_eff),
                    expected_slack_bus=int(demo_slack_bus),
                    expected_compute_nminus1=bool(demo_compute_nminus1),
                )
                if not ok_cfg:
                    logger.info("%s: results.json config mismatch (%s).", case, msg_cfg)
                    results = None
                    status_reasons.append(f"config_mismatch:{msg_cfg}")

            if results is None and generate_missing_results:
                ip = Path(input_path_fallback)
                try:
                    ip_eff = Path(ensure_case_file(str(ip))).resolve()
                except Exception as e:  # noqa: BLE001
                    logger.exception(
                        "Failed to ensure/download input case file for %s: %s",
                        case,
                        str(ip),
                    )
                    status_reasons.append(f"missing_input:{type(e).__name__}")
                else:
                    results = compute_results_for_case(
                        input_path=str(ip_eff),
                        slack_bus=int(demo_slack_bus),
                        dc_mode=str(demo_dc_mode_eff),
                        dc_chunk_size=int(demo_dc_chunk_size),
                        dc_dtype=demo_dc_dtype,
                        margin_factor=float(demo_margin_factor),
                        inj_std_mw=float(demo_inj_std_mw),
                        compute_nminus1=bool(demo_compute_nminus1),
                        nminus1_update_sensitivities=bool(
                            demo_nminus1_update_sensitivities
                        ),
                        nminus1_islanding=str(demo_nminus1_islanding),
                        opf_cfg=opf_cfg,
                        opf_dc_flow_consistency_tol_mw=float(
                            demo_opf_dc_flow_consistency_tol_mw
                        ),
                        opf_bus_balance_tol_mw=float(demo_opf_bus_balance_tol_mw),
                        strict_units=bool(strict_units),
                        allow_phase_shift=bool(allow_phase_shift),
                        path_base_dir=_REPO_ROOT,
                    )
                    rp.parent.mkdir(parents=True, exist_ok=True)
                    rp.write_text(
                        json.dumps(results, indent=4, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    status_reasons.append("generated_results")

        results_status = (
            "ok" if not status_reasons else "WARN(" + ", ".join(status_reasons) + ")"
        )

        if results is None:
            sections.append(
                _case_card_md(
                    case=case,
                    results_status=results_status,
                    vr=None,
                    comparisons={},
                    time_sec=float("nan"),
                )
            )
            rows.append(
                (
                    case,
                    results_status,
                    "n/a",
                    "n/a",
                    "unknown",
                    "n/a",
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                )
            )
            continue

        meta = _get_meta(results)
        time_sec = (
            float(meta.get("compute_time_sec", float("nan")))
            if isinstance(meta, dict)
            else float("nan")
        )

        # Ensure input case file (absolute, deterministic base dir)
        ip_eff = (
            Path(meta.get("input_path", "")).expanduser()
            if isinstance(meta.get("input_path", ""), str)
            else Path(input_path_fallback)
        )
        if not ip_eff.is_absolute():
            # Resolve relative paths against repo root to avoid CWD dependency.
            ip_eff = (_REPO_ROOT / ip_eff).resolve()

        if not ip_eff.exists():
            try:
                ip_eff = Path(
                    ensure_case_file(str(Path(input_path_fallback).resolve()))
                ).resolve()
                status_reasons.append("downloaded_input")
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "Missing input case file for %s and download failed.", case
                )
                status_reasons.append(f"missing_input:{type(e).__name__}")
                results_status = "WARN(" + ", ".join(status_reasons) + ")"
                sections.append(
                    _case_card_md(
                        case=case,
                        results_status=results_status,
                        vr=None,
                        comparisons={},
                        time_sec=time_sec,
                    )
                )
                rows.append(
                    (
                        case,
                        results_status,
                        "n/a",
                        "n/a",
                        "unknown",
                        "n/a",
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        time_sec,
                    )
                )
                continue

        with log_stage(logger, f"{case}: Load network (metadata + comparisons)"):
            net = load_network(ip_eff)

        # Heuristic comparisons (optional)
        comparisons: dict[str, Any] = {}

        try:
            top10 = _top_k_risky_lines(
                results=results, net=net, k=10, radius_field="radius_l2"
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Top-k extraction failed for %s", case)
            top10 = []
            comparisons["top_risky_status"] = f"error:{type(e).__name__}"

        known_pairs: Optional[Sequence[Tuple[int, int]]] = None
        if case == "case30":
            known_pairs = known_case30
        elif case == "case118":
            known_pairs = known_case118

        if known_pairs is not None and top10:
            stats_top = _match_stats_topk(top=top10, known_pairs=known_pairs, k=10)
            comparisons["top_risky_status"] = "ok"
            comparisons["top_risky_match_percent"] = (
                stats_top.match_percent_common_over_k
            )
            comparisons["top_risky_recall_percent"] = (
                stats_top.recall_percent_common_over_known
            )
            comparisons["top_risky_common_k"] = stats_top.common_k
            comparisons["top_risky_known_k"] = stats_top.known_k
        elif known_pairs is not None:
            comparisons["top_risky_status"] = "no_top_lines"

        try:
            n1_stats = _nminus1_critical_match_stats(results=results, net=net)
            comparisons["nminus1_match_status"] = n1_stats.status
            comparisons["nminus1_match_percent"] = n1_stats.match_percent
        except Exception as e:  # noqa: BLE001
            logger.exception("N-1 match stats failed for %s", case)
            comparisons["nminus1_match_status"] = f"error:{type(e).__name__}"

        # Verification (MC + soundness)
        with log_stage(logger, f"{case}: Monte Carlo verification"):
            vr = run_monte_carlo_verification(
                results_path=rp,
                input_case_path=ip_eff,
                slack_bus=int(demo_slack_bus),
                n_samples=int(n_samples),
                seed=int(seed),
                chunk_size=int(chunk_size),
                box_radius_quantile=float(mc_box_radius_quantile),  # ignored
                box_feas_tol_mw=float(mc_box_feas_tol_mw),
                cert_tol_mw=float(mc_cert_tol_mw),
                cert_max_samples=int(mc_cert_max_samples),
                strict_units=bool(strict_units),
                allow_phase_shift=bool(allow_phase_shift),
            )

        # Attach comparisons into VR (immutably).
        vr = replace(vr, comparisons={**vr.comparisons, **comparisons})

        interp = interpret_certificate(vr)

        logger.info(
            "case=%s | base=%s | r*=%.6g | d=%d | cert=%s | usefulness=%s | "
            "p_safe=%.3f%% | p_ball_analytic=%.3f%% | rho=%.6g | time_sec=%s",
            str(case),
            str(vr.base_point.status),
            float(vr.radius.r_star),
            int(vr.inputs.dim_balance),
            str(interp.soundness),
            str(interp.usefulness),
            float(vr.probabilistic.p_safe_gaussian_percent),
            float(vr.probabilistic.p_ball_analytic_percent),
            float(vr.probabilistic.rho),
            _fmt_float_or_na(float(time_sec), digits=6),
        )

        # Table row (comparable metrics only).
        rows.append(
            (
                case,
                vr.overall.status,
                vr.base_point.status,
                vr.radius.status,
                str(interp.soundness),
                str(interp.usefulness),
                float(vr.radius.r_star),
                float(vr.probabilistic.rho),
                float(vr.probabilistic.p_safe_gaussian_percent),
                float(vr.probabilistic.p_ball_analytic_percent),
                float(time_sec),
            )
        )

        sections.append(
            _case_card_md(
                case=case,
                results_status=results_status,
                vr=vr,
                comparisons=comparisons,
                time_sec=time_sec,
            )
        )

    # Cross-case table
    out.append("## Cross-case table (comparable metrics)")
    out.append("")
    out.append(
        "| case | overall | base | radius | cert_soundness | usefulness | r* | rho | p_safe (MC) % | p_ball(analytic) % | time sec |"
    )
    out.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|")

    for (
        case,
        overall,
        base,
        radius,
        cert_soundness,
        usefulness,
        r_star,
        rho,
        p_safe,
        p_ball,
        t,
    ) in rows:
        out.append(
            "| {case} | {overall} | {base} | {radius} | {cert_soundness} | {usefulness} | {r_star} | {rho} | {p_safe} | {p_ball} | {t} |".format(
                case=case,
                overall=overall,
                base=base,
                radius=radius,
                cert_soundness=cert_soundness,
                usefulness=usefulness,
                r_star=_fmt_float_or_na(float(r_star)),
                rho=_fmt_float_or_na(float(rho)),
                p_safe=_fmt_float_or_na(float(p_safe)),
                p_ball=_fmt_float_or_na(float(p_ball)),
                t=_fmt_float_or_na(float(t), digits=6),
            )
        )
    out.append("")
    out.append("## Diagnostics (no hard pass/fail thresholds)")
    out.append("")
    out.append(
        "- Научная корректность здесь определяется **soundness** сертификата (hard check), а не порогами вида `>70%`."
    )
    out.append(
        "- Ключевая семантика для сканирования:\n"
        "  - `cert_soundness=sound` или `trivial_true` ⇒ сертификат корректен в DC-модели.\n"
        "  - `usefulness=zero_radius` ⇒ сертификат корректен, но **неинформативен** (r*=0).\n"
        "  - `cert_soundness=unsound` ⇒ найден контрпример внутри шара (FAIL)."
    )
    out.append("")
    out.append("---")
    out.append("")
    out.extend(sections)

    return "\n".join(out) + "\n"
