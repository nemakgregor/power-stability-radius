from __future__ import annotations

"""
Multi-case verification report generator (Markdown).

Strictness / determinism
------------------------
- No auto-generation of missing results.
- No downloading of missing input case files.
- If a required file/field is missing: raise an explicit error.
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from stability_radius.parsers.matpower import load_network
from stability_radius.utils import log_stage

from .monte_carlo import run_monte_carlo_verification
from .status import summarize_status
from .types import VerificationResult
from .verify_certificate import interpret_certificate

logger = logging.getLogger("stability_radius.verification.generate_report")


@dataclass(frozen=True)
class ReportCaseSpec:
    """Report case specification (YAML/JSON-friendly)."""

    case_id: str
    input_case_path: Path
    results_path: Path
    known_critical_pairs: tuple[tuple[int, int], ...] = ()


def _load_results(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj)}")
    return obj


def _get_meta(results: Dict[str, Any]) -> Dict[str, Any]:
    meta = results.get("__meta__")
    return meta if isinstance(meta, dict) else {}


def _fmt_num(x: Any) -> str:
    """
    Format a scalar numeric value for Markdown.

    Contract
    --------
    - NaN / inf / non-numeric -> "n/a"
    - finite numeric -> compact scientific/decimal via %.6g
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.6g}"


def _fmt_percent(x: Any) -> str:
    """Format percent fields; NaN/inf -> 'n/a' (without trailing '%')."""
    s = _fmt_num(x)
    return "n/a" if s == "n/a" else f"{s}%"


def _case_card_md(
    *,
    case: str,
    results_status: str,
    vr: VerificationResult,
    comparisons: dict[str, Any],
    time_sec: float | None = None,
) -> str:
    """
    Render one Markdown card for a single verification run.

    Notes
    -----
    This function is used by unit tests; it must be stable and handle NaN values
    without emitting strings like "nan%".
    """
    interp = interpret_certificate(vr)

    lines: List[str] = []
    lines.append(f"## {case}")
    lines.append("")
    lines.append(f"- results_status: **{results_status}**")
    lines.append(f"- summary: **{summarize_status(vr)}**")
    lines.append(f"- overall: **{vr.overall.status}**")
    if time_sec is not None:
        lines.append(f"- time_sec: {_fmt_num(time_sec)}")
    if vr.overall.reasons:
        lines.append(f"- reasons: `{list(vr.overall.reasons)}`")
    lines.append("")

    lines.append("### Inputs")
    lines.append("")
    lines.append(f"- slack_bus: {vr.inputs.slack_bus}")
    lines.append(f"- n_bus: {vr.inputs.n_bus}")
    lines.append(f"- n_line: {vr.inputs.n_line}")
    lines.append(f"- d (balanced dim): {vr.inputs.dim_balance}")
    lines.append(f"- sigma_mw: {_fmt_num(vr.inputs.sigma_mw)}")
    lines.append(f"- n_samples: {vr.inputs.n_samples}")
    lines.append(f"- seed: {vr.inputs.seed}")
    lines.append("")

    lines.append("### Base point")
    lines.append("")
    lines.append(f"- status: **{vr.base_point.status}**")
    lines.append(f"- violated_lines: {vr.base_point.violated_lines}")
    lines.append(f"- max_violation: {_fmt_num(vr.base_point.max_violation_mw)}")
    lines.append("")

    lines.append("### Radius")
    lines.append("")
    lines.append(f"- status: **{vr.radius.status}**")
    lines.append(f"- r*: {_fmt_num(vr.radius.r_star)}")
    lines.append(f"- certificate_soundness: **{interp.soundness.upper()}**")
    lines.append(f"- certificate_usefulness: **{interp.usefulness.upper()}**")
    lines.append(f"- argmin_line_idx: {vr.radius.argmin_line_idx}")
    lines.append("")

    lines.append("### Soundness")
    lines.append("")
    lines.append(f"- status: **{vr.soundness.status}**")
    lines.append(f"- n_ball_samples: {vr.soundness.n_ball_samples}")
    lines.append(f"- violation_samples: {vr.soundness.violation_samples}")
    lines.append(f"- max_violation: {_fmt_num(vr.soundness.max_violation_mw)}")
    lines.append(f"- max_violation_line_idx: {vr.soundness.max_violation_line_idx}")
    lines.append("")

    lines.append("### Probabilistic")
    lines.append("")
    lines.append(f"- status: **{vr.probabilistic.status}**")
    lines.append(
        f"- p_safe (MC): {_fmt_percent(vr.probabilistic.p_safe_gaussian_percent)}"
    )
    lines.append(
        f"- p_ball (analytic): {_fmt_percent(vr.probabilistic.p_ball_analytic_percent)}"
    )
    lines.append(f"- p_ball (MC): {_fmt_percent(vr.probabilistic.p_ball_mc_percent)}")
    lines.append(
        f"- eta = P(safe | in ball): {_fmt_percent(vr.probabilistic.eta_safe_given_in_ball_percent)}"
    )
    lines.append(f"- rho: {_fmt_num(vr.probabilistic.rho)}")
    lines.append("")

    if comparisons:
        lines.append("### Diagnostics")
        lines.append("")
        for k in sorted(comparisons.keys()):
            v = comparisons[k]
            lines.append(f"- {k}: {v}")
        lines.append("")

    return "\n".join(lines)


def _meta_has_dc_sigma(meta: dict[str, Any]) -> bool:
    """Return True iff results.json contains __meta__.dc.inj_std_mw in the new schema."""
    dc = meta.get("dc", None)
    if not isinstance(dc, dict):
        return False
    return dc.get("inj_std_mw", None) is not None


def generate_report_text(
    *,
    cases: Sequence[ReportCaseSpec],
    results_dir: Path,
    n_samples: int,
    seed: int,
    chunk_size: int,
    feas_tol: float,
    cert_tol: float,
    cert_max_samples: int,
    strict: bool,
    dc_sigma_override_mw: float | None = None,
    ac_sigma_p_mw: float = 1.0,
    ac_sigma_q_mvar: float = 1.0,
    ac_pf_solver: str,
    ac_lossless: bool,
    ac_basepoint_s_tol_mva: float,
) -> str:
    if not cases:
        raise ValueError("report requires a non-empty cases list.")

    out: List[str] = []
    out.append("# Verification report")
    out.append("")
    out.append("## Setup")
    out.append("")
    out.append(f"- strict: **{bool(strict)}**")
    out.append(f"- n_samples: {int(n_samples)}")
    out.append(f"- seed: {int(seed)}")
    out.append(f"- chunk_size: {int(chunk_size)}")
    out.append(f"- dc_sigma_override_mw: {_fmt_num(dc_sigma_override_mw)}")
    out.append(f"- ac_pf_solver: {str(ac_pf_solver)}")
    out.append(f"- ac_lossless: {bool(ac_lossless)}")
    out.append(f"- ac_basepoint_s_tol_mva: {_fmt_num(ac_basepoint_s_tol_mva)}")
    out.append("")

    for spec in cases:
        case_id = str(spec.case_id)
        rp = Path(spec.results_path)
        if not rp.is_absolute():
            rp = (Path(results_dir) / rp).resolve()

        ip = Path(spec.input_case_path).expanduser()
        if not ip.is_absolute():
            ip = ip.resolve()

        if not rp.exists():
            raise FileNotFoundError(f"Missing results.json for case={case_id}: {rp}")
        if not ip.exists():
            raise FileNotFoundError(f"Missing input case file for case={case_id}: {ip}")

        with log_stage(logger, f"{case_id}: load results"):
            results = _load_results(rp)
            meta = _get_meta(results)

        with log_stage(logger, f"{case_id}: load network"):
            net = load_network(ip)

        has_dc = any(
            isinstance(results.get(f"line_{int(lid)}"), dict)
            and ("radius_l2" in results[f"line_{int(lid)}"])
            for lid in sorted(net.line.index)
        )
        has_ac = any(
            isinstance(results.get(f"line_{int(lid)}"), dict)
            and ("radius_ac_l2" in results[f"line_{int(lid)}"])
            for lid in sorted(net.line.index)
        )

        slack_bus = int(meta.get("slack_bus", 0))

        out.append(f"## {case_id}")
        out.append("")
        out.append(f"- results_path: `{rp}`")
        out.append(f"- input_case_path: `{ip}`")
        out.append(f"- slack_bus (from results meta): {slack_bus}")
        if spec.known_critical_pairs:
            out.append(f"- known_critical_pairs: `{list(spec.known_critical_pairs)}`")
        out.append("")

        if has_dc:
            if dc_sigma_override_mw is None and (not _meta_has_dc_sigma(meta)):
                raise ValueError(
                    f"Case={case_id}: results.json missing __meta__.dc.inj_std_mw. "
                    "Either re-run `compute` with the current schema, or pass "
                    "`report --sigma-override-mw <MW>` (or set report.dc.sigma_override_mw in YAML)."
                )

            vr_dc = run_monte_carlo_verification(
                mode="dc",
                results_path=rp,
                input_case_path=ip,
                slack_bus=slack_bus,
                n_samples=int(n_samples),
                seed=int(seed),
                chunk_size=int(chunk_size),
                feas_tol=float(feas_tol),
                cert_tol=float(cert_tol),
                cert_max_samples=int(cert_max_samples),
                sigma_override_mw=dc_sigma_override_mw,
                allow_download=False,
            )
            out.append("### DC verification")
            out.append("")
            out.append(
                _case_card_md(
                    case=f"{case_id} / DC",
                    results_status="ok",
                    vr=vr_dc,
                    comparisons=vr_dc.comparisons,
                )
            )
            out.append("")
        else:
            if strict:
                raise KeyError(
                    f"Case={case_id}: results.json has no DC fields (radius_l2) but report is strict."
                )
            out.append("### DC verification")
            out.append("")
            out.append("- skipped: no DC fields in results.json")
            out.append("")

        if has_ac:
            vr_ac = run_monte_carlo_verification(
                mode="ac",
                results_path=rp,
                input_case_path=ip,
                slack_bus=slack_bus,
                n_samples=int(n_samples),
                seed=int(seed) + 17_000,
                chunk_size=max(1, int(min(chunk_size, 32))),
                feas_tol=float(feas_tol),
                cert_tol=float(cert_tol),
                cert_max_samples=int(cert_max_samples),
                allow_download=False,
                ac_sigma_p_mw=float(ac_sigma_p_mw),
                ac_sigma_q_mvar=float(ac_sigma_q_mvar),
                ac_pf_solver=str(ac_pf_solver),
                ac_lossless=bool(ac_lossless),
                ac_basepoint_s_tol_mva=float(ac_basepoint_s_tol_mva),
            )
            out.append("### AC verification")
            out.append("")
            out.append(
                _case_card_md(
                    case=f"{case_id} / AC",
                    results_status="ok",
                    vr=vr_ac,
                    comparisons=vr_ac.comparisons,
                )
            )
            out.append("")
        else:
            if strict:
                raise KeyError(
                    f"Case={case_id}: results.json has no AC fields (radius_ac_l2) but report is strict."
                )
            out.append("### AC verification")
            out.append("")
            out.append("- skipped: no AC fields in results.json")
            out.append("")

    return "\n".join(out) + "\n"