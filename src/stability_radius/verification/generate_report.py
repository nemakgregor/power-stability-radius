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
"""

import csv
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


@dataclass(frozen=True)
class ReportCaseSpec:
    """
    Report case specification.

    Notes
    -----
    This is intentionally simple and JSON/YAML-friendly:
    - `results_path` is the results.json path for this case
    - `input_case_path` is the MATPOWER/PGLib .m file path
    - `known_critical_pairs` is optional and used only for heuristic comparisons
      (top-k risky line recall).
    """

    case_id: str
    input_case_path: Path
    results_path: Path
    known_critical_pairs: tuple[tuple[int, int], ...] = ()


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


@dataclass(frozen=True)
class MarginVsRadiusArtifacts:
    """
    Per-line margin vs radius distribution artifacts (CSV + optional scatter plot).

    Paths are stored as *relative* paths intended to be embedded into the Markdown report.
    """

    radius_field: str
    csv_rel_path: str
    plot_rel_path: str | None
    plot_status: str  # ok | skipped_no_matplotlib | skipped_no_finite_points | error
    n_lines: int
    n_plotted: int
    skipped_nonfinite: int
    note: str = ""


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


def _csv_float(x: Any) -> str:
    """Deterministic float formatting for CSV artifacts (round-trip friendly)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)

    if math.isinf(xf):
        return "inf" if xf > 0 else "-inf"
    if math.isnan(xf):
        return "nan"
    return format(xf, ".17g")


def _save_margin_vs_radius_plot(
    *,
    out_path: Path,
    margins: np.ndarray,
    radii: np.ndarray,
    case_id: str,
    radius_field: str,
) -> None:
    """
    Save a scatter plot: x=margin_mw, y=<radius_field> for all finite points.

    Notes
    -----
    - Uses matplotlib if installed.
    - Uses a non-interactive backend when possible.
    """
    try:
        import matplotlib  # type: ignore

        try:
            matplotlib.use(
                "Agg"
            )  # must be called before importing pyplot in most cases
        except Exception:  # noqa: BLE001
            # Backend might already be set by the environment; proceed best-effort.
            pass

        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "matplotlib is required to generate plots for the report assets "
            "(install `matplotlib`)."
        ) from e

    x = np.asarray(margins, dtype=float).reshape(-1)
    y = np.asarray(radii, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("Internal error: margins and radii array size mismatch.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=160)
    ax.scatter(x, y, s=8, alpha=0.45, edgecolors="none")
    ax.set_xlabel("load margin (MW)")
    ax.set_ylabel(f"{radius_field} (MW)")
    ax.set_title(f"{case_id}: load margin vs {radius_field}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_margin_vs_radius_distribution(
    *,
    case_id: str,
    results: Dict[str, Any],
    net: Any,
    out_dir: Path,
    assets_rel_dir: str,
    radius_field: str = "radius_l2",
) -> MarginVsRadiusArtifacts:
    """
    Save per-line distribution artifacts: CSV + scatter plot (if possible).

    CSV columns
    ----------
    - line_idx, from_bus, to_bus
    - flow0_mw, p0_mw, p_limit_mw_est, margin_mw
    - <radius_field>

    Plot
    ----
    Scatter: x=margin_mw, y=<radius_field>, finite points only.

    Failure modes
    -------------
    - CSV write errors: raised to the caller (reported as error in Markdown).
    - Plot errors (missing matplotlib / no finite points): reported via plot_status, report still continues.
    """
    radius_field_eff = str(radius_field).strip()
    if not radius_field_eff:
        raise ValueError("radius_field must be a non-empty string.")

    assets_rel_dir_eff = str(assets_rel_dir).strip()
    if not assets_rel_dir_eff:
        raise ValueError("assets_rel_dir must be a non-empty relative directory name.")

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_name = f"{case_id}_margin_vs_{radius_field_eff}.csv"
    plot_name = f"{case_id}_margin_vs_{radius_field_eff}.png"

    csv_path = (out_dir / csv_name).resolve()
    plot_path = (out_dir / plot_name).resolve()

    csv_rel = f"{assets_rel_dir_eff}/{csv_name}"
    plot_rel = f"{assets_rel_dir_eff}/{plot_name}"

    line_indices = _sorted_line_indices(net)
    n_lines = int(len(line_indices))

    # Collect finite points for plotting.
    x_margin: list[float] = []
    y_radius: list[float] = []
    skipped = 0

    # Use stable ordering to make artifacts reproducible.
    line_tbl = net.line.loc[line_indices, ["from_bus", "to_bus"]]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "line_idx",
                "from_bus",
                "to_bus",
                "flow0_mw",
                "p0_mw",
                "p_limit_mw_est",
                "margin_mw",
                radius_field_eff,
            ]
        )

        for lid, fb, tb in line_tbl.itertuples():
            lid_i = int(lid)
            key = line_key(lid_i)
            row = results.get(key)
            if not isinstance(row, dict):
                raise KeyError(
                    f"results.json is missing required per-line entry: {key}"
                )

            try:
                flow0 = float(row.get("flow0_mw", float("nan")))
            except (TypeError, ValueError):
                flow0 = float("nan")
            try:
                p0 = float(
                    row.get(
                        "p0_mw",
                        abs(flow0) if math.isfinite(flow0) else float("nan"),
                    )
                )
            except (TypeError, ValueError):
                p0 = float("nan")
            try:
                limit = float(row.get("p_limit_mw_est", float("nan")))
            except (TypeError, ValueError):
                limit = float("nan")

            if "margin_mw" in row:
                try:
                    margin = float(row.get("margin_mw", float("nan")))
                except (TypeError, ValueError):
                    margin = float("nan")
            else:
                margin = (
                    float(limit - abs(flow0))
                    if (math.isfinite(limit) and math.isfinite(flow0))
                    else float("nan")
                )

            try:
                radius = float(row.get(radius_field_eff, float("nan")))
            except (TypeError, ValueError):
                radius = float("nan")

            writer.writerow(
                [
                    str(lid_i),
                    str(int(fb)),
                    str(int(tb)),
                    _csv_float(flow0),
                    _csv_float(p0),
                    _csv_float(limit),
                    _csv_float(margin),
                    _csv_float(radius),
                ]
            )

            if math.isfinite(margin) and math.isfinite(radius):
                x_margin.append(float(margin))
                y_radius.append(float(radius))
            else:
                skipped += 1

    logger.info(
        "case=%s: saved per-line margin vs %s CSV (%d lines): %s",
        str(case_id),
        radius_field_eff,
        n_lines,
        str(csv_path),
    )

    note = ""
    plot_status = "skipped_no_matplotlib"
    plot_rel_path: str | None = None

    if x_margin:
        try:
            _save_margin_vs_radius_plot(
                out_path=plot_path,
                margins=np.asarray(x_margin, dtype=float),
                radii=np.asarray(y_radius, dtype=float),
                case_id=str(case_id),
                radius_field=radius_field_eff,
            )
            plot_status = "ok"
            plot_rel_path = plot_rel
            logger.info(
                "case=%s: saved per-line margin vs %s plot (%d points): %s",
                str(case_id),
                radius_field_eff,
                int(len(x_margin)),
                str(plot_path),
            )
        except ImportError as e:
            plot_status = "skipped_no_matplotlib"
            note = str(e)
            logger.error(
                "case=%s: plot skipped (matplotlib missing). Install matplotlib to enable plots.",
                str(case_id),
            )
        except Exception as e:  # noqa: BLE001
            plot_status = "error"
            note = f"{type(e).__name__}: {e}"
            logger.exception(
                "case=%s: failed to generate plot margin vs %s.",
                str(case_id),
                radius_field_eff,
            )
    else:
        plot_status = "skipped_no_finite_points"
        note = "no finite (margin, radius) points to plot"
        logger.warning(
            "case=%s: plot skipped (no finite points). Non-finite points skipped=%d/%d.",
            str(case_id),
            int(skipped),
            int(n_lines),
        )

    return MarginVsRadiusArtifacts(
        radius_field=radius_field_eff,
        csv_rel_path=csv_rel,
        plot_rel_path=plot_rel_path,
        plot_status=str(plot_status),
        n_lines=int(n_lines),
        n_plotted=int(len(x_margin)),
        skipped_nonfinite=int(skipped),
        note=str(note),
    )


def _results_config_matches(
    *,
    results: Dict[str, Any],
    expected_dc_mode: str,
    expected_slack_bus: int,
    expected_compute_nminus1: bool,
    expected_inj_std_mw: float,
    expected_opf_headroom_factor: float,
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

    try:
        inj_std = float(meta.get("inj_std_mw", float("nan")))
    except (TypeError, ValueError):
        inj_std = float("nan")
    if not math.isfinite(inj_std) or abs(inj_std - float(expected_inj_std_mw)) > 1e-12:
        return False, f"inj_std_mw={inj_std!r} != {float(expected_inj_std_mw)!r}"

    try:
        head = float(meta.get("opf_headroom_factor", float("nan")))
    except (TypeError, ValueError):
        head = float("nan")
    if (not math.isfinite(head)) or abs(
        head - float(expected_opf_headroom_factor)
    ) > 1e-12:
        return (
            False,
            f"opf_headroom_factor={head!r} != {float(expected_opf_headroom_factor)!r}",
        )

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
    margin_vs_radius: MarginVsRadiusArtifacts | None = None,
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

    if margin_vs_radius is not None:
        lines.append("### Per-line distribution (load margin vs stability radius)")
        lines.append("")
        lines.append(f"- radius_field: `{margin_vs_radius.radius_field}`")
        if margin_vs_radius.csv_rel_path:
            lines.append(f"- table_csv: `{margin_vs_radius.csv_rel_path}`")
        else:
            lines.append("- table_csv: n/a")

        if margin_vs_radius.plot_rel_path:
            lines.append(
                "- plot_png: "
                f"`{margin_vs_radius.plot_rel_path}` "
                f"(points={margin_vs_radius.n_plotted}/{margin_vs_radius.n_lines}, "
                f"skipped_nonfinite={margin_vs_radius.skipped_nonfinite})"
            )
            lines.append("")
            lines.append(
                f"![{case} margin_vs_{margin_vs_radius.radius_field}]({margin_vs_radius.plot_rel_path})"
            )
        else:
            note = f", note={margin_vs_radius.note}" if margin_vs_radius.note else ""
            lines.append(
                f"- plot_png: n/a (status={margin_vs_radius.plot_status}{note})"
            )
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
        lines.append(f"- compute time_sec: **{time_sec:.3f}**")
    else:
        lines.append("- compute time_sec: n/a")
    lines.append("")

    return "\n".join(lines)


def generate_report_text(
    *,
    cases: Sequence[ReportCaseSpec],
    n_samples: int,
    seed: int,
    chunk_size: int,
    generate_missing_results: bool,
    compute_dc_mode: str,
    compute_slack_bus: int,
    compute_compute_nminus1: bool,
    compute_dc_chunk_size: int = 256,
    compute_dc_dtype: np.dtype = np.float64,
    compute_inj_std_mw: float = 1.0,
    compute_nminus1_update_sensitivities: bool = True,
    compute_nminus1_islanding: str = "skip",
    compute_opf_cfg: OPFConfig | None = None,
    compute_opf_dc_flow_consistency_tol_mw: float = 1e-3,
    compute_opf_bus_balance_tol_mw: float = 1e-6,
    mc_feas_tol_mw: float = DEFAULT_MC.feas_tol_mw,
    mc_cert_tol_mw: float = DEFAULT_MC.cert_tol_mw,
    mc_cert_max_samples: int = DEFAULT_MC.cert_max_samples,
    path_base_dir: Path | None = None,
    per_line_artifacts_dir: Path | None = None,
    per_line_radius_field: str = "radius_l2",
    per_line_assets_rel_dir: str | None = None,
) -> str:
    """
    Generate verification report markdown as a string.

    Parameters
    ----------
    cases:
        Case list (configured in YAML). No hard-coded cases in code.
    path_base_dir:
        Base directory for resolving relative paths in meta/input specs.
        If None, uses current working directory.
    per_line_artifacts_dir:
        If provided, saves per-line artifacts (margin vs radius CSV and plot) into this directory.
    per_line_radius_field:
        Which radius field to use for the per-line scatter plot and CSV (default: radius_l2).
    per_line_assets_rel_dir:
        Relative directory name used in Markdown links to artifacts. If None, uses `per_line_artifacts_dir.name`.
    """
    base_dir = (
        Path(path_base_dir).resolve() if path_base_dir is not None else Path.cwd()
    )
    compute_dc_mode_eff = str(compute_dc_mode).strip().lower()
    if compute_dc_mode_eff not in ("materialize", "operator"):
        raise ValueError("compute_dc_mode must be materialize|operator")

    opf_cfg = compute_opf_cfg if compute_opf_cfg is not None else DEFAULT_OPF

    if not cases:
        raise ValueError("report requires a non-empty cases list.")

    assets_dir: Path | None = None
    assets_rel_dir: str | None = None
    if per_line_artifacts_dir is not None:
        assets_dir = Path(per_line_artifacts_dir).resolve()
        assets_dir.mkdir(parents=True, exist_ok=True)
        assets_rel_dir = (
            str(per_line_assets_rel_dir).strip()
            if per_line_assets_rel_dir
            else assets_dir.name
        )
        if not assets_rel_dir:
            raise ValueError("per_line_assets_rel_dir resolved to an empty string.")

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
        "- **Probabilistic metrics** (soft): Δp is Gaussian in the balanced subspace (slack-invariant)."
    )
    if assets_dir is not None:
        out.append("")
        out.append("### Saved per-line artifacts (this report)")
        out.append("")
        out.append(
            f"- Per-case CSV + scatter plot for **load margin vs `{str(per_line_radius_field).strip()}`** "
            f"are saved under: `{assets_rel_dir}`"
        )
    out.append("")
    out.append("---")
    out.append("")

    for case_spec in cases:
        case = str(case_spec.case_id)
        rp = Path(case_spec.results_path).expanduser()
        if not rp.is_absolute():
            rp = (base_dir / rp).resolve()

        ip_fallback = Path(case_spec.input_case_path).expanduser()
        if not ip_fallback.is_absolute():
            ip_fallback = (base_dir / ip_fallback).resolve()

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
                    expected_dc_mode=str(compute_dc_mode_eff),
                    expected_slack_bus=int(compute_slack_bus),
                    expected_compute_nminus1=bool(compute_compute_nminus1),
                    expected_inj_std_mw=float(compute_inj_std_mw),
                    expected_opf_headroom_factor=float(opf_cfg.headroom_factor),
                )
                if not ok_cfg:
                    logger.info("%s: results.json config mismatch (%s).", case, msg_cfg)
                    results = None
                    status_reasons.append(f"config_mismatch:{msg_cfg}")

            if results is None and generate_missing_results:
                try:
                    ip_eff = Path(ensure_case_file(str(ip_fallback))).resolve()
                except Exception as e:  # noqa: BLE001
                    logger.exception(
                        "Failed to ensure/download input case file for %s: %s",
                        case,
                        str(ip_fallback),
                    )
                    status_reasons.append(f"missing_input:{type(e).__name__}")
                else:
                    results = compute_results_for_case(
                        input_path=str(ip_eff),
                        slack_bus=int(compute_slack_bus),
                        dc_mode=str(compute_dc_mode_eff),
                        dc_chunk_size=int(compute_dc_chunk_size),
                        dc_dtype=compute_dc_dtype,
                        inj_std_mw=float(compute_inj_std_mw),
                        compute_nminus1=bool(compute_compute_nminus1),
                        nminus1_update_sensitivities=bool(
                            compute_nminus1_update_sensitivities
                        ),
                        nminus1_islanding=str(compute_nminus1_islanding),
                        opf_cfg=opf_cfg,
                        opf_dc_flow_consistency_tol_mw=float(
                            compute_opf_dc_flow_consistency_tol_mw
                        ),
                        opf_bus_balance_tol_mw=float(compute_opf_bus_balance_tol_mw),
                        path_base_dir=base_dir,
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
                    margin_vs_radius=None,
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

        # Resolve input case file (prefer meta if it exists, else spec).
        ip_eff = (
            Path(meta.get("input_path", "")).expanduser()
            if isinstance(meta.get("input_path", ""), str)
            else ip_fallback
        )
        if not ip_eff.is_absolute():
            ip_eff = (base_dir / ip_eff).resolve()

        if not ip_eff.exists():
            try:
                ip_eff = Path(ensure_case_file(str(ip_fallback))).resolve()
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
                        margin_vs_radius=None,
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

        # Save per-line artifacts: margin vs radius scatter/CSV
        margin_vs_radius: MarginVsRadiusArtifacts | None = None
        if assets_dir is not None and assets_rel_dir is not None:
            try:
                margin_vs_radius = save_margin_vs_radius_distribution(
                    case_id=case,
                    results=results,
                    net=net,
                    out_dir=assets_dir,
                    assets_rel_dir=assets_rel_dir,
                    radius_field=str(per_line_radius_field),
                )
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "case=%s: failed to generate per-line artifacts (margin vs radius).",
                    case,
                )
                margin_vs_radius = MarginVsRadiusArtifacts(
                    radius_field=str(per_line_radius_field),
                    csv_rel_path="",
                    plot_rel_path=None,
                    plot_status="error",
                    n_lines=int(len(getattr(net, "line", []))),
                    n_plotted=0,
                    skipped_nonfinite=0,
                    note=f"{type(e).__name__}: {e}",
                )

        comparisons: dict[str, Any] = {}

        try:
            top10 = _top_k_risky_lines(
                results=results, net=net, k=10, radius_field="radius_l2"
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Top-k extraction failed for %s", case)
            top10 = []
            comparisons["top_risky_status"] = f"error:{type(e).__name__}"

        known_pairs = list(case_spec.known_critical_pairs)
        if known_pairs and top10:
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
        elif known_pairs:
            comparisons["top_risky_status"] = "no_top_lines"

        try:
            n1_stats = _nminus1_critical_match_stats(results=results, net=net)
            comparisons["nminus1_match_status"] = n1_stats.status
            comparisons["nminus1_match_percent"] = n1_stats.match_percent
        except Exception as e:  # noqa: BLE001
            logger.exception("N-1 match stats failed for %s", case)
            comparisons["nminus1_match_status"] = f"error:{type(e).__name__}"

        with log_stage(logger, f"{case}: Monte Carlo verification"):
            vr = run_monte_carlo_verification(
                results_path=rp,
                input_case_path=ip_eff,
                slack_bus=int(compute_slack_bus),
                n_samples=int(n_samples),
                seed=int(seed),
                chunk_size=int(chunk_size),
                feas_tol_mw=float(mc_feas_tol_mw),
                cert_tol_mw=float(mc_cert_tol_mw),
                cert_max_samples=int(mc_cert_max_samples),
            )

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
                margin_vs_radius=margin_vs_radius,
            )
        )

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
    out.append("---")
    out.append("")
    out.extend(sections)

    return "\n".join(out) + "\n"
