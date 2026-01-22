from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

# Ensure repository root is importable (needed for `import verification.*`).
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stability_radius.dc.dc_model import build_dc_matrices, build_dc_operator
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.common import get_line_base_quantities
from stability_radius.radii.l2 import compute_l2_radius
from stability_radius.radii.metric import compute_metric_radius
from stability_radius.radii.nminus1 import compute_nminus1_l2_radius
from stability_radius.radii.probabilistic import (
    compute_sigma_radius,
    overload_probability_symmetric_limit,
    sigma_radius,
)
from stability_radius.statistics.table import (
    format_radius_summary,
    format_results_csv,
    format_results_table,
)
from stability_radius.utils import setup_logging
from stability_radius.utils.download import download_ieee_case, download_pglib_opf_case

logger = logging.getLogger("power_stability_radius")

_DEFAULT_TABLE_COLUMNS = (
    "flow0_mw",
    "p0_mw",
    "p_limit_mw_est",
    "margin_mw",
    "norm_g",
    "metric_denom",
    "sigma_flow",
    "radius_l2",
    "radius_metric",
    "radius_sigma",
    "overload_probability",
    "radius_nminus1",
    "worst_contingency",
    "worst_contingency_line_idx",
)

# Explicit scalability thresholds (logged, deterministic).
_THRESHOLD_H_ENTRIES = 20_000_000
_THRESHOLD_NMINUS1_LINES = 5_000


def _make_logging_cfg(*, runs_dir: str, level_console: str, level_file: str) -> Any:
    """Build a minimal config object compatible with `stability_radius.utils.setup_logging()`."""
    return SimpleNamespace(
        paths=SimpleNamespace(runs_dir=runs_dir),
        settings=SimpleNamespace(
            logging=SimpleNamespace(level_console=level_console, level_file=level_file)
        ),
    )


def _resolve_under_project_root(p: str) -> str:
    """Resolve a potentially-relative path under the repository root."""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())


def _infer_case_number_from_path(path: str, default: int = 30) -> int:
    """Infer IEEE/MATPOWER case number from filename (first integer substring)."""
    match = re.search(r"(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else default


def _ensure_input_case_file(input_path: str) -> str:
    """
    Ensure the input MATPOWER/PGLib case file exists, downloading it if needed.

    Supported sources:
    - PGLib-OPF: pglib_opf_case*.m
    - MATPOWER IEEE: case{N}.m

    Logging
    -------
    - Existing file: DEBUG (noise for typical CLI usage)
    - Downloads: INFO (important)
    """
    target_path = _resolve_under_project_root(str(input_path))
    if os.path.exists(target_path):
        logger.debug("Using existing input file: %s", target_path)
        return target_path

    base = os.path.basename(target_path)
    if re.match(r"^pglib_opf_case\d+_.*\.m$", base):
        logger.info("Input file not found; downloading PGLib-OPF case: %s", base)
        return download_pglib_opf_case(case_filename=base, target_path=target_path)

    case_number = _infer_case_number_from_path(target_path, default=30)
    logger.info(
        "Input file not found; downloading MATPOWER IEEE case%d into %s",
        case_number,
        target_path,
    )
    return download_ieee_case(case_number=case_number, target_path=target_path)


def _line_like_sort_key(k: str) -> tuple[int, int, str]:
    """Deterministic ordering for per-line keys plus auxiliary keys."""
    if k.startswith("line_"):
        try:
            return (0, int(k.split("_", 1)[1]), k)
        except Exception:
            return (0, 10**18, k)
    return (1, 10**18, k)


def _merge_line_results(*dicts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge multiple per-line result dictionaries (same 'line_<idx>' keys)."""
    keys: set[str] = set()
    for d in dicts:
        keys.update(d.keys())

    merged: dict[str, dict[str, Any]] = {}
    for k in sorted(keys, key=_line_like_sort_key):
        merged[k] = {}
        for d in dicts:
            if k in d:
                merged[k].update(d[k])
    return merged


def _parse_columns(value: str) -> tuple[str, ...]:
    """Parse comma-separated columns string."""
    if not value.strip():
        return tuple(_DEFAULT_TABLE_COLUMNS)
    return tuple(x.strip() for x in value.split(",") if x.strip())


def _dtype_from_str(s: str) -> np.dtype:
    ss = str(s).strip().lower()
    if ss in ("float64", "f64"):
        return np.float64
    if ss in ("float32", "f32"):
        return np.float32
    raise ValueError("dc-dtype must be float64 or float32.")


def _compute_radii_operator_path(
    *,
    dc_op,
    base,
    inj_std_mw: float,
    dc_chunk_size: int,
) -> dict[str, dict[str, Any]]:
    """
    Compute L2/metric/sigma radii without materializing H_full (operator path).

    Notes
    -----
    - Uses ||g_l||_2 from LU solves (chunked).
    - Uses M=I => radius_metric == radius_l2.
    - For Sigma = inj_std^2 I: sigma_flow = inj_std * ||g_l||_2.
    """
    if dc_chunk_size <= 0:
        raise ValueError("dc_chunk_size must be positive.")

    op_line_ids = list(getattr(dc_op, "line_ids", ()))
    if op_line_ids and list(base.line_indices) != [int(x) for x in op_line_ids]:
        raise ValueError(
            "Line ordering mismatch between DC operator and base quantities. "
            "This indicates an internal consistency bug."
        )

    norms = dc_op.row_norms_l2(chunk_size=int(dc_chunk_size))
    if norms.shape != (len(base.line_indices),):
        raise ValueError("Unexpected norms shape from DC operator.")

    out: dict[str, dict[str, Any]] = {}
    for pos, lid in enumerate(base.line_indices):
        margin = float(base.margin_mw[pos])
        norm_g = float(norms[pos])

        r_l2 = float(margin / norm_g) if norm_g > 1e-12 else float("inf")

        # Metric with M=I:
        metric_denom = norm_g
        r_metric = r_l2

        sigma_flow = float(inj_std_mw) * norm_g
        r_sigma = sigma_radius(margin, sigma_flow)

        c = float(base.limit_mw_est[pos])
        f0 = float(base.flow0_mw[pos])
        prob = overload_probability_symmetric_limit(flow0=f0, limit=c, sigma=sigma_flow)

        k = f"line_{int(lid)}"
        out[k] = {
            "flow0_mw": float(base.flow0_mw[pos]),
            "p0_mw": float(base.p0_abs_mw[pos]),
            "p_limit_mw_est": float(base.limit_mw_est[pos]),
            "margin_mw": margin,
            "norm_g": norm_g,
            "radius_l2": r_l2,
            "metric_denom": metric_denom,
            "radius_metric": float(r_metric),
            "sigma_flow": float(sigma_flow),
            "radius_sigma": float(r_sigma),
            "overload_probability": float(prob),
            # N-1 is not computed in operator path for scalability.
            "radius_nminus1": float("nan"),
            "worst_contingency": -1,
            "worst_contingency_line_idx": -1,
        }
    return out


def compute_results_for_case(
    *,
    input_path: str,
    slack_bus: int,
    pf_mode: str,
    dc_mode: str,
    dc_chunk_size: int,
    dc_dtype: np.dtype,
    margin_factor: float,
    inj_std_mw: float,
    nminus1_update_sensitivities: bool,
    nminus1_islanding: str,
) -> dict[str, Any]:
    """
    Compute all per-line radii and return a single results dict (including '__meta__').

    This is the core computation used by:
    - CLI 'demo' (writes into runs/<timestamp>/results.json)
    - verification report auto-generation (writes into verification/results/*.json)

    Important
    ---------
    - This function does NOT configure logging and does NOT write any files.
    - It DOES ensure the input case file exists (downloads if missing).

    Logging
    -------
    Normal pipeline step logs are DEBUG to keep CLI output compact; high-level timing is INFO.
    """
    time_start = time.time()

    input_path_abs = _ensure_input_case_file(str(input_path))
    net = load_network(input_path_abs)

    if margin_factor <= 0:
        raise ValueError("margin_factor must be positive.")
    if inj_std_mw <= 0:
        raise ValueError("inj_std_mw must be positive.")
    if dc_chunk_size <= 0:
        raise ValueError("dc_chunk_size must be positive.")

    pf_mode_eff = str(pf_mode).strip().lower()
    if pf_mode_eff not in ("ac", "dc"):
        raise ValueError("pf_mode must be 'ac' or 'dc'.")

    dc_mode_req = str(dc_mode).strip().lower()
    if dc_mode_req not in ("auto", "materialize", "operator"):
        raise ValueError("dc_mode must be auto|materialize|operator")

    logger.debug(
        "Running base PF (pf_mode=%s) and extracting base quantities...", pf_mode_eff
    )
    base = get_line_base_quantities(
        net, margin_factor=float(margin_factor), pf_mode=pf_mode_eff
    )

    n_bus_guess = int(len(net.bus))
    m_line_guess = int(len(net.line))
    entries = int(n_bus_guess) * int(m_line_guess)

    if dc_mode_req == "auto":
        dc_mode_eff = "materialize" if entries <= _THRESHOLD_H_ENTRIES else "operator"
    else:
        dc_mode_eff = dc_mode_req

    logger.debug(
        "DC mode: requested=%s, effective=%s (n_bus=%d, n_line=%d, m*n=%d, threshold=%d)",
        dc_mode_req,
        dc_mode_eff,
        n_bus_guess,
        m_line_guess,
        entries,
        _THRESHOLD_H_ENTRIES,
    )

    dc_op = None
    H_full = None

    if dc_mode_eff == "materialize":
        # Respect dtype and chunk_size via build_dc_matrices parameters.
        H_full, dc_op = build_dc_matrices(
            net, slack_bus=int(slack_bus), chunk_size=int(dc_chunk_size), dtype=dc_dtype
        )
        n_bus = int(H_full.shape[1])
        m_line = int(H_full.shape[0])
        logger.debug(
            "Materialized H_full: shape=(%d,%d), dtype=%s", m_line, n_bus, H_full.dtype
        )
    else:
        dc_op = build_dc_operator(net, slack_bus=int(slack_bus))
        n_bus = int(dc_op.n_bus)
        m_line = int(dc_op.n_line)
        logger.debug("Built DC operator: n_bus=%d, n_line=%d", n_bus, m_line)

    if H_full is not None:
        l2 = compute_l2_radius(
            net, H_full, margin_factor=float(margin_factor), base=base
        )

        M = np.eye(n_bus, dtype=float)
        metric = compute_metric_radius(
            net, H_full, M, margin_factor=float(margin_factor), base=base
        )

        Sigma_diag = (float(inj_std_mw) ** 2) * np.ones(n_bus, dtype=float)
        sigma = compute_sigma_radius(
            net, H_full, Sigma_diag, margin_factor=float(margin_factor), base=base
        )

        nminus1_computed = False
        if m_line <= _THRESHOLD_NMINUS1_LINES:
            nminus1 = compute_nminus1_l2_radius(
                net,
                H_full,
                margin_factor=float(margin_factor),
                update_sensitivities=bool(nminus1_update_sensitivities),
                islanding=str(nminus1_islanding),
                base=base,
            )
            nminus1_computed = True
        else:
            logger.warning(
                "Skipping N-1 computation: n_line=%d exceeds threshold=%d",
                m_line,
                _THRESHOLD_NMINUS1_LINES,
            )
            nminus1 = {
                f"line_{int(lid)}": {
                    "radius_nminus1": float("nan"),
                    "worst_contingency": -1,
                    "worst_contingency_line_idx": -1,
                }
                for lid in base.line_indices
            }

        results_lines = _merge_line_results(l2, metric, sigma, nminus1)
    else:
        if dc_op is None:
            raise AssertionError("Internal error: operator mode requires dc_op")
        results_lines = _compute_radii_operator_path(
            dc_op=dc_op,
            base=base,
            inj_std_mw=float(inj_std_mw),
            dc_chunk_size=int(dc_chunk_size),
        )
        nminus1_computed = False

    elapsed_sec = float(time.time() - time_start)
    logger.info("Compute time (PF + DC + radii): %.3f sec", elapsed_sec)

    results: dict[str, Any] = {
        "__meta__": {
            "input_path": str(input_path_abs),
            "slack_bus": int(slack_bus),
            "pf_mode": pf_mode_eff,
            "dc_mode_effective": dc_mode_eff,
            "n_bus": int(n_bus),
            "n_line": int(m_line),
            "compute_time_sec": elapsed_sec,
            "nminus1_computed": bool(nminus1_computed),
        }
    }
    results.update(results_lines)
    return results


def run_demo(args: argparse.Namespace) -> int:
    """
    Run the end-to-end demo workflow and save results into runs/<timestamp>/.

    This wrapper:
    - configures logging
    - computes results using `compute_results_for_case(...)`
    - writes results.json + optional CSV
    - optionally exports results.json to a fixed path
    """
    run_dir = setup_logging(
        _make_logging_cfg(
            runs_dir=str(args.runs_dir),
            level_console=str(args.log_level),
            level_file=str(args.log_file_level),
        )
    )
    run_dir_path = Path(run_dir)

    results = compute_results_for_case(
        input_path=str(args.input),
        slack_bus=int(args.slack_bus),
        pf_mode=str(args.pf_mode),
        dc_mode=str(args.dc_mode),
        dc_chunk_size=int(args.dc_chunk_size),
        dc_dtype=_dtype_from_str(str(args.dc_dtype)),
        margin_factor=float(args.margin_factor),
        inj_std_mw=float(args.inj_std_mw),
        nminus1_update_sensitivities=bool(args.nminus1_update_sensitivities),
        nminus1_islanding=str(args.nminus1_islanding),
    )

    results_path = run_dir_path / "results.json"
    results_path.write_text(
        json.dumps(results, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Results written: %s", str(results_path))

    columns = _parse_columns(str(args.table_columns))
    max_rows = int(args.max_rows) if args.max_rows is not None else None

    table_str = format_results_table(results, columns=columns, max_rows=max_rows)
    # Ensure the table never appears in console output.
    logger.debug("Results table:\n%s", table_str, extra={"sr_only_file": True})

    summaries = [
        format_radius_summary(results, radius_field="radius_l2"),
        format_radius_summary(results, radius_field="radius_metric"),
        format_radius_summary(results, radius_field="radius_sigma"),
        format_radius_summary(results, radius_field="radius_nminus1"),
    ]
    for s in summaries:
        logger.info("%s", s)

    if bool(args.save_csv):
        csv_path = run_dir_path / "results_table.csv"
        csv_str = format_results_csv(results, columns=columns, max_rows=max_rows)
        csv_path.write_text(csv_str, encoding="utf-8")
        logger.info("Saved CSV: %s", str(csv_path))

    if str(args.export_results).strip():
        export_path_abs = _resolve_under_project_root(str(args.export_results))
        Path(export_path_abs).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(results_path, export_path_abs)
        logger.info("Exported results to: %s", export_path_abs)

    logger.info("Done. Run directory: %s", str(run_dir_path))
    return 0


def run_monte_carlo(args: argparse.Namespace) -> int:
    """Run Monte Carlo coverage estimation using stored results and a case file."""
    run_dir = setup_logging(
        _make_logging_cfg(
            runs_dir=str(args.runs_dir),
            level_console=str(args.log_level),
            level_file=str(args.log_file_level),
        )
    )
    _ = run_dir  # run dir is used by logging side effects

    from verification.monte_carlo import estimate_coverage_percent  # local import

    stats = estimate_coverage_percent(
        results_path=Path(str(args.results)),
        input_case_path=Path(str(args.input)),
        slack_bus=int(args.slack_bus),
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


def run_report(args: argparse.Namespace) -> int:
    """Generate aggregated verification report (Markdown)."""
    run_dir = setup_logging(
        _make_logging_cfg(
            runs_dir=str(args.runs_dir),
            level_console=str(args.log_level),
            level_file=str(args.log_file_level),
        )
    )
    run_dir_path = Path(run_dir)

    from verification.generate_report import generate_report_text  # local import

    results_dir = Path(_resolve_under_project_root(str(args.results_dir)))
    out_path = Path(_resolve_under_project_root(str(args.out)))

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

    run_copy = run_dir_path / "verification_report.md"
    run_copy.write_text(report_text, encoding="utf-8")
    logger.info("Wrote report copy: %s", str(run_copy))

    return 0


def run_table(args: argparse.Namespace) -> int:
    """Proxy to `python -m stability_radius.statistics.table`."""
    from stability_radius.statistics.table import main as table_main

    argv: list[str] = [str(args.results_json)]
    if args.max_rows is not None:
        argv += ["--max-rows", str(args.max_rows)]
    if str(args.radius_field).strip():
        argv += ["--radius-field", str(args.radius_field)]
    if str(args.columns).strip():
        argv += ["--columns", str(args.columns)]
    if str(args.table_out).strip():
        argv += ["--table-out", str(args.table_out)]
    if str(args.csv_out).strip():
        argv += ["--csv-out", str(args.csv_out)]
    return int(table_main(argv))


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for the unified entrypoint."""
    parser = argparse.ArgumentParser(
        prog="power_stability_radius",
        description="Unified entrypoint for demo runs and verification workflows.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory where per-run folders and run.log are created.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Console logging level (INFO/DEBUG/WARNING/ERROR).",
    )
    parser.add_argument(
        "--log-file-level",
        type=str,
        default="DEBUG",
        help="File logging level (DEBUG recommended to include the full results table).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run the end-to-end demo on a single case.")
    p_demo.add_argument(
        "--input",
        type=str,
        default="data/input/pglib_opf_case30_ieee.m",
        help="Path to MATPOWER/PGLib .m case file (downloaded if missing).",
    )
    p_demo.add_argument("--slack-bus", type=int, default=0)
    p_demo.add_argument("--pf-mode", type=str, default="ac", choices=("ac", "dc"))
    p_demo.add_argument(
        "--dc-mode",
        type=str,
        default="auto",
        choices=("auto", "materialize", "operator"),
    )
    p_demo.add_argument("--dc-chunk-size", type=int, default=256)
    p_demo.add_argument(
        "--dc-dtype", type=str, default="float64", choices=("float64", "float32")
    )

    p_demo.add_argument("--margin-factor", type=float, default=1.0)
    p_demo.add_argument("--inj-std-mw", type=float, default=1.0)

    p_demo.add_argument(
        "--nminus1-update-sensitivities",
        type=int,
        default=1,
        help="1 to update sensitivities (more accurate), 0 to reuse base sensitivities.",
    )
    p_demo.add_argument(
        "--nminus1-islanding",
        type=str,
        default="skip",
        choices=("skip", "raise"),
        help="How to handle islanding-like contingencies when LODF is undefined.",
    )
    p_demo.add_argument(
        "--export-results",
        type=str,
        default="",
        help="Optional path to copy results.json (useful for verification workflows).",
    )
    p_demo.add_argument(
        "--save-csv",
        type=int,
        default=1,
        help="1 to save results_table.csv, 0 to skip.",
    )
    p_demo.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit number of rows in the logged/saved table.",
    )
    p_demo.add_argument(
        "--table-columns",
        type=str,
        default="",
        help="Comma-separated list of table columns (default: full set).",
    )

    p_mc = sub.add_parser("monte-carlo", help="Monte Carlo coverage for one case.")
    p_mc.add_argument("--results", required=True, type=str, help="Path to results.json")
    p_mc.add_argument("--input", required=True, type=str, help="Path to case .m file")
    p_mc.add_argument("--slack-bus", default=0, type=int)
    p_mc.add_argument("--n-samples", default=50_000, type=int)
    p_mc.add_argument("--seed", default=0, type=int)
    p_mc.add_argument("--chunk-size", default=256, type=int)

    p_rep = sub.add_parser("report", help="Generate aggregated verification report.")
    p_rep.add_argument(
        "--results-dir",
        default="verification/results",
        type=str,
        help="Directory containing per-case results JSON files.",
    )
    p_rep.add_argument(
        "--out",
        default="verification/report.md",
        type=str,
        help="Output report path.",
    )
    p_rep.add_argument("--n-samples", default=50_000, type=int)
    p_rep.add_argument("--seed", default=0, type=int)
    p_rep.add_argument("--chunk-size", default=256, type=int)

    p_rep.add_argument(
        "--generate-missing-results",
        type=int,
        default=1,
        help=(
            "1: if verification/results/<case>.json is missing, compute it automatically "
            "(also auto-downloads input cases). 0: do not compute missing results."
        ),
    )
    p_rep.add_argument("--demo-pf-mode", type=str, default="dc", choices=("ac", "dc"))
    p_rep.add_argument(
        "--demo-dc-mode",
        type=str,
        default="auto",
        choices=("auto", "materialize", "operator"),
    )
    p_rep.add_argument("--demo-slack-bus", type=int, default=0)

    p_tab = sub.add_parser(
        "table", help="Print/export a table from an existing results.json."
    )
    p_tab.add_argument("results_json", type=str, help="Path to results.json")
    p_tab.add_argument("--max-rows", type=int, default=None)
    p_tab.add_argument("--radius-field", type=str, default="radius_l2")
    p_tab.add_argument("--columns", type=str, default="")
    p_tab.add_argument("--table-out", type=str, default="")
    p_tab.add_argument("--csv-out", type=str, default="")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "demo":
        return run_demo(args)
    if args.command == "monte-carlo":
        return run_monte_carlo(args)
    if args.command == "report":
        return run_report(args)
    if args.command == "table":
        return run_table(args)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
