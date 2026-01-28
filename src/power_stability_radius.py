from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Ensure repository root is importable (needed for `import verification.*`).
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from stability_radius.config import (
    DEFAULT_DC,
    DEFAULT_LOGGING,
    DEFAULT_MC,
    DEFAULT_OPF,
    DEFAULT_TABLE_COLUMNS,
    DEFAULT_NMINUS1_ISLANDING,
    LoggingConfig,
)
from stability_radius.dc.dc_model import build_dc_matrices, build_dc_operator
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.common import (
    assert_line_limit_sources_present,
    get_line_base_quantities,
)
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
from stability_radius.utils import log_stage, setup_logging

logger = logging.getLogger("stability_radius.cli")


def _resolve_under_project_root(p: str) -> str:
    """Resolve a potentially-relative path under the repository root."""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())


def _ensure_input_case_file(input_path: str) -> str:
    """
    Ensure input case file exists (download if missing and supported).

    Deterministic behavior
    ----------------------
    - No implicit path guessing beyond project-root resolution.
    - If the file is missing AND the filename matches a supported public dataset,
      the file is downloaded deterministically (stable URL ordering):
        * MATPOWER: case<N>.m / ieee<N>.m
        * PGLib-OPF: pglib_opf_*.m

    Raises
    ------
    FileNotFoundError:
        If the file does not exist and its name is not supported for deterministic download.
    RuntimeError:
        If the file name is supported but download failed.
    """
    target_path = _resolve_under_project_root(str(input_path))
    if os.path.exists(target_path):
        logger.debug("Using input file: %s", target_path)
        return target_path

    logger.info(
        "Input case file missing: %s. Trying deterministic download...", target_path
    )

    try:
        from stability_radius.utils.download import ensure_case_file
    except Exception as e:  # noqa: BLE001
        # We keep the error explicit: downloading is an optional feature, but requested by the user.
        raise RuntimeError(
            "Case file is missing and download helpers are unavailable. "
            "Install optional dependencies (e.g., requests) or provide an existing file."
        ) from e

    try:
        ensured = str(ensure_case_file(target_path))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Input case file does not exist: {target_path}. "
            "Auto-download supports only: case<N>.m / ieee<N>.m / pglib_opf_*.m. "
            "Provide an explicit path to an existing MATPOWER/PGLib .m case file."
        ) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to download missing case file: %s", target_path)
        raise RuntimeError(f"Failed to download input case file: {target_path}") from e

    if not os.path.exists(ensured):
        raise RuntimeError(
            f"Internal error: ensure_case_file() returned a non-existent path: {ensured}"
        )

    logger.info("Downloaded case file: %s", ensured)
    return ensured


def _line_like_sort_key(k: str) -> tuple[int, int, str]:
    """Deterministic ordering for per-line keys plus auxiliary keys."""
    if k.startswith("line_"):
        try:
            return (0, int(k.split("_", 1)[1]), k)
        except ValueError:
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
        return tuple(DEFAULT_TABLE_COLUMNS)
    return tuple(x.strip() for x in value.split(",") if x.strip())


def _dtype_from_str(s: str) -> np.dtype:
    ss = str(s).strip().lower()
    if ss in ("float64", "f64"):
        return np.float64
    if ss in ("float32", "f32"):
        return np.float32
    raise ValueError("dc-dtype must be float64 or float32.")


def _run_self_tests(*, project_root: Path) -> int:
    """
    Run repository tests (pytest) inside the current Python process.

    Notes
    -----
    - Deterministic: tests dir only.
    """
    try:
        import pytest
    except ImportError as e:
        raise ImportError(
            "pytest is required to run self-tests. Install dev dependencies or run with --run-tests 0."
        ) from e

    tests_dir = project_root / "tests"
    if not tests_dir.is_dir():
        raise FileNotFoundError(
            f"Tests directory not found: {tests_dir}. "
            "If you installed the package without tests, run with --run-tests 0."
        )

    return int(pytest.main(["-q", str(tests_dir)]))


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
            "radius_nminus1": float("nan"),
            "worst_contingency": -1,
            "worst_contingency_line_idx": -1,
        }
    return out


def compute_results_for_case(
    *,
    input_path: str,
    slack_bus: int,
    dc_mode: str,
    dc_chunk_size: int,
    dc_dtype: np.dtype,
    margin_factor: float,
    inj_std_mw: float,
    compute_nminus1: bool,
    nminus1_update_sensitivities: bool,
    nminus1_islanding: str,
) -> dict[str, Any]:
    """
    Compute per-line radii and return a single results dict (including '__meta__').

    Deterministic pipeline (project policy)
    ---------------------------------------
    Base point is ALWAYS:
      - DC OPF via PyPSA + HiGHS (single snapshot)

    DC model modes
    --------------
    - dc_mode="materialize": materialize H_full and compute all radii (including N-1 if requested)
    - dc_mode="operator": compute L2/metric/sigma radii via operator norms (no N-1)
    """
    time_start = time.time()

    input_path_abs = _ensure_input_case_file(str(input_path))
    case_tag = Path(input_path_abs).stem

    if margin_factor <= 0:
        raise ValueError("margin_factor must be positive.")
    if inj_std_mw <= 0:
        raise ValueError("inj_std_mw must be positive.")
    if dc_chunk_size <= 0:
        raise ValueError("dc_chunk_size must be positive.")

    dc_mode_eff = str(dc_mode).strip().lower()
    if dc_mode_eff not in ("materialize", "operator"):
        raise ValueError("dc_mode must be materialize|operator")

    with log_stage(logger, f"{case_tag}: Read Data"):
        net = load_network(input_path_abs)

        # Hard check requested: at least one thermal limit source must exist
        # (otherwise it's a parser/converter bug, not a radii bug).
        assert_line_limit_sources_present(net)

    with log_stage(
        logger,
        f"{case_tag}: Solve DC OPF (PyPSA, solver={DEFAULT_OPF.highs.solver_name})",
    ):
        base = get_line_base_quantities(net, margin_factor=float(margin_factor))

    H_full = None
    dc_op = None

    with log_stage(logger, f"{case_tag}: Build DC Model (mode={dc_mode_eff})"):
        if dc_mode_eff == "materialize":
            H_full, dc_op = build_dc_matrices(
                net,
                slack_bus=int(slack_bus),
                chunk_size=int(dc_chunk_size),
                dtype=dc_dtype,
            )
            n_bus = int(H_full.shape[1])
            m_line = int(H_full.shape[0])
            logger.debug(
                "Materialized H_full: shape=(%d,%d), dtype=%s",
                m_line,
                n_bus,
                H_full.dtype,
            )
        else:
            dc_op = build_dc_operator(net, slack_bus=int(slack_bus))
            n_bus = int(dc_op.n_bus)
            m_line = int(dc_op.n_line)
            logger.debug("Built DC operator: n_bus=%d, n_line=%d", n_bus, m_line)

    with log_stage(logger, f"{case_tag}: Compute Radii"):
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

            if bool(compute_nminus1):
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
                nminus1 = {
                    f"line_{int(lid)}": {
                        "radius_nminus1": float("nan"),
                        "worst_contingency": -1,
                        "worst_contingency_line_idx": -1,
                    }
                    for lid in base.line_indices
                }
                nminus1_computed = False

            results_lines = _merge_line_results(l2, metric, sigma, nminus1)
        else:
            if dc_op is None:
                raise AssertionError("Internal error: operator mode requires dc_op")
            if bool(compute_nminus1):
                raise ValueError(
                    "compute_nminus1=1 requires dc_mode=materialize (N-1 needs H_full)."
                )

            results_lines = _compute_radii_operator_path(
                dc_op=dc_op,
                base=base,
                inj_std_mw=float(inj_std_mw),
                dc_chunk_size=int(dc_chunk_size),
            )
            nminus1_computed = False

    elapsed_sec = float(time.time() - time_start)
    logger.info(
        "%s: Total compute time (read+opf+dc+radii): %.3f sec", case_tag, elapsed_sec
    )

    results: dict[str, Any] = {
        "__meta__": {
            "input_path": str(input_path_abs),
            "slack_bus": int(slack_bus),
            "dispatch_mode": "opf_pypsa",
            "opf_solver": str(DEFAULT_OPF.highs.solver_name),
            "opf_solver_threads": int(DEFAULT_OPF.highs.threads),
            "opf_solver_random_seed": int(DEFAULT_OPF.highs.random_seed),
            "opf_status": str(base.opf_status)
            if base.opf_status is not None
            else "n/a",
            "opf_objective": float(base.opf_objective)
            if base.opf_objective is not None
            else float("nan"),
            "dc_mode": str(dc_mode_eff),
            "dc_dtype": str(np.dtype(dc_dtype)),
            "dc_chunk_size": int(dc_chunk_size),
            "margin_factor": float(margin_factor),
            "inj_std_mw": float(inj_std_mw),
            "compute_time_sec": elapsed_sec,
            "n_bus": int(n_bus),
            "n_line": int(m_line),
            "nminus1_computed": bool(nminus1_computed),
        }
    }
    results.update(results_lines)
    return results


def run_demo(args: argparse.Namespace) -> int:
    """Run the end-to-end single-case workflow and save results into runs/<timestamp>/."""
    run_dir = setup_logging(
        LoggingConfig(
            runs_dir=str(args.runs_dir),
            level_console=str(args.log_level),
            level_file=str(args.log_file_level),
        )
    )
    run_dir_path = Path(run_dir)

    logger.info(
        "Workflow (demo): Read Data -> Solve DC OPF (PyPSA+HiGHS) -> Build DC Model -> Compute Radii -> Save Outputs"
    )

    results = compute_results_for_case(
        input_path=str(args.input),
        slack_bus=int(args.slack_bus),
        dc_mode=str(args.dc_mode),
        dc_chunk_size=int(args.dc_chunk_size),
        dc_dtype=_dtype_from_str(str(args.dc_dtype)),
        margin_factor=float(args.margin_factor),
        inj_std_mw=float(args.inj_std_mw),
        compute_nminus1=bool(args.compute_nminus1),
        nminus1_update_sensitivities=bool(args.nminus1_update_sensitivities),
        nminus1_islanding=str(args.nminus1_islanding),
    )

    with log_stage(logger, "Write Results (JSON)"):
        results_path = run_dir_path / "results.json"
        results_path.write_text(
            json.dumps(results, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.info("Results written: %s", str(results_path))

    columns = _parse_columns(str(args.table_columns))
    max_rows = int(args.max_rows) if args.max_rows is not None else None

    table_str = format_results_table(results, columns=columns, max_rows=max_rows)
    logging.getLogger("stability_radius.fileonly").debug(
        "Results table:\n%s", table_str
    )

    summaries = [
        format_radius_summary(results, radius_field="radius_l2"),
        format_radius_summary(results, radius_field="radius_metric"),
        format_radius_summary(results, radius_field="radius_sigma"),
        format_radius_summary(results, radius_field="radius_nminus1"),
    ]
    for s in summaries:
        logger.info("%s", s)

    if bool(args.save_csv):
        with log_stage(logger, "Write Results (CSV)"):
            csv_path = run_dir_path / "results_table.csv"
            csv_str = format_results_csv(results, columns=columns, max_rows=max_rows)
            csv_path.write_text(csv_str, encoding="utf-8")
            logger.info("Saved CSV: %s", str(csv_path))

    if str(args.export_results).strip():
        with log_stage(logger, "Export Results (copy results.json)"):
            export_path_abs = _resolve_under_project_root(str(args.export_results))
            Path(export_path_abs).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(results_path, export_path_abs)
            logger.info("Exported results to: %s", export_path_abs)

    logger.info("Done. Run directory: %s", str(run_dir_path))
    return 0


def run_monte_carlo(args: argparse.Namespace) -> int:
    """Run Monte Carlo coverage estimation using stored results and a case file."""
    setup_logging(
        LoggingConfig(
            runs_dir=str(args.runs_dir),
            level_console=str(args.log_level),
            level_file=str(args.log_file_level),
        )
    )

    logger.info(
        "Workflow (monte-carlo): Read Results -> Read Data -> Build DC Model -> Run Monte Carlo Coverage"
    )

    from verification.monte_carlo import estimate_coverage_percent

    with log_stage(logger, "Monte Carlo Coverage"):
        stats = estimate_coverage_percent(
            results_path=Path(str(args.results)),
            input_case_path=Path(str(args.input)),
            slack_bus=int(args.slack_bus),
            n_samples=int(args.n_samples),
            seed=int(args.seed),
            chunk_size=int(args.chunk_size),
            box_radius_quantile=float(args.box_radius_quantile),
        )

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


def run_report(args: argparse.Namespace) -> int:
    """Generate aggregated verification report (Markdown)."""
    run_dir = setup_logging(
        LoggingConfig(
            runs_dir=str(args.runs_dir),
            level_console=str(args.log_level),
            level_file=str(args.log_file_level),
        )
    )
    run_dir_path = Path(run_dir)

    logger.info(
        "Workflow (report): Ensure results -> Monte Carlo Coverage -> Generate Report [dc_mode=%s]",
        str(args.dc_mode),
    )

    from verification.generate_report import generate_report_text

    results_dir = Path(_resolve_under_project_root(str(args.results_dir)))
    out_path = Path(_resolve_under_project_root(str(args.out)))

    with log_stage(logger, "Generate Report (all cases)"):
        report_text = generate_report_text(
            results_dir=results_dir,
            n_samples=int(args.n_samples),
            seed=int(args.seed),
            chunk_size=int(args.chunk_size),
            generate_missing_results=bool(args.generate_missing_results),
            demo_dc_mode=str(args.dc_mode),
            demo_slack_bus=int(args.slack_bus),
            demo_compute_nminus1=bool(args.compute_nminus1),
            mc_box_radius_quantile=float(args.box_radius_quantile),
        )

    with log_stage(logger, "Write Report"):
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
        description="Unified entrypoint for stability radius workflows.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=DEFAULT_LOGGING.runs_dir,
        help="Directory where per-run folders and run.log are created.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOGGING.level_console,
        help="Console logging level (INFO/DEBUG/WARNING/ERROR).",
    )
    parser.add_argument(
        "--log-file-level",
        type=str,
        default=DEFAULT_LOGGING.level_file,
        help="File logging level (DEBUG recommended to include the full results table).",
    )
    parser.add_argument(
        "--run-tests",
        type=int,
        default=1,
        help="1: run pytest suite before executing any command, 0: skip.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run single-case OPF->radii workflow.")
    p_demo.add_argument(
        "--input",
        type=str,
        default="data/input/pglib_opf_case30_ieee.m",
        help=(
            "Path to MATPOWER/PGLib .m case file. If missing and the filename matches "
            "a supported dataset (case<N>.m/ieee<N>.m/pglib_opf_*.m), it will be downloaded."
        ),
    )
    p_demo.add_argument("--slack-bus", type=int, default=0)

    p_demo.add_argument(
        "--dc-mode",
        type=str,
        default=DEFAULT_DC.mode,
        choices=("materialize", "operator"),
        help="DC model mode: materialize H_full or use operator norms (no N-1 in operator mode).",
    )
    p_demo.add_argument("--dc-chunk-size", type=int, default=DEFAULT_DC.chunk_size)
    p_demo.add_argument(
        "--dc-dtype", type=str, default=DEFAULT_DC.dtype, choices=("float64", "float32")
    )

    p_demo.add_argument("--margin-factor", type=float, default=1.0)
    p_demo.add_argument("--inj-std-mw", type=float, default=1.0)

    p_demo.add_argument(
        "--compute-nminus1",
        type=int,
        default=0,
        help="1 to compute effective N-1 radii (requires --dc-mode materialize), 0 to skip.",
    )
    p_demo.add_argument(
        "--nminus1-update-sensitivities",
        type=int,
        default=1,
        help="1 to update sensitivities (more accurate), 0 to reuse base sensitivities.",
    )
    p_demo.add_argument(
        "--nminus1-islanding",
        type=str,
        default=DEFAULT_NMINUS1_ISLANDING,
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
    p_mc.add_argument("--n-samples", default=DEFAULT_MC.n_samples, type=int)
    p_mc.add_argument("--seed", default=DEFAULT_MC.seed, type=int)
    p_mc.add_argument("--chunk-size", default=DEFAULT_MC.chunk_size, type=int)
    p_mc.add_argument(
        "--box-radius-quantile",
        default=DEFAULT_MC.box_radius_quantile,
        type=float,
        help="Quantile of finite radius_l2 used to scale the MC sampling box.",
    )

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
    p_rep.add_argument("--n-samples", default=DEFAULT_MC.n_samples, type=int)
    p_rep.add_argument("--seed", default=DEFAULT_MC.seed, type=int)
    p_rep.add_argument("--chunk-size", default=DEFAULT_MC.chunk_size, type=int)
    p_rep.add_argument(
        "--box-radius-quantile",
        default=DEFAULT_MC.box_radius_quantile,
        type=float,
        help="Quantile of finite radius_l2 used to scale the MC sampling box.",
    )
    p_rep.add_argument(
        "--generate-missing-results",
        type=int,
        default=1,
        help="1: compute missing/invalid results automatically (will also download supported input cases). 0: do not.",
    )
    p_rep.add_argument(
        "--dc-mode",
        type=str,
        default=DEFAULT_DC.mode,
        choices=("materialize", "operator"),
        help="DC model mode used for auto-generated results.",
    )
    p_rep.add_argument("--slack-bus", type=int, default=0)
    p_rep.add_argument(
        "--compute-nminus1",
        type=int,
        default=0,
        help="1 to compute effective N-1 radii (requires --dc-mode materialize), 0 to skip.",
    )

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

    if bool(getattr(args, "run_tests", 0)):
        code = _run_self_tests(project_root=_PROJECT_ROOT)
        if code != 0:
            print(
                f"[ERROR] Self-tests failed (pytest exit code={code}). Aborting.",
                file=sys.stderr,
            )
            return int(code if code > 0 else 1)

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
