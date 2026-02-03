from __future__ import annotations

"""
Unified CLI implementation (argparse + Hydra-style YAML defaults).

Key goals
---------
- Single entrypoint via `src/power_stability_radius.py`
- Defaults and constants live in `conf/config.yaml`
- CLI flags override config values
- Each run creates a run folder and saves:
  - run.log
  - config.yaml (effective)
  - config_source.yaml (copied)
  - argv.txt

Supported commands
------------------
- compute: compute per-line radii for a single case
- report: run Monte Carlo verification + generate Markdown report for configured cases
"""

import argparse
import json
import logging
import shutil
import sys
from collections.abc import Mapping, Sequence as SequenceABC
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from stability_radius.config import (
    DEFAULT_DC,
    DEFAULT_LOGGING,
    DEFAULT_MC,
    DEFAULT_NMINUS1_ISLANDING,
    DEFAULT_OPF,
    DEFAULT_TABLE_COLUMNS,
    HAVE_OMEGACONF,
    HiGHSConfig,
    LoggingConfig,
    OPFConfig,
    OmegaConf,
    load_project_config,
)
from stability_radius.statistics.table import (
    format_radius_summary,
    format_results_csv,
    format_results_table,
)
from stability_radius.utils import log_stage, setup_logging
from stability_radius.workflows import compute_results_for_case

logger = logging.getLogger("stability_radius.cli")

_SUPPORTED_COMMANDS: tuple[str, ...] = ("compute", "report")


def _dtype_from_str(s: str) -> np.dtype:
    ss = str(s).strip().lower()
    if ss in ("float64", "f64"):
        return np.float64
    if ss in ("float32", "f32"):
        return np.float32
    raise ValueError("dc-dtype must be float64 or float32.")


def _parse_columns(value: str, *, default_columns: Sequence[str]) -> tuple[str, ...]:
    """Parse comma-separated columns string."""
    if not str(value).strip():
        return tuple(default_columns)
    return tuple(x.strip() for x in str(value).split(",") if x.strip())


def _resolve_path(p: str) -> str:
    """
    Resolve a potentially-relative path against current working directory.

    This is intentionally CWD-based (not repo-root based) to behave well when the repo
    is used as a library/tool from other projects.
    """
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str(path.resolve())


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


def _preparse_config_path(argv: Sequence[str] | None) -> str:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=str,
        default="conf/config.yaml",
        help="Path to OmegaConf-compatible YAML config (supports `extends:`).",
    )
    ns, _ = pre.parse_known_args(list(argv) if argv is not None else None)
    return str(ns.config)


def _unknown_is_tail(argv: Sequence[str], unknown: Sequence[str]) -> bool:
    """
    Return True if `unknown` equals the trailing slice of argv.

    Used to safely inject an implicit subcommand from config when the user omitted it.
    """
    u = list(unknown)
    a = list(argv)
    if not u:
        return True
    if len(u) > len(a):
        return False
    return a[-len(u) :] == u


def _load_yaml_config(path: Path) -> Any:
    """Load YAML config (OmegaConf) with support for `extends:` composition."""
    return load_project_config(path, allow_missing=False)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Safe config lookup for OmegaConf-based config objects."""
    if cfg is None:
        return default
    if not HAVE_OMEGACONF or OmegaConf is None:
        return default
    try:
        v = OmegaConf.select(cfg, key)
    except Exception:  # noqa: BLE001
        return default
    return default if v is None else v


def _infer_default_command(cfg_loaded: Any) -> str | None:
    """Infer default command from config (top-level `command:` key)."""
    v = _cfg_get(cfg_loaded, "command", None)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _table_default_columns(cfg: Any) -> tuple[str, ...]:
    """
    Extract table.default_columns as a tuple of strings.

    Notes
    -----
    OmegaConf returns ListConfig/DictConfig objects which are not plain `list`/`dict`,
    so we accept any non-string Sequence here for runtime compatibility.
    """
    cols = _cfg_get(cfg, "table.default_columns", None)
    if cols is None:
        return tuple(DEFAULT_TABLE_COLUMNS)

    # Allow a comma-separated string for robustness (explicit, deterministic parsing).
    if isinstance(cols, str):
        parsed = [x.strip() for x in cols.split(",") if x.strip()]
        return tuple(parsed) if parsed else tuple(DEFAULT_TABLE_COLUMNS)

    if isinstance(cols, SequenceABC):
        out: list[str] = []
        for x in cols:
            sx = str(x).strip()
            if sx:
                out.append(sx)
        if out:
            return tuple(out)

    return tuple(DEFAULT_TABLE_COLUMNS)


def build_parser(cfg: Any) -> argparse.ArgumentParser:
    """Create CLI parser (defaults are taken from the loaded YAML config)."""
    parser = argparse.ArgumentParser(
        prog="power_stability_radius",
        description="Unified entrypoint for stability radius workflows.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="conf/config.yaml",
        help="Path to OmegaConf-compatible YAML config (supports `extends:`).",
    )

    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(_cfg_get(cfg, "logging.runs_dir", DEFAULT_LOGGING.runs_dir)),
        help="Directory where per-run folders and run.log are created.",
    )
    parser.add_argument(
        "--run-dir-mode",
        type=str,
        default=str(
            _cfg_get(cfg, "logging.run_dir_mode", DEFAULT_LOGGING.run_dir_mode)
        ),
        choices=("timestamp", "overwrite"),
        help="Run directory behavior: timestamp (new folder) or overwrite (reuse run-name).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=str(_cfg_get(cfg, "logging.run_name", DEFAULT_LOGGING.run_name)),
        help="Run folder name used when --run-dir-mode overwrite.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=str(
            _cfg_get(cfg, "logging.level_console", DEFAULT_LOGGING.level_console)
        ),
        help="Console logging level (INFO/DEBUG/WARNING/ERROR).",
    )
    parser.add_argument(
        "--log-file-level",
        type=str,
        default=str(_cfg_get(cfg, "logging.level_file", DEFAULT_LOGGING.level_file)),
        help="File logging level (DEBUG recommended to include the full results table).",
    )
    parser.add_argument(
        "--run-tests",
        type=int,
        default=int(_cfg_get(cfg, "run_tests", 1)),
        help="1: run pytest suite before executing any command, 0: skip.",
    )

    # OPF settings
    parser.add_argument(
        "--opf-solver-name",
        type=str,
        default=str(_cfg_get(cfg, "opf.solver_name", DEFAULT_OPF.highs.solver_name)),
        help="OPF solver name (project policy: must be 'highs').",
    )
    parser.add_argument(
        "--opf-threads",
        type=int,
        default=int(_cfg_get(cfg, "opf.threads", DEFAULT_OPF.highs.threads)),
        help="HiGHS threads (1 recommended for determinism).",
    )
    parser.add_argument(
        "--opf-random-seed",
        type=int,
        default=int(_cfg_get(cfg, "opf.random_seed", DEFAULT_OPF.highs.random_seed)),
        help="HiGHS random seed.",
    )
    parser.add_argument(
        "--opf-headroom-factor",
        type=float,
        default=float(
            _cfg_get(cfg, "opf.headroom_factor", DEFAULT_OPF.headroom_factor)
        ),
        help="OPF line limit tightening factor (security headroom), e.g. 0.95 means 5% headroom.",
    )
    parser.add_argument(
        "--opf-unconstrained-line-nom-mw",
        type=float,
        default=float(
            _cfg_get(
                cfg,
                "opf.unconstrained_line_nom_mw",
                DEFAULT_OPF.unconstrained_line_nom_mw,
            )
        ),
        help="Surrogate MW limit used by OPF for unconstrained lines (+inf/NaN).",
    )

    # OPF->DC consistency tolerances
    parser.add_argument(
        "--opf-dc-flow-consistency-tol-mw",
        type=float,
        default=float(_cfg_get(cfg, "tolerances.opf_dc_flow_consistency_tol_mw", 1e-3)),
        help="Max allowed OPF->DC reconstructed flow difference (MW).",
    )
    parser.add_argument(
        "--opf-bus-balance-tol-mw",
        type=float,
        default=float(_cfg_get(cfg, "tolerances.opf_bus_balance_tol_mw", 1e-6)),
        help="Max allowed imbalance of OPF injections sum (MW).",
    )

    sub = parser.add_subparsers(dest="command", required=False)

    # ---------- compute ----------
    p_compute = sub.add_parser("compute", help="Compute radii for a single case.")
    p_compute.add_argument(
        "--input",
        type=str,
        default=str(
            _cfg_get(cfg, "compute.input", "data/input/pglib_opf_case30_ieee.m")
        ),
        help=(
            "Path to MATPOWER/PGLib .m case file. If missing and the filename matches "
            "a supported dataset (case<N>.m/ieee<N>.m/pglib_opf_*.m), it will be downloaded."
        ),
    )
    p_compute.add_argument(
        "--slack-bus", type=int, default=int(_cfg_get(cfg, "compute.slack_bus", 0))
    )

    p_compute.add_argument(
        "--dc-mode",
        type=str,
        default=str(_cfg_get(cfg, "compute.dc_mode", DEFAULT_DC.mode)),
        choices=("materialize", "operator"),
        help="DC model mode: materialize H_full or use operator norms (no N-1 in operator mode).",
    )
    p_compute.add_argument(
        "--dc-chunk-size",
        type=int,
        default=int(_cfg_get(cfg, "compute.dc_chunk_size", DEFAULT_DC.chunk_size)),
    )
    p_compute.add_argument(
        "--dc-dtype",
        type=str,
        default=str(_cfg_get(cfg, "compute.dc_dtype", DEFAULT_DC.dtype)),
        choices=("float64", "float32"),
    )

    p_compute.add_argument(
        "--inj-std-mw",
        type=float,
        default=float(_cfg_get(cfg, "compute.inj_std_mw", 1.0)),
        help="Sigma (MW) for Gaussian probabilistic metrics derived from the L2 norms.",
    )

    p_compute.add_argument(
        "--compute-nminus1",
        type=int,
        default=int(_cfg_get(cfg, "compute.compute_nminus1", 0)),
        help="1 to compute effective N-1 radii (requires --dc-mode materialize), 0 to skip.",
    )
    p_compute.add_argument(
        "--nminus1-update-sensitivities",
        type=int,
        default=int(_cfg_get(cfg, "compute.nminus1_update_sensitivities", 1)),
        help="1 to update sensitivities (more accurate), 0 to reuse base sensitivities.",
    )
    p_compute.add_argument(
        "--nminus1-islanding",
        type=str,
        default=str(
            _cfg_get(cfg, "compute.nminus1_islanding", DEFAULT_NMINUS1_ISLANDING)
        ),
        choices=("skip", "raise"),
        help="How to handle islanding-like contingencies when LODF is undefined.",
    )

    p_compute.add_argument(
        "--export-results",
        type=str,
        default=str(_cfg_get(cfg, "compute.export_results", "")),
        help="Optional path to copy results.json (useful for report workflows).",
    )
    p_compute.add_argument(
        "--save-csv",
        type=int,
        default=int(_cfg_get(cfg, "compute.save_csv", 1)),
        help="1 to save results_table.csv, 0 to skip.",
    )
    p_compute.add_argument(
        "--max-rows",
        type=int,
        default=_cfg_get(cfg, "compute.max_rows", None),
        help="Limit number of rows in the logged/saved table.",
    )
    p_compute.add_argument(
        "--table-columns",
        type=str,
        default=str(_cfg_get(cfg, "compute.table_columns", "")),
        help="Comma-separated list of table columns (empty -> config table.default_columns).",
    )

    # ---------- report ----------
    p_rep = sub.add_parser("report", help="Generate aggregated verification report.")
    p_rep.add_argument(
        "--results-dir",
        default=str(_cfg_get(cfg, "report.results_dir", "verification/results")),
        type=str,
        help="Directory containing per-case results JSON files.",
    )
    p_rep.add_argument(
        "--out",
        default=str(_cfg_get(cfg, "report.out", "verification/report.md")),
        type=str,
        help="Output report path.",
    )
    p_rep.add_argument(
        "--n-samples",
        default=int(_cfg_get(cfg, "report.n_samples", DEFAULT_MC.n_samples)),
        type=int,
    )
    p_rep.add_argument(
        "--seed", default=int(_cfg_get(cfg, "report.seed", DEFAULT_MC.seed)), type=int
    )
    p_rep.add_argument(
        "--chunk-size",
        default=int(_cfg_get(cfg, "report.chunk_size", DEFAULT_MC.chunk_size)),
        type=int,
    )
    p_rep.add_argument(
        "--feas-tol-mw",
        default=float(_cfg_get(cfg, "report.feas_tol_mw", DEFAULT_MC.feas_tol_mw)),
        type=float,
        help="Feasibility tolerance in MW (passed to Monte Carlo evaluation).",
    )
    p_rep.add_argument(
        "--cert-tol-mw",
        default=float(_cfg_get(cfg, "report.cert_tol_mw", DEFAULT_MC.cert_tol_mw)),
        type=float,
        help="Certificate feasibility tolerance in MW for in-ball check.",
    )
    p_rep.add_argument(
        "--cert-max-samples",
        default=int(
            _cfg_get(cfg, "report.cert_max_samples", DEFAULT_MC.cert_max_samples)
        ),
        type=int,
        help="Max number of samples used for certificate in-ball soundness check.",
    )

    p_rep.add_argument(
        "--generate-missing-results",
        type=int,
        default=int(_cfg_get(cfg, "report.generate_missing_results", 1)),
        help="1: compute missing/invalid results automatically (will also download supported input cases). 0: do not.",
    )

    # Compute-generation parameters (used only when --generate-missing-results=1)
    p_rep.add_argument(
        "--dc-mode",
        type=str,
        default=str(_cfg_get(cfg, "report.compute.dc_mode", DEFAULT_DC.mode)),
        choices=("materialize", "operator"),
        help="DC model mode used for auto-generated results.",
    )
    p_rep.add_argument(
        "--slack-bus",
        type=int,
        default=int(_cfg_get(cfg, "report.compute.slack_bus", 0)),
    )
    p_rep.add_argument(
        "--compute-nminus1",
        type=int,
        default=int(_cfg_get(cfg, "report.compute.compute_nminus1", 0)),
        help="1 to compute effective N-1 radii (requires --dc-mode materialize), 0 to skip.",
    )
    p_rep.add_argument(
        "--compute-dc-chunk-size",
        type=int,
        default=int(
            _cfg_get(cfg, "report.compute.dc_chunk_size", DEFAULT_DC.chunk_size)
        ),
        help="DC chunk size for auto-generated results.",
    )
    p_rep.add_argument(
        "--compute-dc-dtype",
        type=str,
        default=str(_cfg_get(cfg, "report.compute.dc_dtype", DEFAULT_DC.dtype)),
        choices=("float64", "float32"),
        help="DC dtype for auto-generated results.",
    )
    p_rep.add_argument(
        "--compute-inj-std-mw",
        type=float,
        default=float(_cfg_get(cfg, "report.compute.inj_std_mw", 1.0)),
        help="inj_std_mw for auto-generated results.",
    )
    p_rep.add_argument(
        "--compute-nminus1-update-sensitivities",
        type=int,
        default=int(_cfg_get(cfg, "report.compute.nminus1_update_sensitivities", 1)),
        help="N-1 sensitivity update flag for auto-generated results.",
    )
    p_rep.add_argument(
        "--compute-nminus1-islanding",
        type=str,
        default=str(
            _cfg_get(cfg, "report.compute.nminus1_islanding", DEFAULT_NMINUS1_ISLANDING)
        ),
        choices=("skip", "raise"),
        help="How to handle islanding contingencies for auto-generated results.",
    )

    return parser


def _write_run_artifacts(
    *,
    run_dir: Path,
    cfg_source_path: Path,
    cfg_used: dict[str, Any],
    argv: Sequence[str],
) -> None:
    """Write reproducibility artifacts into the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "argv.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")

    if cfg_source_path.exists():
        shutil.copyfile(cfg_source_path, run_dir / "config_source.yaml")

    cfg_json = json.dumps(cfg_used, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    (run_dir / "config.json").write_text(cfg_json, encoding="utf-8")

    cfg_yaml = OmegaConf.to_yaml(OmegaConf.create(cfg_used))  # type: ignore[union-attr]
    (run_dir / "config.yaml").write_text(cfg_yaml, encoding="utf-8")


def _make_opf_cfg(args: argparse.Namespace) -> OPFConfig:
    """Build OPFConfig object from CLI args."""
    highs = HiGHSConfig(
        solver_name=str(args.opf_solver_name),
        threads=int(args.opf_threads),
        random_seed=int(args.opf_random_seed),
    )
    return OPFConfig(
        highs=highs,
        unconstrained_line_nom_mw=float(args.opf_unconstrained_line_nom_mw),
        headroom_factor=float(args.opf_headroom_factor),
    )


def _setup_run_and_logging(args: argparse.Namespace) -> Path:
    """Create run directory and configure logging."""
    run_dir = Path(
        setup_logging(
            LoggingConfig(
                runs_dir=str(args.runs_dir),
                level_console=str(args.log_level),
                level_file=str(args.log_file_level),
                run_dir_mode=str(args.run_dir_mode),
                run_name=str(args.run_name),
            )
        )
    )
    return run_dir


def run_compute(
    args: argparse.Namespace, *, cfg_loaded: Any, cfg_path: Path, argv: Sequence[str]
) -> int:
    """Run the end-to-end single-case workflow and save results into the run folder."""
    if not str(getattr(args, "input", "")).strip():
        raise ValueError(
            "compute requires --input or config key compute.input (got an empty string)."
        )

    run_dir = _setup_run_and_logging(args)

    cfg_used: dict[str, Any] = {
        "config_path": str(cfg_path),
        "command": "compute",
        "run_tests": int(getattr(args, "run_tests", 0)),
        "logging": {
            "runs_dir": str(args.runs_dir),
            "run_dir_mode": str(args.run_dir_mode),
            "run_name": str(args.run_name),
            "level_console": str(args.log_level),
            "level_file": str(args.log_file_level),
        },
        "opf": {
            "solver_name": str(args.opf_solver_name),
            "threads": int(args.opf_threads),
            "random_seed": int(args.opf_random_seed),
            "headroom_factor": float(args.opf_headroom_factor),
            "unconstrained_line_nom_mw": float(args.opf_unconstrained_line_nom_mw),
        },
        "tolerances": {
            "opf_dc_flow_consistency_tol_mw": float(
                args.opf_dc_flow_consistency_tol_mw
            ),
            "opf_bus_balance_tol_mw": float(args.opf_bus_balance_tol_mw),
        },
        "table": {"default_columns": list(_table_default_columns(cfg_loaded))},
        "compute": {
            "input": str(args.input),
            "slack_bus": int(args.slack_bus),
            "dc_mode": str(args.dc_mode),
            "dc_chunk_size": int(args.dc_chunk_size),
            "dc_dtype": str(args.dc_dtype),
            "inj_std_mw": float(args.inj_std_mw),
            "compute_nminus1": int(args.compute_nminus1),
            "nminus1_update_sensitivities": int(args.nminus1_update_sensitivities),
            "nminus1_islanding": str(args.nminus1_islanding),
            "export_results": str(args.export_results),
            "save_csv": int(args.save_csv),
            "max_rows": args.max_rows if args.max_rows is None else int(args.max_rows),
            "table_columns": str(args.table_columns),
        },
    }
    _write_run_artifacts(
        run_dir=run_dir, cfg_source_path=cfg_path, cfg_used=cfg_used, argv=argv
    )

    logger.info(
        "Workflow (compute): Read Data -> Solve DC OPF -> Build DC Model -> Consistency Check -> Compute Radii -> Save Outputs"
    )

    opf_cfg = _make_opf_cfg(args)

    results = compute_results_for_case(
        input_path=str(args.input),
        slack_bus=int(args.slack_bus),
        dc_mode=str(args.dc_mode),
        dc_chunk_size=int(args.dc_chunk_size),
        dc_dtype=_dtype_from_str(str(args.dc_dtype)),
        inj_std_mw=float(args.inj_std_mw),
        compute_nminus1=bool(args.compute_nminus1),
        nminus1_update_sensitivities=bool(args.nminus1_update_sensitivities),
        nminus1_islanding=str(args.nminus1_islanding),
        opf_cfg=opf_cfg,
        opf_dc_flow_consistency_tol_mw=float(args.opf_dc_flow_consistency_tol_mw),
        opf_bus_balance_tol_mw=float(args.opf_bus_balance_tol_mw),
        path_base_dir=Path.cwd(),
    )

    with log_stage(logger, "Write Results (JSON)"):
        results_path = run_dir / "results.json"
        results_path.write_text(
            json.dumps(results, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        logger.info("Results written: %s", str(results_path))

    default_cols = _table_default_columns(cfg_loaded)
    columns = _parse_columns(str(args.table_columns), default_columns=default_cols)
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
            csv_path = run_dir / "results_table.csv"
            csv_str = format_results_csv(results, columns=columns, max_rows=max_rows)
            csv_path.write_text(csv_str, encoding="utf-8")
            logger.info("Saved CSV: %s", str(csv_path))

    if str(args.export_results).strip():
        with log_stage(logger, "Export Results (copy results.json)"):
            export_path_abs = _resolve_path(str(args.export_results))
            Path(export_path_abs).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(results_path, export_path_abs)
            logger.info("Exported results to: %s", export_path_abs)

    logger.info("Done. Run directory: %s", str(run_dir))
    return 0


def _parse_report_cases_from_cfg(
    *, cfg_loaded: Any, results_dir_abs: Path, base_dir: Path
) -> list[dict[str, Any]]:
    """
    Extract report.cases from OmegaConf config.

    Returns a JSON-friendly list of dicts to be saved into run artifacts.

    Important
    ---------
    OmegaConf returns ListConfig/DictConfig containers which are not instances of
    built-in `list`/`dict`. We therefore validate using collections.abc (Sequence/Mapping)
    to avoid false negatives.
    """
    raw = _cfg_get(cfg_loaded, "report.cases", None)

    def _is_seq_not_str(x: Any) -> bool:
        return isinstance(x, SequenceABC) and not isinstance(x, (str, bytes, bytearray))

    if raw is None:
        raise ValueError(
            "Missing required config key `report.cases` (must be a non-empty list)."
        )
    if not _is_seq_not_str(raw):
        raise ValueError(
            "report.cases must be a non-empty list/sequence in the YAML config under `report.cases` "
            f"(got type={type(raw)})."
        )
    if len(raw) == 0:
        raise ValueError(
            "report.cases must be a non-empty list/sequence in the YAML config under `report.cases`."
        )

    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"report.cases[{i}] must be a mapping/dict, got {type(item)}"
            )

        case_id = str(item.get("id", "")).strip()
        input_path = str(item.get("input", "")).strip()
        results_name = str(item.get("results", "")).strip()
        if not case_id:
            raise ValueError(f"report.cases[{i}].id is required")
        if not input_path:
            raise ValueError(f"report.cases[{i}].input is required")
        if not results_name:
            raise ValueError(f"report.cases[{i}].results is required")

        # Resolve results relative to results_dir unless absolute
        rp = Path(results_name).expanduser()
        rp_abs = rp if rp.is_absolute() else (results_dir_abs / rp).resolve()

        ip = Path(input_path).expanduser()
        ip_abs = ip if ip.is_absolute() else (base_dir / ip).resolve()

        known = item.get("known_critical_pairs", None)
        known_pairs: list[list[int]] = []
        if known is not None:
            if not _is_seq_not_str(known):
                raise ValueError(
                    f"report.cases[{i}].known_critical_pairs must be a list of 2-element pairs (got type={type(known)})."
                )
            for j, p in enumerate(known):
                if not _is_seq_not_str(p) or len(p) != 2:
                    raise ValueError(
                        f"report.cases[{i}].known_critical_pairs[{j}] must be a 2-element pair, got {p!r}"
                    )
                try:
                    a = int(p[0])
                    b = int(p[1])
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"report.cases[{i}].known_critical_pairs[{j}] elements must be int-castable, got {p!r}"
                    ) from e
                known_pairs.append([a, b])

        out.append(
            {
                "id": case_id,
                "input": str(ip_abs),
                "results": str(rp_abs),
                "known_critical_pairs": known_pairs,
            }
        )

    logger.debug(
        "Parsed report.cases: count=%d, results_dir=%s", len(out), str(results_dir_abs)
    )
    return out


def run_report(
    args: argparse.Namespace, *, cfg_loaded: Any, cfg_path: Path, argv: Sequence[str]
) -> int:
    """Generate aggregated verification report (Markdown)."""
    if cfg_loaded is None:
        raise ValueError(
            "report requires a loaded YAML config (conf/config.yaml) because the case list is configured via report.cases."
        )

    if not str(getattr(args, "results_dir", "")).strip():
        raise ValueError(
            "report requires --results-dir or config key report.results_dir (got empty)."
        )
    if not str(getattr(args, "out", "")).strip():
        raise ValueError("report requires --out or config key report.out (got empty).")

    # Fail fast before creating run directory (no side effects on invalid config).
    results_dir = Path(_resolve_path(str(args.results_dir)))
    out_path = Path(_resolve_path(str(args.out)))

    base_dir = Path.cwd().resolve()
    cases_cfg = _parse_report_cases_from_cfg(
        cfg_loaded=cfg_loaded, results_dir_abs=results_dir.resolve(), base_dir=base_dir
    )

    run_dir = _setup_run_and_logging(args)

    assets_dir_name = f"{out_path.stem}_assets"
    assets_dir_run = (run_dir / assets_dir_name).resolve()

    cfg_used: dict[str, Any] = {
        "config_path": str(cfg_path),
        "command": "report",
        "run_tests": int(getattr(args, "run_tests", 0)),
        "logging": {
            "runs_dir": str(args.runs_dir),
            "run_dir_mode": str(args.run_dir_mode),
            "run_name": str(args.run_name),
            "level_console": str(args.log_level),
            "level_file": str(args.log_file_level),
        },
        "opf": {
            "solver_name": str(args.opf_solver_name),
            "threads": int(args.opf_threads),
            "random_seed": int(args.opf_random_seed),
            "headroom_factor": float(args.opf_headroom_factor),
            "unconstrained_line_nom_mw": float(args.opf_unconstrained_line_nom_mw),
        },
        "tolerances": {
            "opf_dc_flow_consistency_tol_mw": float(
                args.opf_dc_flow_consistency_tol_mw
            ),
            "opf_bus_balance_tol_mw": float(args.opf_bus_balance_tol_mw),
        },
        "report": {
            "results_dir": str(results_dir),
            "out": str(out_path),
            "n_samples": int(args.n_samples),
            "seed": int(args.seed),
            "chunk_size": int(args.chunk_size),
            "feas_tol_mw": float(args.feas_tol_mw),
            "cert_tol_mw": float(args.cert_tol_mw),
            "cert_max_samples": int(args.cert_max_samples),
            "generate_missing_results": int(args.generate_missing_results),
            "assets_dir_name": str(assets_dir_name),
            "per_line_distribution": {
                "margin_field": "margin_mw",
                "radius_field": "radius_l2",
            },
            "compute": {
                "dc_mode": str(args.dc_mode),
                "slack_bus": int(args.slack_bus),
                "compute_nminus1": int(args.compute_nminus1),
                "dc_chunk_size": int(args.compute_dc_chunk_size),
                "dc_dtype": str(args.compute_dc_dtype),
                "inj_std_mw": float(args.compute_inj_std_mw),
                "nminus1_update_sensitivities": int(
                    args.compute_nminus1_update_sensitivities
                ),
                "nminus1_islanding": str(args.compute_nminus1_islanding),
            },
            "cases": cases_cfg,
        },
    }
    _write_run_artifacts(
        run_dir=run_dir, cfg_source_path=cfg_path, cfg_used=cfg_used, argv=argv
    )

    logger.info(
        "Workflow (report): Ensure results -> Monte Carlo Verification -> Generate Report [dc_mode=%s]",
        str(args.dc_mode),
    )
    logger.info("Report assets directory (per-run): %s", str(assets_dir_run))

    from stability_radius.verification.generate_report import (
        ReportCaseSpec,
        generate_report_text,
    )

    # Convert JSON-friendly dict cases to typed specs.
    case_specs: list[ReportCaseSpec] = []
    for item in cases_cfg:
        known_pairs = tuple(
            (int(a), int(b)) for a, b in (item.get("known_critical_pairs") or [])
        )
        case_specs.append(
            ReportCaseSpec(
                case_id=str(item["id"]),
                input_case_path=Path(str(item["input"])),
                results_path=Path(str(item["results"])),
                known_critical_pairs=known_pairs,
            )
        )

    opf_cfg = _make_opf_cfg(args)

    with log_stage(logger, "Generate Report (all cases)"):
        report_text = generate_report_text(
            cases=case_specs,
            n_samples=int(args.n_samples),
            seed=int(args.seed),
            chunk_size=int(args.chunk_size),
            generate_missing_results=bool(args.generate_missing_results),
            compute_dc_mode=str(args.dc_mode),
            compute_slack_bus=int(args.slack_bus),
            compute_compute_nminus1=bool(args.compute_nminus1),
            compute_dc_chunk_size=int(args.compute_dc_chunk_size),
            compute_dc_dtype=_dtype_from_str(str(args.compute_dc_dtype)),
            compute_inj_std_mw=float(args.compute_inj_std_mw),
            compute_nminus1_update_sensitivities=bool(
                args.compute_nminus1_update_sensitivities
            ),
            compute_nminus1_islanding=str(args.compute_nminus1_islanding),
            compute_opf_cfg=opf_cfg,
            compute_opf_dc_flow_consistency_tol_mw=float(
                args.opf_dc_flow_consistency_tol_mw
            ),
            compute_opf_bus_balance_tol_mw=float(args.opf_bus_balance_tol_mw),
            mc_feas_tol_mw=float(args.feas_tol_mw),
            mc_cert_tol_mw=float(args.cert_tol_mw),
            mc_cert_max_samples=int(args.cert_max_samples),
            path_base_dir=base_dir,
            per_line_artifacts_dir=assets_dir_run,
            per_line_radius_field="radius_l2",
        )

    with log_stage(logger, "Write Report"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_text, encoding="utf-8")
        logger.info("Wrote report: %s", str(out_path))

        run_copy = run_dir / "verification_report.md"
        run_copy.write_text(report_text, encoding="utf-8")
        logger.info("Wrote report copy: %s", str(run_copy))

    # Copy assets next to the user-requested report path so relative links in Markdown work.
    assets_dir_out = (out_path.parent / assets_dir_name).resolve()
    if assets_dir_run.exists():
        try:
            if assets_dir_out != assets_dir_run:
                if assets_dir_out.exists():
                    shutil.rmtree(assets_dir_out)
                shutil.copytree(assets_dir_run, assets_dir_out)
                logger.info(
                    "Copied report assets: %s -> %s",
                    str(assets_dir_run),
                    str(assets_dir_out),
                )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to copy report assets to: %s", str(assets_dir_out))

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint."""
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    cfg_path_str = _preparse_config_path(argv_list)
    cfg_path = Path(cfg_path_str)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()

    try:
        cfg_loaded = _load_yaml_config(cfg_path)
    except Exception as e:  # noqa: BLE001
        print(
            f"[ERROR] Failed to load config: {str(cfg_path)} ({e})",
            file=sys.stderr,
        )
        return 2

    parser = build_parser(cfg_loaded)

    args, unknown = parser.parse_known_args(argv_list)
    argv_effective = list(argv_list)

    if args.command is None:
        default_cmd = _infer_default_command(cfg_loaded)

        if default_cmd is None:
            parser.print_help(sys.stderr)
            print(
                "\n[ERROR] No command specified. "
                "Either pass a subcommand (compute|report) or set `command: <...>` "
                "at the top level of the YAML config.",
                file=sys.stderr,
            )
            return 2

        if default_cmd not in _SUPPORTED_COMMANDS:
            print(
                f"[ERROR] Invalid config default `command: {default_cmd}`. "
                f"Supported: {', '.join(_SUPPORTED_COMMANDS)}.",
                file=sys.stderr,
            )
            return 2

        if unknown and not _unknown_is_tail(argv_list, unknown):
            print(
                "[ERROR] Cannot infer command from config because command-specific arguments are interleaved "
                "with global arguments. Specify the command explicitly.",
                file=sys.stderr,
            )
            return 2

        prefix = argv_list[: len(argv_list) - len(unknown)] if unknown else argv_list
        argv_effective = [*prefix, default_cmd, *list(unknown)]
        args = parser.parse_args(argv_effective)
    else:
        if unknown:
            args = parser.parse_args(argv_list)

    if args.command is None:
        raise AssertionError("Internal error: command is still None after parsing.")

    if bool(getattr(args, "run_tests", 0)):
        project_root = Path(__file__).resolve().parents[2]
        code = _run_self_tests(project_root=project_root)
        if code != 0:
            print(
                f"[ERROR] Self-tests failed (pytest exit code={code}). Aborting.",
                file=sys.stderr,
            )
            return int(code if code > 0 else 1)

    logger.debug("CLI command=%s, argv_effective=%s", str(args.command), argv_effective)

    if args.command == "compute":
        return run_compute(
            args, cfg_loaded=cfg_loaded, cfg_path=cfg_path, argv=argv_effective
        )
    if args.command == "report":
        return run_report(
            args, cfg_loaded=cfg_loaded, cfg_path=cfg_path, argv=argv_effective
        )

    raise AssertionError(f"Unhandled command: {args.command}")
