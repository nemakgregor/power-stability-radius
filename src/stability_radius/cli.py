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

Config inheritance (minimal)
----------------------------
In addition to plain OmegaConf YAML files, we support a minimal inheritance mechanism:

- `extends: ../config.yaml` at the top-level of an experiment YAML.
- The extended config is loaded first, then current file overrides it.
- `extends` paths are resolved relative to the experiment file.

This is intentionally NOT full Hydra composition; it's a small deterministic subset.

Optional "default command" from config
--------------------------------------
If the CLI is called without a subcommand, it can take it from the YAML config:

  command: report   # or demo / monte-carlo / table

This allows:
  python src/power_stability_radius.py --config conf/experiments/report.yaml

The explicit CLI subcommand still works and always overrides config.
"""

import argparse
import json
import logging
import shutil
import sys
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
    DEFAULT_UNITS,
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

_SUPPORTED_COMMANDS: tuple[str, ...] = ("demo", "monte-carlo", "report", "table")


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


def _argv_has_explicit_config_flag(argv: Sequence[str]) -> bool:
    """Return True if argv explicitly contains a --config option."""
    for t in argv:
        if t == "--config" or str(t).startswith("--config="):
            return True
    return False


def _unknown_is_tail(argv: Sequence[str], unknown: Sequence[str]) -> bool:
    """
    Return True if `unknown` equals the trailing slice of argv.

    This is used to safely inject an implicit subcommand from config when the user
    omitted it but did provide subcommand-specific args.
    """
    u = list(unknown)
    a = list(argv)
    if not u:
        return True
    if len(u) > len(a):
        return False
    return a[-len(u) :] == u


def _load_yaml_config(path: Path, *, allow_missing: bool) -> Any:
    """
    Load YAML config (OmegaConf) with support for `extends:` composition.

    Returns `None` if allow_missing=True and file does not exist (CLI will fall back to Python defaults).
    """
    return load_project_config(path, allow_missing=bool(allow_missing))


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
    """
    Infer default command from config.

    Convention:
      command: demo|monte-carlo|report|table
    """
    v = _cfg_get(cfg_loaded, "command", None)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _table_default_columns(cfg: Any) -> tuple[str, ...]:
    cols = _cfg_get(cfg, "table.default_columns", None)
    if isinstance(cols, (list, tuple)) and all(isinstance(x, str) for x in cols):
        return tuple(cols)
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

    # Unit/physics validation policy
    parser.add_argument(
        "--strict-units",
        type=int,
        default=int(
            _cfg_get(cfg, "units.strict_units", int(DEFAULT_UNITS.strict_units))
        ),
        help=(
            "1: strict unit validation (vn_kv>0, x_ohm>0, sn_mva>0; no silent 0.0 fallbacks). "
            "0: legacy permissive behavior (may skip invalid branches)."
        ),
    )
    parser.add_argument(
        "--allow-phase-shift",
        type=int,
        default=int(
            _cfg_get(
                cfg, "units.allow_phase_shift", int(DEFAULT_UNITS.allow_phase_shift)
            )
        ),
        help=(
            "1: allow transformers with non-zero shift_degree (phase shift is ignored by the project's DC model). "
            "0: raise on shift_degree != 0 (recommended)."
        ),
    )

    # OPF settings (constants previously hard-coded in DEFAULT_OPF)
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

    # NOTE:
    # - required=False to allow "command from config" behavior.
    # - main() still enforces that a command exists (CLI arg or config key).
    sub = parser.add_subparsers(dest="command", required=False)

    # ---------- demo ----------
    p_demo = sub.add_parser("demo", help="Run single-case OPF->radii workflow.")
    p_demo.add_argument(
        "--input",
        type=str,
        default=str(_cfg_get(cfg, "demo.input", "data/input/pglib_opf_case30_ieee.m")),
        help=(
            "Path to MATPOWER/PGLib .m case file. If missing and the filename matches "
            "a supported dataset (case<N>.m/ieee<N>.m/pglib_opf_*.m), it will be downloaded."
        ),
    )
    p_demo.add_argument(
        "--slack-bus", type=int, default=int(_cfg_get(cfg, "demo.slack_bus", 0))
    )

    p_demo.add_argument(
        "--dc-mode",
        type=str,
        default=str(_cfg_get(cfg, "demo.dc_mode", DEFAULT_DC.mode)),
        choices=("materialize", "operator"),
        help="DC model mode: materialize H_full or use operator norms (no N-1 in operator mode).",
    )
    p_demo.add_argument(
        "--dc-chunk-size",
        type=int,
        default=int(_cfg_get(cfg, "demo.dc_chunk_size", DEFAULT_DC.chunk_size)),
    )
    p_demo.add_argument(
        "--dc-dtype",
        type=str,
        default=str(_cfg_get(cfg, "demo.dc_dtype", DEFAULT_DC.dtype)),
        choices=("float64", "float32"),
    )

    p_demo.add_argument(
        "--margin-factor",
        type=float,
        default=float(_cfg_get(cfg, "demo.margin_factor", 1.0)),
    )
    p_demo.add_argument(
        "--inj-std-mw", type=float, default=float(_cfg_get(cfg, "demo.inj_std_mw", 1.0))
    )

    p_demo.add_argument(
        "--compute-nminus1",
        type=int,
        default=int(_cfg_get(cfg, "demo.compute_nminus1", 0)),
        help="1 to compute effective N-1 radii (requires --dc-mode materialize), 0 to skip.",
    )
    p_demo.add_argument(
        "--nminus1-update-sensitivities",
        type=int,
        default=int(_cfg_get(cfg, "demo.nminus1_update_sensitivities", 1)),
        help="1 to update sensitivities (more accurate), 0 to reuse base sensitivities.",
    )
    p_demo.add_argument(
        "--nminus1-islanding",
        type=str,
        default=str(_cfg_get(cfg, "demo.nminus1_islanding", DEFAULT_NMINUS1_ISLANDING)),
        choices=("skip", "raise"),
        help="How to handle islanding-like contingencies when LODF is undefined.",
    )
    p_demo.add_argument(
        "--export-results",
        type=str,
        default=str(_cfg_get(cfg, "demo.export_results", "")),
        help="Optional path to copy results.json (useful for verification workflows).",
    )
    p_demo.add_argument(
        "--save-csv",
        type=int,
        default=int(_cfg_get(cfg, "demo.save_csv", 1)),
        help="1 to save results_table.csv, 0 to skip.",
    )
    p_demo.add_argument(
        "--max-rows",
        type=int,
        default=_cfg_get(cfg, "demo.max_rows", None),
        help="Limit number of rows in the logged/saved table.",
    )
    p_demo.add_argument(
        "--table-columns",
        type=str,
        default=str(_cfg_get(cfg, "demo.table_columns", "")),
        help="Comma-separated list of table columns (empty -> config table.default_columns).",
    )

    # ---------- monte-carlo ----------
    p_mc = sub.add_parser("monte-carlo", help="Monte Carlo verification for one case.")
    # NOTE: not `required=True` so config can supply defaults; validated explicitly in run_monte_carlo().
    p_mc.add_argument(
        "--results",
        type=str,
        default=str(_cfg_get(cfg, "monte_carlo.results", "")),
        help="Path to results.json (required via CLI or config key monte_carlo.results).",
    )
    p_mc.add_argument(
        "--input",
        type=str,
        default=str(_cfg_get(cfg, "monte_carlo.input", "")),
        help="Path to case .m file (required via CLI or config key monte_carlo.input).",
    )
    p_mc.add_argument(
        "--slack-bus", default=int(_cfg_get(cfg, "monte_carlo.slack_bus", 0)), type=int
    )
    p_mc.add_argument(
        "--n-samples",
        default=int(_cfg_get(cfg, "monte_carlo.n_samples", DEFAULT_MC.n_samples)),
        type=int,
    )
    p_mc.add_argument(
        "--seed",
        default=int(_cfg_get(cfg, "monte_carlo.seed", DEFAULT_MC.seed)),
        type=int,
    )
    p_mc.add_argument(
        "--chunk-size",
        default=int(_cfg_get(cfg, "monte_carlo.chunk_size", DEFAULT_MC.chunk_size)),
        type=int,
    )

    p_mc.add_argument(
        "--box-radius-quantile",
        default=float(
            _cfg_get(
                cfg, "monte_carlo.box_radius_quantile", DEFAULT_MC.box_radius_quantile
            )
        ),
        type=float,
        help=(
            "DEPRECATED (ignored): legacy parameter from the old box-based coverage experiment. "
            "Kept for backward-compatibility only."
        ),
    )
    p_mc.add_argument(
        "--box-feas-tol-mw",
        default=float(
            _cfg_get(cfg, "monte_carlo.box_feas_tol_mw", DEFAULT_MC.box_feas_tol_mw)
        ),
        type=float,
        help="Feasibility tolerance in MW (applies to Gaussian feasibility checks).",
    )
    p_mc.add_argument(
        "--cert-tol-mw",
        default=float(_cfg_get(cfg, "monte_carlo.cert_tol_mw", DEFAULT_MC.cert_tol_mw)),
        type=float,
        help="Certificate feasibility tolerance in MW for in-ball check.",
    )
    p_mc.add_argument(
        "--cert-max-samples",
        default=int(
            _cfg_get(cfg, "monte_carlo.cert_max_samples", DEFAULT_MC.cert_max_samples)
        ),
        type=int,
        help="Max number of samples used for certificate in-ball soundness check.",
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
        "--box-radius-quantile",
        default=float(
            _cfg_get(cfg, "report.box_radius_quantile", DEFAULT_MC.box_radius_quantile)
        ),
        type=float,
        help=(
            "DEPRECATED (ignored): legacy parameter from the old box-based coverage experiment. "
            "Kept for backward-compatibility only."
        ),
    )
    p_rep.add_argument(
        "--box-feas-tol-mw",
        default=float(
            _cfg_get(cfg, "report.box_feas_tol_mw", DEFAULT_MC.box_feas_tol_mw)
        ),
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

    # Demo-generation parameters (used only when --generate-missing-results=1)
    p_rep.add_argument(
        "--dc-mode",
        type=str,
        default=str(_cfg_get(cfg, "report.demo.dc_mode", DEFAULT_DC.mode)),
        choices=("materialize", "operator"),
        help="DC model mode used for auto-generated results.",
    )
    p_rep.add_argument(
        "--slack-bus", type=int, default=int(_cfg_get(cfg, "report.demo.slack_bus", 0))
    )
    p_rep.add_argument(
        "--compute-nminus1",
        type=int,
        default=int(_cfg_get(cfg, "report.demo.compute_nminus1", 0)),
        help="1 to compute effective N-1 radii (requires --dc-mode materialize), 0 to skip.",
    )
    p_rep.add_argument(
        "--demo-dc-chunk-size",
        type=int,
        default=int(_cfg_get(cfg, "report.demo.dc_chunk_size", DEFAULT_DC.chunk_size)),
        help="DC chunk size for auto-generated results.",
    )
    p_rep.add_argument(
        "--demo-dc-dtype",
        type=str,
        default=str(_cfg_get(cfg, "report.demo.dc_dtype", DEFAULT_DC.dtype)),
        choices=("float64", "float32"),
        help="DC dtype for auto-generated results.",
    )
    p_rep.add_argument(
        "--demo-margin-factor",
        type=float,
        default=float(_cfg_get(cfg, "report.demo.margin_factor", 1.0)),
        help="Margin factor for auto-generated results.",
    )
    p_rep.add_argument(
        "--demo-inj-std-mw",
        type=float,
        default=float(_cfg_get(cfg, "report.demo.inj_std_mw", 1.0)),
        help="inj_std_mw for auto-generated results.",
    )
    p_rep.add_argument(
        "--demo-nminus1-update-sensitivities",
        type=int,
        default=int(_cfg_get(cfg, "report.demo.nminus1_update_sensitivities", 1)),
        help="N-1 sensitivity update flag for auto-generated results.",
    )
    p_rep.add_argument(
        "--demo-nminus1-islanding",
        type=str,
        default=str(
            _cfg_get(cfg, "report.demo.nminus1_islanding", DEFAULT_NMINUS1_ISLANDING)
        ),
        choices=("skip", "raise"),
        help="How to handle islanding contingencies for auto-generated results.",
    )

    # ---------- table ----------
    p_tab = sub.add_parser(
        "table", help="Print/export a table from an existing results.json."
    )
    p_tab.add_argument("results_json", type=str, help="Path to results.json")
    p_tab.add_argument(
        "--max-rows", type=int, default=_cfg_get(cfg, "table_cmd.max_rows", None)
    )
    p_tab.add_argument(
        "--radius-field",
        type=str,
        default=str(_cfg_get(cfg, "table_cmd.radius_field", "radius_l2")),
    )
    p_tab.add_argument(
        "--columns", type=str, default=str(_cfg_get(cfg, "table_cmd.columns", ""))
    )
    p_tab.add_argument(
        "--table-out", type=str, default=str(_cfg_get(cfg, "table_cmd.table_out", ""))
    )
    p_tab.add_argument(
        "--csv-out", type=str, default=str(_cfg_get(cfg, "table_cmd.csv_out", ""))
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

    # Always keep a copy of the original config file for audit.
    if cfg_source_path.exists():
        shutil.copyfile(cfg_source_path, run_dir / "config_source.yaml")

    # Effective config (after CLI overrides).
    cfg_json = json.dumps(cfg_used, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    (run_dir / "config.json").write_text(cfg_json, encoding="utf-8")

    if HAVE_OMEGACONF and OmegaConf is not None:
        cfg_yaml = OmegaConf.to_yaml(OmegaConf.create(cfg_used))  # type: ignore[union-attr]
        (run_dir / "config.yaml").write_text(cfg_yaml, encoding="utf-8")
    else:
        # JSON is still better than nothing.
        (run_dir / "config.yaml").write_text(
            "# OmegaConf is not installed; see config.json\n", encoding="utf-8"
        )


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


def run_demo(
    args: argparse.Namespace, *, cfg_loaded: Any, cfg_path: Path, argv: Sequence[str]
) -> int:
    """Run the end-to-end single-case workflow and save results into the run folder."""
    if not str(getattr(args, "input", "")).strip():
        raise ValueError(
            "demo requires --input or config key demo.input (got an empty string)."
        )

    run_dir = _setup_run_and_logging(args)

    cfg_used: dict[str, Any] = {
        "config_path": str(cfg_path),
        "command": "demo",
        "run_tests": int(getattr(args, "run_tests", 0)),
        "units": {
            "strict_units": int(
                getattr(args, "strict_units", int(DEFAULT_UNITS.strict_units))
            ),
            "allow_phase_shift": int(
                getattr(args, "allow_phase_shift", int(DEFAULT_UNITS.allow_phase_shift))
            ),
        },
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
            "unconstrained_line_nom_mw": float(args.opf_unconstrained_line_nom_mw),
        },
        "tolerances": {
            "opf_dc_flow_consistency_tol_mw": float(
                args.opf_dc_flow_consistency_tol_mw
            ),
            "opf_bus_balance_tol_mw": float(args.opf_bus_balance_tol_mw),
        },
        "table": {"default_columns": list(_table_default_columns(cfg_loaded))},
        "demo": {
            "input": str(args.input),
            "slack_bus": int(args.slack_bus),
            "dc_mode": str(args.dc_mode),
            "dc_chunk_size": int(args.dc_chunk_size),
            "dc_dtype": str(args.dc_dtype),
            "margin_factor": float(args.margin_factor),
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
        "Workflow (demo): Read Data -> Solve DC OPF -> Build DC Model -> Consistency Check -> Compute Radii -> Save Outputs"
    )

    opf_cfg = _make_opf_cfg(args)

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
        opf_cfg=opf_cfg,
        opf_dc_flow_consistency_tol_mw=float(args.opf_dc_flow_consistency_tol_mw),
        opf_bus_balance_tol_mw=float(args.opf_bus_balance_tol_mw),
        strict_units=bool(int(args.strict_units)),
        allow_phase_shift=bool(int(args.allow_phase_shift)),
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


def run_monte_carlo(
    args: argparse.Namespace, *, cfg_loaded: Any, cfg_path: Path, argv: Sequence[str]
) -> int:
    """Run Monte Carlo verification using stored results and a case file."""
    results_path_s = str(getattr(args, "results", "")).strip()
    if not results_path_s:
        raise ValueError(
            "monte-carlo requires --results or config key monte_carlo.results (got empty)."
        )

    input_path_s = str(getattr(args, "input", "")).strip()
    if not input_path_s:
        raise ValueError(
            "monte-carlo requires --input or config key monte_carlo.input (got empty)."
        )

    # Resolve paths early (avoid relative paths leaking into logs/artifacts).
    results_path_abs = Path(_resolve_path(results_path_s))
    input_path_abs = Path(_resolve_path(input_path_s))

    # Fail fast before creating run directory to avoid side effects on invalid config/CLI.
    run_dir = _setup_run_and_logging(args)

    cfg_used: dict[str, Any] = {
        "config_path": str(cfg_path),
        "command": "monte-carlo",
        "run_tests": int(getattr(args, "run_tests", 0)),
        "units": {
            "strict_units": int(
                getattr(args, "strict_units", int(DEFAULT_UNITS.strict_units))
            ),
            "allow_phase_shift": int(
                getattr(args, "allow_phase_shift", int(DEFAULT_UNITS.allow_phase_shift))
            ),
        },
        "logging": {
            "runs_dir": str(args.runs_dir),
            "run_dir_mode": str(args.run_dir_mode),
            "run_name": str(args.run_name),
            "level_console": str(args.log_level),
            "level_file": str(args.log_file_level),
        },
        "monte_carlo": {
            "results": str(results_path_abs),
            "input": str(input_path_abs),
            "slack_bus": int(args.slack_bus),
            "n_samples": int(args.n_samples),
            "seed": int(args.seed),
            "chunk_size": int(args.chunk_size),
            "box_radius_quantile": float(args.box_radius_quantile),
            "box_feas_tol_mw": float(args.box_feas_tol_mw),
            "cert_tol_mw": float(args.cert_tol_mw),
            "cert_max_samples": int(args.cert_max_samples),
        },
    }
    _write_run_artifacts(
        run_dir=run_dir, cfg_source_path=cfg_path, cfg_used=cfg_used, argv=argv
    )

    logger.info(
        "Workflow (monte-carlo): Read Results -> Read Data -> Build DC Model -> "
        "Gaussian feasibility MC + analytic ball mass -> L2-ball certificate soundness check"
    )

    from stability_radius.verification.monte_carlo import run_monte_carlo_verification

    with log_stage(logger, "Monte Carlo Verification"):
        vr = run_monte_carlo_verification(
            results_path=results_path_abs,
            input_case_path=input_path_abs,
            slack_bus=int(args.slack_bus),
            n_samples=int(args.n_samples),
            seed=int(args.seed),
            chunk_size=int(args.chunk_size),
            # DEPRECATED: legacy box parameter kept for CLI backward-compatibility.
            box_radius_quantile=float(args.box_radius_quantile),
            box_feas_tol_mw=float(args.box_feas_tol_mw),
            cert_tol_mw=float(args.cert_tol_mw),
            cert_max_samples=int(args.cert_max_samples),
            strict_units=bool(int(args.strict_units)),
            allow_phase_shift=bool(int(args.allow_phase_shift)),
        )

    stats_dict = vr.to_dict()

    stats_path = run_dir / "monte_carlo_stats.json"
    stats_path.write_text(
        json.dumps(stats_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Saved Monte Carlo stats: %s", str(stats_path))

    print(json.dumps(stats_dict, indent=2, ensure_ascii=False))
    return 0


def run_report(
    args: argparse.Namespace, *, cfg_loaded: Any, cfg_path: Path, argv: Sequence[str]
) -> int:
    """Generate aggregated verification report (Markdown)."""
    if not str(getattr(args, "results_dir", "")).strip():
        raise ValueError(
            "report requires --results-dir or config key report.results_dir (got empty)."
        )
    if not str(getattr(args, "out", "")).strip():
        raise ValueError("report requires --out or config key report.out (got empty).")

    run_dir = _setup_run_and_logging(args)

    cfg_used: dict[str, Any] = {
        "config_path": str(cfg_path),
        "command": "report",
        "run_tests": int(getattr(args, "run_tests", 0)),
        "units": {
            "strict_units": int(
                getattr(args, "strict_units", int(DEFAULT_UNITS.strict_units))
            ),
            "allow_phase_shift": int(
                getattr(args, "allow_phase_shift", int(DEFAULT_UNITS.allow_phase_shift))
            ),
        },
        "logging": {
            "runs_dir": str(args.runs_dir),
            "run_dir_mode": str(args.run_dir_mode),
            "run_name": str(args.run_name),
            "level_console": str(args.log_level),
            "level_file": str(args.log_file_level),
        },
        "report": {
            "results_dir": str(args.results_dir),
            "out": str(args.out),
            "n_samples": int(args.n_samples),
            "seed": int(args.seed),
            "chunk_size": int(args.chunk_size),
            "box_radius_quantile": float(args.box_radius_quantile),
            "box_feas_tol_mw": float(args.box_feas_tol_mw),
            "cert_tol_mw": float(args.cert_tol_mw),
            "cert_max_samples": int(args.cert_max_samples),
            "generate_missing_results": int(args.generate_missing_results),
            "demo": {
                "dc_mode": str(args.dc_mode),
                "slack_bus": int(args.slack_bus),
                "compute_nminus1": int(args.compute_nminus1),
                "dc_chunk_size": int(args.demo_dc_chunk_size),
                "dc_dtype": str(args.demo_dc_dtype),
                "margin_factor": float(args.demo_margin_factor),
                "inj_std_mw": float(args.demo_inj_std_mw),
                "nminus1_update_sensitivities": int(
                    args.demo_nminus1_update_sensitivities
                ),
                "nminus1_islanding": str(args.demo_nminus1_islanding),
            },
        },
        "opf": {
            "solver_name": str(args.opf_solver_name),
            "threads": int(args.opf_threads),
            "random_seed": int(args.opf_random_seed),
            "unconstrained_line_nom_mw": float(args.opf_unconstrained_line_nom_mw),
        },
        "tolerances": {
            "opf_dc_flow_consistency_tol_mw": float(
                args.opf_dc_flow_consistency_tol_mw
            ),
            "opf_bus_balance_tol_mw": float(args.opf_bus_balance_tol_mw),
        },
    }
    _write_run_artifacts(
        run_dir=run_dir, cfg_source_path=cfg_path, cfg_used=cfg_used, argv=argv
    )

    logger.info(
        "Workflow (report): Ensure results -> Monte Carlo Verification -> Generate Report [dc_mode=%s]",
        str(args.dc_mode),
    )

    from stability_radius.verification.generate_report import generate_report_text

    results_dir = Path(_resolve_path(str(args.results_dir)))
    out_path = Path(_resolve_path(str(args.out)))

    opf_cfg = _make_opf_cfg(args)

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
            demo_dc_chunk_size=int(args.demo_dc_chunk_size),
            demo_dc_dtype=_dtype_from_str(str(args.demo_dc_dtype)),
            demo_margin_factor=float(args.demo_margin_factor),
            demo_inj_std_mw=float(args.demo_inj_std_mw),
            demo_nminus1_update_sensitivities=bool(
                args.demo_nminus1_update_sensitivities
            ),
            demo_nminus1_islanding=str(args.demo_nminus1_islanding),
            demo_opf_cfg=opf_cfg,
            demo_opf_dc_flow_consistency_tol_mw=float(
                args.opf_dc_flow_consistency_tol_mw
            ),
            demo_opf_bus_balance_tol_mw=float(args.opf_bus_balance_tol_mw),
            mc_box_radius_quantile=float(args.box_radius_quantile),  # legacy, ignored
            mc_box_feas_tol_mw=float(args.box_feas_tol_mw),
            mc_cert_tol_mw=float(args.cert_tol_mw),
            mc_cert_max_samples=int(args.cert_max_samples),
            strict_units=bool(int(args.strict_units)),
            allow_phase_shift=bool(int(args.allow_phase_shift)),
        )

    with log_stage(logger, "Write Report"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_text, encoding="utf-8")
        logger.info("Wrote report: %s", str(out_path))

        run_copy = run_dir / "verification_report.md"
        run_copy.write_text(report_text, encoding="utf-8")
        logger.info("Wrote report copy: %s", str(run_copy))

    return 0


def run_table(
    args: argparse.Namespace, *, cfg_loaded: Any, cfg_path: Path, argv: Sequence[str]
) -> int:
    """Proxy to `python -m stability_radius.statistics.table`."""
    # Table printing is intentionally kept as a pure function CLI; still create run dir for reproducibility.
    run_dir = _setup_run_and_logging(args)

    cfg_used: dict[str, Any] = {
        "config_path": str(cfg_path),
        "command": "table",
        "run_tests": int(getattr(args, "run_tests", 0)),
        "logging": {
            "runs_dir": str(args.runs_dir),
            "run_dir_mode": str(args.run_dir_mode),
            "run_name": str(args.run_name),
            "level_console": str(args.log_level),
            "level_file": str(args.log_file_level),
        },
        "table_cmd": {
            "results_json": str(args.results_json),
            "max_rows": args.max_rows if args.max_rows is None else int(args.max_rows),
            "radius_field": str(args.radius_field),
            "columns": str(args.columns),
            "table_out": str(args.table_out),
            "csv_out": str(args.csv_out),
        },
    }
    _write_run_artifacts(
        run_dir=run_dir, cfg_source_path=cfg_path, cfg_used=cfg_used, argv=argv
    )

    from stability_radius.statistics.table import main as table_main

    argv2: list[str] = [str(args.results_json)]
    if args.max_rows is not None:
        argv2 += ["--max-rows", str(args.max_rows)]
    if str(args.radius_field).strip():
        argv2 += ["--radius-field", str(args.radius_field)]
    if str(args.columns).strip():
        argv2 += ["--columns", str(args.columns)]
    if str(args.table_out).strip():
        argv2 += ["--table-out", str(args.table_out)]
    if str(args.csv_out).strip():
        argv2 += ["--csv-out", str(args.csv_out)]
    return int(table_main(argv2))


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint."""
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    cfg_path_str = _preparse_config_path(argv_list)
    cfg_path = Path(cfg_path_str)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()

    # If user explicitly requested a config via --config, fail fast on missing file.
    # The default config path (when --config is not provided) can still be missing
    # and the CLI will fall back to built-in defaults.
    user_provided_config = _argv_has_explicit_config_flag(argv_list)

    try:
        cfg_loaded = _load_yaml_config(cfg_path, allow_missing=not user_provided_config)
    except Exception as e:  # noqa: BLE001
        print(
            f"[ERROR] Failed to load config: {str(cfg_path)} ({e})",
            file=sys.stderr,
        )
        return 2

    parser = build_parser(cfg_loaded)

    # First pass: allow missing command to support config-driven default.
    args, unknown = parser.parse_known_args(argv_list)

    argv_effective = list(argv_list)

    if args.command is None:
        default_cmd = _infer_default_command(cfg_loaded)

        if default_cmd is None:
            # Match argparse-like UX, but with a clearer hint.
            parser.print_help(sys.stderr)
            print(
                "\n[ERROR] No command specified. "
                "Either pass a subcommand (demo|monte-carlo|report|table) or set `command: <...>` "
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

        # If the user provided extra args without specifying a command, we can only safely
        # inject the command if those args form a trailing slice. Otherwise we refuse to guess.
        if unknown and not _unknown_is_tail(argv_list, unknown):
            print(
                "[ERROR] Cannot infer command from config because command-specific arguments are interleaved "
                "with global arguments. Specify the command explicitly, e.g.:\n"
                f"  power_stability_radius {default_cmd} ...",
                file=sys.stderr,
            )
            return 2

        prefix = argv_list[: len(argv_list) - len(unknown)] if unknown else argv_list
        argv_effective = [*prefix, default_cmd, *list(unknown)]

        # Second pass: parse with the injected command so subcommand defaults/args exist.
        args = parser.parse_args(argv_effective)
    else:
        # Strict mode: if there are unknown args, re-run strict parse to trigger argparse error.
        if unknown:
            args = parser.parse_args(argv_list)

    if args.command is None:
        # Defensive: should never happen now.
        raise AssertionError("Internal error: command is still None after parsing.")

    # Run self-tests early (before creating a run directory).
    if bool(getattr(args, "run_tests", 0)):
        project_root = (
            Path(__file__).resolve().parents[2]
        )  # <repo>/src/stability_radius/cli.py
        code = _run_self_tests(project_root=project_root)
        if code != 0:
            print(
                f"[ERROR] Self-tests failed (pytest exit code={code}). Aborting.",
                file=sys.stderr,
            )
            return int(code if code > 0 else 1)

    logger.debug("CLI command=%s, argv_effective=%s", str(args.command), argv_effective)

    if args.command == "demo":
        return run_demo(
            args, cfg_loaded=cfg_loaded, cfg_path=cfg_path, argv=argv_effective
        )
    if args.command == "monte-carlo":
        return run_monte_carlo(
            args, cfg_loaded=cfg_loaded, cfg_path=cfg_path, argv=argv_effective
        )
    if args.command == "report":
        return run_report(
            args, cfg_loaded=cfg_loaded, cfg_path=cfg_path, argv=argv_effective
        )
    if args.command == "table":
        return run_table(
            args, cfg_loaded=cfg_loaded, cfg_path=cfg_path, argv=argv_effective
        )

    raise AssertionError(f"Unhandled command: {args.command}")
