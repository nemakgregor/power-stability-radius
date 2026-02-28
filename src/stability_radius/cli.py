from __future__ import annotations

"""
Unified CLI implementation (argparse + Hydra-style YAML defaults).

Commands
--------
- compute
- monte-carlo   (mode: dc|ac)
- report        (strict, DC+AC sections)
- table
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
    HAVE_OMEGACONF,
    HiGHSConfig,
    LoggingConfig,
    OPFConfig,
    OmegaConf,
    load_project_config,
)
from stability_radius.statistics.table import (
    DEFAULT_AC_COLUMNS,
    DEFAULT_DC_COLUMNS,
    format_radius_summary,
    format_results_csv_sections,
    format_results_table,
    format_results_table_sections,
    infer_default_flat_columns,
)
from stability_radius.utils import log_stage, setup_logging
from stability_radius.workflows import (
    ACExtensionsConfig,
    DCExtensionsConfig,
    compute_results_for_case,
)

logger = logging.getLogger("stability_radius.cli")

_SUPPORTED_COMMANDS: tuple[str, ...] = (
    "compute",
    "demo",
    "monte-carlo",
    "report",
    "table",
)


def _dtype_from_str(s: str) -> np.dtype:
    ss = str(s).strip().lower()
    if ss in ("float64", "f64"):
        return np.float64
    if ss in ("float32", "f32"):
        return np.float32
    raise ValueError("dc-dtype must be float64 or float32.")


def _parse_columns(value: str, *, default_columns: Sequence[str]) -> tuple[str, ...]:
    if not str(value).strip():
        return tuple(default_columns)
    return tuple(x.strip() for x in str(value).split(",") if x.strip())


def _resolve_path(p: str) -> str:
    path = Path(str(p).strip()).expanduser()
    return str(path.resolve())


def _run_self_tests(*, project_root: Path) -> int:
    import pytest  # type: ignore

    tests_dir = project_root / "tests"
    if not tests_dir.is_dir():
        raise FileNotFoundError(f"Tests directory not found: {tests_dir}")
    return int(pytest.main(["-q", str(tests_dir)]))


def _preparse_config_path(argv: Sequence[str] | None) -> str:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="conf/config.yaml")
    ns, _ = pre.parse_known_args(list(argv) if argv is not None else None)
    return str(ns.config)


def _unknown_is_tail(argv: Sequence[str], unknown: Sequence[str]) -> bool:
    u = list(unknown)
    a = list(argv)
    if not u:
        return True
    if len(u) > len(a):
        return False
    return a[-len(u) :] == u


def _load_yaml_config(path: Path) -> Any:
    return load_project_config(path, allow_missing=False)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None or (not HAVE_OMEGACONF) or OmegaConf is None:
        return default
    try:
        v = OmegaConf.select(cfg, key)
    except Exception:  # noqa: BLE001
        return default
    return default if v is None else v


def _infer_default_command(cfg_loaded: Any) -> str | None:
    v = _cfg_get(cfg_loaded, "command", None)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _table_columns_from_cfg(
    cfg: Any, key: str, fallback: Sequence[str]
) -> tuple[str, ...]:
    cols = _cfg_get(cfg, key, None)
    if cols is None:
        return tuple(fallback)
    if isinstance(cols, str):
        parsed = [x.strip() for x in cols.split(",") if x.strip()]
        return tuple(parsed) if parsed else tuple(fallback)
    if isinstance(cols, SequenceABC) and not isinstance(cols, (str, bytes, bytearray)):
        out: list[str] = []
        for x in cols:
            sx = str(x).strip()
            if sx:
                out.append(sx)
        return tuple(out) if out else tuple(fallback)
    return tuple(fallback)


def _compose_section_columns(
    cfg: Any, *, section: str, fallback_extra: Sequence[str]
) -> tuple[str, ...]:
    base = _table_columns_from_cfg(cfg, "table.columns", ())
    extra = _table_columns_from_cfg(
        cfg, f"table.{section}_extra_columns", fallback_extra
    )
    return tuple(base) + tuple(extra)


def build_parser(cfg: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="power_stability_radius")

    parser.add_argument("--config", type=str, default="conf/config.yaml")

    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(_cfg_get(cfg, "logging.runs_dir", DEFAULT_LOGGING.runs_dir)),
    )
    parser.add_argument(
        "--run-dir-mode",
        type=str,
        default=str(
            _cfg_get(cfg, "logging.run_dir_mode", DEFAULT_LOGGING.run_dir_mode)
        ),
        choices=("timestamp", "overwrite"),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=str(_cfg_get(cfg, "logging.run_name", DEFAULT_LOGGING.run_name)),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=str(
            _cfg_get(cfg, "logging.level_console", DEFAULT_LOGGING.level_console)
        ),
    )
    parser.add_argument(
        "--log-file-level",
        type=str,
        default=str(_cfg_get(cfg, "logging.level_file", DEFAULT_LOGGING.level_file)),
    )
    parser.add_argument(
        "--run-tests", type=int, default=int(_cfg_get(cfg, "run_tests", 1))
    )

    parser.add_argument(
        "--allow-download", type=int, default=int(_cfg_get(cfg, "io.allow_download", 0))
    )

    # OPF settings
    parser.add_argument(
        "--opf-solver-name",
        type=str,
        default=str(_cfg_get(cfg, "opf.solver_name", DEFAULT_OPF.highs.solver_name)),
    )
    parser.add_argument(
        "--opf-threads",
        type=int,
        default=int(_cfg_get(cfg, "opf.threads", DEFAULT_OPF.highs.threads)),
    )
    parser.add_argument(
        "--opf-random-seed",
        type=int,
        default=int(_cfg_get(cfg, "opf.random_seed", DEFAULT_OPF.highs.random_seed)),
    )
    parser.add_argument(
        "--opf-headroom-factor",
        type=float,
        default=float(
            _cfg_get(cfg, "opf.headroom_factor", DEFAULT_OPF.headroom_factor)
        ),
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
    )
    parser.add_argument(
        "--opf-ext-grid-marginal-cost-base",
        type=float,
        default=float(
            _cfg_get(
                cfg,
                "opf.ext_grid_marginal_cost_base",
                DEFAULT_OPF.ext_grid_marginal_cost_base,
            )
        ),
    )

    # tolerances
    parser.add_argument(
        "--opf-dc-flow-consistency-tol-mw",
        type=float,
        default=float(_cfg_get(cfg, "tolerances.opf_dc_flow_consistency_tol_mw", 1e-3)),
    )
    parser.add_argument(
        "--opf-bus-balance-tol-mw",
        type=float,
        default=float(_cfg_get(cfg, "tolerances.opf_bus_balance_tol_mw", 1e-6)),
    )

    sub = parser.add_subparsers(dest="command", required=False)

    # ---------- compute ----------
    p_compute = sub.add_parser("compute", aliases=["demo"])
    p_compute.add_argument(
        "--input",
        type=str,
        default=str(
            _cfg_get(cfg, "compute.input", "data/input/pglib_opf_case30_ieee.m")
        ),
    )
    p_compute.add_argument(
        "--slack-bus", type=int, default=int(_cfg_get(cfg, "compute.slack_bus", 0))
    )

    p_compute.add_argument(
        "--base-dispatch",
        type=str,
        default=str(_cfg_get(cfg, "compute.base_dispatch", "case")),
        choices=("case", "dc_opf"),
    )

    # DC
    p_compute.add_argument(
        "--compute-dc", type=int, default=int(_cfg_get(cfg, "compute.dc.compute", 1))
    )
    p_compute.add_argument(
        "--dc-mode",
        type=str,
        default=str(_cfg_get(cfg, "dc.mode", DEFAULT_DC.mode)),
        choices=("materialize", "operator"),
    )
    p_compute.add_argument(
        "--dc-chunk-size",
        type=int,
        default=int(_cfg_get(cfg, "dc.chunk_size", DEFAULT_DC.chunk_size)),
    )
    p_compute.add_argument(
        "--dc-dtype",
        type=str,
        default=str(_cfg_get(cfg, "dc.dtype", DEFAULT_DC.dtype)),
        choices=("float64", "float32"),
    )
    p_compute.add_argument(
        "--inj-std-mw",
        type=float,
        default=float(_cfg_get(cfg, "compute.dc.inj_std_mw", 1.0)),
    )

    # DC extensions (advanced): keep out of the main option list.
    g_dc_ext = p_compute.add_argument_group("DC extensions (advanced; optional)")
    g_dc_ext.add_argument(
        "--compute-dc-probabilistic",
        type=int,
        default=int(_cfg_get(cfg, "compute.dc.probabilistic.enabled", 0)),
        help="If 1, compute DC sigma-radius and overload_probability (post-processing). Default: 0.",
    )
    g_dc_ext.add_argument(
        "--compute-nminus1",
        type=int,
        default=int(_cfg_get(cfg, "compute.dc.nminus1.enabled", 0)),
        help="If 1, compute effective N-1 DC radii (requires --dc-mode materialize). Default: 0.",
    )
    g_dc_ext.add_argument(
        "--nminus1-update-sensitivities",
        type=int,
        default=int(_cfg_get(cfg, "compute.dc.nminus1.update_sensitivities", 1)),
        help="If 1, update sensitivities via LODF approximation (more accurate, slower). Default: 1.",
    )
    g_dc_ext.add_argument(
        "--nminus1-islanding",
        type=str,
        default=str(
            _cfg_get(cfg, "compute.dc.nminus1.islanding", DEFAULT_NMINUS1_ISLANDING)
        ),
        choices=("skip", "raise"),
        help="How to handle islanding/undefined LODF contingencies. Default: skip.",
    )

    # AC
    p_compute.add_argument(
        "--compute-ac", type=int, default=int(_cfg_get(cfg, "compute.ac.compute", 0))
    )
    p_compute.add_argument(
        "--ac-chunk-size",
        type=int,
        default=int(_cfg_get(cfg, "compute.ac.chunk_size", 256)),
    )
    p_compute.add_argument(
        "--ac-balance", type=int, default=int(_cfg_get(cfg, "compute.ac.balance", 1))
    )
    p_compute.add_argument(
        "--ac-pf-solver",
        type=str,
        default=str(_cfg_get(cfg, "ac.pf_solver", "pandapower")),
        choices=("pandapower", "pypsa"),
    )
    p_compute.add_argument(
        "--ac-pf-init",
        type=str,
        default=str(_cfg_get(cfg, "compute.ac.pf_init", "flat")),
        choices=("flat", "dc", "pp"),
    )
    p_compute.add_argument(
        "--ac-lossless", type=int, default=int(_cfg_get(cfg, "ac.lossless", 1))
    )

    # AC extensions (sigma-radius, metric-radius, h-vector saving)
    p_compute.add_argument(
        "--ac-sigma-p-source",
        type=str,
        default=str(_cfg_get(cfg, "compute.ac.sigma.sigma_p_mw_source", "")),
    )
    p_compute.add_argument(
        "--ac-sigma-q-source",
        type=str,
        default=str(_cfg_get(cfg, "compute.ac.sigma.sigma_q_mvar_source", "")),
    )
    p_compute.add_argument(
        "--ac-sigma-p-uniform",
        type=float,
        default=float(_cfg_get(cfg, "compute.ac.sigma.sigma_p_mw_uniform", 1.0)),
    )
    p_compute.add_argument(
        "--ac-sigma-q-uniform",
        type=float,
        default=float(_cfg_get(cfg, "compute.ac.sigma.sigma_q_mvar_uniform", 1.0)),
    )
    p_compute.add_argument(
        "--ac-metric-enabled",
        type=int,
        default=int(_cfg_get(cfg, "compute.ac.metric.enabled", 0)),
    )
    p_compute.add_argument(
        "--ac-save-h-vectors",
        type=int,
        default=int(_cfg_get(cfg, "compute.ac.save_h_vectors", 0)),
    )

    # outputs
    p_compute.add_argument(
        "--export-results",
        type=str,
        default=str(_cfg_get(cfg, "compute.output.export_results", "")),
    )
    p_compute.add_argument(
        "--save-csv", type=int, default=int(_cfg_get(cfg, "compute.output.save_csv", 1))
    )
    p_compute.add_argument(
        "--max-rows", type=int, default=_cfg_get(cfg, "compute.output.max_rows", None)
    )
    p_compute.add_argument(
        "--table-columns",
        type=str,
        default=str(_cfg_get(cfg, "compute.output.table_columns", "")),
    )

    # ---------- monte-carlo ----------
    p_mc = sub.add_parser("monte-carlo")
    p_mc.add_argument(
        "--mode",
        type=str,
        default=str(_cfg_get(cfg, "monte_carlo.mode", "dc")),
        choices=("dc", "ac"),
    )
    p_mc.add_argument(
        "--results", type=str, default=str(_cfg_get(cfg, "monte_carlo.results", ""))
    )
    p_mc.add_argument(
        "--input", type=str, default=str(_cfg_get(cfg, "monte_carlo.input", ""))
    )
    p_mc.add_argument(
        "--slack-bus",
        type=int,
        default=int(
            _cfg_get(
                cfg, "monte_carlo.slack_bus", _cfg_get(cfg, "compute.slack_bus", 0)
            )
        ),
    )

    p_mc.add_argument(
        "--n-samples",
        type=int,
        default=int(
            _cfg_get(cfg, "monte_carlo.sampling.n_samples", DEFAULT_MC.n_samples)
        ),
    )
    p_mc.add_argument(
        "--seed",
        type=int,
        default=int(_cfg_get(cfg, "monte_carlo.sampling.seed", DEFAULT_MC.seed)),
    )
    p_mc.add_argument(
        "--chunk-size",
        type=int,
        default=int(
            _cfg_get(cfg, "monte_carlo.sampling.chunk_size", DEFAULT_MC.chunk_size)
        ),
    )

    p_mc.add_argument(
        "--feas-tol",
        type=float,
        default=float(
            _cfg_get(cfg, "monte_carlo.tolerances.feas_tol", DEFAULT_MC.feas_tol_mw)
        ),
    )
    p_mc.add_argument(
        "--cert-tol",
        type=float,
        default=float(
            _cfg_get(cfg, "monte_carlo.tolerances.cert_tol", DEFAULT_MC.cert_tol_mw)
        ),
    )
    p_mc.add_argument(
        "--cert-max-samples",
        type=int,
        default=int(
            _cfg_get(
                cfg,
                "monte_carlo.tolerances.cert_max_samples",
                DEFAULT_MC.cert_max_samples,
            )
        ),
    )

    # DC-only
    p_mc.add_argument(
        "--sigma-override-mw",
        type=float,
        default=_cfg_get(cfg, "monte_carlo.dc.sigma_override_mw", None),
    )

    # AC-only distribution
    p_mc.add_argument(
        "--ac-sigma-p-mw",
        type=float,
        default=_cfg_get(cfg, "monte_carlo.ac.sigma_p_mw", None),
    )
    p_mc.add_argument(
        "--ac-sigma-q-mvar",
        type=float,
        default=_cfg_get(cfg, "monte_carlo.ac.sigma_q_mvar", None),
    )

    # AC PF backend policy (shared with compute)
    p_mc.add_argument(
        "--ac-pf-solver",
        type=str,
        default=str(_cfg_get(cfg, "ac.pf_solver", "pandapower")),
        choices=("pandapower", "pypsa"),
    )
    p_mc.add_argument(
        "--ac-lossless", type=int, default=int(_cfg_get(cfg, "ac.lossless", 1))
    )
    p_mc.add_argument(
        "--ac-basepoint-s-tol-mva",
        type=float,
        default=float(_cfg_get(cfg, "ac.basepoint_s_tol_mva", 1e-3)),
    )

    # ---------- report ----------
    p_rep = sub.add_parser("report")
    p_rep.add_argument(
        "--results-dir",
        type=str,
        default=str(_cfg_get(cfg, "report.io.results_dir", "verification/results")),
    )
    p_rep.add_argument(
        "--out",
        type=str,
        default=str(_cfg_get(cfg, "report.io.out", "verification/report.md")),
    )
    p_rep.add_argument(
        "--n-samples",
        type=int,
        default=int(_cfg_get(cfg, "report.sampling.n_samples", DEFAULT_MC.n_samples)),
    )
    p_rep.add_argument(
        "--seed",
        type=int,
        default=int(_cfg_get(cfg, "report.sampling.seed", DEFAULT_MC.seed)),
    )
    p_rep.add_argument(
        "--chunk-size",
        type=int,
        default=int(_cfg_get(cfg, "report.sampling.chunk_size", DEFAULT_MC.chunk_size)),
    )

    p_rep.add_argument(
        "--feas-tol",
        type=float,
        default=float(
            _cfg_get(cfg, "report.tolerances.feas_tol", DEFAULT_MC.feas_tol_mw)
        ),
    )
    p_rep.add_argument(
        "--cert-tol",
        type=float,
        default=float(
            _cfg_get(cfg, "report.tolerances.cert_tol", DEFAULT_MC.cert_tol_mw)
        ),
    )
    p_rep.add_argument(
        "--cert-max-samples",
        type=int,
        default=int(
            _cfg_get(
                cfg, "report.tolerances.cert_max_samples", DEFAULT_MC.cert_max_samples
            )
        ),
    )
    p_rep.add_argument(
        "--strict", type=int, default=int(_cfg_get(cfg, "report.strict", 1))
    )

    # DC-only (report verification): override Gaussian sigma (MW). If null, uses results.__meta__.dc.inj_std_mw.
    p_rep.add_argument(
        "--sigma-override-mw",
        type=float,
        default=_cfg_get(cfg, "report.dc.sigma_override_mw", None),
    )

    # AC distribution for report verification
    p_rep.add_argument(
        "--ac-sigma-p-mw",
        type=float,
        default=float(_cfg_get(cfg, "report.ac.sigma_p_mw", 1.0)),
    )
    p_rep.add_argument(
        "--ac-sigma-q-mvar",
        type=float,
        default=float(_cfg_get(cfg, "report.ac.sigma_q_mvar", 1.0)),
    )

    # AC PF backend policy (shared)
    p_rep.add_argument(
        "--ac-pf-solver",
        type=str,
        default=str(_cfg_get(cfg, "ac.pf_solver", "pandapower")),
        choices=("pandapower", "pypsa"),
    )
    p_rep.add_argument(
        "--ac-lossless", type=int, default=int(_cfg_get(cfg, "ac.lossless", 1))
    )
    p_rep.add_argument(
        "--ac-basepoint-s-tol-mva",
        type=float,
        default=float(_cfg_get(cfg, "ac.basepoint_s_tol_mva", 1e-3)),
    )

    # ---------- table ----------
    p_table = sub.add_parser("table")
    p_table.add_argument("results_json", type=str)
    p_table.add_argument("--max-rows", type=int, default=None)
    p_table.add_argument(
        "--format",
        type=str,
        choices=("sections", "flat"),
        default=str(_cfg_get(cfg, "table.format", "sections")),
    )
    p_table.add_argument("--radius-field", type=str, default="radius_l2")
    # IMPORTANT: do not default from YAML list values (OmegaConf ListConfig -> "[]").
    # For flat mode, empty means "infer defaults from results".
    p_table.add_argument("--columns", type=str, default="")
    p_table.add_argument("--table-out", type=str, default="")
    p_table.add_argument("--csv-out", type=str, default="")

    return parser


def _write_run_artifacts(
    *,
    run_dir: Path,
    cfg_source_path: Path,
    cfg_used: dict[str, Any],
    argv: Sequence[str],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "argv.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")
    if cfg_source_path.exists():
        shutil.copyfile(cfg_source_path, run_dir / "config_source.yaml")
    cfg_json = json.dumps(cfg_used, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    (run_dir / "config.json").write_text(cfg_json, encoding="utf-8")
    if HAVE_OMEGACONF and OmegaConf is not None:
        cfg_yaml = OmegaConf.to_yaml(OmegaConf.create(cfg_used))  # type: ignore[union-attr]
        (run_dir / "config.yaml").write_text(cfg_yaml, encoding="utf-8")


def _make_opf_cfg(args: argparse.Namespace) -> OPFConfig:
    highs = HiGHSConfig(
        solver_name=str(args.opf_solver_name),
        threads=int(args.opf_threads),
        random_seed=int(args.opf_random_seed),
    )
    return OPFConfig(
        highs=highs,
        unconstrained_line_nom_mw=float(args.opf_unconstrained_line_nom_mw),
        headroom_factor=float(args.opf_headroom_factor),
        ext_grid_marginal_cost_base=float(args.opf_ext_grid_marginal_cost_base),
    )


def _setup_run_and_logging(args: argparse.Namespace) -> Path:
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


def _results_has_field(results: dict[str, Any], field: str) -> bool:
    """Return True iff at least one per-line dict contains `field`."""
    for k, v in results.items():
        if not k.startswith("line_") or not isinstance(v, dict):
            continue
        if field in v:
            return True
    return False


def run_compute(
    args: argparse.Namespace, *, cfg_loaded: Any, cfg_path: Path, argv: Sequence[str]
) -> int:
    if not str(getattr(args, "input", "")).strip():
        raise ValueError("compute requires --input (empty).")

    run_dir = _setup_run_and_logging(args)

    cfg_used: dict[str, Any] = {
        "config_path": str(cfg_path),
        "command": str(args.command),
        "allow_download": int(getattr(args, "allow_download", 0)),
        "compute": {
            "input": str(args.input),
            "slack_bus": int(args.slack_bus),
            "base_dispatch": str(args.base_dispatch),
            "dc": {
                "compute": int(args.compute_dc),
                "mode": str(args.dc_mode),
                "chunk_size": int(args.dc_chunk_size),
                "dtype": str(args.dc_dtype),
                "inj_std_mw": float(args.inj_std_mw),
                "probabilistic": {
                    "enabled": int(args.compute_dc_probabilistic),
                },
                "nminus1": {
                    "enabled": int(args.compute_nminus1),
                    "update_sensitivities": int(args.nminus1_update_sensitivities),
                    "islanding": str(args.nminus1_islanding),
                },
            },
            "ac": {
                "compute": int(args.compute_ac),
                "chunk_size": int(args.ac_chunk_size),
                "balance": int(args.ac_balance),
                "pf_solver": str(args.ac_pf_solver),
                "pf_init": str(args.ac_pf_init),
                "lossless": int(args.ac_lossless),
                "sigma": {
                    "sigma_p_mw_source": str(args.ac_sigma_p_source),
                    "sigma_q_mvar_source": str(args.ac_sigma_q_source),
                    "sigma_p_mw_uniform": float(args.ac_sigma_p_uniform),
                    "sigma_q_mvar_uniform": float(args.ac_sigma_q_uniform),
                },
                "metric": {
                    "enabled": int(args.ac_metric_enabled),
                },
                "save_h_vectors": int(args.ac_save_h_vectors),
            },
            "output": {
                "export_results": str(args.export_results),
                "save_csv": int(args.save_csv),
                "max_rows": args.max_rows,
                "table_columns": str(args.table_columns),
            },
        },
    }
    _write_run_artifacts(
        run_dir=run_dir, cfg_source_path=cfg_path, cfg_used=cfg_used, argv=argv
    )

    opf_cfg = _make_opf_cfg(args)

    dc_ext = DCExtensionsConfig(
        probabilistic_enabled=bool(int(args.compute_dc_probabilistic)),
        nminus1_enabled=bool(int(args.compute_nminus1)),
        nminus1_update_sensitivities=bool(int(args.nminus1_update_sensitivities)),
        nminus1_islanding=str(args.nminus1_islanding),
    )

    ac_ext = ACExtensionsConfig(
        sigma_p_mw_source=str(args.ac_sigma_p_source),
        sigma_q_mvar_source=str(args.ac_sigma_q_source),
        sigma_p_mw_uniform=float(args.ac_sigma_p_uniform),
        sigma_q_mvar_uniform=float(args.ac_sigma_q_uniform),
        metric_enabled=bool(int(args.ac_metric_enabled)),
        save_h_vectors=bool(int(args.ac_save_h_vectors)),
    )

    results = compute_results_for_case(
        input_path=str(args.input),
        slack_bus=int(args.slack_bus),
        base_dispatch=str(args.base_dispatch),
        compute_dc=bool(args.compute_dc),
        dc_mode=str(args.dc_mode),
        dc_chunk_size=int(args.dc_chunk_size),
        dc_dtype=_dtype_from_str(str(args.dc_dtype)),
        dc_inj_std_mw=float(args.inj_std_mw),
        dc_extensions=dc_ext,
        compute_ac=bool(args.compute_ac),
        ac_chunk_size=int(args.ac_chunk_size),
        ac_balance=bool(args.ac_balance),
        ac_pf_init=str(args.ac_pf_init),
        ac_pf_solver=str(args.ac_pf_solver),
        ac_lossless=bool(int(args.ac_lossless)),
        ac_extensions=ac_ext,
        opf_cfg=opf_cfg,
        opf_dc_flow_consistency_tol_mw=float(args.opf_dc_flow_consistency_tol_mw),
        opf_bus_balance_tol_mw=float(args.opf_bus_balance_tol_mw),
        path_base_dir=Path.cwd(),
        allow_download=bool(args.allow_download),
    )

    # Save h-vectors to .npz if present (non-JSON-serializable, must be extracted first).
    h_vectors_data = results.pop("_h_vectors", None)
    if h_vectors_data is not None:
        h_path = run_dir / "h_vectors.npz"
        with log_stage(logger, "Write h-vectors (.npz)"):
            np.savez_compressed(str(h_path), **h_vectors_data)
            logger.info("Saved h-vectors: %s", str(h_path))

    with log_stage(logger, "Write Results (JSON)"):
        (run_dir / "results.json").write_text(
            json.dumps(results, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # Table output:
    max_rows = int(args.max_rows) if args.max_rows is not None else None
    if str(args.table_columns).strip():
        cols = _parse_columns(
            str(args.table_columns), default_columns=DEFAULT_DC_COLUMNS
        )
        table_str = format_results_table(results, columns=cols, max_rows=max_rows)
        (run_dir / "results_table.txt").write_text(table_str + "\n", encoding="utf-8")
        if bool(args.save_csv):
            csv_text = format_results_csv_sections(
                results, dc_columns=cols, ac_columns=()
            )
            (run_dir / "results_table.csv").write_text(
                csv_text.get("dc", ""), encoding="utf-8"
            )
    else:
        dc_cols = _compose_section_columns(
            cfg_loaded, section="dc", fallback_extra=DEFAULT_DC_COLUMNS
        )
        ac_cols = _compose_section_columns(
            cfg_loaded, section="ac", fallback_extra=DEFAULT_AC_COLUMNS
        )
        table_str = format_results_table_sections(
            results, dc_columns=dc_cols, ac_columns=ac_cols, max_rows=max_rows
        )
        (run_dir / "results_table.txt").write_text(table_str + "\n", encoding="utf-8")

        if bool(args.save_csv):
            csvs = format_results_csv_sections(
                results, dc_columns=dc_cols, ac_columns=ac_cols, max_rows=max_rows
            )
            if "dc" in csvs:
                (run_dir / "results_table_dc.csv").write_text(
                    csvs["dc"], encoding="utf-8"
                )
            if "ac" in csvs:
                (run_dir / "results_table_ac.csv").write_text(
                    csvs["ac"], encoding="utf-8"
                )

    # Summaries: only for fields that exist (no "compat" spam).
    summary_fields: list[str] = []
    if _results_has_field(results, "radius_l2"):
        summary_fields.append("radius_l2")
    if _results_has_field(results, "radius_sigma"):
        summary_fields.append("radius_sigma")
    if _results_has_field(results, "radius_nminus1"):
        summary_fields.append("radius_nminus1")
    if _results_has_field(results, "radius_ac_l2"):
        summary_fields.append("radius_ac_l2")
    if _results_has_field(results, "radius_ac_sigma"):
        summary_fields.append("radius_ac_sigma")
    if _results_has_field(results, "radius_ac_metric"):
        summary_fields.append("radius_ac_metric")

    for field in summary_fields:
        logger.info("%s", format_radius_summary(results, radius_field=field))

    if str(args.export_results).strip():
        export_path_abs = _resolve_path(str(args.export_results))
        Path(export_path_abs).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / "results.json", export_path_abs)

    logger.info("Done. Run directory: %s", str(run_dir))
    return 0


def _parse_report_cases_from_cfg(
    *, cfg_loaded: Any, results_dir_abs: Path, base_dir: Path
) -> list[dict[str, Any]]:
    raw = _cfg_get(cfg_loaded, "report.cases", None)

    def _is_seq_not_str(x: Any) -> bool:
        return isinstance(x, SequenceABC) and not isinstance(x, (str, bytes, bytearray))

    if raw is None or not _is_seq_not_str(raw) or len(raw) == 0:
        raise ValueError(
            "Missing required config key `report.cases` (must be a non-empty list)."
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
        if not case_id or not input_path or not results_name:
            raise ValueError(f"report.cases[{i}] must have id/input/results.")

        rp = Path(results_name).expanduser()
        rp_abs = rp if rp.is_absolute() else (results_dir_abs / rp).resolve()

        ip = Path(input_path).expanduser()
        ip_abs = ip if ip.is_absolute() else (base_dir / ip).resolve()

        known = item.get("known_critical_pairs", None)
        known_pairs: list[list[int]] = []
        if known is not None:
            if not _is_seq_not_str(known):
                raise ValueError(
                    f"report.cases[{i}].known_critical_pairs must be a list."
                )
            for j, p in enumerate(known):
                if not _is_seq_not_str(p) or len(p) != 2:
                    raise ValueError(
                        f"report.cases[{i}].known_critical_pairs[{j}] must be a 2-element pair."
                    )
                known_pairs.append([int(p[0]), int(p[1])])

        out.append(
            {
                "id": case_id,
                "input": str(ip_abs),
                "results": str(rp_abs),
                "known_critical_pairs": known_pairs,
            }
        )
    return out


def run_report(
    args: argparse.Namespace, *, cfg_loaded: Any, cfg_path: Path, argv: Sequence[str]
) -> int:
    if cfg_loaded is None:
        raise ValueError("report requires a loaded YAML config (report.cases).")

    results_dir = Path(_resolve_path(str(args.results_dir)))
    out_path = Path(_resolve_path(str(args.out)))

    base_dir = Path.cwd().resolve()
    cases_cfg = _parse_report_cases_from_cfg(
        cfg_loaded=cfg_loaded, results_dir_abs=results_dir.resolve(), base_dir=base_dir
    )

    run_dir = _setup_run_and_logging(args)

    from stability_radius.verification.generate_report import (
        ReportCaseSpec,
        generate_report_text,
    )

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

    report_text = generate_report_text(
        cases=case_specs,
        results_dir=results_dir.resolve(),
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
        feas_tol=float(args.feas_tol),
        cert_tol=float(args.cert_tol),
        cert_max_samples=int(args.cert_max_samples),
        strict=bool(int(args.strict)),
        dc_sigma_override_mw=(
            None if args.sigma_override_mw is None else float(args.sigma_override_mw)
        ),
        ac_sigma_p_mw=float(args.ac_sigma_p_mw),
        ac_sigma_q_mvar=float(args.ac_sigma_q_mvar),
        ac_pf_solver=str(args.ac_pf_solver),
        ac_lossless=bool(int(args.ac_lossless)),
        ac_basepoint_s_tol_mva=float(args.ac_basepoint_s_tol_mva),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    (run_dir / "verification_report.md").write_text(report_text, encoding="utf-8")
    logger.info("Wrote report: %s", str(out_path))
    return 0


def run_monte_carlo(
    args: argparse.Namespace, *, cfg_loaded: Any, cfg_path: Path, argv: Sequence[str]
) -> int:
    results_path_raw = str(getattr(args, "results", "")).strip()
    input_path_raw = str(getattr(args, "input", "")).strip()
    if not results_path_raw:
        raise ValueError("monte-carlo requires --results (path to results.json).")
    if not input_path_raw:
        raise ValueError("monte-carlo requires --input (path to .m case).")

    results_path = Path(_resolve_path(results_path_raw))
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found: {results_path}")

    input_case_path = Path(_resolve_path(input_path_raw))

    run_dir = _setup_run_and_logging(args)

    cfg_used: dict[str, Any] = {
        "config_path": str(cfg_path),
        "command": "monte-carlo",
        "allow_download": int(getattr(args, "allow_download", 0)),
        "monte_carlo": {
            "mode": str(args.mode),
            "results": str(results_path),
            "input": str(input_case_path),
            "slack_bus": int(args.slack_bus),
            "n_samples": int(args.n_samples),
            "seed": int(args.seed),
            "chunk_size": int(args.chunk_size),
            "feas_tol": float(args.feas_tol),
            "cert_tol": float(args.cert_tol),
            "cert_max_samples": int(args.cert_max_samples),
            "sigma_override_mw": args.sigma_override_mw
            if args.sigma_override_mw is None
            else float(args.sigma_override_mw),
            "ac_sigma_p_mw": args.ac_sigma_p_mw,
            "ac_sigma_q_mvar": args.ac_sigma_q_mvar,
            "ac_pf_solver": str(args.ac_pf_solver),
            "ac_lossless": int(args.ac_lossless),
            "ac_basepoint_s_tol_mva": float(args.ac_basepoint_s_tol_mva),
        },
    }
    _write_run_artifacts(
        run_dir=run_dir, cfg_source_path=cfg_path, cfg_used=cfg_used, argv=argv
    )

    from stability_radius.verification.monte_carlo import run_monte_carlo_verification

    vr = run_monte_carlo_verification(
        mode=str(args.mode),
        results_path=results_path,
        input_case_path=input_case_path,
        slack_bus=int(args.slack_bus),
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
        feas_tol=float(args.feas_tol),
        cert_tol=float(args.cert_tol),
        cert_max_samples=int(args.cert_max_samples),
        sigma_override_mw=args.sigma_override_mw
        if args.sigma_override_mw is None
        else float(args.sigma_override_mw),
        allow_download=bool(args.allow_download),
        ac_sigma_p_mw=args.ac_sigma_p_mw
        if args.ac_sigma_p_mw is None
        else float(args.ac_sigma_p_mw),
        ac_sigma_q_mvar=args.ac_sigma_q_mvar
        if args.ac_sigma_q_mvar is None
        else float(args.ac_sigma_q_mvar),
        ac_pf_solver=str(args.ac_pf_solver),
        ac_lossless=bool(int(args.ac_lossless)),
        ac_basepoint_s_tol_mva=float(args.ac_basepoint_s_tol_mva),
    )

    stats_json = (
        json.dumps(vr.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    (run_dir / "monte_carlo_stats.json").write_text(stats_json, encoding="utf-8")
    sys.stdout.write(stats_json)
    return 0


def run_table(
    args: argparse.Namespace, *, cfg_loaded: Any, cfg_path: Path, argv: Sequence[str]
) -> int:
    results_path = Path(_resolve_path(str(getattr(args, "results_json", "")).strip()))
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found: {results_path}")

    run_dir = _setup_run_and_logging(args)

    obj = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("results.json must contain a JSON object.")
    results: dict[str, Any] = obj

    max_rows = int(args.max_rows) if args.max_rows is not None else None
    if str(args.format) == "flat":
        if str(args.columns).strip():
            cols = tuple(x.strip() for x in str(args.columns).split(",") if x.strip())
        else:
            cols = infer_default_flat_columns(results)
        table_str = format_results_table(results, columns=cols, max_rows=max_rows)
    else:
        dc_cols = _compose_section_columns(
            cfg_loaded, section="dc", fallback_extra=DEFAULT_DC_COLUMNS
        )
        ac_cols = _compose_section_columns(
            cfg_loaded, section="ac", fallback_extra=DEFAULT_AC_COLUMNS
        )
        table_str = format_results_table_sections(
            results, dc_columns=dc_cols, ac_columns=ac_cols, max_rows=max_rows
        )

    summary = format_radius_summary(results, radius_field=str(args.radius_field))
    print(table_str)
    print(summary)
    (run_dir / "results_table.txt").write_text(table_str + "\n", encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    cfg_path_str = _preparse_config_path(argv_list)
    cfg_path = Path(cfg_path_str).expanduser()
    cfg_path = (
        (Path.cwd() / cfg_path).resolve()
        if not cfg_path.is_absolute()
        else cfg_path.resolve()
    )

    try:
        cfg_loaded = _load_yaml_config(cfg_path)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Failed to load config: {str(cfg_path)} ({e})", file=sys.stderr)
        return 2

    parser = build_parser(cfg_loaded)
    args, unknown = parser.parse_known_args(argv_list)
    argv_effective = list(argv_list)

    if args.command is None:
        default_cmd = _infer_default_command(cfg_loaded)
        if default_cmd is None:
            parser.print_help(sys.stderr)
            print(
                "\n[ERROR] No command specified and no `command:` in config.",
                file=sys.stderr,
            )
            return 2
        if default_cmd not in _SUPPORTED_COMMANDS:
            print(
                f"[ERROR] Invalid config default `command: {default_cmd}`.",
                file=sys.stderr,
            )
            return 2
        if unknown and not _unknown_is_tail(argv_list, unknown):
            print(
                "[ERROR] Cannot infer command from config: interleaved command args.",
                file=sys.stderr,
            )
            return 2
        prefix = argv_list[: len(argv_list) - len(unknown)] if unknown else argv_list
        argv_effective = [*prefix, default_cmd, *list(unknown)]
        args = parser.parse_args(argv_effective)
    else:
        if unknown:
            args = parser.parse_args(argv_list)

    if bool(getattr(args, "run_tests", 0)):
        project_root = Path(__file__).resolve().parents[2]
        code = _run_self_tests(project_root=project_root)
        if code != 0:
            print(
                f"[ERROR] Self-tests failed (pytest exit code={code}).", file=sys.stderr
            )
            return int(code if code > 0 else 1)

    if args.command in {"compute", "demo"}:
        return run_compute(
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
