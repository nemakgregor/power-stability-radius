"""Experiment 1: DC vs AC radius sweep across PGLib networks.

Reads ``experiments/configs/pglib_sweep.yaml``, computes DC and AC L2 radii
for each listed PGLib case, and produces:

- Per-case JSON results in ``experiments/output/pglib_sweep/``
- ``summary.json`` with aggregated metrics
- **Table 1** (printed to stdout): case, n_b, n_l, r*_DC, r*_AC, AC/DC, time, bottleneck
- **Fig. 1** (saved as PNG): bar chart comparing r*_DC and r*_AC across cases

Usage::

    python -m experiments.run_pglib_sweep
    python -m experiments.run_pglib_sweep --config experiments/configs/pglib_sweep.yaml

Key design choice
-----------------
DC and AC radii are computed in a **single** ``compute_results_for_case`` call
with ``compute_dc=True, compute_ac=True``.  This ensures both share the same
OPF dispatch and base point, making the comparison valid.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from stability_radius.config import OPFConfig
from stability_radius.parsers.matpower import load_network
from stability_radius.utils.download import ensure_case_file
from stability_radius.workflows import (
    DCExtensionsConfig,
    compute_results_for_case,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "pglib_sweep.yaml"


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _numpy_serialiser(obj: object) -> object:
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _detect_slack_bus(net) -> int:
    """Auto-detect the slack bus from the pandapower ext_grid table.

    PGLib-OPF cases converted via ``from_ppc`` map the MATPOWER type-3
    (slack) bus generator to ``net.ext_grid``.  We pick the bus of the
    first in-service ext_grid entry.
    """
    if hasattr(net, "ext_grid") and net.ext_grid is not None and len(net.ext_grid):
        for _, row in net.ext_grid.iterrows():
            if bool(row.get("in_service", True)):
                return int(row["bus"])
    # Fallback: first bus in sorted order.
    return int(sorted(net.bus.index)[0])


def _compute_case(
    *,
    input_path: str,
    slack_bus: int,
    base_dispatch: str,
    dc_cfg: dict,
    ac_cfg: dict,
    ac_fpf_cfg: dict,
    opf_cfg: OPFConfig,
    allow_download: bool,
    opf_dc_flow_consistency_tol_mw: float,
) -> dict:
    """Run compute_results_for_case with DC+AC both enabled (shared base point)."""
    return compute_results_for_case(
        input_path=input_path,
        slack_bus=slack_bus,
        base_dispatch=base_dispatch,
        # DC
        compute_dc=True,
        dc_mode=str(dc_cfg.get("mode", "materialize")),
        dc_chunk_size=int(dc_cfg.get("chunk_size", 64)),
        dc_dtype=np.dtype(dc_cfg.get("dtype", "float64")),
        dc_inj_std_mw=float(dc_cfg.get("inj_std_mw", 10.0)),
        dc_extensions=DCExtensionsConfig(probabilistic_enabled=True),
        # AC
        compute_ac=True,
        ac_chunk_size=int(ac_cfg.get("chunk_size", 64)),
        ac_balance=bool(ac_cfg.get("balance", True)),
        ac_pf_init=str(ac_cfg.get("pf_init", "flat")),
        ac_pf_solver=str(ac_cfg.get("pf_solver", "pandapower")),
        ac_lossless=bool(ac_cfg.get("lossless", True)),
        ac_distributed_slack=bool(ac_cfg.get("distributed_slack", False)),
        ac_trafo_model=str(ac_cfg.get("trafo_model", "pi")),
        # AC FPF
        ac_fpf_pg0_source=str(ac_fpf_cfg.get("pg0_source", "case")),
        # OPF
        opf_cfg=opf_cfg,
        opf_dc_flow_consistency_tol_mw=float(opf_dc_flow_consistency_tol_mw),
        # shared
        allow_download=allow_download,
    )


def _case_worker(
    result_queue: mp.Queue, kwargs: dict, log_level: int = logging.INFO
) -> None:
    """Run ``_compute_case`` in a child process.

    With ``fork`` context (Linux default) the parent's logging config is
    inherited so solver messages still appear in debug.log.
    On Windows (``spawn`` context) the child must configure logging itself.
    """
    # Ensure logging is configured in the child process (critical for Windows
    # where ``spawn`` gives us a fresh interpreter with no handlers).
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s [subprocess] %(name)s: %(message)s",
        force=True,  # override if already configured (e.g. Linux fork)
    )
    child_logger = logging.getLogger(__name__)
    case_name = kwargs.get("input_path", "unknown")
    child_logger.info(
        "Subprocess started for case: %s (PID=%d)", case_name, os.getpid()
    )
    try:
        result = _compute_case(**kwargs)
        result.pop("_h_vectors", None)
        child_logger.info("Subprocess finished OK for case: %s", case_name)
        result_queue.put(("ok", result))
    except Exception:
        import traceback

        tb = traceback.format_exc()
        child_logger.error("Subprocess FAILED for case %s:\n%s", case_name, tb)
        result_queue.put(("error", tb))


def _run_case_isolated(timeout: int = 900, **kwargs) -> dict:
    """Run ``_compute_case`` in a subprocess for crash isolation.

    pandapower can trigger C-level memory corruption
    (``realloc: invalid next size`` / ``Aborted``) on certain networks,
    which kills the entire process.  Running each case in a child process
    ensures the sweep runner survives and continues with the next case.
    """
    case_name = kwargs.get("input_path", "unknown")
    logger.info("Launching subprocess for case: %s (timeout=%ds)", case_name, timeout)
    result_queue: mp.Queue = mp.Queue(maxsize=1)

    proc = mp.Process(target=_case_worker, args=(result_queue, kwargs))
    proc.start()
    logger.info("Subprocess started: PID=%s, waiting up to %ds...", proc.pid, timeout)
    proc.join(timeout=timeout)

    if proc.exitcode is None:
        logger.error(
            "Subprocess PID=%s TIMED OUT after %ds for case: %s. Terminating.",
            proc.pid,
            timeout,
            case_name,
        )
        proc.terminate()
        proc.join(timeout=10)
        raise RuntimeError(
            f"Case computation timed out after {timeout}s (PID {proc.pid} terminated)."
        )

    if proc.exitcode != 0:
        logger.error(
            "Subprocess PID=%s CRASHED with exit code %d for case: %s",
            proc.pid,
            proc.exitcode,
            case_name,
        )
        raise RuntimeError(
            f"Subprocess crashed with exit code {proc.exitcode}. "
            "Likely a C-level crash (segfault/abort) in the power flow solver."
        )

    logger.info(
        "Subprocess PID=%s exited OK (code=0) for case: %s", proc.pid, case_name
    )

    try:
        status, payload = result_queue.get_nowait()
    except Exception:
        raise RuntimeError("Subprocess completed but produced no result.")

    if status == "error":
        raise RuntimeError(f"Case computation failed in subprocess:\n{payload}")

    return payload


def _find_bottleneck(results: dict) -> tuple[int, float, str]:
    """Find bottleneck line: the line with the smallest finite AC L2 radius.

    Returns (line_idx, margin, mode) where mode is 'ac' if AC radius was
    available, otherwise 'dc'.  Falls back to DC if no AC radii exist.
    """
    best_line = -1
    best_radius = float("inf")
    best_margin = float("nan")
    mode = "dc"

    for key, val in results.items():
        if not key.startswith("line_") or not isinstance(val, dict):
            continue
        lid = int(key.split("_", 1)[1])

        # Prefer AC radius for bottleneck identification.
        r_ac = val.get("radius_ac_l2")
        if r_ac is not None and np.isfinite(r_ac):
            if r_ac < best_radius:
                best_radius = float(r_ac)
                best_line = lid
                best_margin = float(val.get("margin_ac_mva", float("nan")))
                mode = "ac"
        else:
            r_dc = val.get("radius_l2")
            if r_dc is not None and np.isfinite(r_dc) and mode != "ac":
                if r_dc < best_radius:
                    best_radius = float(r_dc)
                    best_line = lid
                    best_margin = float(val.get("margin_mw", float("nan")))
                    mode = "dc"

    return best_line, best_margin, mode


def _print_table(rows: list[dict]) -> None:
    """Print Table 1 to stdout in a fixed-width format."""
    header = (
        f"{'Case':<28s} {'n_b':>5s} {'n_l':>5s} "
        f"{'r*_DC (MW)':>12s} {'r*_AC (MW)':>12s} {'AC/DC':>7s} "
        f"{'AC feas':>7s} "
        f"{'T_tot (s)':>10s} "
        f"{'Bottleneck':>11s} {'Margin':>10s} "
        f"{'Status':>12s} {'PF attempt':>12s}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("Table 1: DC vs AC L2 Stability Radius across PGLib-OPF cases")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in rows:
        ratio_str = (
            f"{r['ac_dc_ratio']:.3f}" if np.isfinite(r["ac_dc_ratio"]) else "n/a"
        )
        bn_str = f"L{r['bottleneck_line']}" if r["bottleneck_line"] >= 0 else "n/a"
        margin_str = (
            f"{r['bottleneck_margin']:.2f}"
            if np.isfinite(r["bottleneck_margin"])
            else "n/a"
        )
        feas_str = r.get("ac_feasible", "n/a")
        if isinstance(feas_str, bool):
            feas_str = "YES" if feas_str else "NO"
        status_str = str(r.get("status", "ok"))
        pf_attempt_str = str(r.get("ac_pf_attempt", "n/a"))

        # Handle failed cases (n_buses/n_lines may be 0, radii may be NaN).
        if status_str != "ok":
            dc_str = "n/a"
            ac_str = "n/a"
            buses_str = f"{'---':>5s}"
            lines_str = f"{'---':>5s}"
            print(
                f"{r['case']:<28s} {buses_str} {lines_str} "
                f"{dc_str:>12s} {ac_str:>12s} {'n/a':>7s} "
                f"{'n/a':>7s} "
                f"{r['time_total']:>10.2f} "
                f"{'n/a':>11s} {'n/a':>10s} "
                f"{status_str:>12s} {pf_attempt_str:>12s}"
            )
        else:
            print(
                f"{r['case']:<28s} {r['n_buses']:>5d} {r['n_lines']:>5d} "
                f"{r['dc_r_star']:>12.4f} {r['ac_r_star']:>12.4f} {ratio_str:>7s} "
                f"{feas_str:>7s} "
                f"{r['time_total']:>10.2f} "
                f"{bn_str:>11s} {margin_str:>10s} "
                f"{status_str:>12s} {pf_attempt_str:>12s}"
            )

    print(sep)
    print()


def _plot_bar_chart(rows: list[dict], output_dir: Path) -> Path:
    """Generate Fig. 1: bar chart comparing r*_DC and r*_AC across cases."""
    # Only plot cases that completed successfully (have finite radii).
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok"]
    if not ok_rows:
        ok_rows = rows  # fallback: plot everything even if all failed

    labels = [r["case"].replace("pglib_opf_", "") for r in ok_rows]
    dc_vals = [r["dc_r_star"] if np.isfinite(r["dc_r_star"]) else 0.0 for r in ok_rows]
    ac_vals = [r["ac_r_star"] if np.isfinite(r["ac_r_star"]) else 0.0 for r in ok_rows]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    ax.bar(
        x - width / 2,
        dc_vals,
        width,
        label=r"$r^*_{\mathrm{DC}}$ (L2)",
        color="#4C72B0",
    )
    ax.bar(
        x + width / 2,
        ac_vals,
        width,
        label=r"$r^*_{\mathrm{AC}}$ (L2)",
        color="#DD8452",
    )

    ax.set_xlabel("PGLib-OPF Case")
    ax.set_ylabel("Stability Radius (MW)")
    ax.set_title("Fig. 1: DC vs AC L2 Stability Radius")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = output_dir / "fig1_dc_vs_ac_radius.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    return plot_path


def _setup_logging(output_dir: Path) -> logging.FileHandler:
    """Configure root logger to also write DEBUG-level logs to output_dir/debug.log."""
    log_path = output_dir / "debug.log"
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(fh)
    logger.info("Debug log: %s", log_path)
    return fh


def run(config_path: Path) -> None:
    cfg = _load_config(config_path)
    cases = cfg["cases"]
    compute_cfg = cfg.get("compute", {})
    data_dir = Path(cfg.get("data_dir", "data/input"))
    output_dir = Path(cfg.get("output_dir", "experiments/output/pglib_sweep"))
    allow_download = bool(cfg.get("allow_download", False))
    case_timeout = int(cfg.get("case_timeout_sec", 900))

    dc_cfg = compute_cfg.get("dc", {})
    ac_cfg = compute_cfg.get("ac", {})
    ac_fpf_cfg = compute_cfg.get("ac_fpf", {})
    opf_yaml = compute_cfg.get("opf", {})
    base_dispatch = str(compute_cfg.get("base_dispatch", "dc_opf"))

    # Consistency check tolerance (MW).  Phase-shifting transformers in PGLib
    # cause DC model flow reconstruction to deviate from OPF flows.
    # A large tolerance accepts this mismatch while still logging it.
    consistency_tol = float(compute_cfg.get("consistency_tol_mw", 1e6))

    opf_cfg = OPFConfig(
        headroom_factor=float(opf_yaml.get("headroom_factor", 0.9)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- File logging (debug.log) ----
    file_handler = _setup_logging(output_dir)

    logger.info("Config: %s", config_path)
    logger.info("OPFConfig: headroom_factor=%.4f", opf_cfg.headroom_factor)
    logger.info("Consistency tolerance: %.2f MW", consistency_tol)
    logger.info("Base dispatch: %s", base_dispatch)
    logger.info("Case timeout: %d sec", case_timeout)
    logger.info("Cases: %d", len(cases))

    summary_rows: list[dict] = []
    n_cases = len(cases)

    for case_idx, case in enumerate(cases, 1):
        name = case["name"]
        filename = case["file"]
        input_path = str(data_dir / filename)

        logger.info("=" * 60)
        logger.info("Processing [%d/%d] %s", case_idx, n_cases, name)
        logger.info("=" * 60)

        # ---- Per-case OPFConfig override (headroom_factor) ----
        case_headroom = case.get("headroom_factor")
        if case_headroom is not None:
            case_opf_cfg = OPFConfig(
                headroom_factor=float(case_headroom),
            )
            logger.info(
                "%s: per-case headroom_factor=%.4f (overrides global %.4f)",
                name,
                float(case_headroom),
                opf_cfg.headroom_factor,
            )
        else:
            case_opf_cfg = opf_cfg

        # ---- Per-case AC config overrides ----
        case_ac_overrides = case.get("ac", {})
        if case_ac_overrides:
            case_ac_cfg = {**ac_cfg, **case_ac_overrides}
            logger.info("%s: per-case AC overrides: %s", name, case_ac_overrides)
        else:
            case_ac_cfg = ac_cfg

        # ---- Ensure file exists & auto-detect slack bus ----
        try:
            input_path_abs = ensure_case_file(input_path)
            net = load_network(input_path_abs)
            slack_bus = _detect_slack_bus(net)
            logger.info(
                "%s: auto-detected slack_bus=%d (ext_grid bus)", name, slack_bus
            )
        except Exception:
            logger.exception("Failed to load network for %s", name)
            continue

        # ---- Single combined run: DC+AC share the same base point ----
        case_status = "ok"  # "ok" | "dc_opf_infeasible" | "dc_consistency_warning" | "ac_pf_failed" | "crashed" | "error"
        case_error_msg = ""
        try:
            t_start = time.perf_counter()
            combined = _run_case_isolated(
                timeout=case_timeout,
                input_path=input_path_abs,
                slack_bus=slack_bus,
                base_dispatch=base_dispatch,
                dc_cfg=dc_cfg,
                ac_cfg=case_ac_cfg,
                ac_fpf_cfg=ac_fpf_cfg,
                opf_cfg=case_opf_cfg,
                allow_download=False,
                opf_dc_flow_consistency_tol_mw=consistency_tol,
            )
            time_total = time.perf_counter() - t_start
        except Exception as exc:
            logger.exception("Failed computation for %s", name)
            time_total = time.perf_counter() - t_start if "t_start" in dir() else 0.0
            case_error_msg = str(exc)
            # Classify the failure.
            exc_str = str(exc).lower()
            if "subprocess crashed" in exc_str or "exit code" in exc_str:
                case_status = "crashed"
            elif "timed out" in exc_str:
                case_status = "timeout"
            elif "infeasible" in exc_str and "opf" in exc_str:
                case_status = "dc_opf_infeasible"
            elif "consistency" in exc_str:
                case_status = "dc_consistency_failed"
            elif "pandapower" in exc_str or "pf" in exc_str or "converge" in exc_str:
                case_status = "ac_pf_failed"
            else:
                case_status = "error"
            # Write a failed summary row so the case is tracked.
            row = {
                "case": name,
                "n_buses": 0,
                "n_lines": 0,
                "dc_r_star": float("nan"),
                "ac_r_star": float("nan"),
                "ac_dc_ratio": float("nan"),
                "ac_feasible": "n/a",
                "ac_n_violated": 0,
                "headroom_factor_used": None,
                "time_total": time_total,
                "bottleneck_line": -1,
                "bottleneck_margin": float("nan"),
                "status": case_status,
                "error": case_error_msg[:200],
                "ac_pf_attempt": "n/a",
                "ac_pf_repairs": [],
                "dc_consistency_passed": None,
                "dc_consistency_max_diff_mw": float("nan"),
            }
            summary_rows.append(row)
            continue

        # Remove non-serialisable h-vectors before saving.
        combined.pop("_h_vectors", None)

        case_output = output_dir / f"{name}.json"
        with case_output.open("w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2, default=_numpy_serialiser)
        logger.info("Results written: %s", case_output)

        # ---- Extract metrics ----
        meta = combined.get("__meta__", {})
        bp_dc = meta.get("base_point_dc") or {}
        bp_ac = meta.get("base_point_ac") or {}
        n_buses = len(bp_dc.get("bus_ids", bp_ac.get("bus_ids", [])))
        n_lines = sum(1 for k in combined if k.startswith("line_"))

        # DC global radius (min over all lines).
        dc_radii = []
        for key, val in combined.items():
            if key.startswith("line_") and isinstance(val, dict):
                r = val.get("radius_l2")
                if r is not None and np.isfinite(r):
                    dc_radii.append(float(r))
        dc_r_star = float(min(dc_radii)) if dc_radii else float("nan")

        # AC global radius (min over all lines).
        ac_radii = []
        for key, val in combined.items():
            if key.startswith("line_") and isinstance(val, dict):
                r = val.get("radius_ac_l2")
                if r is not None and np.isfinite(r):
                    ac_radii.append(float(r))
        ac_r_star = float(min(ac_radii)) if ac_radii else float("nan")

        # AC/DC ratio.
        if np.isfinite(dc_r_star) and dc_r_star > 0 and np.isfinite(ac_r_star):
            ac_dc_ratio = ac_r_star / dc_r_star
        else:
            ac_dc_ratio = float("nan")

        # Bottleneck line.
        bn_line, bn_margin, _ = _find_bottleneck(combined)

        # AC feasibility info.
        ac_meta = meta.get("ac", {})
        ac_feas = ac_meta.get("feasibility")
        ac_feasible: bool | str = "n/a"
        ac_n_violated = 0
        if isinstance(ac_feas, dict):
            ac_feasible = bool(ac_feas.get("is_feasible", True))
            ac_n_violated = int(ac_feas.get("n_constrained_violated", 0))
            if not ac_feasible:
                logger.warning(
                    "%s: AC base point INFEASIBLE: %d constrained lines violated "
                    "(worst_margin=%.4f MVA on line %d)",
                    name,
                    ac_n_violated,
                    float(ac_feas.get("worst_margin_mva", float("nan"))),
                    int(ac_feas.get("worst_line_id", -1)),
                )

        # Headroom factor actually used (may differ from configured due to adaptive schedule).
        opf_meta = meta.get("opf", {})
        hf_used = opf_meta.get("headroom_factor_used", float("nan"))
        hf_configured = opf_meta.get("headroom_factor_configured", float("nan"))
        if (
            np.isfinite(hf_used)
            and np.isfinite(hf_configured)
            and hf_used != hf_configured
        ):
            logger.info(
                "%s: Adaptive headroom: configured=%.4f, used=%.4f",
                name,
                hf_configured,
                hf_used,
            )

        # Log consistency check info from meta.
        consistency_max_diff = meta.get("opf_dc_flow_max_abs_diff_mw", float("nan"))
        if np.isfinite(consistency_max_diff):
            logger.info(
                "%s: OPF->DC consistency max|Δf|=%.4f MW", name, consistency_max_diff
            )

        # AC PF repair metadata from __meta__.
        ac_pf_attempt = ac_meta.get("pf_attempt", "n/a")
        ac_pf_repairs = list(ac_meta.get("pf_repairs", []))

        # DC consistency metadata from __meta__.
        dc_consistency_passed = meta.get("opf_dc_consistency_passed")
        dc_consistency_max_diff = meta.get("opf_dc_flow_max_abs_diff_mw", float("nan"))

        # Ext_grid absorption metadata from __meta__.
        ext_absorb_mw = float(opf_meta.get("ext_grid_absorption_mw", 0.0))

        row = {
            "case": name,
            "n_buses": n_buses,
            "n_lines": n_lines,
            "dc_r_star": dc_r_star,
            "ac_r_star": ac_r_star,
            "ac_dc_ratio": ac_dc_ratio,
            "ac_feasible": ac_feasible,
            "ac_n_violated": ac_n_violated,
            "headroom_factor_used": float(hf_used) if np.isfinite(hf_used) else None,
            "time_total": time_total,
            "bottleneck_line": bn_line,
            "bottleneck_margin": bn_margin,
            "status": "ok",
            "ac_pf_attempt": ac_pf_attempt,
            "ac_pf_repairs": ac_pf_repairs,
            "dc_consistency_passed": dc_consistency_passed,
            "dc_consistency_max_diff_mw": float(dc_consistency_max_diff)
            if np.isfinite(dc_consistency_max_diff)
            else None,
            "ext_grid_absorption_mw": ext_absorb_mw if ext_absorb_mw > 1e-3 else 0.0,
        }
        summary_rows.append(row)

    if not summary_rows:
        logger.error("No cases completed successfully.")
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        return

    # ---- Write summary JSON ----
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_rows, fh, indent=2, default=_numpy_serialiser)
    logger.info("Summary written: %s", summary_path)

    # ---- Print Table 1 ----
    _print_table(summary_rows)

    # ---- Generate Fig. 1 ----
    plot_path = _plot_bar_chart(summary_rows, output_dir)
    logger.info("Plot saved: %s", plot_path)
    print(f"Figure saved: {plot_path}")

    # ---- Cleanup ----
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Experiment 1: DC vs AC radius sweep across PGLib networks.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to pglib_sweep.yaml config.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
