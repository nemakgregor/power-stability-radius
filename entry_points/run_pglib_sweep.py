"""Experiment 1: DC vs AC radius sweep across PGLib networks.

Reads ``experiments/configs/pglib_sweep.yaml``, computes DC and AC L2 radii
for each listed PGLib case, and produces:

- Per-case JSON results in ``run_artifacts/run_pglib_sweep/``
- ``summary.json`` with aggregated metrics
- **Table 1** (printed to stdout): case, n_b, n_l, r*_DC, r*_AC, AC/DC, time, bottleneck
- **Fig. 1** (saved as PNG): bar chart comparing r*_DC and r*_AC across cases

Usage::

    python entry_points/run_pglib_sweep.py
    python entry_points/run_pglib_sweep.py --config experiments/configs/pglib_sweep.yaml

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
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from stability_radius.config import OPFConfig
from stability_radius.parsers.matpower import load_network
from stability_radius.utils import (
    create_module_output_dir,
    numpy_to_builtin,
    resolve_artifacts_root,
)
from stability_radius.utils.download import ensure_case_file
from stability_radius.workflows import (
    DCExtensionsConfig,
    compute_results_for_case,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "experiments" / "configs" / "pglib_sweep.yaml"
)


def _load_config(path: Path) -> dict:
    """Internal helper for module-local processing."""
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


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
    # Deterministic default: first bus in sorted order.
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
    dc_checkpoint_path: str | None = None,
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
        ac_fpf_vm_min_pu=float(ac_fpf_cfg.get("vm_min_pu", 0.9)),
        ac_fpf_vm_max_pu=float(ac_fpf_cfg.get("vm_max_pu", 1.1)),
        ac_fpf_max_iteration=int(ac_fpf_cfg.get("max_iteration", 300)),
        ac_fpf_max_loading_percent=float(ac_fpf_cfg.get("max_loading_percent", 99.0)),
        ac_fpf_init=str(ac_fpf_cfg.get("init", "dc")),
        ac_fpf_max_attempts=int(ac_fpf_cfg.get("max_attempts", 1)),
        ac_fpf_per_attempt_timeout=float(ac_fpf_cfg.get("per_attempt_timeout", 0)),
        # OPF
        opf_cfg=opf_cfg,
        opf_dc_flow_consistency_tol_mw=float(opf_dc_flow_consistency_tol_mw),
        # shared
        allow_download=allow_download,
        dc_checkpoint_path=dc_checkpoint_path,
    )


def _case_worker(
    result_queue: mp.Queue,
    kwargs: dict,
    log_level: int = logging.INFO,
    log_path: str | None = None,
    checkpoint_path: str | None = None,
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

    # Add file handler so subprocess logs go to debug.log.
    if log_path:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s [subprocess] %(name)s: %(message)s"
            )
        )
        logging.getLogger().addHandler(fh)
        # Enable DEBUG for project loggers only; third-party libs stay at INFO
        # (inherited from root) so numba/scipy/etc. DEBUG spam is excluded.
        logging.getLogger("stability_radius").setLevel(logging.DEBUG)
        logging.getLogger("__main__").setLevel(logging.DEBUG)
    child_logger = logging.getLogger(__name__)
    case_name = kwargs.get("input_path", "unknown")
    child_logger.info(
        "Subprocess started for case: %s (PID=%d)", case_name, os.getpid()
    )
    try:
        compute_kwargs = dict(kwargs)
        if checkpoint_path:
            compute_kwargs["dc_checkpoint_path"] = checkpoint_path
        result = _compute_case(**compute_kwargs)
        result.pop("_h_vectors", None)
        child_logger.info("Subprocess finished OK for case: %s", case_name)
        result_queue.put(("ok", result))
    except Exception:
        import traceback

        tb = traceback.format_exc()
        child_logger.error("Subprocess FAILED for case %s:\n%s", case_name, tb)
        result_queue.put(("error", tb))


def _run_case_isolated(
    timeout: int = 900,
    log_path: str | None = None,
    checkpoint_dir: str | None = None,
    **kwargs,
) -> dict:
    """Run ``_compute_case`` in a subprocess for crash isolation.

    pandapower can trigger C-level memory corruption
    (``realloc: invalid next size`` / ``Aborted``) on certain networks,
    which kills the entire process.  Running each case in a child process
    ensures the sweep runner survives and continues with the next case.

    If *checkpoint_dir* is provided, DC-only intermediate results are written
    there by the child process.  On timeout, the parent attempts to recover
    the checkpoint so at least DC radii are available.
    """
    case_name = kwargs.get("input_path", "unknown")
    logger.info("Launching subprocess for case: %s (timeout=%ds)", case_name, timeout)
    result_queue: mp.Queue = mp.Queue(maxsize=1)

    # Build checkpoint path for the child process.
    checkpoint_path: str | None = None
    if checkpoint_dir:
        case_stem = Path(case_name).stem
        checkpoint_path = str(Path(checkpoint_dir) / f".{case_stem}.dc_checkpoint.json")

    proc = mp.Process(
        target=_case_worker,
        args=(result_queue, kwargs),
        kwargs={
            "log_path": log_path,
            "checkpoint_path": checkpoint_path,
        },
    )
    proc.start()
    logger.info("Subprocess started: PID=%s, waiting up to %ds...", proc.pid, timeout)

    # Drain the result queue BEFORE joining.  Python docs warn that
    # proc.join() can deadlock if the child put data on a Queue whose
    # internal pipe buffer is full — the feeder thread blocks and the
    # child cannot exit.  Reading the queue first avoids this.
    result_tuple: tuple | None = None
    effective_timeout = timeout if timeout > 0 else None  # 0 means no timeout
    try:
        result_tuple = result_queue.get(timeout=effective_timeout)
    except Exception:
        pass

    # Now safe to join (queue is drained).
    join_timeout = 30 if timeout > 0 else None
    proc.join(timeout=join_timeout)

    if proc.exitcode is None:
        logger.error(
            "Subprocess PID=%s TIMED OUT after %ds for case: %s. Terminating.",
            proc.pid,
            timeout,
            case_name,
        )
        proc.terminate()
        proc.join(timeout=10)
        # Try to recover DC checkpoint written by the child before timeout.
        if checkpoint_path and Path(checkpoint_path).is_file():
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as fh:
                    dc_result = json.load(fh)
                dc_result.setdefault("__meta__", {})["ac_timeout"] = True
                logger.info("Recovered DC checkpoint for %s after timeout.", case_name)
                return dc_result
            except Exception:
                logger.warning(
                    "DC checkpoint exists but could not be loaded for %s.",
                    case_name,
                    exc_info=True,
                )
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

    if result_tuple is None:
        raise RuntimeError("Subprocess completed but produced no result.")

    status, payload = result_tuple

    if status == "error":
        raise RuntimeError(f"Case computation failed in subprocess:\n{payload}")

    # Clean up DC checkpoint on success (no longer needed).
    if checkpoint_path and Path(checkpoint_path).is_file():
        try:
            Path(checkpoint_path).unlink()
        except OSError:
            pass

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


def _extract_summary_row(name: str, combined: dict, time_total: float) -> dict:
    """Extract summary metrics from a per-case result dict.

    Used both for freshly computed results and for reused JSON files.
    """
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

    # Headroom factor actually used.
    opf_meta = meta.get("opf", {})
    hf_used = opf_meta.get("headroom_factor_used", float("nan"))

    # Consistency check info.
    dc_consistency_passed = meta.get("opf_dc_consistency_passed")
    dc_consistency_max_diff = meta.get("opf_dc_flow_max_abs_diff_mw", float("nan"))

    # AC PF repair metadata.
    ac_pf_attempt = ac_meta.get("pf_attempt", "n/a")
    ac_pf_repairs = list(ac_meta.get("pf_repairs", []))

    # Ext_grid absorption.
    ext_absorb_mw = float(opf_meta.get("ext_grid_absorption_mw", 0.0))

    # Dispatch method tracking.
    dispatch_method = meta.get("base_dispatch", "")
    dispatch_requested = meta.get("base_dispatch_requested", dispatch_method)
    dispatch_changed = str(dispatch_method) != str(dispatch_requested)

    # Determine status based on result content.
    ac_available = np.isfinite(ac_r_star)
    dc_available = np.isfinite(dc_r_star)

    if meta.get("dc_checkpoint") or meta.get("ac_timeout"):
        status = "dc_only_timeout"
    elif not ac_available and dc_available:
        status = "dc_only"
    elif ac_available and ac_r_star < 0:
        status = "ac_infeasible"
    elif dc_available and dc_r_star < 0:
        status = "dc_negative"
    else:
        status = "ok"

    return {
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
        "status": status,
        "ac_pf_attempt": ac_pf_attempt,
        "ac_pf_repairs": ac_pf_repairs,
        "dc_consistency_passed": dc_consistency_passed,
        "dc_consistency_max_diff_mw": float(dc_consistency_max_diff)
        if np.isfinite(dc_consistency_max_diff)
        else None,
        "ext_grid_absorption_mw": ext_absorb_mw if ext_absorb_mw > 1e-3 else 0.0,
        "dispatch_method": dispatch_method,
        "dispatch_changed": dispatch_changed,
    }


def _update_summary_and_plot(summary_rows: list[dict], output_dir: Path) -> None:
    """Write summary.json and regenerate the plot with rows sorted by n_buses."""
    sorted_rows = sorted(summary_rows, key=lambda r: (r["n_buses"], r["case"]))
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(sorted_rows, fh, indent=2, default=numpy_to_builtin)
    try:
        _plot_bar_chart(sorted_rows, output_dir)
    except Exception:
        logger.warning("Could not update plot (matplotlib error)", exc_info=True)


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
        # Statuses with data: ok, ac_infeasible, dc_negative, dc_only, dc_only_timeout
        # Statuses without data: timeout, crashed, dc_opf_infeasible, error
        has_data = (
            status_str
            in ("ok", "ac_infeasible", "dc_negative", "dc_only", "dc_only_timeout")
            and r.get("n_buses", 0) > 0
        )
        if not has_data:
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
            dc_str = (
                f"{r['dc_r_star']:>12.4f}"
                if np.isfinite(r["dc_r_star"])
                else f"{'n/a':>12s}"
            )
            ac_str = (
                f"{r['ac_r_star']:>12.4f}"
                if np.isfinite(r["ac_r_star"])
                else f"{'n/a':>12s}"
            )
            print(
                f"{r['case']:<28s} {r['n_buses']:>5d} {r['n_lines']:>5d} "
                f"{dc_str} {ac_str} {ratio_str:>7s} "
                f"{feas_str:>7s} "
                f"{r['time_total']:>10.2f} "
                f"{bn_str:>11s} {margin_str:>10s} "
                f"{status_str:>12s} {pf_attempt_str:>12s}"
            )

    print(sep)
    print()


def _plot_bar_chart(rows: list[dict], output_dir: Path) -> Path:
    """Generate Fig. 1: bar chart comparing r*_DC and r*_AC across cases.

    Visual encoding:
    - Solid bars: normal positive radii
    - Red bars with hatching: negative radii (infeasible base point)
    - x marker: AC unavailable (dc_only / timeout)
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Plot ALL cases that have at least some data (DC or AC radii).
    plot_rows = [
        r
        for r in rows
        if (
            np.isfinite(r.get("dc_r_star", float("nan")))
            or np.isfinite(r.get("ac_r_star", float("nan")))
        )
    ]
    if not plot_rows:
        plot_rows = rows

    # Sort by n_buses for readability.
    plot_rows = sorted(
        plot_rows, key=lambda r: (r.get("n_buses", 0), r.get("case", ""))
    )

    labels = [r["case"].replace("pglib_opf_", "") for r in plot_rows]

    COLOR_DC = "#4C72B0"
    COLOR_AC = "#DD8452"
    COLOR_NEGATIVE = "#C44E52"
    COLOR_MISSING = "#999999"

    dc_vals = []
    ac_vals = []
    dc_colors = []
    ac_colors = []
    dc_hatches = []
    ac_hatches = []
    ac_missing_indices = []

    for i, r in enumerate(plot_rows):
        status = r.get("status", "ok")
        dc_r = r.get("dc_r_star", float("nan"))
        ac_r = r.get("ac_r_star", float("nan"))

        # DC bar
        if np.isfinite(dc_r):
            dc_vals.append(abs(dc_r))
            dc_colors.append(COLOR_NEGATIVE if dc_r < 0 else COLOR_DC)
            dc_hatches.append("//" if dc_r < 0 else "")
        else:
            dc_vals.append(0.0)
            dc_colors.append(COLOR_DC)
            dc_hatches.append("")

        # AC bar
        if status in ("dc_only", "dc_only_timeout", "timeout", "crashed", "error"):
            ac_vals.append(0.0)
            ac_colors.append(COLOR_AC)
            ac_hatches.append("")
            ac_missing_indices.append(i)
        elif np.isfinite(ac_r):
            ac_vals.append(abs(ac_r))
            ac_colors.append(COLOR_NEGATIVE if ac_r < 0 else COLOR_AC)
            ac_hatches.append("//" if ac_r < 0 else "")
        else:
            ac_vals.append(0.0)
            ac_colors.append(COLOR_AC)
            ac_hatches.append("")
            ac_missing_indices.append(i)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))

    # Draw bars one at a time to support per-bar colors and hatching.
    for i in range(len(labels)):
        ax.bar(
            x[i] - width / 2,
            dc_vals[i],
            width,
            color=dc_colors[i],
            edgecolor="black",
            linewidth=0.5,
            hatch=dc_hatches[i],
        )
        ax.bar(
            x[i] + width / 2,
            ac_vals[i],
            width,
            color=ac_colors[i],
            edgecolor="black",
            linewidth=0.5,
            hatch=ac_hatches[i],
        )

    # Mark AC-unavailable cases.
    if ac_missing_indices:
        y_offset = max(dc_vals) * 0.02 if dc_vals else 0.5
        ax.scatter(
            [x[i] + width / 2 for i in ac_missing_indices],
            [y_offset] * len(ac_missing_indices),
            marker="x",
            color=COLOR_MISSING,
            s=40,
            zorder=5,
        )

    # Legend with proxy artists.
    legend_elements = [
        Patch(facecolor=COLOR_DC, edgecolor="black", label=r"$r^*_{\mathrm{DC}}$ (L2)"),
        Patch(facecolor=COLOR_AC, edgecolor="black", label=r"$r^*_{\mathrm{AC}}$ (L2)"),
    ]
    # Only add negative/missing legend items if they exist.
    has_negative = any(h == "//" for h in dc_hatches + ac_hatches)
    if has_negative:
        legend_elements.append(
            Patch(
                facecolor=COLOR_NEGATIVE,
                edgecolor="black",
                hatch="//",
                label="Negative (infeasible)",
            )
        )
    if ac_missing_indices:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="x",
                color="w",
                markeredgecolor=COLOR_MISSING,
                markersize=8,
                label="AC unavailable",
            )
        )

    ax.set_xlabel("PGLib-OPF Case")
    ax.set_ylabel(r"$|r^*|$ Stability Radius (MW)")
    ax.set_title("DC vs AC L2 Stability Radius")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = output_dir / "fig1_dc_vs_ac_radius.png"
    fig.savefig(str(plot_path), dpi=300)
    # Also save PDF for LaTeX.
    pdf_path = output_dir / "fig1_dc_vs_ac_radius.pdf"
    fig.savefig(str(pdf_path))
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
    # Enable DEBUG for project loggers only; third-party libs stay at INFO
    # (inherited from root) so numba/scipy/etc. DEBUG spam is excluded.
    logging.getLogger("stability_radius").setLevel(logging.DEBUG)
    logging.getLogger("__main__").setLevel(logging.DEBUG)
    logger.info("Debug log: %s", log_path)
    return fh


def run(config_path: Path, reuse_dir: Path | None = None) -> None:
    """Run the configured workflow."""
    cfg = _load_config(config_path)
    cases = cfg["cases"]
    compute_cfg = cfg.get("compute", {})
    data_dir = Path(cfg.get("data_dir", "data/input"))
    artifacts_root = resolve_artifacts_root(cfg)
    output_dir = create_module_output_dir(
        module_name="run_pglib_sweep",
        runs_dir=artifacts_root,
        requested_output_dir=cfg.get("output_dir", None),
    )
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

    os.makedirs(str(output_dir), exist_ok=True)

    # ---- File logging (debug.log) ----
    file_handler = _setup_logging(output_dir)
    log_path = str(output_dir / "debug.log")

    logger.info("Config: %s", config_path)
    logger.info("OPFConfig: headroom_factor=%.4f", opf_cfg.headroom_factor)
    logger.info("Consistency tolerance: %.2f MW", consistency_tol)
    logger.info("Base dispatch: %s", base_dispatch)
    logger.info("Case timeout: %d sec", case_timeout)
    if reuse_dir is not None:
        logger.info("Reuse dir: %s (will skip already-solved cases)", reuse_dir)
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

        # ---- Reuse existing result if available ----
        if reuse_dir is not None:
            reuse_path = Path(reuse_dir) / f"{name}.json"
            if reuse_path.is_file():
                try:
                    with reuse_path.open("r", encoding="utf-8") as fh:
                        reused = json.load(fh)
                    row = _extract_summary_row(name, reused, time_total=0.0)
                    summary_rows.append(row)
                    logger.info(
                        "%s: REUSED from %s (n_buses=%d, dc_r*=%.4f, ac_r*=%.4f)",
                        name,
                        reuse_path,
                        row["n_buses"],
                        row["dc_r_star"],
                        row["ac_r_star"],
                    )
                    # Copy JSON into output_dir so it is self-contained.
                    dest = output_dir / f"{name}.json"
                    if dest.resolve() != reuse_path.resolve():
                        shutil.copy2(str(reuse_path), str(dest))
                    _update_summary_and_plot(summary_rows, output_dir)
                    continue
                except Exception:
                    logger.warning(
                        "%s: could not reuse %s, will recompute",
                        name,
                        reuse_path,
                        exc_info=True,
                    )

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

        # ---- Per-case AC FPF config overrides ----
        case_ac_fpf_overrides = case.get("ac_fpf", {})
        if case_ac_fpf_overrides:
            case_ac_fpf_cfg = {**ac_fpf_cfg, **case_ac_fpf_overrides}
            logger.info(
                "%s: per-case AC FPF overrides: %s", name, case_ac_fpf_overrides
            )
        else:
            case_ac_fpf_cfg = ac_fpf_cfg

        # ---- Per-case base_dispatch override ----
        case_base_dispatch_override = case.get("base_dispatch")
        if case_base_dispatch_override is not None:
            case_base_dispatch = str(case_base_dispatch_override)
            logger.info(
                "%s: per-case base_dispatch='%s' (overrides global '%s')",
                name,
                case_base_dispatch,
                base_dispatch,
            )
        else:
            case_base_dispatch = base_dispatch

        # ---- Per-case timeout override ----
        case_timeout_override = case.get("timeout")
        if case_timeout_override is not None:
            case_timeout_eff = int(case_timeout_override)
            logger.info(
                "%s: per-case timeout=%d sec (overrides global %d sec)",
                name,
                case_timeout_eff,
                case_timeout,
            )
        else:
            case_timeout_eff = case_timeout

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
                timeout=case_timeout_eff,
                log_path=log_path,
                checkpoint_dir=str(output_dir),
                input_path=input_path_abs,
                slack_bus=slack_bus,
                base_dispatch=case_base_dispatch,
                dc_cfg=dc_cfg,
                ac_cfg=case_ac_cfg,
                ac_fpf_cfg=case_ac_fpf_cfg,
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
            _update_summary_and_plot(summary_rows, output_dir)
            continue
        # Remove non-serialisable h-vectors before saving.
        combined.pop("_h_vectors", None)

        case_output = output_dir / f"{name}.json"
        with case_output.open("w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2, default=numpy_to_builtin)
        logger.info("Results written: %s", case_output)

        # ---- Extract metrics ----
        row = _extract_summary_row(name, combined, time_total)

        # If this result was recovered from a DC checkpoint (timeout with
        # partial DC results), mark the status accordingly.
        meta = combined.get("__meta__", {})
        if meta.get("dc_checkpoint") or meta.get("ac_timeout"):
            row["status"] = "dc_only_timeout"

        # Log notable conditions.
        meta = combined.get("__meta__", {})
        ac_meta = meta.get("ac", {})
        ac_feas = ac_meta.get("feasibility")
        if isinstance(ac_feas, dict) and not ac_feas.get("is_feasible", True):
            logger.warning(
                "%s: AC base point INFEASIBLE: %d constrained lines violated "
                "(worst_margin=%.4f MVA on line %d)",
                name,
                int(ac_feas.get("n_constrained_violated", 0)),
                float(ac_feas.get("worst_margin_mva", float("nan"))),
                int(ac_feas.get("worst_line_id", -1)),
            )
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
        consistency_max_diff = meta.get("opf_dc_flow_max_abs_diff_mw", float("nan"))
        if np.isfinite(consistency_max_diff):
            logger.info(
                "%s: OPF->DC consistency max|Δf|=%.4f MW", name, consistency_max_diff
            )

        summary_rows.append(row)
        _update_summary_and_plot(summary_rows, output_dir)

    if not summary_rows:
        logger.error("No cases completed successfully.")
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        return

    # ---- Final summary (sorted by n_buses) ----
    sorted_rows = sorted(summary_rows, key=lambda r: (r["n_buses"], r["case"]))
    _update_summary_and_plot(sorted_rows, output_dir)
    logger.info("Summary written: %s", output_dir / "summary.json")

    # ---- Print Table 1 ----
    _print_table(sorted_rows)

    plot_path = output_dir / "fig1_dc_vs_ac_radius.png"
    logger.info("Plot saved: %s", plot_path)
    print(f"Figure saved: {plot_path}")

    # ---- Cleanup ----
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()


def main() -> None:
    """Run the command-line entry point."""
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
    parser.add_argument(
        "--reuse-dir",
        type=Path,
        default=None,
        help="Reuse existing per-case JSON results from this directory; "
        "only solve cases without a result file.",
    )
    args = parser.parse_args()
    run(args.config, reuse_dir=args.reuse_dir)


if __name__ == "__main__":
    raise SystemExit(main())
