from __future__ import annotations

"""
High-level workflows (library API).

This module contains the deterministic end-to-end single-case pipeline used by:
- the unified CLI (`src/power_stability_radius.py`)
- verification/report generation scripts

Public API
----------
- compute_results_for_case(...)
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from stability_radius.config import DEFAULT_OPF, OPFConfig
from stability_radius.dc.dc_model import build_dc_matrices, build_dc_operator
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.common import (
    LineBaseQuantities,
    assert_line_limit_sources_present,
    get_line_base_quantities,
)
from stability_radius.radii.l2 import compute_l2_radius
from stability_radius.radii.nminus1 import compute_nminus1_l2_radius
from stability_radius.radii.probabilistic import (
    overload_probability_symmetric_limit,
    sigma_radius,
)
from stability_radius.utils import log_stage

logger = logging.getLogger(__name__)

_DEFAULT_OPF_DC_FLOW_CONSISTENCY_TOL_MW = 1e-3
_DEFAULT_OPF_BUS_BALANCE_TOL_MW = 1e-6


def _resolve_path(p: str | os.PathLike[str], *, base_dir: Path | None) -> Path:
    """
    Resolve a potentially-relative path.

    Parameters
    ----------
    p:
        A path-like object.
    base_dir:
        Base directory for relative paths. If None, uses current working directory.

    Returns
    -------
    Path
        Absolute resolved path.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    root = base_dir if base_dir is not None else Path.cwd()
    return (root / path).resolve()


def _ensure_input_case_file(input_path: str, *, base_dir: Path | None) -> str:
    """
    Ensure input case file exists (download if missing and supported).

    Deterministic behavior
    ----------------------
    - No implicit path guessing beyond `base_dir` resolution.
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
    target_path = _resolve_path(input_path, base_dir=base_dir)
    if target_path.exists():
        logger.debug("Using input file: %s", str(target_path))
        return str(target_path)

    logger.info(
        "Input case file missing: %s. Trying deterministic download...", target_path
    )

    try:
        from stability_radius.utils.download import ensure_case_file
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Case file is missing and download helpers are unavailable. "
            "Install optional dependencies (e.g., requests) or provide an existing file."
        ) from e

    try:
        ensured = Path(ensure_case_file(str(target_path)))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Input case file does not exist: {target_path}. "
            "Auto-download supports only: case<N>.m / ieee<N>.m / pglib_opf_*.m. "
            "Provide an explicit path to an existing MATPOWER/PGLib .m case file."
        ) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to download missing case file: %s", target_path)
        raise RuntimeError(f"Failed to download input case file: {target_path}") from e

    if not ensured.exists():
        raise RuntimeError(
            f"Internal error: ensure_case_file() returned a non-existent path: {ensured}"
        )

    logger.info("Downloaded case file: %s", str(ensured))
    return str(ensured)


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


def _compute_projected_norms_from_operator(*, dc_op, chunk_size: int) -> np.ndarray:
    """
    Compute per-line sensitivity norms for the balanced subspace sum(Δp)=0.

    Norm definition
    ---------------
    For a row g in full bus coordinates (defined up to adding a constant 1-vector),
    the correct dual norm on the balanced subspace (with full Euclidean norm) is:

        ||Proj(g)||_2 = ||g - mean(g)*1||_2

    Using the identity:
        ||Proj(g)||^2 = ||g||^2 - (sum(g))^2 / n_bus

    Implementation notes
    --------------------
    - DCOperator provides g values for non-slack buses only, with slack component == 0.
      This is a valid representative of the equivalence class modulo constants.
    - We therefore compute:
        ||Proj(g)||^2 = sum(g_red^2) - (sum(g_red))^2 / n_bus
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    m = int(dc_op.n_line)
    n_bus = int(dc_op.n_bus)
    norms = np.zeros(m, dtype=float)

    start = 0
    while start < m:
        end = min(m, start + int(chunk_size))
        block = np.arange(start, end, dtype=int)

        # Y: (n_bus-1, k), column j is g_red^T for the corresponding line.
        Y = dc_op.row_sensitivities_transposed(block)

        t = np.sum(Y * Y, axis=0)  # ||g||^2 (slack component is 0)
        s = np.sum(Y, axis=0)  # sum(g) (slack component is 0)

        # Projected norm^2 = ||g||^2 - sum(g)^2 / n_bus
        proj2 = t - (s * s) / float(n_bus)
        norms[start:end] = np.sqrt(np.maximum(proj2, 0.0))

        start = end

    return norms


def _compute_sigma_from_l2_results(
    *, l2_results: dict[str, dict[str, Any]], inj_std_mw: float
) -> dict[str, dict[str, Any]]:
    """
    Compute sigma-radii and overload probabilities using the L2 row norms.

    Assumes the probabilistic model used throughout the project:
    - Δp is Gaussian in the balanced subspace sum(Δp)=0
    - isotropic with parameter sigma_mw in an orthonormal basis (d = n_bus - 1)

    Then for each line:
        sigma_flow = sigma_mw * ||Proj(g)||_2
        radius_sigma = margin / sigma_flow
    """
    sigma: dict[str, dict[str, Any]] = {}

    s = float(inj_std_mw)
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("inj_std_mw must be finite and positive.")

    for k, row in l2_results.items():
        if not isinstance(row, dict):
            continue

        margin = float(row.get("margin_mw", float("nan")))
        norm_g = float(row.get("norm_g", float("nan")))
        flow0 = float(row.get("flow0_mw", float("nan")))
        limit = float(row.get("p_limit_mw_est", float("nan")))

        sigma_flow = s * norm_g
        r_sigma = sigma_radius(margin, sigma_flow)
        prob = overload_probability_symmetric_limit(
            flow0=flow0, limit=limit, sigma=sigma_flow
        )

        sigma[k] = {
            "sigma_flow": float(sigma_flow),
            "radius_sigma": float(r_sigma),
            "overload_probability": float(prob),
        }

    return sigma


def _compute_radii_operator_path(
    *,
    dc_op,
    base: LineBaseQuantities,
    inj_std_mw: float,
    dc_chunk_size: int,
) -> dict[str, dict[str, Any]]:
    """
    Compute L2/metric/sigma radii without materializing H_full (operator path).

    Disturbance model (project-wide)
    --------------------------------
    - Δp is constrained to the balanced subspace: sum(Δp)=0.
    - Disturbance size is measured in the full-bus Euclidean norm ||Δp||_2.
    - Sensitivity norms use the projected norm ||Proj(g)||_2, which is slack-invariant.

    Notes
    -----
    - Uses LU solves via DCOperator to obtain g rows (chunked).
    - Default workflow uses M=I => radius_metric == radius_l2.
    - Gaussian model in balanced subspace: sigma_flow = inj_std * ||Proj(g)||_2.
    """
    if dc_chunk_size <= 0:
        raise ValueError("dc_chunk_size must be positive.")

    op_line_ids = list(getattr(dc_op, "line_ids", ()))
    if op_line_ids and list(base.line_indices) != [int(x) for x in op_line_ids]:
        raise ValueError(
            "Line ordering mismatch between DC operator and base quantities. "
            "This indicates an internal consistency bug."
        )

    norms = _compute_projected_norms_from_operator(
        dc_op=dc_op, chunk_size=int(dc_chunk_size)
    )
    if norms.shape != (len(base.line_indices),):
        raise ValueError("Unexpected norms shape from DC operator.")

    out: dict[str, dict[str, Any]] = {}
    for pos, lid in enumerate(base.line_indices):
        margin = float(base.margin_mw[pos])
        norm_g = float(norms[pos])

        r_l2 = float(margin / norm_g) if norm_g > 1e-12 else float("inf")

        sigma_flow = float(inj_std_mw) * norm_g
        r_sigma = sigma_radius(margin, sigma_flow)

        c = float(base.limit_mva_assumed_mw[pos])
        f0 = float(base.flow0_mw[pos])
        prob = overload_probability_symmetric_limit(flow0=f0, limit=c, sigma=sigma_flow)

        k = f"line_{int(lid)}"
        out[k] = {
            "flow0_mw": float(base.flow0_mw[pos]),
            "p0_mw": float(base.p0_abs_mw[pos]),
            "p_limit_mw_est": float(base.limit_mva_assumed_mw[pos]),
            "margin_mw": margin,
            "norm_g": norm_g,
            "radius_l2": r_l2,
            "metric_denom": norm_g,
            "radius_metric": float(r_l2),
            "sigma_flow": float(sigma_flow),
            "radius_sigma": float(r_sigma),
            "overload_probability": float(prob),
            "radius_nminus1": float("nan"),
            "worst_contingency": -1,
            "worst_contingency_line_idx": -1,
        }
    return out


def _check_opf_dc_consistency(
    *,
    net: Any,
    dc_op,
    base: LineBaseQuantities,
    tol_flow_mw: float,
    tol_balance_mw: float,
) -> dict[str, float]:
    """
    Validate that OPF base flows are consistent with the project's DCOperator.

    This is a hard correctness check:
    - we reconstruct line flows from OPF bus injections via DCOperator
    - we compare against OPF-reported base flows f0 for monitored lines

    Raises
    ------
    ValueError
        If bus ordering is inconsistent, injections are missing, or max|Δf| > tol_flow_mw.

    Returns
    -------
    dict
        Diagnostic scalars for results meta.
    """
    if base.bus_ids is None or base.bus_injections_mw is None:
        raise ValueError(
            "Internal error: OPF base quantities must include bus_ids and bus_injections_mw "
            "(required for OPF->DC consistency checks). Regenerate results with the current version."
        )

    bus_ids = tuple(int(x) for x in base.bus_ids)
    op_bus_ids = tuple(int(x) for x in getattr(dc_op, "bus_ids", ()))
    if bus_ids != op_bus_ids:
        raise ValueError(
            "Bus ordering mismatch between OPF base point and DC operator. "
            f"opf_bus_ids[:10]={list(bus_ids)[:10]}..., dc_bus_ids[:10]={list(op_bus_ids)[:10]}..."
        )

    p = np.asarray(base.bus_injections_mw, dtype=float).reshape(-1)
    if p.shape != (len(bus_ids),):
        raise ValueError(
            f"bus_injections_mw shape mismatch: got {p.shape}, expected ({len(bus_ids)},)"
        )

    inj_sum = float(np.sum(p))
    if abs(inj_sum) > float(tol_balance_mw):
        raise ValueError(
            "OPF bus injections are not balanced within tolerance. "
            f"sum(injections)={inj_sum:.6g} MW (tol={float(tol_balance_mw):.6g})."
        )

    f0_opf = np.asarray(base.flow0_mw, dtype=float).reshape(-1)
    f0_dc = np.asarray(dc_op.flows_from_delta_injections(p), dtype=float).reshape(-1)

    if f0_dc.shape != f0_opf.shape:
        raise ValueError(
            f"OPF/DC flow vector shape mismatch: opf={f0_opf.shape}, dc={f0_dc.shape}"
        )

    diff = f0_dc - f0_opf
    if diff.size:
        abs_diff = np.abs(diff)
        abs_diff_safe = np.nan_to_num(
            abs_diff, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
        )
        argmax_pos = int(np.argmax(abs_diff_safe))
        max_abs = float(abs_diff_safe[argmax_pos])
        argmax_line_idx = (
            int(base.line_indices[argmax_pos])
            if 0 <= argmax_pos < len(base.line_indices)
            else -1
        )
        argmax_opf = float(f0_opf[argmax_pos])
        argmax_dc = float(f0_dc[argmax_pos])
        argmax_diff = float(diff[argmax_pos])

        # Helpful diagnostics (debug-only) to avoid huge console logs.
        top_k = min(5, int(diff.size))
        candidates = [
            (float(abs_diff_safe[i]), int(base.line_indices[i]), int(i))
            for i in range(int(diff.size))
        ]
        candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
        top = candidates[:top_k]
        top_summary = [
            (
                pos,
                line_idx,
                float(f0_opf[pos]),
                float(f0_dc[pos]),
                float(diff[pos]),
                float(abs_diff_safe[pos]),
            )
            for _, line_idx, pos in top
        ]
        logger.debug(
            "OPF->DC flow diffs: argmax_pos=%d line_idx=%d opf=%.6g dc=%.6g diff=%.6g abs=%.6g; top5(pos,line,opf,dc,diff,abs)=%s",
            argmax_pos,
            argmax_line_idx,
            argmax_opf,
            argmax_dc,
            argmax_diff,
            max_abs,
            top_summary,
        )

        # Fast local diagnostics for the argmax line (for debugging mismatches on real cases).
        try:
            if (
                hasattr(net, "line")
                and net.line is not None
                and int(argmax_line_idx) in net.line.index
            ):
                row = net.line.loc[int(argmax_line_idx)]
                fb = int(row.get("from_bus", -1))
                tb = int(row.get("to_bus", -1))
                x_ohm_per_km = float(row.get("x_ohm_per_km", float("nan")))
                length_km = float(row.get("length_km", float("nan")))
                parallel = float(row.get("parallel", 1.0))
                in_service = bool(row.get("in_service", True))

                # dc_op.b is aligned with monitored line ordering, so we can log b for that position.
                b_mw_per_rad = float(getattr(dc_op, "b")[argmax_pos])

                logger.error(
                    "OPF->DC mismatch details: line_idx=%d pos=%d in_service=%s from_bus=%d to_bus=%d "
                    "x_ohm_per_km=%.6g length_km=%.6g parallel=%.6g b_mw_per_rad=%.6g "
                    "flow_opf=%.6g flow_dc=%.6g diff=%.6g",
                    int(argmax_line_idx),
                    int(argmax_pos),
                    bool(in_service),
                    int(fb),
                    int(tb),
                    float(x_ohm_per_km),
                    float(length_km),
                    float(parallel),
                    float(b_mw_per_rad),
                    float(argmax_opf),
                    float(argmax_dc),
                    float(argmax_diff),
                )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Failed to produce detailed mismatch diagnostics.", exc_info=True
            )
    else:
        max_abs = 0.0
        argmax_line_idx = -1
        argmax_pos = -1
        argmax_opf = float("nan")
        argmax_dc = float("nan")
        argmax_diff = float("nan")

    logger.info(
        "OPF->DC consistency check: max|Δf|=%.6g MW (tol=%.6g MW), sum(inj)=%.6g MW",
        float(max_abs),
        float(tol_flow_mw),
        inj_sum,
    )

    if not np.isfinite(max_abs) or max_abs > float(tol_flow_mw):
        raise ValueError(
            "OPF->DC consistency check failed: base flows from PyPSA do not match DCOperator flows "
            f"(max|Δf|={float(max_abs):.6g} MW, tol={float(tol_flow_mw):.6g} MW). "
            f"argmax_line_pos={int(argmax_pos)}, argmax_line_idx={int(argmax_line_idx)}, "
            f"flow_opf={float(argmax_opf):.6g} MW, flow_dc={float(argmax_dc):.6g} MW, diff={float(argmax_diff):.6g} MW. "
            "This indicates a model/data mismatch between OPF construction and the DC operator."
        )

    return {
        "opf_bus_balance_abs_mw": float(abs(inj_sum)),
        "opf_dc_flow_max_abs_diff_mw": float(max_abs),
        "opf_dc_flow_tol_mw": float(tol_flow_mw),
        "opf_bus_balance_tol_mw": float(tol_balance_mw),
    }


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
    opf_cfg: OPFConfig | None = None,
    opf_dc_flow_consistency_tol_mw: float = _DEFAULT_OPF_DC_FLOW_CONSISTENCY_TOL_MW,
    opf_bus_balance_tol_mw: float = _DEFAULT_OPF_BUS_BALANCE_TOL_MW,
    strict_units: bool = True,
    allow_phase_shift: bool = False,
    path_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compute per-line radii and return a single results dict (including '__meta__').

    Deterministic pipeline (project policy)
    ---------------------------------------
    Base point is ALWAYS produced by a DC OPF:
      - PyPSA + HiGHS (single snapshot)

    DC model modes
    --------------
    - dc_mode="materialize": materialize H_full and compute all radii (including N-1 if requested)
    - dc_mode="operator": compute L2/metric/sigma radii via operator norms (no N-1)

    Disturbance / norm convention (important)
    -----------------------------------------
    The L2 certificate is defined on **balanced** injections:
        sum(Δp) = 0
    with the full-bus Euclidean norm ||Δp||_2. Sensitivity norms therefore use the
    projected norm ||g - mean(g)*1||_2 (slack-invariant).

    Units
    -----
    See UNITS_CONTRACT.md in the repository root.

    Unit validation
    ---------------
    - strict_units=True: fail fast on vn_kv<=0, x_ohm<=0, sn_mva<=0 (no silent 0.0 fallbacks)
    - allow_phase_shift=False: reject any transformer with shift_degree != 0 (recommended)

    Correctness checks
    ------------------
    The pipeline enforces an OPF->DCOperator consistency check:
    base flows from OPF must match DCOperator flows reconstructed from OPF bus injections.

    Parameters
    ----------
    path_base_dir:
        Base dir for resolving relative paths (input case path). If None, uses current working directory.

    Returns
    -------
    dict
        Results object with '__meta__' and 'line_<idx>' entries.
    """
    time_start = time.time()

    base_dir = Path(path_base_dir).resolve() if path_base_dir is not None else None
    input_path_abs = _ensure_input_case_file(str(input_path), base_dir=base_dir)
    case_tag = Path(input_path_abs).stem

    cfg = opf_cfg if opf_cfg is not None else DEFAULT_OPF

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
        f"{case_tag}: Solve DC OPF (PyPSA, solver={cfg.highs.solver_name})",
    ):
        base = get_line_base_quantities(
            net,
            margin_factor=float(margin_factor),
            opf_cfg=cfg,
            strict_units=bool(strict_units),
            allow_phase_shift=bool(allow_phase_shift),
        )

    H_full = None
    dc_op = None

    with log_stage(logger, f"{case_tag}: Build DC Model (mode={dc_mode_eff})"):
        if dc_mode_eff == "materialize":
            H_full, dc_op = build_dc_matrices(
                net,
                slack_bus=int(slack_bus),
                strict_units=bool(strict_units),
                allow_phase_shift=bool(allow_phase_shift),
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
            dc_op = build_dc_operator(
                net,
                slack_bus=int(slack_bus),
                strict_units=bool(strict_units),
                allow_phase_shift=bool(allow_phase_shift),
            )
            n_bus = int(dc_op.n_bus)
            m_line = int(dc_op.n_line)
            logger.debug("Built DC operator: n_bus=%d, n_line=%d", n_bus, m_line)

    if dc_op is None:
        raise AssertionError("Internal error: DC operator was not created.")

    with log_stage(logger, f"{case_tag}: Consistency Check (OPF -> DCOperator)"):
        consistency = _check_opf_dc_consistency(
            net=net,
            dc_op=dc_op,
            base=base,
            tol_flow_mw=float(opf_dc_flow_consistency_tol_mw),
            tol_balance_mw=float(opf_bus_balance_tol_mw),
        )

    with log_stage(logger, f"{case_tag}: Compute Radii"):
        if H_full is not None:
            l2 = compute_l2_radius(
                net, H_full, margin_factor=float(margin_factor), base=base
            )

            # Performance/memory note:
            # Default workflow uses M=I => radius_metric == radius_l2. Avoid allocating I_n.
            metric = {
                k: {
                    "metric_denom": float(v.get("norm_g", float("nan"))),
                    "radius_metric": float(v.get("radius_l2", float("nan"))),
                }
                for k, v in l2.items()
            }

            sigma = _compute_sigma_from_l2_results(
                l2_results=l2, inj_std_mw=float(inj_std_mw)
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
            "opf_solver": str(cfg.highs.solver_name),
            "opf_solver_threads": int(cfg.highs.threads),
            "opf_solver_random_seed": int(cfg.highs.random_seed),
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
            "strict_units": bool(strict_units),
            "allow_phase_shift": bool(allow_phase_shift),
            "compute_time_sec": elapsed_sec,
            "n_bus": int(n_bus),
            "n_line": int(m_line),
            "nminus1_computed": bool(nminus1_computed),
            **consistency,
        }
    }
    results.update(results_lines)
    return results
