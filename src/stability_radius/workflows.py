from __future__ import annotations

"""
High-level workflows (library API).

Main contract
-------------
- AC stability radius is computed around an AC PF base point (NOT DC OPF).
- DC OPF is optionally used only as a dispatch source (if compute.base_dispatch=dc_opf).

Determinism policy
------------------
- No implicit downloading unless allow_download=True.
- Stable ordering: sorted bus/line indices.
- No hidden "compatibility" results: optional radii are computed only when explicitly enabled.

Determinism note (important)
----------------------------
Some configuration parameters must be identical regardless of entry point (YAML vs Python).
In particular, OPFConfig.unconstrained_line_nom_mw is used as a finite surrogate limit for
"unconstrained" lines in PyPSA, so it must be reproducible.
"""

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stability_radius.base_point import (
    build_dc_base_point_case,
    build_dc_base_point_dc_opf,
    build_dc_base_point_from_acpf,
    solve_ac_fpf_base_point,
    solve_ac_pf_base_point,
)
from stability_radius.config import DEFAULT_OPF, OPFConfig
from stability_radius.dc.dc_model import build_dc_matrices, build_dc_operator
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.ac_feasibility import (
    ACFeasibilityResult,
    check_ac_base_point_feasibility,
)
from stability_radius.radii.common import (
    LineBaseQuantities,
    assert_line_limit_sources_present,
    classify_constraint_certificate,
    estimate_line_limit_mva_with_flag,
    signed_radius_from_margin_norm,
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
_DEFAULT_OPF_BUS_BALANCE_TOL_MW = 1


@dataclass(frozen=True)
class DCExtensionsConfig:
    """
    Optional DC post-processing extensions.

    This groups "non-core" DC options to keep the public workflow signature manageable.

    Notes
    -----
    - probabilistic_enabled controls sigma-radius and overload probability post-processing.
    - nminus1_enabled requires dc_mode="materialize" (needs H_full).
    """

    probabilistic_enabled: bool = False
    nminus1_enabled: bool = False
    nminus1_update_sensitivities: bool = True
    nminus1_islanding: str = "skip"  # "skip" | "raise"


@dataclass(frozen=True)
class ACExtensionsConfig:
    """
    Optional AC post-processing extensions (sigma-radius, metric-radius).

    Notes
    -----
    - sigma_p_mw_source / sigma_q_mvar_source control how sigma arrays are built:
        * "uniform" : broadcast scalar sigma_p_mw_uniform / sigma_q_mvar_uniform to all buses.
        * "uc_jl"   : use per-bus arrays from sigma_p_mw_array / sigma_q_mvar_array
                       (typically loaded from a UnitCommitment.jl instance via parsers.uc_jl).
        * ""        : disabled (no sigma arrays → sigma/metric radii are skipped).
    - metric_enabled gates AC metric-radius computation alongside sigma-radius.
      When enabled and sigma arrays are available, M = diag(1/sigma^2) is used,
      which should reproduce the sigma-radius (serves as a cross-check).
    - save_h_vectors: if True, h-vectors are returned under the "_h_vectors" key
      for the caller (CLI) to save as a compressed .npz file.
    """

    sigma_p_mw_source: str = ""  # "uniform" | "uc_jl" | ""
    sigma_q_mvar_source: str = ""  # "uniform" | "uc_jl" | ""
    sigma_p_mw_uniform: float = 1.0
    sigma_q_mvar_uniform: float = 1.0
    sigma_p_mw_array: np.ndarray | None = (
        None  # per-bus array (n_bus,) for "uc_jl" source
    )
    sigma_q_mvar_array: np.ndarray | None = (
        None  # per-bus array (n_bus,) for "uc_jl" source
    )
    sigma_n_timesteps: int | None = None  # number of timesteps from UC.jl instance
    metric_enabled: bool = False
    save_h_vectors: bool = False
    nonlinear_validation_enabled: bool = False
    nonlinear_validation_top_k: int = 20
    nonlinear_validation_scale_max: float = 5.0
    nonlinear_validation_tol: float = 0.01
    nonlinear_validation_max_iter: int = 20


def _resolve_path(p: str | os.PathLike[str], *, base_dir: Path | None) -> Path:
    """Resolve a potentially-relative path (with "~" expansion)."""
    path = Path(p).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        root = base_dir if base_dir is not None else Path.cwd()
        resolved = (root / path).resolve()
    logger.debug(
        "Resolved path: %s -> %s (base_dir=%s)", str(p), str(resolved), base_dir
    )
    return resolved


def _ensure_input_case_file(
    input_path: str, *, base_dir: Path | None, allow_download: bool
) -> str:
    """
    Ensure input case file exists (deterministic).

    - missing & allow_download=False -> FileNotFoundError
    - missing & allow_download=True  -> deterministic download via ensure_case_file()
    """
    target_path = _resolve_path(input_path, base_dir=base_dir)
    if target_path.exists():
        return str(target_path)

    if not bool(allow_download):
        raise FileNotFoundError(
            f"Input case file not found: {target_path}. "
            "Set io.allow_download=true in config or pass --allow-download 1."
        )

    from stability_radius.utils.download import ensure_case_file

    ensured = Path(ensure_case_file(str(target_path))).resolve()
    if not ensured.exists():
        raise RuntimeError(
            f"Internal error: ensure_case_file() returned non-existent path: {ensured}"
        )
    return str(ensured)


def _assert_and_log_effective_unconstrained_line_nom_mw(
    *, cfg: OPFConfig, source: str
) -> float:
    """
    Validate and log effective `opf.unconstrained_line_nom_mw`.

    This serves two purposes:
    1) Determinism: make the value visible at workflow start (helps debugging YAML vs Python paths).
    2) Fail-fast: prevent silent invalid values (NaN/inf/<=0) from propagating into OPF models.

    Parameters
    ----------
    cfg:
        Effective OPFConfig used by the workflow.
    source:
        Human-readable source label ("DEFAULT_OPF" / "caller_provided" / etc.)

    Returns
    -------
    float
        The validated value (MW).
    """
    v = float(cfg.unconstrained_line_nom_mw)
    if (not np.isfinite(v)) or v <= 0.0:
        logger.error(
            "Invalid opf.unconstrained_line_nom_mw: %r (source=%s). Must be finite and >0.",
            cfg.unconstrained_line_nom_mw,
            str(source),
        )
        raise ValueError("opf.unconstrained_line_nom_mw must be finite and >0.")

    logger.info(
        "Determinism: effective opf.unconstrained_line_nom_mw=%.6g MW (source=%s)",
        float(v),
        str(source),
    )
    return float(v)


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
    """Compute per-line sensitivity norms for the balanced subspace sum(Δp)=0."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    m = int(dc_op.n_line)
    n_bus = int(dc_op.n_bus)
    norms = np.zeros(m, dtype=float)

    start = 0
    while start < m:
        end = min(m, start + int(chunk_size))
        block = np.arange(start, end, dtype=int)

        Y = dc_op.row_sensitivities_transposed(block)  # (n_bus-1, k)
        t = np.sum(Y * Y, axis=0)
        s = np.sum(Y, axis=0)
        proj2 = t - (s * s) / float(n_bus)
        norms[start:end] = np.sqrt(np.maximum(proj2, 0.0))
        start = end

    return norms


def _compute_probabilistic_from_l2_results(
    *, l2_results: dict[str, dict[str, Any]], inj_std_mw: float
) -> dict[str, dict[str, Any]]:
    """
    Compute sigma-radii and overload probabilities using L2 row norms.

    This is OPTIONAL (AC-focused defaults do not compute DC probabilistic post-processing).
    """
    s = float(inj_std_mw)
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("inj_std_mw must be finite and positive.")

    out: dict[str, dict[str, Any]] = {}
    for k, row in l2_results.items():
        if not isinstance(row, dict):
            continue

        margin = float(row.get("margin_mw", float("nan")))
        norm_g = float(row.get("norm_g", float("nan")))
        flow0 = float(row.get("flow0_mw", float("nan")))
        limit = float(row.get("p_limit_mw_est", float("nan")))

        sigma_flow = s * norm_g
        r_sigma = sigma_radius(margin, sigma_flow)
        status, cert_radius, signed_distance = classify_constraint_certificate(
            margin=margin,
            dual_norm=sigma_flow,
            eps=1e-12,
            is_unconstrained=bool(row.get("is_unconstrained", False)),
        )
        prob = overload_probability_symmetric_limit(
            flow0=flow0, limit=limit, sigma=sigma_flow
        )

        out[k] = {
            "sigma_flow": float(sigma_flow),
            "radius_sigma": float(r_sigma),
            "certificate_radius_sigma": float(cert_radius),
            "signed_distance_sigma": float(signed_distance),
            "constraint_status_sigma": str(status),
            "overload_probability": float(prob),
        }
    return out


def _compute_radii_operator_path(
    *,
    dc_op: Any,
    base: LineBaseQuantities,
    dc_chunk_size: int,
    net: Any,
) -> dict[str, dict[str, Any]]:
    """
    Compute DC L2 radii without materializing H_full (operator path).

    Important
    ---------
    This path returns ONLY the core DC L2 fields.
    Optional radii (probabilistic / N-1) are not computed here by design.
    """
    if dc_chunk_size <= 0:
        raise ValueError("dc_chunk_size must be positive.")

    norms = _compute_projected_norms_from_operator(
        dc_op=dc_op, chunk_size=int(dc_chunk_size)
    )
    if norms.shape != (len(base.line_indices),):
        raise ValueError("Unexpected norms shape from DC operator.")

    # Unconstrained flags: prefer base if provided, otherwise extract from net deterministically.
    if base.is_unconstrained is not None:
        is_unconstrained = np.asarray(base.is_unconstrained, dtype=bool).reshape(-1)
        if is_unconstrained.shape != (len(base.line_indices),):
            raise ValueError("base.is_unconstrained shape mismatch.")
    else:
        is_unconstrained = np.zeros(len(base.line_indices), dtype=bool)
        for pos, lid in enumerate(base.line_indices):
            _, is_uc = estimate_line_limit_mva_with_flag(net, net.line.loc[int(lid)])
            is_unconstrained[pos] = bool(is_uc)

    out: dict[str, dict[str, Any]] = {}
    for pos, lid in enumerate(base.line_indices):
        margin = float(base.margin_mw[pos])
        norm_g = float(norms[pos])
        r_l2 = signed_radius_from_margin_norm(
            margin=margin, dual_norm=norm_g, eps=1e-12
        )
        status, cert_radius, signed_distance = classify_constraint_certificate(
            margin=margin,
            dual_norm=norm_g,
            eps=1e-12,
            is_unconstrained=bool(is_unconstrained[pos]),
        )

        k = f"line_{int(lid)}"
        out[k] = {
            "flow0_mw": float(base.flow0_mw[pos]),
            "p0_mw": float(base.p0_abs_mw[pos]),
            "p_limit_mw_est": float(base.limit_mva_assumed_mw[pos]),
            "is_unconstrained": bool(is_unconstrained[pos]),
            "margin_mw": margin,
            "norm_g": norm_g,
            "radius_l2": float(r_l2),
            "certificate_radius_l2": float(cert_radius),
            "signed_distance_l2": float(signed_distance),
            "constraint_status_l2": str(status),
        }
    return out


def _check_opf_dc_consistency(
    *,
    dc_op,
    base: LineBaseQuantities,
    tol_flow_mw: float,
    tol_balance_mw: float,
) -> dict[str, float]:
    """Validate that OPF base flows are consistent with DCOperator reconstruction."""
    if base.bus_ids is None or base.bus_injections_mw is None:
        raise ValueError(
            "OPF base quantities must include bus_ids and bus_injections_mw."
        )

    bus_ids = tuple(int(x) for x in base.bus_ids)
    op_bus_ids = tuple(int(x) for x in getattr(dc_op, "bus_ids", ()))
    if bus_ids != op_bus_ids:
        raise ValueError(
            "Bus ordering mismatch between OPF base point and DC operator."
        )

    p = np.asarray(base.bus_injections_mw, dtype=float).reshape(-1)
    if p.shape != (len(bus_ids),):
        raise ValueError("bus_injections_mw shape mismatch.")

    inj_sum = float(np.sum(p))
    if abs(inj_sum) > float(tol_balance_mw):
        raise ValueError(
            "OPF bus injections are not balanced within tolerance. "
            f"sum(injections)={inj_sum:.6g} MW (tol={float(tol_balance_mw):.6g})."
        )

    f0_opf = np.asarray(base.flow0_mw, dtype=float).reshape(-1)
    f0_dc = np.asarray(dc_op.flows_from_bus_injections_mw(p), dtype=float).reshape(-1)

    abs_diff = np.abs(f0_dc - f0_opf)
    abs_diff_safe = np.nan_to_num(
        abs_diff, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
    )
    argmax_pos = int(np.argmax(abs_diff_safe)) if abs_diff_safe.size else -1
    max_abs = float(abs_diff_safe[argmax_pos]) if abs_diff_safe.size else 0.0

    logger.info(
        "OPF->DC consistency check: max|Δf|=%.6g MW (tol=%.6g MW), sum(inj)=%.6g MW",
        float(max_abs),
        float(tol_flow_mw),
        inj_sum,
    )

    if not np.isfinite(max_abs) or max_abs > float(tol_flow_mw):
        argmax_line_idx = (
            int(base.line_indices[argmax_pos])
            if 0 <= argmax_pos < len(base.line_indices)
            else -1
        )
        logger.warning(
            "OPF->DC consistency check EXCEEDED tolerance: "
            "max|Δf|=%.6g MW > tol=%.6g MW. "
            "argmax_line_pos=%d, argmax_line_idx=%d. "
            "Results may be less accurate for this case (e.g. phase-shifting transformers).",
            float(max_abs),
            float(tol_flow_mw),
            int(argmax_pos),
            int(argmax_line_idx),
        )

    return {
        "opf_bus_balance_abs_mw": float(abs(inj_sum)),
        "opf_dc_flow_max_abs_diff_mw": float(max_abs),
        "opf_dc_flow_tol_mw": float(tol_flow_mw),
        "opf_bus_balance_tol_mw": float(tol_balance_mw),
        "opf_dc_consistency_passed": bool(
            np.isfinite(max_abs) and max_abs <= float(tol_flow_mw)
        ),
    }


def expand_h_reduced_to_full(
    h_reduced: np.ndarray,
    *,
    n_bus: int,
    slack_pos: int,
    pq_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Expand reduced h-vectors to full dimension (2*n_bus,).

    The h-vector is the injection-space gradient: h = J^{-T} (d|S|/dx),
    with blocks [h_P_red | h_Q_red].

    When ``pq_mask`` is None (all non-slack buses are PQ), the reduced
    layout is ``[h_P(n_red) | h_Q(n_red)]`` and we insert a zero at
    ``slack_pos`` in each block.

    When ``pq_mask`` is provided (networks with PV generator buses),
    the reduced layout is ``[h_P(n_theta) | h_Q(n_pq)]`` where
    ``n_theta = n_bus - 1`` (all non-slack) and ``n_pq = sum(pq_mask)``.
    The P block inserts a zero at slack_pos.  The Q block scatters
    ``n_pq`` values to PQ bus positions; PV and slack buses get zero.

    Parameters
    ----------
    h_reduced : (m, n_theta + n_pq) array
    n_bus     : total bus count (including slack)
    slack_pos : position of the slack bus in the sorted bus ordering
    pq_mask   : (n_bus,) bool array, True for PQ buses.  None means all-PQ.

    Returns
    -------
    (m, 2*n_bus) array.  Layout: [h_P_full | h_Q_full].
    """
    h = np.asarray(h_reduced, dtype=float)
    n_red = n_bus - 1

    if pq_mask is None:
        # Default reduced layout: all non-slack buses are PQ.
        if h.ndim != 2 or h.shape[1] != 2 * n_red:
            raise ValueError(f"h_reduced shape must be (m, {2 * n_red}), got {h.shape}")
        m = h.shape[0]
        p_red = h[:, :n_red]
        q_red = h[:, n_red:]

        p_full = np.insert(p_red, slack_pos, 0.0, axis=1)
        q_full = np.insert(q_red, slack_pos, 0.0, axis=1)
        return np.hstack([p_full, q_full])

    # PV-aware path.
    pq = np.asarray(pq_mask, dtype=bool)
    n_pq = int(np.sum(pq))
    n_vars = n_red + n_pq

    if h.ndim != 2 or h.shape[1] != n_vars:
        raise ValueError(
            f"h_reduced shape must be (m, {n_vars}), got {h.shape} "
            f"(n_theta={n_red}, n_pq={n_pq})"
        )

    m = h.shape[0]
    p_red = h[:, :n_red]  # (m, n_theta)
    q_red = h[:, n_red:]  # (m, n_pq)

    # P block: insert zero at slack_pos -> (m, n_bus)
    p_full = np.insert(p_red, slack_pos, 0.0, axis=1)

    # Q block: scatter PQ-only values to full bus dimension
    q_full = np.zeros((m, n_bus), dtype=float)
    pq_indices = np.where(pq)[0]
    q_full[:, pq_indices] = q_red

    return np.hstack([p_full, q_full])


def _build_sigma_arrays(
    *, ac_ext: ACExtensionsConfig, n_bus: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-bus sigma_p_mw and sigma_q_mvar arrays from config.

    Returns (sigma_p, sigma_q) each of shape (n_bus,).
    """
    src_p = str(ac_ext.sigma_p_mw_source).strip().lower()
    src_q = str(ac_ext.sigma_q_mvar_source).strip().lower()

    if src_p == "uniform":
        v = float(ac_ext.sigma_p_mw_uniform)
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"sigma_p_mw_uniform must be finite and >0, got {v}")
        sigma_p = np.full(n_bus, v, dtype=float)
    elif src_p == "uc_jl":
        if ac_ext.sigma_p_mw_array is None:
            raise ValueError(
                "sigma_p_mw_source='uc_jl' requires sigma_p_mw_array to be set."
            )
        sigma_p = np.asarray(ac_ext.sigma_p_mw_array, dtype=float).reshape(-1)
        if sigma_p.shape != (n_bus,):
            raise ValueError(
                f"sigma_p_mw_array must have shape ({n_bus},), got {sigma_p.shape}"
            )
    else:
        raise ValueError(
            f"sigma_p_mw_source must be 'uniform' or 'uc_jl' when sigma is enabled, got {src_p!r}"
        )

    if src_q == "uniform":
        v = float(ac_ext.sigma_q_mvar_uniform)
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"sigma_q_mvar_uniform must be finite and >0, got {v}")
        sigma_q = np.full(n_bus, v, dtype=float)
    elif src_q == "uc_jl":
        if ac_ext.sigma_q_mvar_array is None:
            raise ValueError(
                "sigma_q_mvar_source='uc_jl' requires sigma_q_mvar_array to be set."
            )
        sigma_q = np.asarray(ac_ext.sigma_q_mvar_array, dtype=float).reshape(-1)
        if sigma_q.shape != (n_bus,):
            raise ValueError(
                f"sigma_q_mvar_array must have shape ({n_bus},), got {sigma_q.shape}"
            )
    else:
        raise ValueError(
            f"sigma_q_mvar_source must be 'uniform' or 'uc_jl' when sigma is enabled, got {src_q!r}"
        )

    return sigma_p, sigma_q


def extract_binding_end_data(
    *, ac_results: dict[str, dict[str, Any]], h_from: np.ndarray, h_to: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Select binding-end h-vectors, s0, and limits from AC L2 results.

    Returns (h_bind, s0_mva, s_limit_mva, line_ids) where h_bind has
    shape (n_lines, d) with each row being the binding-end h-vector.
    """
    line_keys = sorted(
        (k for k in ac_results if k.startswith("line_")),
        key=lambda k: int(k.split("_", 1)[1]),
    )
    n_lines = len(line_keys)
    d = h_from.shape[1]

    h_bind = np.empty((n_lines, d), dtype=float)
    s0_mva = np.empty(n_lines, dtype=float)
    s_limit_mva = np.empty(n_lines, dtype=float)
    line_ids: list[int] = []

    for pos, k in enumerate(line_keys):
        row = ac_results[k]
        lid = int(k.split("_", 1)[1])
        line_ids.append(lid)

        binding_end = str(row["binding_end"])
        if binding_end == "from":
            h_bind[pos, :] = h_from[pos, :]
            s0_mva[pos] = float(row["ac_s0_from_mva"])
        else:
            h_bind[pos, :] = h_to[pos, :]
            s0_mva[pos] = float(row["ac_s0_to_mva"])

        s_limit_mva[pos] = float(row["ac_s_limit_mva"])

    return h_bind, s0_mva, s_limit_mva, line_ids


def _max_finite_or_nan(values: list[float]) -> float:
    """Internal helper for module-local processing."""
    finite = [float(x) for x in values if math.isfinite(float(x))]
    return float(max(finite)) if finite else float("nan")


def _percentile_or_nan(values: list[float], q: float) -> float:
    """Internal helper for module-local processing."""
    finite = [float(x) for x in values if math.isfinite(float(x))]
    if not finite:
        return float("nan")
    return float(np.percentile(np.asarray(finite, dtype=float), float(q)))


def _run_ac_nonlinear_validation_topk(
    *,
    net: Any,
    case_tag: str,
    results_lines: dict[str, dict[str, Any]],
    h_bind: np.ndarray,
    s0_mva: np.ndarray,
    s_limit_mva: np.ndarray,
    line_ids: list[int],
    n_bus: int,
    pq_mask: np.ndarray | None,
    balance: bool,
    lossless: bool,
    gen_dispatch_mw_by_name: dict[str, float] | None,
    top_k: int,
    scale_max: float,
    tol: float,
    max_iter: int,
) -> dict[str, Any]:
    """
    Replay nonlinear AC PF for the top-k smallest finite AC L2 radii.

    The replay perturbation is built with the same balanced geometry as the AC
    L2 denominator.  The full trajectory is returned as a separate report,
    while compact per-line fields are merged into ``results_lines``.
    """
    from stability_radius.geometry.balanced import (
        make_ac_block_specs,
        worst_case_l2_direction,
    )
    from stability_radius.verification.verify_worst_case import find_violation_scale

    top_k_eff = int(top_k)
    if top_k_eff < 1:
        raise ValueError("nonlinear_validation_top_k must be >= 1.")
    if float(scale_max) <= 0.0 or not math.isfinite(float(scale_max)):
        raise ValueError("nonlinear_validation_scale_max must be finite and > 0.")
    if float(tol) <= 0.0 or not math.isfinite(float(tol)):
        raise ValueError("nonlinear_validation_tol must be finite and > 0.")
    if int(max_iter) < 1:
        raise ValueError("nonlinear_validation_max_iter must be >= 1.")

    h_arr = np.asarray(h_bind, dtype=float)
    s0_arr = np.asarray(s0_mva, dtype=float).reshape(-1)
    limit_arr = np.asarray(s_limit_mva, dtype=float).reshape(-1)
    if h_arr.ndim != 2 or h_arr.shape[0] != len(line_ids):
        raise ValueError("h_bind must have one row per AC line.")
    if s0_arr.shape != (len(line_ids),) or limit_arr.shape != (len(line_ids),):
        raise ValueError("s0_mva and s_limit_mva must align with line_ids.")

    candidates: list[tuple[float, int, int, str, dict[str, Any]]] = []
    for pos, lid in enumerate(line_ids):
        key = f"line_{int(lid)}"
        row = results_lines.get(key)
        if not isinstance(row, dict):
            continue
        status = str(row.get("constraint_status_ac_l2", ""))
        if status and status != "ok_finite":
            continue
        radius = float(
            row.get(
                "certificate_radius_ac_l2",
                row.get("radius_ac_l2_linear", row.get("radius_ac_l2", float("nan"))),
            )
        )
        if not math.isfinite(radius) or radius <= 0.0:
            continue
        if not math.isfinite(float(s0_arr[pos])) or not math.isfinite(
            float(limit_arr[pos])
        ):
            continue
        candidates.append((radius, int(lid), int(pos), key, row))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = candidates[:top_k_eff]

    q_bus_indices: np.ndarray | None = None
    if pq_mask is not None:
        pq = np.asarray(pq_mask, dtype=bool).reshape(-1)
        if pq.shape != (int(n_bus),):
            raise ValueError("pq_mask must have shape (n_bus,).")
        q_bus_indices = np.where(pq)[0]

    blocks = make_ac_block_specs(
        int(n_bus), balance=bool(balance), q_bus_indices=q_bus_indices
    )

    report_lines: list[dict[str, Any]] = []
    for rank, (radius, lid, pos, key, row) in enumerate(selected, start=1):
        h_vec = np.asarray(h_arr[pos], dtype=float).reshape(-1)
        direction = worst_case_l2_direction(h_vec, blocks)
        direction_norm = float(np.linalg.norm(direction, ord=2))
        if direction_norm <= 0.0 or not math.isfinite(direction_norm):
            continue

        s0 = float(s0_arr[pos])
        limit = float(limit_arr[pos])
        margin = float(limit - s0)
        binding_end = str(row.get("binding_end", ""))
        delta_u_boundary = direction * float(radius)

        search = find_violation_scale(
            net=net,
            line_id=int(lid),
            h_vec=h_vec,
            radius=float(radius),
            s0_mva=float(s0),
            limit_mva=float(limit),
            delta_u_unit=delta_u_boundary,
            binding_end=binding_end if binding_end in {"from", "to"} else None,
            lossless=bool(lossless),
            scale_max=float(scale_max),
            tol=float(tol),
            max_iter=int(max_iter),
            gen_dispatch_mw_by_name=gen_dispatch_mw_by_name,
        )

        trajectory: list[dict[str, Any]] = []
        rel_errors: list[float] = []
        safe_scales: list[float] = []
        pf_failed_calls = 0
        for item in search.scale_trajectory:
            scale = float(item.get("scale", float("nan")))
            actual = float(item.get("actual_s_mva", float("nan")))
            pf_converged = bool(item.get("pf_converged", False))
            violated = bool(item.get("violated", False))
            predicted = (
                float(s0 + scale * margin) if math.isfinite(scale) else float("nan")
            )
            rel_error = (
                abs(predicted - actual) / actual
                if pf_converged and math.isfinite(actual) and abs(actual) > 1e-12
                else float("nan")
            )
            if math.isfinite(rel_error):
                rel_errors.append(float(rel_error))
            if pf_converged and not violated and math.isfinite(scale):
                safe_scales.append(float(scale))
            if not pf_converged:
                pf_failed_calls += 1
            trajectory.append(
                {
                    "scale": float(scale),
                    "predicted_s_mva": float(predicted),
                    "actual_s_mva": float(actual),
                    "relative_error": float(rel_error),
                    "violated": bool(violated),
                    "pf_converged": bool(pf_converged),
                }
            )

        gamma = float(search.conservatism_ratio)
        validation_scale_violation = float(search.actual_violation_scale)
        validation_scale_safe = _max_finite_or_nan(safe_scales)
        max_rel_error = _max_finite_or_nan(rel_errors)

        if math.isinf(gamma):
            validated_radius = float(radius)
        elif math.isfinite(gamma):
            validated_radius = float(radius) * min(max(float(gamma), 0.0), 1.0)
        else:
            validated_radius = float("nan")

        pf_replay_status = "pf_failed" if pf_failed_calls else "converged"
        if pf_failed_calls and (not math.isfinite(gamma) or gamma < 1.0):
            linearization_status = "nonlinear_unvalidated"
        elif math.isfinite(gamma) and gamma < 1.0 - float(tol):
            linearization_status = "nonlinear_optimistic"
        else:
            linearization_status = "validated_local"

        compact = {
            "radius_ac_l2_validated": float(validated_radius),
            "validation_scale_safe": float(validation_scale_safe),
            "validation_scale_violation": float(validation_scale_violation),
            "nonlinear_conservatism_ratio": float(gamma),
            "pf_replay_status": str(pf_replay_status),
            "max_replay_rel_error": float(max_rel_error),
            "nonlinear_validation_n_pf_calls": int(search.n_pf_calls),
            "linearization_status": str(linearization_status),
        }
        results_lines[key].update(compact)

        report_lines.append(
            {
                "rank": int(rank),
                "line_id": int(lid),
                "line_key": str(key),
                "binding_end": str(binding_end),
                "radius_ac_l2_linear": float(radius),
                "radius_ac_l2_validated": float(validated_radius),
                "s0_mva": float(s0),
                "limit_mva": float(limit),
                "margin_mva": float(margin),
                "validation_scale_safe": float(validation_scale_safe),
                "validation_scale_violation": float(validation_scale_violation),
                "nonlinear_conservatism_ratio": float(gamma),
                "pf_replay_status": str(pf_replay_status),
                "linearization_status": str(linearization_status),
                "max_replay_rel_error": float(max_rel_error),
                "n_pf_calls": int(search.n_pf_calls),
                "n_pf_failed": int(pf_failed_calls),
                "search_converged": bool(search.converged),
                "actual_s_at_violation_mva": float(search.actual_s_at_violation),
                "trajectory": trajectory,
            }
        )

    gammas = [float(x["nonlinear_conservatism_ratio"]) for x in report_lines]
    n_pf_calls = int(sum(int(x["n_pf_calls"]) for x in report_lines))
    n_pf_failed = int(sum(int(x["n_pf_failed"]) for x in report_lines))
    finite_gammas = [float(x) for x in gammas if math.isfinite(float(x))]
    summary = {
        "n_pf_calls": int(n_pf_calls),
        "n_converged": int(n_pf_calls - n_pf_failed),
        "n_pf_failed": int(n_pf_failed),
        "n_lines_pf_failed": int(
            sum(1 for x in report_lines if str(x["pf_replay_status"]) == "pf_failed")
        ),
        "n_lines_gamma_lt_1": int(sum(1 for x in finite_gammas if x < 1.0)),
        "n_lines_no_violation_up_to_scale_max": int(
            sum(1 for x in gammas if math.isinf(float(x)))
        ),
        "median_gamma": _percentile_or_nan(finite_gammas, 50.0),
        "p05_gamma": _percentile_or_nan(finite_gammas, 5.0),
    }

    return {
        "schema_version": 1,
        "case": str(case_tag),
        "top_k_requested": int(top_k_eff),
        "top_k_replayed": int(len(report_lines)),
        "scale_max": float(scale_max),
        "tol": float(tol),
        "max_iter": int(max_iter),
        "model_policy": {
            "lossless": bool(lossless),
            "balance": bool(balance),
            "q_balance_block": "pq_buses" if q_bus_indices is not None else "all_buses",
            "gen_dispatch_reapplied": bool(gen_dispatch_mw_by_name),
        },
        "summary": summary,
        "lines": report_lines,
    }


# ---------------------------------------------------------------------------
# Adaptive headroom for DC OPF
# ---------------------------------------------------------------------------


# Default schedule: configured headroom first, then relax towards 1.0.
def _solve_dc_opf_once(
    *,
    net: Any,
    slack_bus: int,
    opf_cfg: OPFConfig,
    limit_factor: float,
    case_tag: str,
) -> tuple[Any, "LineBaseQuantities", float]:
    """Solve DC OPF once with the configured headroom factor.

    Returns (bp_dc, base_dc, used_headroom_factor). Raises RuntimeError if the
    configured OPF is infeasible or otherwise fails.
    """
    bp_dc, base_dc = build_dc_base_point_dc_opf(
        net=net,
        slack_bus=int(slack_bus),
        opf_cfg=opf_cfg,
        limit_factor=float(limit_factor),
    )
    logger.info(
        "%s: DC OPF solved with headroom_factor=%.4f",
        case_tag,
        float(opf_cfg.headroom_factor),
    )
    return bp_dc, base_dc, float(opf_cfg.headroom_factor)


def compute_results_for_case(
    *,
    input_path: str,
    slack_bus: int,
    base_dispatch: str,  # case | dc_opf
    # DC
    compute_dc: bool,
    dc_mode: str,
    dc_chunk_size: int,
    dc_dtype: np.dtype,
    dc_inj_std_mw: float,
    dc_extensions: DCExtensionsConfig | None = None,
    # AC
    compute_ac: bool,
    ac_chunk_size: int,
    ac_balance: bool,
    ac_pf_init: str,
    ac_pf_solver: str,
    ac_lossless: bool,
    ac_distributed_slack: bool = False,
    ac_trafo_model: str = "pi",
    ac_extensions: ACExtensionsConfig | None = None,
    # AC FPF
    ac_fpf_pg0_source: str = "case",
    ac_fpf_vm_min_pu: float = 0.9,
    ac_fpf_vm_max_pu: float = 1.1,
    ac_fpf_max_iteration: int = 300,
    ac_fpf_max_loading_percent: float = 99.0,
    ac_fpf_init: str = "dc",
    ac_fpf_max_attempts: int = 1,
    ac_fpf_per_attempt_timeout: float = 0,
    # shared
    opf_cfg: OPFConfig | None = None,
    opf_dc_flow_consistency_tol_mw: float = _DEFAULT_OPF_DC_FLOW_CONSISTENCY_TOL_MW,
    opf_bus_balance_tol_mw: float = _DEFAULT_OPF_BUS_BALANCE_TOL_MW,
    path_base_dir: str | os.PathLike[str] | None = None,
    allow_download: bool = False,
    dc_checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """
    Compute per-line radii and return a single results dict (including '__meta__').

    Important
    ---------
    AC certificate is always computed around an AC PF base point (AC PF, not DC OPF).
    DC OPF is optional and serves only as a dispatch source.

    Extensions policy (AC-focused defaults)
    --------------------------------------
    - DC probabilistic post-processing is computed only if enabled.
    - DC N-1 radii are computed only if enabled AND dc.mode=materialize.
    """
    t0 = time.time()

    cfg = opf_cfg if opf_cfg is not None else DEFAULT_OPF
    ext = dc_extensions if dc_extensions is not None else DCExtensionsConfig()
    ac_ext = ac_extensions if ac_extensions is not None else ACExtensionsConfig()

    bd = str(base_dispatch).strip().lower()
    if bd not in {"case", "dc_opf", "acpf", "ac_fpf"}:
        raise ValueError("base_dispatch must be case|dc_opf|acpf|ac_fpf")

    # Determinism visibility / fail-fast validation (logged at compute start).
    _assert_and_log_effective_unconstrained_line_nom_mw(
        cfg=cfg,
        source=("caller_provided" if opf_cfg is not None else "DEFAULT_OPF"),
    )

    if not bool(compute_dc) and not bool(compute_ac):
        raise ValueError("At least one of compute_dc or compute_ac must be enabled.")

    if bool(compute_ac) and not bool(ac_lossless):
        raise NotImplementedError(
            "AC certificate with ac.lossless=false is not implemented: the current "
            "ACOperator is a series-only model and would be inconsistent with a "
            "full pandapower PF containing charging/shunts. Use ac.lossless=true "
            "or implement the full pi-model Jacobian first."
        )

    dc_probabilistic_enabled = bool(ext.probabilistic_enabled)
    dc_nminus1_enabled = bool(ext.nminus1_enabled)
    dc_nminus1_update_sensitivities = bool(ext.nminus1_update_sensitivities)
    dc_nminus1_islanding = str(ext.nminus1_islanding).strip().lower() or "skip"
    if dc_nminus1_islanding not in {"skip", "raise"}:
        raise ValueError("dc_extensions.nminus1_islanding must be 'skip'|'raise'")

    logger.info(
        "DC extensions: probabilistic=%s nminus1=%s (update_sensitivities=%s islanding=%s)",
        dc_probabilistic_enabled,
        dc_nminus1_enabled,
        dc_nminus1_update_sensitivities,
        dc_nminus1_islanding,
    )

    base_dir = Path(path_base_dir).resolve() if path_base_dir is not None else None
    input_path_abs = _ensure_input_case_file(
        str(input_path), base_dir=base_dir, allow_download=bool(allow_download)
    )
    case_tag = Path(input_path_abs).stem

    with log_stage(logger, f"{case_tag}: Read Data"):
        net = load_network(input_path_abs)
        assert_line_limit_sources_present(net)

    # ---------- DC base dispatch / base quantities ----------
    bp_dc_meta: dict[str, Any] | None = None
    base_dc: LineBaseQuantities | None = None
    gen_dispatch_for_ac: dict[str, float] = {}
    used_headroom_factor: float = float(cfg.headroom_factor)

    if bd == "dc_opf":
        with log_stage(logger, f"{case_tag}: Base dispatch via DC OPF (PyPSA+HiGHS)"):
            bp_dc, base_dc, used_headroom_factor = _solve_dc_opf_once(
                net=net,
                slack_bus=int(slack_bus),
                opf_cfg=cfg,
                limit_factor=1.0,
                case_tag=case_tag,
            )
            bp_dc_meta = bp_dc.to_meta_dict()
            gen_dispatch_for_ac = dict(bp_dc.gen_dispatch_mw_by_name)
    elif bd == "acpf":
        logger.info("%s: base_dispatch=acpf: solving AC PF base point.", case_tag)
    elif bd == "ac_fpf":
        logger.info("%s: base_dispatch=ac_fpf: solving AC FPF base point.", case_tag)
    else:
        logger.info("%s: base_dispatch=case: using case dispatch (NO OPF).", case_tag)

    # ---------- ACPF: Solve AC PF early to extract bus injections ----------
    from stability_radius.base_point.types import BasePointAC
    from stability_radius.base_point.pypsa_pf import PyPSAAPFResult as _PyPSAAPFResult

    acpf_bp_ac: BasePointAC | None = None
    acpf_base_pf: _PyPSAAPFResult | None = None
    acpf_loss_correction_mw: float = 0.0

    # ---------- Explicit AC base point resolution (ac_fpf / acpf) ----------
    if bd in {"acpf", "ac_fpf"} and (bool(compute_dc) or bool(compute_ac)):
        line_indices_sorted = [int(x) for x in sorted(net.line.index)]
        if bd == "ac_fpf":
            from stability_radius.base_point.pandapower_opp import (
                ACFPFConfig as _ACFPFConfig,
            )

            with log_stage(logger, f"{case_tag}: AC FPF (runopp) base dispatch"):
                n_buses_net = (
                    int(len(net.bus))
                    if hasattr(net, "bus") and net.bus is not None
                    else 0
                )
                n_lines_net = (
                    int(len(net.line))
                    if hasattr(net, "line") and net.line is not None
                    else 0
                )
                fpf_cfg = _ACFPFConfig(
                    pg0_source=str(ac_fpf_pg0_source),
                    vm_min_pu=float(ac_fpf_vm_min_pu),
                    vm_max_pu=float(ac_fpf_vm_max_pu),
                    max_iteration=int(ac_fpf_max_iteration),
                    max_loading_percent=float(ac_fpf_max_loading_percent),
                    init=str(ac_fpf_init),
                    max_attempts=int(ac_fpf_max_attempts),
                    per_attempt_timeout=float(ac_fpf_per_attempt_timeout),
                )
                logger.info(
                    "%s: AC FPF: starting runopp solve (buses=%d, lines=%d, "
                    "lossless=%s, pg0_source=%s, max_attempts=%d)",
                    case_tag,
                    n_buses_net,
                    n_lines_net,
                    ac_lossless,
                    fpf_cfg.pg0_source,
                    fpf_cfg.max_attempts,
                )
                acpf_bp_ac, acpf_base_pf = solve_ac_fpf_base_point(
                    net=net,
                    slack_bus=int(slack_bus),
                    lossless=bool(ac_lossless),
                    fpf_cfg=fpf_cfg,
                    opf_cfg=cfg,
                    line_indices=line_indices_sorted,
                )
                logger.info(
                    "%s: AC FPF: runopp solve completed (status=%s, attempt=%s, repairs=%s)",
                    case_tag,
                    acpf_base_pf.status if acpf_base_pf else "n/a",
                    acpf_base_pf.pf_attempt if acpf_base_pf else "n/a",
                    acpf_base_pf.pf_repairs if acpf_base_pf else [],
                )
                if acpf_base_pf.bus_p_mw is None:
                    raise RuntimeError(
                        "AC FPF mode requires bus_p_mw from runopp solver."
                    )
                acpf_loss_correction_mw = float(np.sum(acpf_base_pf.bus_p_mw))
                logger.info(
                    "%s: AC FPF solved; AC loss imbalance=%.6g MW "
                    "(will be absorbed by slack for DC model).",
                    case_tag,
                    acpf_loss_correction_mw,
                )
        else:
            with log_stage(logger, f"{case_tag}: AC PF base dispatch"):
                acpf_bp_ac, acpf_base_pf = solve_ac_pf_base_point(
                    net=net,
                    slack_bus=int(slack_bus),
                    pf_solver=str(ac_pf_solver),
                    pf_init=str(ac_pf_init),
                    lossless=bool(ac_lossless),
                    gen_dispatch_mw_by_name={},
                    line_indices=line_indices_sorted,
                    distributed_slack=bool(ac_distributed_slack),
                    trafo_model=str(ac_trafo_model),
                )
                if acpf_base_pf.bus_p_mw is None:
                    raise RuntimeError("AC PF mode requires bus_p_mw from runpp.")
                acpf_loss_correction_mw = float(np.sum(acpf_base_pf.bus_p_mw))
                logger.info(
                    "%s: AC PF solved; AC loss imbalance=%.6g MW "
                    "(will be absorbed by slack for DC model).",
                    case_tag,
                    acpf_loss_correction_mw,
                )

    # ---------- DC model stage ----------
    results_lines: dict[str, dict[str, Any]] = {}
    consistency: dict[str, float] = {}
    nminus1_computed = False
    probabilistic_computed = False

    H_full = None
    dc_op = None
    if bool(compute_dc):
        dc_mode_eff = str(dc_mode).strip().lower()
        if dc_mode_eff not in {"operator", "materialize"}:
            raise ValueError("dc_mode must be operator|materialize")
        if dc_chunk_size <= 0:
            raise ValueError("dc_chunk_size must be positive")
        if float(dc_inj_std_mw) <= 0:
            raise ValueError("dc.inj_std_mw must be positive")

        with log_stage(logger, f"{case_tag}: Build DC Model (mode={dc_mode_eff})"):
            if dc_mode_eff == "materialize":
                H_full, dc_op = build_dc_matrices(
                    net,
                    slack_bus=int(slack_bus),
                    chunk_size=int(dc_chunk_size),
                    dtype=dc_dtype,
                )
            else:
                dc_op = build_dc_operator(net, slack_bus=int(slack_bus))

        if base_dc is None:
            if (
                bd in {"acpf", "ac_fpf"}
                and acpf_base_pf is not None
                and acpf_base_pf.bus_p_mw is not None
            ):
                with log_stage(
                    logger,
                    f"{case_tag}: Build DC base point from ACPF bus injections",
                ):
                    bp_dc_a, base_dc_a, dc_op_a = build_dc_base_point_from_acpf(
                        net=net,
                        slack_bus=int(slack_bus),
                        acpf_bus_p_mw=acpf_base_pf.bus_p_mw,
                        acpf_bus_ids=list(acpf_base_pf.bus_ids),
                        dc_op=dc_op,
                        limit_factor=1.0,
                    )
                    bp_dc_meta = bp_dc_a.to_meta_dict()
                    base_dc = base_dc_a
                    dc_op = dc_op_a
            else:
                with log_stage(
                    logger, f"{case_tag}: Build DC base point from case injections"
                ):
                    bp_dc2, base_dc2, dc_op2 = build_dc_base_point_case(
                        net=net, slack_bus=int(slack_bus), dc_op=dc_op, limit_factor=1.0
                    )
                    bp_dc_meta = bp_dc2.to_meta_dict()
                    base_dc = base_dc2
                    dc_op = dc_op2

        if bd == "dc_opf":
            with log_stage(
                logger, f"{case_tag}: Consistency Check (OPF -> DCOperator)"
            ):
                dc_op_ck = (
                    dc_op
                    if dc_op is not None
                    else build_dc_operator(net, slack_bus=int(slack_bus))
                )
                consistency = _check_opf_dc_consistency(
                    dc_op=dc_op_ck,
                    base=base_dc,
                    tol_flow_mw=float(opf_dc_flow_consistency_tol_mw),
                    tol_balance_mw=float(opf_bus_balance_tol_mw),
                )
        else:
            consistency = {
                "opf_bus_balance_abs_mw": float("nan"),
                "opf_dc_flow_max_abs_diff_mw": float("nan"),
                "opf_dc_flow_tol_mw": float(opf_dc_flow_consistency_tol_mw),
                "opf_bus_balance_tol_mw": float(opf_bus_balance_tol_mw),
            }

        with log_stage(logger, f"{case_tag}: Compute Radii (DC)"):
            if H_full is not None:
                l2 = compute_l2_radius(net, H_full, base=base_dc)

                parts: list[dict[str, dict[str, Any]]] = [l2]

                if bool(dc_probabilistic_enabled):
                    prob = _compute_probabilistic_from_l2_results(
                        l2_results=l2, inj_std_mw=float(dc_inj_std_mw)
                    )
                    parts.append(prob)
                    probabilistic_computed = True

                if bool(dc_nminus1_enabled):
                    nminus1 = compute_nminus1_l2_radius(
                        net,
                        H_full,
                        update_sensitivities=bool(dc_nminus1_update_sensitivities),
                        islanding=str(dc_nminus1_islanding),
                        base=base_dc,
                    )
                    parts.append(nminus1)
                    nminus1_computed = True

                results_lines = _merge_line_results(*parts)
            else:
                if bool(dc_nminus1_enabled):
                    raise ValueError(
                        "dc_extensions.nminus1_enabled=1 requires dc.mode=materialize (N-1 needs H_full)."
                    )
                if dc_op is None:
                    raise AssertionError("Internal error: DC operator missing.")

                l2 = _compute_radii_operator_path(
                    dc_op=dc_op,
                    base=base_dc,
                    dc_chunk_size=int(dc_chunk_size),
                    net=net,
                )
                results_lines = l2

                if bool(dc_probabilistic_enabled):
                    prob = _compute_probabilistic_from_l2_results(
                        l2_results=l2, inj_std_mw=float(dc_inj_std_mw)
                    )
                    results_lines = _merge_line_results(results_lines, prob)
                    probabilistic_computed = True

                nminus1_computed = False

        # ---- Write DC checkpoint (partial results) for timeout recovery ----
        if dc_checkpoint_path and results_lines:
            import json as _json
            import tempfile as _tempfile

            dc_checkpoint = {
                "__meta__": {
                    "dc_checkpoint": True,
                    "base_dispatch": str(bd),
                    "base_dispatch_requested": str(base_dispatch),
                    "base_point_dc": bp_dc_meta if bp_dc_meta is not None else {},
                },
                **results_lines,
            }
            try:
                cp_dir = os.path.dirname(dc_checkpoint_path)
                fd, tmp_path = _tempfile.mkstemp(
                    dir=cp_dir, suffix=".tmp", prefix=".dc_cp_"
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as fh:
                        _json.dump(
                            dc_checkpoint,
                            fh,
                            indent=2,
                            default=lambda o: (
                                float(o)
                                if isinstance(o, (np.floating, np.integer))
                                else (
                                    o.tolist() if isinstance(o, np.ndarray) else str(o)
                                )
                            ),
                        )
                    os.replace(tmp_path, dc_checkpoint_path)
                    logger.debug(
                        "%s: DC checkpoint written to %s",
                        case_tag,
                        dc_checkpoint_path,
                    )
                except Exception:
                    # Clean up temp file on error.
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise
            except Exception:
                logger.warning(
                    "%s: Failed to write DC checkpoint (non-fatal).",
                    case_tag,
                    exc_info=True,
                )

    # ---------- AC stage (base point is AC PF) ----------
    bp_ac_meta: dict[str, Any] | None = None
    ac_pf_status = "n/a"
    ac_pf_attempt = "n/a"
    ac_pf_repairs: list[str] = []
    ac_distributed_slack_used = False
    ac_q_limit_hit = False
    ac_q_limit_events: list[dict[str, Any]] = []
    ac_sigma_computed = False
    ac_metric_computed = False
    ac_nonlinear_validation_computed = False
    ac_feasibility: ACFeasibilityResult | None = None
    h_vectors_saved: dict[str, np.ndarray] | None = None
    ac_validation_report: dict[str, Any] | None = None

    # Determine whether h-vectors are needed (sigma, metric, validation, or save).
    ac_sigma_enabled = bool(
        str(ac_ext.sigma_p_mw_source).strip()
        and str(ac_ext.sigma_q_mvar_source).strip()
    )
    ac_nonlinear_validation_enabled = bool(ac_ext.nonlinear_validation_enabled)
    ac_need_h = (
        ac_sigma_enabled
        or bool(ac_ext.metric_enabled)
        or bool(ac_ext.save_h_vectors)
        or bool(ac_nonlinear_validation_enabled)
    )

    if bool(compute_ac):
        if ac_chunk_size <= 0:
            raise ValueError("ac.chunk_size must be positive")
        apfi = str(ac_pf_init).strip().lower()
        if apfi not in {"flat", "dc", "pp"}:
            raise ValueError("ac.pf_init must be flat|dc|pp")

        try:
            if (
                bd in {"acpf", "ac_fpf"}
                and acpf_bp_ac is not None
                and acpf_base_pf is not None
            ):
                # Reuse early AC PF/FPF solved for ACPF/AC_FPF base dispatch.
                logger.info("%s: Reusing early AC result for %s mode.", case_tag, bd)
                bp_ac = acpf_bp_ac
                base_pf = acpf_base_pf
                bp_ac_meta = bp_ac.to_meta_dict()
                ac_pf_status = str(bp_ac.status)
                ac_pf_attempt = str(bp_ac.pf_attempt)
                ac_pf_repairs = list(bp_ac.pf_repairs)
                ac_distributed_slack_used = bool(bp_ac.distributed_slack_used)
                ac_q_limit_hit = bool(getattr(bp_ac, "q_limit_hit", False))
                ac_q_limit_events = list(getattr(bp_ac, "q_limit_events", ()) or ())
            else:
                with log_stage(
                    logger,
                    f"{case_tag}: Solve AC PF base point (solver={ac_pf_solver})",
                ):
                    bp_ac, base_pf = solve_ac_pf_base_point(
                        net=net,
                        slack_bus=int(slack_bus),
                        pf_solver=str(ac_pf_solver),
                        pf_init=str(ac_pf_init),
                        lossless=bool(ac_lossless),
                        gen_dispatch_mw_by_name=gen_dispatch_for_ac
                        if bd == "dc_opf"
                        else {},
                        line_indices=[int(x) for x in sorted(net.line.index)],
                        distributed_slack=bool(ac_distributed_slack),
                        trafo_model=str(ac_trafo_model),
                    )
                    bp_ac_meta = bp_ac.to_meta_dict()
                    ac_pf_status = str(bp_ac.status)
                    ac_pf_attempt = str(bp_ac.pf_attempt)
                    ac_pf_repairs = list(bp_ac.pf_repairs)
                    ac_distributed_slack_used = bool(bp_ac.distributed_slack_used)
                    ac_q_limit_hit = bool(getattr(bp_ac, "q_limit_hit", False))
                    ac_q_limit_events = list(getattr(bp_ac, "q_limit_events", ()) or ())
        except Exception as exc:
            raise RuntimeError(
                f"{case_tag}: AC power flow base point failed; "
                "AC radius computation requires a solved AC PF/FPF regime."
            ) from exc

        # ---------- AC feasibility gate ----------
        if bool(compute_ac):
            with log_stage(logger, f"{case_tag}: AC Feasibility Check"):
                ac_feasibility = check_ac_base_point_feasibility(
                    net=net, base_pf=base_pf
                )
                if not ac_feasibility.is_feasible:
                    logger.warning(
                        "%s: AC base point violates %d constrained line limits. "
                        "AC radii on those lines will be negative. "
                        "Consider using a smaller headroom_factor or ACOPF dispatch.",
                        case_tag,
                        ac_feasibility.n_constrained_violated,
                    )

        if bool(compute_ac):
            h_vecs_raw: dict[str, np.ndarray] | None = None
            h_from_full: np.ndarray | None = None
            h_to_full: np.ndarray | None = None
            h_bind: np.ndarray | None = None
            s0_mva: np.ndarray | None = None
            s_limit_mva: np.ndarray | None = None
            line_ids_ac: list[int] | None = None
            bus_ids_ac: list[int] | None = None
            n_bus_ac: int | None = None
            pq_mask_ac: np.ndarray | None = None

            with log_stage(logger, f"{case_tag}: Compute Radii (AC L2)"):
                from stability_radius.radii.ac_l2 import compute_ac_l2_radius

                ac = compute_ac_l2_radius(
                    net,
                    base_pf=base_pf,
                    slack_bus=int(slack_bus),
                    chunk_size=int(ac_chunk_size),
                    balance=bool(ac_balance),
                    lossless=bool(ac_lossless),
                    return_h_vectors=bool(ac_need_h),
                )

                # Extract h-vector data before merging (the "_h_vectors" key is not per-line).
                h_vecs_raw: dict[str, np.ndarray] | None = None
                if ac_need_h and "_h_vectors" in ac:
                    h_vecs_raw = ac.pop("_h_vectors")

                results_lines = _merge_line_results(results_lines, ac)

            if h_vecs_raw is not None:
                bus_ids_ac = [int(x) for x in sorted(net.bus.index)]
                n_bus_ac = len(bus_ids_ac)
                # Use the slack position the AC operator actually eliminated.
                slack_pos = int(h_vectors_ac["slack_pos"])
                slack_bus_id = int(h_vectors_ac["slack_bus_id"])
                pq_mask_ac = h_vecs_raw.get("pq_mask")

                h_from_full = expand_h_reduced_to_full(
                    h_vecs_raw["h_from"],
                    n_bus=n_bus_ac,
                    slack_pos=slack_pos,
                    pq_mask=pq_mask_ac,
                )
                h_to_full = expand_h_reduced_to_full(
                    h_vecs_raw["h_to"],
                    n_bus=n_bus_ac,
                    slack_pos=slack_pos,
                    pq_mask=pq_mask_ac,
                )

                h_bind, s0_mva, s_limit_mva, line_ids_ac = extract_binding_end_data(
                    ac_results=ac, h_from=h_from_full, h_to=h_to_full
                )

            # ---------- AC sigma/metric post-processing ----------
            if (
                ac_sigma_enabled
                and h_bind is not None
                and s0_mva is not None
                and s_limit_mva is not None
                and line_ids_ac is not None
                and n_bus_ac is not None
            ):
                with log_stage(logger, f"{case_tag}: Compute Radii (AC Sigma)"):
                    sigma_p, sigma_q = _build_sigma_arrays(
                        ac_ext=ac_ext, n_bus=n_bus_ac
                    )

                    from stability_radius.radii.ac_sigma_radius import (
                        compute_ac_sigma_radius,
                    )

                    ac_sigma = compute_ac_sigma_radius(
                        h_vectors=h_bind,
                        s_limit_mva=s_limit_mva,
                        s0_mva=s0_mva,
                        sigma_p_mw=sigma_p,
                        sigma_q_mvar=sigma_q,
                        line_ids=line_ids_ac,
                        balance=bool(ac_balance),
                        pq_mask=pq_mask_ac,
                    )
                    results_lines = _merge_line_results(results_lines, ac_sigma)
                    ac_sigma_computed = True

                if bool(ac_ext.metric_enabled):
                    with log_stage(logger, f"{case_tag}: Compute Radii (AC Metric)"):
                        from stability_radius.radii.ac_metric_radius import (
                            compute_ac_metric_radius,
                        )

                        # M = diag(1/sigma^2) — inverse covariance (sigma-radius cross-check).
                        M_diag = 1.0 / np.concatenate(
                            [sigma_p * sigma_p, sigma_q * sigma_q]
                        )

                        ac_metric = compute_ac_metric_radius(
                            h_vectors=h_bind,
                            s_limit_mva=s_limit_mva,
                            s0_mva=s0_mva,
                            M=M_diag,
                            line_ids=line_ids_ac,
                            balance=bool(ac_balance),
                            pq_mask=pq_mask_ac,
                        )
                        results_lines = _merge_line_results(results_lines, ac_metric)
                        ac_metric_computed = True

            # ---------- AC nonlinear replay validation ----------
            if (
                ac_nonlinear_validation_enabled
                and h_bind is not None
                and s0_mva is not None
                and s_limit_mva is not None
                and line_ids_ac is not None
                and n_bus_ac is not None
            ):
                with log_stage(
                    logger, f"{case_tag}: Validate AC L2 (nonlinear replay)"
                ):
                    replay_gen_dispatch = dict(gen_dispatch_for_ac)
                    if not replay_gen_dispatch:
                        replay_gen_dispatch = dict(
                            getattr(bp_ac, "gen_dispatch_mw_by_name", ()) or ()
                        )
                    ac_validation_report = _run_ac_nonlinear_validation_topk(
                        net=net,
                        case_tag=case_tag,
                        results_lines=results_lines,
                        h_bind=h_bind,
                        s0_mva=s0_mva,
                        s_limit_mva=s_limit_mva,
                        line_ids=line_ids_ac,
                        n_bus=int(n_bus_ac),
                        pq_mask=pq_mask_ac,
                        balance=bool(ac_balance),
                        lossless=bool(ac_lossless),
                        gen_dispatch_mw_by_name=replay_gen_dispatch,
                        top_k=int(ac_ext.nonlinear_validation_top_k),
                        scale_max=float(ac_ext.nonlinear_validation_scale_max),
                        tol=float(ac_ext.nonlinear_validation_tol),
                        max_iter=int(ac_ext.nonlinear_validation_max_iter),
                    )
                    ac_nonlinear_validation_computed = True

            if bool(ac_q_limit_hit):
                # Q-limit saturation at the base point is now handled inside
                # compute_ac_l2_radius: fully saturated PV buses are linearized
                # as PQ, so the certificate matches the PF's converged active
                # set.  The flags below are kept for accounting; the
                # linearization is only marked invalid when saturation could
                # NOT be absorbed (partially saturated multi-gen buses).
                for row in results_lines.values():
                    if not isinstance(row, dict) or "radius_ac_l2" not in row:
                        continue
                    row["q_limit_hit"] = True
                    row["pv_pq_switch_detected"] = True
                    row["linearization_status"] = "q_limit_absorbed_as_pq"
            elif bool(compute_ac):
                for row in results_lines.values():
                    if not isinstance(row, dict) or "radius_ac_l2" not in row:
                        continue
                    row.setdefault("q_limit_hit", False)
                    row.setdefault("pv_pq_switch_detected", False)

            # ---------- h-vector saving ----------
            if (
                bool(ac_ext.save_h_vectors)
                and h_from_full is not None
                and h_to_full is not None
                and bus_ids_ac is not None
            ):
                h_vectors_saved = {
                    "h_from": h_from_full,
                    "h_to": h_to_full,
                    "bus_ids": np.array(bus_ids_ac, dtype=int),
                    "line_ids": np.array(
                        [int(x) for x in sorted(net.line.index)], dtype=int
                    ),
                }

    elapsed = float(time.time() - t0)
    logger.info("%s: Total compute time: %.3f sec", case_tag, elapsed)

    # ---- meta ----
    # Build sigma serialisation for __meta__.ac.
    _sigma_p_meta: list[float] | float | None = None
    _sigma_q_meta: list[float] | float | None = None
    _sigma_source: str | None = None
    _sigma_n_ts: int | None = None

    if ac_sigma_computed:
        src_p = str(ac_ext.sigma_p_mw_source).strip().lower()
        _sigma_source = src_p if src_p else None
        _sigma_n_ts = ac_ext.sigma_n_timesteps

        if src_p == "uniform":
            _sigma_p_meta = float(ac_ext.sigma_p_mw_uniform)
        elif src_p == "uc_jl" and ac_ext.sigma_p_mw_array is not None:
            _arr = ac_ext.sigma_p_mw_array
            _sigma_p_meta = _arr.tolist() if hasattr(_arr, "tolist") else list(_arr)
        src_q = str(ac_ext.sigma_q_mvar_source).strip().lower()
        if src_q == "uniform":
            _sigma_q_meta = float(ac_ext.sigma_q_mvar_uniform)
        elif src_q == "uc_jl" and ac_ext.sigma_q_mvar_array is not None:
            _arr = ac_ext.sigma_q_mvar_array
            _sigma_q_meta = _arr.tolist() if hasattr(_arr, "tolist") else list(_arr)

    results: dict[str, Any] = {
        "__meta__": {
            "schema_version": 3,
            "input_path": str(input_path_abs),
            "slack_bus": int(slack_bus),
            "base_dispatch": str(bd),
            "base_dispatch_requested": str(base_dispatch),
            "allow_download": bool(allow_download),
            "compute_dc": bool(compute_dc),
            "compute_ac": bool(compute_ac),
            "dc": {
                "mode": str(dc_mode).strip().lower(),
                "dtype": str(np.dtype(dc_dtype)),
                "chunk_size": int(dc_chunk_size),
                "inj_std_mw": float(dc_inj_std_mw),
                "probabilistic_enabled": bool(dc_probabilistic_enabled),
                "probabilistic_computed": bool(probabilistic_computed),
                "nminus1_enabled": bool(dc_nminus1_enabled),
                "nminus1_computed": bool(nminus1_computed),
                "nminus1_update_sensitivities": bool(dc_nminus1_update_sensitivities),
                "nminus1_islanding": str(dc_nminus1_islanding),
            },
            "ac": {
                "pf_solver": str(ac_pf_solver),
                "pf_init": str(ac_pf_init),
                "lossless": bool(ac_lossless),
                "distributed_slack_requested": bool(ac_distributed_slack),
                "distributed_slack": bool(ac_distributed_slack_used),
                "trafo_model": str(ac_trafo_model),
                "chunk_size": int(ac_chunk_size),
                "balance": bool(ac_balance),
                "pf_status": str(ac_pf_status),
                "pf_attempt": str(ac_pf_attempt),
                "pf_repairs": list(ac_pf_repairs),
                "q_limit_hit": bool(ac_q_limit_hit),
                "q_limit_events": list(ac_q_limit_events),
                "feasibility": ac_feasibility.to_meta_dict()
                if ac_feasibility is not None
                else None,
                "sigma_source": _sigma_source,
                "sigma_p_mw": _sigma_p_meta,
                "sigma_q_mvar": _sigma_q_meta,
                "sigma_n_timesteps": _sigma_n_ts,
                "sigma_computed": bool(ac_sigma_computed),
                "metric_enabled": bool(ac_ext.metric_enabled),
                "metric_computed": bool(ac_metric_computed),
                "save_h_vectors": bool(ac_ext.save_h_vectors),
                "nonlinear_validation_enabled": bool(
                    ac_ext.nonlinear_validation_enabled
                ),
                "nonlinear_validation_computed": bool(ac_nonlinear_validation_computed),
                "nonlinear_validation_top_k": int(ac_ext.nonlinear_validation_top_k),
                "nonlinear_validation_scale_max": float(
                    ac_ext.nonlinear_validation_scale_max
                ),
                "nonlinear_validation_tol": float(ac_ext.nonlinear_validation_tol),
                "nonlinear_validation_max_iter": int(
                    ac_ext.nonlinear_validation_max_iter
                ),
            },
            # critical artifacts for reproducibility & MC consistency checks
            "base_point_dc": bp_dc_meta,
            "base_point_ac": bp_ac_meta,
            "opf": {
                "solver": str(cfg.highs.solver_name) if bd == "dc_opf" else "n/a",
                "threads": int(cfg.highs.threads) if bd == "dc_opf" else -1,
                "random_seed": int(cfg.highs.random_seed) if bd == "dc_opf" else -1,
                "headroom_factor_configured": float(cfg.headroom_factor)
                if bd == "dc_opf"
                else float("nan"),
                "headroom_factor_used": float(used_headroom_factor)
                if bd == "dc_opf"
                else float("nan"),
                "unconstrained_line_nom_mw": float(cfg.unconstrained_line_nom_mw)
                if bd == "dc_opf"
                else float("nan"),
                "ext_grid_absorption_mw": float(base_dc.opf_ext_grid_absorption_mw)
                if bd == "dc_opf" and base_dc is not None
                else 0.0,
            },
            "acpf_slack_loss_correction_mw": float(acpf_loss_correction_mw)
            if bd in {"acpf", "ac_fpf"}
            else None,
            "ac_fpf_pg0_source": str(ac_fpf_pg0_source) if bd == "ac_fpf" else None,
            "compute_time_sec": float(elapsed),
            **(consistency if consistency else {}),
        }
    }
    results.update(results_lines)

    if h_vectors_saved is not None:
        results["_h_vectors"] = h_vectors_saved
    if ac_validation_report is not None:
        results["_validation_report"] = ac_validation_report

    return results
