from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np

from stability_radius.config import DEFAULT_OPF, OPFConfig

logger = logging.getLogger(__name__)

_RATING_ZERO_EPS = 1e-12


class ConstraintStatus(str, Enum):
    """Per-constraint certificate status used in result rows."""

    OK_FINITE = "ok_finite"
    OK_INFINITE = "ok_infinite"
    BASE_INFEASIBLE = "base_infeasible"
    DEGENERATE_SENSITIVITY = "degenerate_sensitivity"
    UNCONSTRAINED_LIMIT = "unconstrained_limit"
    PF_FAILED = "pf_failed"
    JACOBIAN_SINGULAR = "jacobian_singular"
    NONLINEAR_UNVALIDATED = "nonlinear_unvalidated"
    NONLINEAR_OPTIMISTIC = "nonlinear_optimistic"
    POST_CONTINGENCY_INFEASIBLE = "post_contingency_infeasible"
    NONDIFFERENTIABLE_APPARENT_POWER = "nondifferentiable_apparent_power"


def classify_constraint_certificate(
    *,
    margin: float,
    dual_norm: float,
    eps: float,
    is_unconstrained: bool = False,
) -> tuple[str, float, float]:
    """
    Classify one affine thermal constraint and return nonnegative radius fields.

    Returns
    -------
    (status, certificate_radius, signed_distance)
        ``certificate_radius`` is never negative. ``signed_distance`` preserves
        the signed margin/dual-norm distance for diagnostics and sorting.
    """
    margin_f = float(margin)
    norm_f = float(dual_norm)
    eps_f = float(eps)

    if bool(is_unconstrained):
        return (
            ConstraintStatus.UNCONSTRAINED_LIMIT.value,
            float("inf"),
            float("inf"),
        )

    if not math.isfinite(margin_f) or not math.isfinite(norm_f) or norm_f < 0.0:
        return (
            ConstraintStatus.DEGENERATE_SENSITIVITY.value,
            float("nan"),
            float("nan"),
        )

    if margin_f < 0.0:
        signed = margin_f / norm_f if norm_f > eps_f else float("-inf")
        return ConstraintStatus.BASE_INFEASIBLE.value, 0.0, float(signed)

    if norm_f <= eps_f:
        return ConstraintStatus.OK_INFINITE.value, float("inf"), float("inf")

    radius = margin_f / norm_f
    return ConstraintStatus.OK_FINITE.value, float(max(radius, 0.0)), float(radius)


@dataclass(frozen=True)
class LineBaseQuantities:
    """
    Container for per-line base quantities used by radius calculations.

    Notes
    -----
    - flow0_mw is signed (PyPSA's convention for Line.p0 with bus0->bus1 direction).
    - p0_abs_mw is abs(flow0_mw).
    - limit_mva_assumed_mw is the thermal limit extracted from the case (typically MVA),
      and then used as MW under the DC PF=1 convention.
    - margin_mw = limit - abs(flow0).  Can be negative if the base flow exceeds the limit.

    Unconstrained lines
    -------------------
    In MATPOWER/PGLib convention, rating=0 means "unconstrained", not "zero limit".
    For correctness and to avoid false bottlenecks, we use a large finite surrogate limit
    (see estimate_line_limit_mva*()), and store a per-line flag:
      - is_unconstrained[pos] == True  -> the extracted limit is a surrogate, not a real constraint.

    OPF metadata
    ------------
    Base point in this project is OPF-based (PyPSA+HiGHS), therefore:
    - opf_status
    - opf_objective
    are expected to be populated for pipeline runs.

    Bus injections (for consistency checks)
    ---------------------------------------
    - bus_ids is the stable bus ordering used across the project (sorted pandapower net.bus.index).
    - bus_injections_mw is aligned with bus_ids and corresponds to the OPF dispatch result
      (sum gens at bus - sum loads at bus), used to validate OPF -> DCOperator consistency.

    Generator dispatch (for AC PF reuse)
    ------------------------------------
    - opf_gen_dispatch_mw_by_name stores generator active power dispatch (MW) keyed by PyPSA
      generator name (e.g. gen_0, ext_0). This is used by the AC PF base-point builder to
      avoid re-solving OPF when AC computations are requested.

    Units (project contract)
    ------------------------
    - P, f0, Δp, c, margin: MW
    - rateA/sn_mva/max_mva: MVA in source data
      (used as MW under PF=1 assumption in lossless DC)
    """

    line_indices: list[int]
    flow0_mw: np.ndarray  # shape (m,)
    p0_abs_mw: np.ndarray  # shape (m,)
    limit_mva_assumed_mw: np.ndarray  # shape (m,)
    margin_mw: np.ndarray  # shape (m,)

    # Optional, but when present should be aligned with line_indices.
    is_unconstrained: np.ndarray | None = None  # shape (m,), dtype=bool

    opf_status: str | None = None
    opf_objective: float | None = None

    bus_ids: list[int] | None = None
    bus_injections_mw: np.ndarray | None = None

    # Optional diagnostics (not required by radii computations).
    opf_limits_mw: np.ndarray | None = None

    # Optional: per-generator dispatch (PyPSA generator names -> P(MW)).
    opf_gen_dispatch_mw_by_name: tuple[tuple[str, float], ...] | None = None

    # Optional: ext_grid absorption used by OPF (MW).
    opf_ext_grid_absorption_mw: float = 0.0


def _line_row_id(line_row: object) -> str:
    """Best-effort human-readable line id for error messages."""
    name = getattr(line_row, "name", None)
    return str(name) if name is not None else "unknown"


def _bus_vn_kv(net: object, bus_id: int) -> float:
    """Return bus nominal voltage vn_kv if available, else NaN."""
    bus_tbl = getattr(net, "bus", None)
    if bus_tbl is None or len(bus_tbl) == 0:
        return float("nan")
    if bus_id not in bus_tbl.index:
        return float("nan")
    if "vn_kv" not in bus_tbl.columns:
        return float("nan")
    return float(bus_tbl.loc[bus_id, "vn_kv"])


def assert_line_limit_sources_present(net: object) -> None:
    """
    Fail-fast sanity check: ensure the loaded network contains at least one supported
    source of thermal limits for lines.
    """
    line_tbl = getattr(net, "line", None)
    bus_tbl = getattr(net, "bus", None)

    if line_tbl is None:
        raise ValueError(
            "Loaded network has no 'net.line' table (pandapower conversion failed)."
        )
    if bus_tbl is None:
        raise ValueError(
            "Loaded network has no 'net.bus' table (pandapower conversion failed)."
        )
    if len(line_tbl) == 0:
        raise ValueError(
            "Loaded network has zero net.line entries. This project computes radii for pandapower net.line; "
            "check the MATPOWER/PGLib converter output."
        )

    explicit_cols = ("rateA", "rate_a_mva", "sn_mva", "max_mva")

    has_explicit = False
    for c in explicit_cols:
        if c not in getattr(line_tbl, "columns", ()):
            continue
        try:
            has_explicit = bool(line_tbl[c].notna().any())
        except Exception:  # noqa: BLE001
            has_explicit = True
        if has_explicit:
            break

    has_current_based = False
    if "max_i_ka" in getattr(line_tbl, "columns", ()) and "vn_kv" in getattr(
        bus_tbl, "columns", ()
    ):
        try:
            has_current_based = bool(
                line_tbl["max_i_ka"].notna().any() and bus_tbl["vn_kv"].notna().any()
            )
        except Exception:  # noqa: BLE001
            has_current_based = True

    logger.debug(
        "Thermal rating sources detected: explicit=%s, current_based=%s",
        bool(has_explicit),
        bool(has_current_based),
    )

    if has_explicit or has_current_based:
        return

    try:
        line_cols = list(getattr(line_tbl, "columns", []))
    except Exception:
        line_cols = []
    try:
        bus_cols = list(getattr(bus_tbl, "columns", []))
    except Exception:
        bus_cols = []

    raise ValueError(
        "Missing line thermal rating sources after loading the network. "
        "Expected either MATPOWER ratings on net.line "
        "(rateA/rate_a_mva/sn_mva/max_mva) or a current-based rating from "
        "(net.line.max_i_ka + net.bus.vn_kv). "
        f"Available net.line columns={line_cols!r}, net.bus columns={bus_cols!r}. "
        "This indicates a parser/converter issue; fix stability_radius.parsers.matpower."
    )


def _resolve_unconstrained_surrogate_mva(
    unconstrained_surrogate_mva: float | None,
) -> float:
    """
    Return a deterministic, *finite* surrogate limit used for unconstrained lines.

    Notes
    -----
    - We intentionally do NOT return +inf, because:
        * downstream consumers (tables/plots) should not treat it as a true constraint,
          and a finite surrogate is easier to handle consistently.
    - Default value is aligned with OPFConfig.unconstrained_line_nom_mw.
    """
    v = (
        float(DEFAULT_OPF.unconstrained_line_nom_mw)
        if unconstrained_surrogate_mva is None
        else float(unconstrained_surrogate_mva)
    )
    if (not math.isfinite(v)) or v <= 0.0:
        raise ValueError(
            "unconstrained_surrogate_mva must be finite and >0. "
            f"Got unconstrained_surrogate_mva={unconstrained_surrogate_mva!r} -> resolved={v!r}"
        )
    return float(v)


def estimate_line_limit_mva_with_flag(
    net: Any,
    line_row: Any,
    *,
    unconstrained_surrogate_mva: float | None = None,
) -> tuple[float, bool]:
    """
    Extract a line thermal limit in MVA and return (limit_mva, is_unconstrained).

    Key correctness convention
    --------------------------
    MATPOWER/PGLib (and thus many PGLib-derived pandapower nets) use:
      rateA == 0  => "unconstrained"
    not "zero thermal limit".

    Therefore:
    - For explicit rating columns (rateA/rate_a_mva/sn_mva/max_mva):
        * v in {0, NaN, +inf} is treated as unconstrained and mapped to a large finite surrogate.
    - For current-based rating (max_i_ka):
        * i_ka in {0, NaN, +inf} is treated as unconstrained and mapped to the same surrogate.

    Parameters
    ----------
    net:
        pandapower net.
    line_row:
        A net.line row (pandas Series-like).
    unconstrained_surrogate_mva:
        Optional finite surrogate (default: DEFAULT_OPF.unconstrained_line_nom_mw).

    Returns
    -------
    (limit_mva, is_unconstrained)
    """
    surrogate = _resolve_unconstrained_surrogate_mva(unconstrained_surrogate_mva)

    try:
        max_loading_percent = float(line_row.get("max_loading_percent", 100.0))
    except (TypeError, ValueError):
        max_loading_percent = 100.0
    if not np.isfinite(max_loading_percent) or max_loading_percent <= 0:
        max_loading_percent = 100.0
    mult = float(max_loading_percent) / 100.0

    def _unconstrained_surrogate(reason: str) -> tuple[float, bool]:
        # DEBUG only: can be many lines in large cases.
        """Internal helper for module-local processing."""
        logger.debug(
            "Line %s: unconstrained (%s) -> using surrogate limit=%.6g MVA (mult=%.6g)",
            _line_row_id(line_row),
            reason,
            float(surrogate) * float(mult),
            float(mult),
        )
        return float(surrogate) * float(mult), True

    # Prefer explicit columns (MATPOWER/PGLib convention).
    for k in ("rateA", "rate_a_mva", "sn_mva", "max_mva"):
        if k not in line_row:
            continue

        try:
            v = float(line_row[k])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Line {_line_row_id(line_row)}: failed to parse {k} as float: {line_row.get(k)!r}"
            ) from e

        # Negative ratings are invalid (including -inf).
        if math.isfinite(v) and v < 0:
            raise ValueError(
                f"Line {_line_row_id(line_row)}: invalid negative line rating {k}={v!r}"
            )
        if math.isinf(v) and v < 0:
            raise ValueError(
                f"Line {_line_row_id(line_row)}: invalid negative infinite line rating {k}={v!r}"
            )

        # Unconstrained semantics: v in {0, NaN, +inf}.
        if (not math.isfinite(v)) or abs(v) <= _RATING_ZERO_EPS:
            return _unconstrained_surrogate(f"{k}={v!r}")

        return float(v) * float(mult), False

    # Fall back to current-based rating if explicit rating is missing.
    if "max_i_ka" in line_row:
        try:
            i_ka = float(line_row.get("max_i_ka", float("nan")))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Line {_line_row_id(line_row)}: failed to parse max_i_ka as float: {line_row.get('max_i_ka')!r}"
            ) from e

        if math.isfinite(i_ka) and i_ka < 0:
            raise ValueError(
                f"Line {_line_row_id(line_row)}: invalid negative max_i_ka={i_ka!r}"
            )
        if math.isinf(i_ka) and i_ka < 0:
            raise ValueError(
                f"Line {_line_row_id(line_row)}: invalid negative infinite max_i_ka={i_ka!r}"
            )

        if (not math.isfinite(i_ka)) or abs(i_ka) <= _RATING_ZERO_EPS:
            return _unconstrained_surrogate(f"max_i_ka={i_ka!r}")

        fb = int(line_row.get("from_bus", -1))
        vn_kv = _bus_vn_kv(net, fb)
        if not np.isfinite(vn_kv) or vn_kv <= 0:
            raise ValueError(
                f"Line {_line_row_id(line_row)}: cannot derive limit from max_i_ka "
                f"because net.bus.vn_kv is missing/invalid for from_bus={fb} (vn_kv={vn_kv!r})."
            )

        s_mva = math.sqrt(3.0) * float(vn_kv) * float(i_ka)
        if not np.isfinite(s_mva) or s_mva < 0:
            raise ValueError(
                f"Line {_line_row_id(line_row)}: derived invalid S_MVA={s_mva!r} "
                f"from vn_kv={vn_kv!r}, max_i_ka={i_ka!r}."
            )
        return float(s_mva) * float(mult), False

    try:
        available = list(getattr(line_row, "index", []))
    except Exception:
        available = []

    raise ValueError(
        "Missing explicit line thermal rating. Expected one of: "
        "rateA, rate_a_mva, sn_mva, max_mva, or (max_i_ka + net.bus.vn_kv). "
        f"line={_line_row_id(line_row)}, available_columns={available!r}"
    )


def estimate_line_limit_mva(
    net: Any,
    line_row: Any,
    *,
    unconstrained_surrogate_mva: float | None = None,
) -> float:
    """
    Extract a line thermal limit in MVA using explicit case / converted data.

    See also
    --------
    estimate_line_limit_mva_with_flag : returns (limit_mva, is_unconstrained).
    """
    limit, _is_unconstrained = estimate_line_limit_mva_with_flag(
        net,
        line_row,
        unconstrained_surrogate_mva=unconstrained_surrogate_mva,
    )
    return float(limit)


def sorted_line_limits_mva(net: Any) -> tuple[list[int], np.ndarray]:
    """Return sorted pandapower line ids and their MVA thermal limits."""
    line_ids = [int(x) for x in sorted(net.line.index)]
    limits = np.empty(len(line_ids), dtype=float)
    for pos, lid in enumerate(line_ids):
        limits[pos] = float(estimate_line_limit_mva(net, net.line.loc[lid]))
    return line_ids, limits


def get_line_base_quantities(
    net: Any,
    *,
    limit_factor: float = 1.0,
    line_indices: Sequence[int] | None = None,
    opf_cfg: OPFConfig | None = None,
) -> LineBaseQuantities:
    """
    Extract per-line base flows, limits, and margins around an OPF base point.

    Project policy: base point is PyPSA DC OPF (HiGHS).

    Important behavior for unconstrained lines
    ------------------------------------------
    - If a line is "unconstrained" in MATPOWER/PGLib sense (rateA==0 / NaN / +inf),
      a large finite surrogate limit is used.
    - OPF headroom is applied ONLY to *constrained* lines.
      For unconstrained lines, the surrogate is not scaled by headroom to keep the
      surrogate purely numerical (matching historical inf->unconstrained_nom behavior).
    """
    cfg = opf_cfg if opf_cfg is not None else DEFAULT_OPF

    if limit_factor <= 0:
        raise ValueError("limit_factor must be positive.")

    opf_headroom = float(getattr(cfg, "headroom_factor", 1.0))
    if not math.isfinite(opf_headroom) or opf_headroom <= 0.0:
        raise ValueError("opf_cfg.headroom_factor must be finite and >0.")

    idx = (
        sorted(net.line.index)
        if line_indices is None
        else [int(x) for x in line_indices]
    )

    unconstrained_surrogate_mva = float(
        getattr(cfg, "unconstrained_line_nom_mw", DEFAULT_OPF.unconstrained_line_nom_mw)
    )
    limits_mva = np.empty(len(idx), dtype=float)
    is_unconstrained = np.zeros(len(idx), dtype=bool)

    for pos, (_, line_row) in enumerate(net.line.loc[idx].iterrows()):
        s_limit_mva, is_uc = estimate_line_limit_mva_with_flag(
            net,
            line_row,
            unconstrained_surrogate_mva=unconstrained_surrogate_mva,
        )
        limits_mva[pos] = float(s_limit_mva) * float(limit_factor)
        is_unconstrained[pos] = bool(is_uc)

    limits_mva_assumed_mw = limits_mva.copy()

    if np.isnan(limits_mva_assumed_mw).any():
        bad = np.where(np.isnan(limits_mva_assumed_mw))[0]
        raise ValueError(
            "Line limit extraction produced NaN. This indicates invalid rating data. "
            f"Bad line positions count={int(bad.size)} (first 10: {bad[:10].tolist()})."
        )

    if np.any(np.isfinite(limits_mva_assumed_mw) & (limits_mva_assumed_mw < 0.0)):
        bad = np.where(
            np.isfinite(limits_mva_assumed_mw) & (limits_mva_assumed_mw < 0.0)
        )[0]
        raise ValueError(
            "Negative line limit encountered after scaling. "
            f"Bad line positions count={int(bad.size)} (first 10: {bad[:10].tolist()})."
        )

    opf_limits = limits_mva_assumed_mw.copy()
    # Apply headroom to constrained lines only.
    finite_and_constrained = np.isfinite(opf_limits) & (~is_unconstrained)
    opf_limits[finite_and_constrained] = opf_limits[finite_and_constrained] * float(
        opf_headroom
    )

    from stability_radius.base_point.pypsa_opf import (
        solve_dc_opf_base_flows_from_pandapower,
    )

    logger.info(
        "Solving OPF base point via PyPSA DC OPF (solver=%s, threads=%d, headroom_factor=%s, limit_factor=%s)...",
        str(cfg.highs.solver_name),
        int(cfg.highs.threads),
        float(opf_headroom),
        float(limit_factor),
    )
    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=idx,
        line_limits_mw=opf_limits,
        opf_cfg=cfg,
    )
    flow0 = np.asarray(opf_res.line_flows_mw, dtype=float)

    p0_abs = np.abs(flow0)
    margins = limits_mva_assumed_mw - p0_abs

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    if tuple(bus_ids) != tuple(opf_res.bus_ids):
        raise ValueError(
            "Internal consistency error: PyPSA OPF returned bus_ids not matching pandapower net.bus ordering. "
            f"pandapower(sorted)={bus_ids[:10]}..., pypsa={list(opf_res.bus_ids)[:10]}..."
        )

    bus_inj = np.asarray(opf_res.bus_injections_mw, dtype=float).reshape(-1)
    if bus_inj.shape != (len(bus_ids),):
        raise ValueError(
            f"Unexpected bus_injections_mw shape from OPF: got {bus_inj.shape}, expected ({len(bus_ids)},)"
        )

    return LineBaseQuantities(
        line_indices=idx,
        flow0_mw=flow0,
        p0_abs_mw=p0_abs,
        limit_mva_assumed_mw=limits_mva_assumed_mw,
        margin_mw=margins,
        is_unconstrained=is_unconstrained,
        opf_status=str(opf_res.status),
        opf_objective=float(opf_res.objective),
        bus_ids=bus_ids,
        bus_injections_mw=bus_inj,
        opf_limits_mw=opf_limits,
        opf_gen_dispatch_mw_by_name=getattr(opf_res, "gen_dispatch_mw_by_name", None),
        opf_ext_grid_absorption_mw=float(
            getattr(opf_res, "ext_grid_absorption_mw", 0.0)
        ),
    )


def line_key(line_idx: int) -> str:
    """Stable external key format for per-line result dictionaries."""
    return f"line_{int(line_idx)}"


def as_2d_square_matrix(x: np.ndarray, n: int, *, name: str) -> np.ndarray:
    """Validate and return x as a (n,n) float matrix."""
    X = np.asarray(x, dtype=float)
    if X.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n},{n}); got {X.shape}.")
    return X


def as_1d_vector(x: np.ndarray, n: int, *, name: str) -> np.ndarray:
    """Validate and return x as a (n,) float vector."""
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},); got {v.shape}.")
    return v
