from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LineBaseQuantities:
    """
    Container for per-line base quantities used by radius calculations.

    Notes
    -----
    - flow0_mw is signed (pandapower's p_from_mw convention).
    - p0_abs_mw is abs(flow0_mw).
    - limit_mw_est is the thermal limit from the case (treated as MW proxy).
    - margin_mw = max(limit - abs(flow0), 0).

    OPF metadata
    ------------
    Base point in this project is OPF-based (PyPSA+HiGHS), therefore:
    - opf_status
    - opf_objective
    are always expected to be populated for pipeline runs.
    """

    line_indices: list[int]
    flow0_mw: np.ndarray  # shape (m,)
    p0_abs_mw: np.ndarray  # shape (m,)
    limit_mw_est: np.ndarray  # shape (m,)
    margin_mw: np.ndarray  # shape (m,)

    opf_status: str | None = None
    opf_objective: float | None = None


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

    Required sources (MATPOWER/PGLib policy)
    ----------------------------------------
    At least one of:
    - explicit rating columns on net.line:
        rateA / rate_a_mva / sn_mva / max_mva
    - or a deterministic fallback source:
        net.line.max_i_ka AND net.bus.vn_kv

    Why this exists
    ---------------
    If neither source exists, that's a MATPOWER->pandapower conversion/parsing issue
    (the radii code cannot "guess" missing limits). This check provides a clearer,
    earlier error than failing deep inside radii/OPF.
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
        except Exception:  # noqa: BLE001 - defensive for non-standard tables
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
        "(rateA/rate_a_mva/sn_mva/max_mva) or a fallback based on "
        "(net.line.max_i_ka + net.bus.vn_kv). "
        f"Available net.line columns={line_cols!r}, net.bus columns={bus_cols!r}. "
        "This indicates a parser/converter issue; fix stability_radius.parsers.matpower."
    )


def estimate_line_limit_mva(net, line_row) -> float:
    """
    Extract a line thermal limit in MVA using explicit case / converted data.

    Supported explicit sources (deterministic)
    ------------------------------------------
    1) MATPOWER/PGLib-style rating columns (if present on pandapower net.line):
        - rateA / rate_a_mva / sn_mva / max_mva

       MATPOWER convention:
       - rateA == 0 means unconstrained => +inf.

    2) pandapower current rating + nominal voltage:
        - max_i_ka and net.bus.vn_kv
        - S_MVA = sqrt(3) * V_kV * I_kA

    max_loading_percent (if present) is applied as a deterministic multiplier.

    Raises
    ------
    ValueError
        If no supported rating data is found or if found data is invalid.

    Returns
    -------
    float
        Limit in MVA (or +inf for unconstrained).
    """
    try:
        max_loading_percent = float(line_row.get("max_loading_percent", 100.0))
    except (TypeError, ValueError):
        max_loading_percent = 100.0
    if not np.isfinite(max_loading_percent) or max_loading_percent <= 0:
        max_loading_percent = 100.0
    mult = float(max_loading_percent) / 100.0

    for k in ("rateA", "rate_a_mva", "sn_mva", "max_mva"):
        if k not in line_row:
            continue

        try:
            v = float(line_row[k])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Line {_line_row_id(line_row)}: failed to parse {k} as float: {line_row.get(k)!r}"
            ) from e

        if math.isnan(v):
            logger.debug(
                "Line %s: rating column %s is NaN; trying other sources.",
                _line_row_id(line_row),
                k,
            )
            continue

        if math.isinf(v):
            return float("inf")

        if not np.isfinite(v):
            raise ValueError(
                f"Line {_line_row_id(line_row)}: invalid non-finite line rating {k}={v!r}"
            )

        # MATPOWER: rateA==0 => unconstrained. We keep the same meaning for equivalent columns.
        if abs(v) <= 1e-12:
            return float("inf")
        if v < 0:
            raise ValueError(
                f"Line {_line_row_id(line_row)}: invalid negative line rating {k}={v!r}"
            )

        return float(v) * mult

    if "max_i_ka" in line_row:
        try:
            i_ka = float(line_row.get("max_i_ka", float("nan")))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Line {_line_row_id(line_row)}: failed to parse max_i_ka as float: {line_row.get('max_i_ka')!r}"
            ) from e

        if math.isnan(i_ka):
            logger.debug(
                "Line %s: max_i_ka is NaN; cannot derive limit from current.",
                _line_row_id(line_row),
            )
        else:
            if math.isinf(i_ka):
                return float("inf")
            if not np.isfinite(i_ka):
                raise ValueError(
                    f"Line {_line_row_id(line_row)}: invalid non-finite max_i_ka={i_ka!r}"
                )

            if abs(i_ka) <= 1e-12:
                return float("inf")
            if i_ka < 0:
                raise ValueError(
                    f"Line {_line_row_id(line_row)}: invalid negative max_i_ka={i_ka!r}"
                )

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
            return float(s_mva) * mult

    try:
        available = list(getattr(line_row, "index", []))
    except Exception:
        available = []

    raise ValueError(
        "Missing explicit line thermal rating. Expected one of: "
        "rateA, rate_a_mva, sn_mva, max_mva, or (max_i_ka + net.bus.vn_kv). "
        f"line={_line_row_id(line_row)}, available_columns={available!r}"
    )


def get_line_base_quantities(
    net,
    *,
    margin_factor: float = 1.0,
    line_indices: Sequence[int] | None = None,
) -> LineBaseQuantities:
    """
    Extract per-line base flows, limits, and margins around an OPF base point.

    Project policy
    --------------
    Base point is ALWAYS:
      - PyPSA DC OPF solved by HiGHS

    Limits
    ------
    Extracted only from explicit converted data (`estimate_line_limit_mva`).
    If rating information is missing/invalid, ValueError is raised.

    Parameters
    ----------
    net:
        pandapower network.
    margin_factor:
        Multiplier applied to extracted limits (e.g., 0.9 for conservative).
    line_indices:
        Optional explicit ordering of line indices. Defaults to sorted(net.line.index).

    Returns
    -------
    LineBaseQuantities
    """
    if margin_factor <= 0:
        raise ValueError("margin_factor must be positive.")

    idx = (
        sorted(net.line.index)
        if line_indices is None
        else [int(x) for x in line_indices]
    )

    limits = np.empty(len(idx), dtype=float)
    for pos, (_, line_row) in enumerate(net.line.loc[idx].iterrows()):
        s_limit = estimate_line_limit_mva(net, line_row)
        limits[pos] = float(s_limit) * float(margin_factor)

    if np.isnan(limits).any():
        bad = np.where(np.isnan(limits))[0]
        raise ValueError(
            "Line limit extraction produced NaN. This indicates invalid rating data. "
            f"Bad line positions count={int(bad.size)} (first 10: {bad[:10].tolist()})."
        )

    if np.any(np.isfinite(limits) & (limits < 0.0)):
        bad = np.where(np.isfinite(limits) & (limits < 0.0))[0]
        raise ValueError(
            "Negative line limit encountered after scaling. "
            f"Bad line positions count={int(bad.size)} (first 10: {bad[:10].tolist()})."
        )

    from stability_radius.opf.pypsa_opf import solve_dc_opf_base_flows_from_pandapower

    logger.info(
        "Solving OPF base point via PyPSA DC OPF (HiGHS, margin_factor=%s)...",
        float(margin_factor),
    )
    opf_res = solve_dc_opf_base_flows_from_pandapower(
        net=net,
        line_indices=idx,
        line_limits_mw=limits,
    )
    flow0 = np.asarray(opf_res.line_flows_mw, dtype=float)

    p0_abs = np.abs(flow0)
    margins = np.maximum(limits - p0_abs, 0.0)

    return LineBaseQuantities(
        line_indices=idx,
        flow0_mw=flow0,
        p0_abs_mw=p0_abs,
        limit_mw_est=limits,
        margin_mw=margins,
        opf_status=str(opf_res.status),
        opf_objective=float(opf_res.objective),
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
