from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    _HAVE_SCIPY = True
except ImportError:
    sp = None
    spla = None
    _HAVE_SCIPY = False

_X_TOTAL_EPS = 1e-12
_TAP_RATIO_EPS = 1e-12
_SHIFT_DEG_EPS = 1e-9


def _pp_row_id(row: Any) -> str:
    """Best-effort row id for diagnostics (works for pandas Series/DataFrame rows)."""
    try:
        name = getattr(row, "name", None)
        return str(name) if name is not None else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass(frozen=True)
class DCOperator:
    """
    DC linear operator for PTDF-like computations without necessarily materializing H_full.

    Represents:
        theta_red = Bred^{-1} * p_red
        f_lines = diag(b_lines) * A_lines_red * theta_red

    Notes
    -----
    - bus_ids and line_ids are sorted to keep deterministic ordering.
    - slack bus is eliminated (reduced system).

    Important modeling contract
    ---------------------------
    This operator models a *lossless DC* network with a single reference angle (slack).
    Transformer tap ratios (if any) must be handled consistently with the OPF construction.

    In this repository, we follow a MATPOWER-style DC approximation for 2-winding transformers:
      - tap ratio is applied as a scaling of branch susceptance: b_eff = b / tap
      - phase shift (shift_degree) is NOT modeled by this operator. If you have phase shifters,
        you must either:
          * reject them (default, recommended), or
          * explicitly allow and accept that shift is ignored (allow_phase_shift=True).

    Units
    -----
    - vn_kv: kV
    - x/r: Ohm
    - theta: rad
    - p / f: MW
    - b: MW/rad (computed as V_kV^2 / X_ohm)
    """

    bus_ids: tuple[int, ...]
    line_ids: tuple[int, ...]
    slack_pos: int

    from_bus_pos: np.ndarray  # (m_lines,)
    to_bus_pos: np.ndarray  # (m_lines,)
    b: np.ndarray  # (m_lines,)

    mask_non_slack: np.ndarray  # (n,), bool
    red_pos_of_bus_pos: np.ndarray  # (n,), -1 for slack else 0..n-2

    Bred_lu: Any  # scipy.sparse.linalg.SuperLU
    W: Any  # scipy sparse matrix (m_lines x (n-1)) = diag(b_lines)*A_lines_red

    @property
    def n_bus(self) -> int:
        """Number of buses in the model (including slack)."""
        return int(len(self.bus_ids))

    @property
    def n_line(self) -> int:
        """Number of monitored lines in the model (pandapower net.line)."""
        return int(len(self.line_ids))

    def solve_Bred(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve Bred * x = rhs using cached LU factorization.

        Parameters
        ----------
        rhs:
            Array of shape (n-1,) or (n-1, k).

        Returns
        -------
        np.ndarray
            Solution with same shape as rhs.
        """
        rhs_arr = np.asarray(rhs, dtype=float)
        if rhs_arr.ndim == 1:
            if rhs_arr.shape != (self.n_bus - 1,):
                raise ValueError(
                    f"rhs must have shape ({self.n_bus - 1},) got {rhs_arr.shape}"
                )
            return np.asarray(self.Bred_lu.solve(rhs_arr), dtype=float)
        if rhs_arr.ndim == 2:
            if rhs_arr.shape[0] != self.n_bus - 1:
                raise ValueError(
                    f"rhs must have shape ({self.n_bus - 1}, k) got {rhs_arr.shape}"
                )
            return np.asarray(self.Bred_lu.solve(rhs_arr), dtype=float)
        raise ValueError(f"rhs must be 1D or 2D, got ndim={rhs_arr.ndim}")

    def flows_from_delta_injections(self, delta: np.ndarray) -> np.ndarray:
        """
        Compute flow changes on monitored lines for given balanced injections.

        Parameters
        ----------
        delta:
            Balanced nodal injections, shape (k, n_bus) or (n_bus,).
            Caller must enforce sum(delta)=0 for each sample.

        Returns
        -------
        np.ndarray
            Flow deltas on monitored lines, shape (k, n_line).
        """
        d = np.asarray(delta, dtype=float)
        if d.ndim == 1:
            d = d.reshape(1, -1)
        if d.ndim != 2:
            raise ValueError(f"delta must be 1D or 2D; got {d.shape}")
        if d.shape[1] != self.n_bus:
            raise ValueError(
                f"delta must have n_bus={self.n_bus} columns; got {d.shape}"
            )

        rhs = d[:, self.mask_non_slack].T  # (n-1, k)
        theta = self.solve_Bred(rhs)  # (n-1, k)
        flow = (self.W @ theta).T  # (k, m)
        return np.asarray(flow, dtype=float)

    def row_sensitivities_transposed(self, line_positions: np.ndarray) -> np.ndarray:
        """
        Compute columns of H_red^T for selected monitored lines.

        Returns
        -------
        np.ndarray
            Matrix (n-1, k), column j equals g_{line_positions[j]}^T.
        """
        pos = np.asarray(line_positions, dtype=int).reshape(-1)
        m = self.n_line
        if pos.size == 0:
            return np.zeros((self.n_bus - 1, 0), dtype=float)
        if np.any(pos < 0) or np.any(pos >= m):
            raise ValueError("line_positions contains out-of-range indices")

        k = int(pos.size)
        rhs = np.zeros((self.n_bus - 1, k), dtype=float)

        for j, p in enumerate(pos.tolist()):
            b_p = float(self.b[p])
            if not np.isfinite(b_p) or abs(b_p) < 1e-18:
                continue

            fb_pos = int(self.from_bus_pos[p])
            tb_pos = int(self.to_bus_pos[p])

            rfb = int(self.red_pos_of_bus_pos[fb_pos])
            rtb = int(self.red_pos_of_bus_pos[tb_pos])

            if rfb >= 0:
                rhs[rfb, j] += b_p
            if rtb >= 0:
                rhs[rtb, j] -= b_p

        return self.solve_Bred(rhs)

    def row_norms_l2(self, *, chunk_size: int = 256) -> np.ndarray:
        """
        Compute per-line L2 norms ||g_l||_2 for all monitored lines.

        Parameters
        ----------
        chunk_size:
            Number of lines per LU-solve block.

        Returns
        -------
        np.ndarray
            norms array of shape (m,).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        m = self.n_line
        norms = np.zeros(m, dtype=float)

        start = 0
        while start < m:
            end = min(m, start + int(chunk_size))
            block = np.arange(start, end, dtype=int)
            Y = self.row_sensitivities_transposed(block)  # (n-1, k)
            norms[start:end] = np.linalg.norm(Y, ord=2, axis=0)
            start = end

        return norms

    def materialize_H_full(
        self, *, dtype: np.dtype = np.float64, chunk_size: int = 256
    ) -> np.ndarray:
        """
        Materialize dense H_full (m_lines x n_bus) for monitored lines.

        Parameters
        ----------
        dtype:
            Output dtype (float64 or float32 recommended).
        chunk_size:
            Number of lines per LU-solve block.

        Returns
        -------
        np.ndarray
            Dense H_full matrix.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        m, n = self.n_line, self.n_bus
        H_full = np.zeros((m, n), dtype=dtype, order="C")

        start = 0
        while start < m:
            end = min(m, start + int(chunk_size))
            block = np.arange(start, end, dtype=int)

            Y = self.row_sensitivities_transposed(block)  # (n-1, k)
            H_full[start:end, self.mask_non_slack] = Y.T.astype(dtype, copy=False)

            start = end

        return H_full


def _resolve_slack_pos(bus_ids: list[int], slack_bus: int) -> int:
    """
    Resolve slack bus as either:
    - exact bus id (present in bus_ids)
    - positional index in current bus ordering
    """
    bus_pos = {int(bid): pos for pos, bid in enumerate(bus_ids)}
    if int(slack_bus) in bus_pos:
        return int(bus_pos[int(slack_bus)])
    if 0 <= int(slack_bus) < len(bus_ids):
        return int(slack_bus)
    raise ValueError(
        f"slack_bus must be a valid bus id or position. Got {slack_bus!r}; "
        f"valid positions: [0, {len(bus_ids) - 1}]"
    )


def _is_in_service(row: Any) -> bool:
    """Return pandapower element in_service flag."""
    return bool(row.get("in_service", True))


def _bus_vn_kv(net: Any, bus_id: int) -> float:
    """Return bus nominal voltage vn_kv if available, else NaN."""
    if bus_id in net.bus.index and "vn_kv" in net.bus.columns:
        return float(net.bus.loc[bus_id, "vn_kv"])
    return float("nan")


def _line_x_total_ohm(line_row: Any, *, strict_units: bool) -> float:
    """
    Compute total series reactance of a pandapower line in Ohm.

    strict_units behavior
    ---------------------
    - If strict_units=True: raises ValueError on non-finite, near-zero, or non-positive x_ohm.
    - If strict_units=False: returns 0.0 for non-finite/near-zero x_ohm (legacy behavior),
      and allows negative x_ohm.
    """
    x_ohm_per_km = float(line_row.get("x_ohm_per_km", 0.0))
    length_km = float(line_row.get("length_km", 0.0))
    parallel = float(line_row.get("parallel", 1.0))
    if not np.isfinite(parallel) or parallel <= 0:
        if strict_units:
            raise ValueError(
                f"Line {_pp_row_id(line_row)}: parallel must be finite and >0; got {parallel!r}"
            )
        parallel = 1.0

    x_total = x_ohm_per_km * length_km / parallel
    if (not np.isfinite(x_total)) or abs(float(x_total)) <= _X_TOTAL_EPS:
        if strict_units:
            raise ValueError(
                f"Line {_pp_row_id(line_row)}: x_total_ohm must be finite and non-zero; got {x_total!r} "
                f"(x_ohm_per_km={x_ohm_per_km!r}, length_km={length_km!r}, parallel={parallel!r})"
            )
        return 0.0

    if strict_units and float(x_total) <= 0.0:
        raise ValueError(
            f"Line {_pp_row_id(line_row)}: x_total_ohm must be >0 (strict_units); got {x_total!r}"
        )

    return float(x_total)


def _line_b_mw_from_row(net: Any, line_row: Any, *, strict_units: bool) -> float:
    """
    Compute DC branch coefficient b for a pandapower line in MW/rad.

    b ≈ V_kV^2 / X_ohm

    strict_units behavior
    ---------------------
    - vn_kv must be finite and >0
    - x_ohm must be finite and >0
    """
    fb = int(line_row.get("from_bus", -1))
    vn_kv = _bus_vn_kv(net, fb)
    if not np.isfinite(vn_kv) or float(vn_kv) <= 0.0:
        if strict_units:
            raise ValueError(
                f"Line {_pp_row_id(line_row)}: from_bus={fb} has invalid vn_kv={vn_kv!r} (must be >0)."
            )
        return 0.0

    x_ohm = _line_x_total_ohm(line_row, strict_units=bool(strict_units))
    if not np.isfinite(x_ohm) or abs(float(x_ohm)) <= _X_TOTAL_EPS:
        if strict_units:
            raise ValueError(
                f"Line {_pp_row_id(line_row)}: invalid x_total_ohm={x_ohm!r}."
            )
        return 0.0

    if strict_units and float(x_ohm) <= 0.0:
        # Should already be caught in _line_x_total_ohm, keep defensive.
        raise ValueError(
            f"Line {_pp_row_id(line_row)}: x_total_ohm must be >0 (strict_units); got {x_ohm!r}"
        )

    return float((vn_kv * vn_kv) / x_ohm)


def trafo_x_total_ohm(net: Any, trafo_row: Any, *, strict_units: bool = False) -> float:
    """
    Approximate transformer series reactance in Ohms from pandapower trafo parameters.

    Uses:
      z_pu = vk_percent / 100
      r_pu = vkr_percent / 100
      x_pu = sqrt(max(z_pu^2 - r_pu^2, 0))
      Z_base = (V_kV^2) / S_MVA   [Ohm]
      x_ohm = x_pu * Z_base

    strict_units behavior
    ---------------------
    - sn_mva must be finite and >0
    - vn_hv_kv must be finite and >0 (either trafo field or hv bus)
    - x_ohm must be finite and >0
    """
    vk_percent = float(trafo_row.get("vk_percent", np.nan))
    if not np.isfinite(vk_percent):
        if strict_units:
            raise ValueError(
                f"Trafo {_pp_row_id(trafo_row)}: missing/invalid vk_percent={vk_percent!r}"
            )
        return 0.0

    sn_mva = float(trafo_row.get("sn_mva", np.nan))
    if not np.isfinite(sn_mva) or float(sn_mva) <= 0.0:
        if strict_units:
            raise ValueError(
                f"Trafo {_pp_row_id(trafo_row)}: sn_mva must be finite and >0; got {sn_mva!r}"
            )
        return 0.0

    z_pu = float(vk_percent) / 100.0
    r_pu = float(trafo_row.get("vkr_percent", 0.0)) / 100.0
    x_pu2 = z_pu * z_pu - r_pu * r_pu
    x_pu = float(np.sqrt(max(float(x_pu2), 0.0)))

    vn_hv_kv = float(trafo_row.get("vn_hv_kv", np.nan))
    if not np.isfinite(vn_hv_kv) or float(vn_hv_kv) <= 0.0:
        hv_bus = int(trafo_row.get("hv_bus", -1))
        vn_hv_kv = _bus_vn_kv(net, hv_bus)

    if not np.isfinite(vn_hv_kv) or float(vn_hv_kv) <= 0.0:
        if strict_units:
            raise ValueError(
                f"Trafo {_pp_row_id(trafo_row)}: vn_hv_kv must be finite and >0; got {vn_hv_kv!r}"
            )
        return 0.0

    z_base_ohm = (float(vn_hv_kv) * float(vn_hv_kv)) / float(sn_mva)
    x_ohm = float(x_pu) * float(z_base_ohm)

    if (not np.isfinite(x_ohm)) or abs(float(x_ohm)) <= _X_TOTAL_EPS:
        if strict_units:
            raise ValueError(
                f"Trafo {_pp_row_id(trafo_row)}: x_ohm must be finite and non-zero; got {x_ohm!r}"
            )
        return 0.0

    if strict_units and float(x_ohm) <= 0.0:
        # Defensive (x_pu and z_base are non-negative in the formula).
        raise ValueError(
            f"Trafo {_pp_row_id(trafo_row)}: x_ohm must be >0 (strict_units); got {x_ohm!r}"
        )

    return float(x_ohm)


def trafo_tap_ratio(trafo_row: Any) -> float:
    """
    Compute an *effective* transformer tap ratio magnitude from pandapower trafo fields.

    Interpretation (pandapower-style)
    ---------------------------------
    When a transformer has a tap changer:
      tap = 1 + (tap_pos - tap_neutral) * tap_step_percent / 100

    If tap_side == "lv", we return 1/tap to represent the same ratio as a "hv-side"
    tap in a MATPOWER-style DC approximation, to keep a single consistent convention.

    Returns
    -------
    float
        Tap ratio magnitude (>=0), defaults to 1.0 if no tap data is present.

    Raises
    ------
    ValueError
        If computed tap ratio is non-finite or <= 0.
    """
    try:
        tap_side = str(trafo_row.get("tap_side", "")).strip().lower()
    except Exception:  # noqa: BLE001
        tap_side = ""

    if tap_side not in {"hv", "lv"}:
        return 1.0

    try:
        tap_step_percent = float(trafo_row.get("tap_step_percent", 0.0))
    except (TypeError, ValueError):
        tap_step_percent = 0.0
    if not np.isfinite(tap_step_percent) or abs(tap_step_percent) <= _TAP_RATIO_EPS:
        return 1.0

    try:
        tap_pos = float(trafo_row.get("tap_pos", 0.0))
    except (TypeError, ValueError):
        tap_pos = 0.0
    try:
        tap_neutral = float(trafo_row.get("tap_neutral", 0.0))
    except (TypeError, ValueError):
        tap_neutral = 0.0

    if not np.isfinite(tap_neutral):
        tap_neutral = 0.0
    if not np.isfinite(tap_pos):
        tap_pos = tap_neutral

    delta = tap_pos - tap_neutral
    tap = 1.0 + (delta * tap_step_percent / 100.0)

    if not np.isfinite(tap) or tap <= 0.0:
        raise ValueError(
            "Invalid tap computed from tap_pos/tap_neutral/tap_step_percent: "
            f"tap_side={tap_side!r}, tap_pos={tap_pos!r}, tap_neutral={tap_neutral!r}, tap_step_percent={tap_step_percent!r}, tap={tap!r}"
        )

    if tap_side == "lv":
        tap = 1.0 / tap

    return float(tap)


def _trafo_b_mw_from_row(net: Any, trafo_row: Any, *, strict_units: bool) -> float:
    """
    Compute DC branch coefficient b for a pandapower transformer in MW/rad.

    Uses hv-side nominal voltage:
        b ≈ V_hv_kV^2 / X_ohm

    Notes
    -----
    Tap ratios are handled outside of this function (see build_dc_operator).
    """
    hv_bus = int(trafo_row.get("hv_bus", -1))

    vn_kv = float(trafo_row.get("vn_hv_kv", np.nan))
    if not np.isfinite(vn_kv) or vn_kv <= 0:
        vn_kv = _bus_vn_kv(net, hv_bus)

    if not np.isfinite(vn_kv) or float(vn_kv) <= 0.0:
        if strict_units:
            raise ValueError(
                f"Trafo {_pp_row_id(trafo_row)}: invalid hv-side vn_kv={vn_kv!r} (must be >0)."
            )
        return 0.0

    x_ohm = float(trafo_x_total_ohm(net, trafo_row, strict_units=bool(strict_units)))
    if not np.isfinite(x_ohm) or abs(x_ohm) <= _X_TOTAL_EPS:
        if strict_units:
            raise ValueError(
                f"Trafo {_pp_row_id(trafo_row)}: invalid/zero x_ohm={x_ohm!r}."
            )
        return 0.0

    if strict_units and float(x_ohm) <= 0.0:
        raise ValueError(
            f"Trafo {_pp_row_id(trafo_row)}: x_ohm must be >0 (strict_units); got {x_ohm!r}"
        )

    return float((vn_kv * vn_kv) / x_ohm)


def _impedance_x_total_ohm(net: Any, imp_row: Any, *, strict_units: bool) -> float:
    """
    Approximate impedance element series reactance in Ohms.

    x_ohm ≈ x_pu * (V_kV^2 / S_MVA)
    """
    x_pu = float(imp_row.get("xft_pu", np.nan))
    if not np.isfinite(x_pu) or abs(float(x_pu)) <= _X_TOTAL_EPS:
        if strict_units:
            raise ValueError(
                f"Impedance {_pp_row_id(imp_row)}: xft_pu must be finite and non-zero; got {x_pu!r}"
            )
        return 0.0

    if strict_units and float(x_pu) <= 0.0:
        raise ValueError(
            f"Impedance {_pp_row_id(imp_row)}: xft_pu must be >0 (strict_units); got {x_pu!r}"
        )

    sn_mva = float(getattr(net, "sn_mva", np.nan))
    if not np.isfinite(sn_mva) or float(sn_mva) <= 0.0:
        if strict_units:
            raise ValueError(
                f"pandapower net.sn_mva must be finite and >0 (strict_units); got {sn_mva!r}"
            )
        return 0.0

    fb = int(imp_row.get("from_bus", -1))
    vn_kv = _bus_vn_kv(net, fb)
    if not np.isfinite(vn_kv) or float(vn_kv) <= 0.0:
        if strict_units:
            raise ValueError(
                f"Impedance {_pp_row_id(imp_row)}: from_bus={fb} has invalid vn_kv={vn_kv!r}."
            )
        return 0.0

    z_base_ohm = (float(vn_kv) * float(vn_kv)) / float(sn_mva)
    x_ohm = float(x_pu) * float(z_base_ohm)
    if (not np.isfinite(x_ohm)) or abs(float(x_ohm)) <= _X_TOTAL_EPS:
        if strict_units:
            raise ValueError(
                f"Impedance {_pp_row_id(imp_row)}: x_ohm must be finite and non-zero; got {x_ohm!r}"
            )
        return 0.0

    if strict_units and float(x_ohm) <= 0.0:
        raise ValueError(
            f"Impedance {_pp_row_id(imp_row)}: x_ohm must be >0 (strict_units); got {x_ohm!r}"
        )

    return float(x_ohm)


def _impedance_b_mw_from_row(net: Any, imp_row: Any, *, strict_units: bool) -> float:
    """
    Compute DC branch coefficient b for a pandapower impedance element in MW/rad.

    Uses from-bus nominal voltage:
        b ≈ V_kV^2 / X_ohm
    """
    fb = int(imp_row.get("from_bus", -1))

    vn_kv = _bus_vn_kv(net, fb)
    if not np.isfinite(vn_kv) or float(vn_kv) <= 0.0:
        if strict_units:
            raise ValueError(
                f"Impedance {_pp_row_id(imp_row)}: from_bus={fb} has invalid vn_kv={vn_kv!r}."
            )
        return 0.0

    x_ohm = float(_impedance_x_total_ohm(net, imp_row, strict_units=bool(strict_units)))
    if not np.isfinite(x_ohm) or abs(x_ohm) <= _X_TOTAL_EPS:
        if strict_units:
            raise ValueError(
                f"Impedance {_pp_row_id(imp_row)}: invalid/zero x_ohm={x_ohm!r}."
            )
        return 0.0

    if strict_units and float(x_ohm) <= 0.0:
        raise ValueError(
            f"Impedance {_pp_row_id(imp_row)}: x_ohm must be >0 (strict_units); got {x_ohm!r}"
        )

    return float((vn_kv * vn_kv) / x_ohm)


def _check_connected_to_slack(
    *,
    n_bus: int,
    slack_pos: int,
    edges: list[tuple[int, int]],
    bus_ids: list[int],
) -> None:
    """
    Ensure all buses are connected to slack via provided undirected edges.

    Raises ValueError with a diagnostic message if disconnected buses exist.
    """
    parent = np.arange(n_bus, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        if 0 <= a < n_bus and 0 <= b < n_bus and a != b:
            union(int(a), int(b))

    root_slack = find(int(slack_pos))
    disconnected_pos = [i for i in range(n_bus) if find(i) != root_slack]

    if disconnected_pos:
        disconnected_bus_ids = [int(bus_ids[i]) for i in disconnected_pos]
        msg = (
            "Network is disconnected from the chosen slack bus under the DC branch model. "
            f"Disconnected buses count={len(disconnected_bus_ids)} (first 20 ids: {disconnected_bus_ids[:20]}). "
            "This typically happens when some branches were converted into transformers/impedances "
            "and were ignored, or when too many branches have invalid/zero reactance."
        )
        raise ValueError(msg)


def build_dc_operator(
    net,
    slack_bus: int = 0,
    *,
    strict_units: bool = True,
    allow_phase_shift: bool = False,
) -> DCOperator:
    """
    Build a DCOperator for a pandapower network.

    Requirements
    ------------
    SciPy is required (sparse LU factorization).

    Behavior
    --------
    - Monitored elements: net.line (ordering: sorted(net.line.index))
    - B matrix assembly includes: net.line + net.trafo + net.impedance
      (in service, nonzero reactance).

    Transformer taps
    ----------------
    For pandapower net.trafo entries, we apply a MATPOWER-style DC approximation:
        b_eff = b / tap
    where tap is derived from tap_pos/tap_step_percent (see trafo_tap_ratio()).
    This improves consistency with DC-OPF formulations that account for tap ratios.

    Phase shifts (shift_degree)
    ---------------------------
    The project's DC model does NOT model phase shifts. Therefore:
    - if allow_phase_shift=False (default): any in-service transformer with shift_degree != 0 raises ValueError.
    - if allow_phase_shift=True: we log a WARNING and ignore the phase shift.

    strict_units
    ------------
    When True (default), this function fails fast on invalid units/physics:
    - vn_kv must be >0
    - x_ohm must be >0
    - sn_mva must be >0
    and it does not silently return 0.0 branch coefficients.

    Logging
    -------
    Uses DEBUG-level logs for normal progress to keep CLI output clean.
    """
    if not _HAVE_SCIPY:
        raise ImportError(
            "SciPy is required for DCOperator and DC matrices in this project. Install scipy."
        )

    strict = bool(strict_units)
    allow_shift = bool(allow_phase_shift)

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)
    if n_bus == 0:
        raise ValueError("Network has no buses.")

    slack_pos = _resolve_slack_pos(bus_ids, int(slack_bus))

    line_ids = [int(x) for x in sorted(net.line.index)]
    m_line = len(line_ids)
    if m_line == 0:
        raise ValueError("Network has no lines; cannot build DC operator.")

    bus_pos = {bid: pos for pos, bid in enumerate(bus_ids)}

    # ---------- Build monitored line incidence A_lines and b_lines ----------
    from_bus_pos = np.zeros(m_line, dtype=int)
    to_bus_pos = np.zeros(m_line, dtype=int)
    b_lines = np.zeros(m_line, dtype=float)

    A_line_row: list[int] = []
    A_line_col: list[int] = []
    A_line_data: list[float] = []

    valid_monitored = 0

    for row_pos, lid in enumerate(line_ids):
        row = net.line.loc[lid]
        fb = int(row.get("from_bus", -1))
        tb = int(row.get("to_bus", -1))

        if fb not in bus_pos or tb not in bus_pos:
            raise ValueError(
                f"Monitored line {lid} references missing buses ({fb} -> {tb})."
            )

        fpos = int(bus_pos[fb])
        tpos = int(bus_pos[tb])
        from_bus_pos[row_pos] = fpos
        to_bus_pos[row_pos] = tpos

        if not _is_in_service(row):
            continue

        try:
            b_i = float(_line_b_mw_from_row(net, row, strict_units=strict))
        except ValueError as e:
            logger.error(
                "Unit/parameter validation failed for monitored line %d (strict_units=%s): %s",
                int(lid),
                bool(strict),
                str(e),
            )
            raise

        if not np.isfinite(b_i) or abs(b_i) <= 0.0:
            raise ValueError(
                f"Monitored in-service line {lid} has invalid/zero reactance or voltage (b={b_i})."
            )

        b_lines[row_pos] = b_i
        valid_monitored += 1

        A_line_row.extend([row_pos, row_pos])
        A_line_col.extend([fpos, tpos])
        A_line_data.extend([1.0, -1.0])

    A_lines = sp.csr_matrix(
        (A_line_data, (A_line_row, A_line_col)),
        shape=(m_line, n_bus),
        dtype=float,
    )

    # ---------- Build full branch set for B (lines + trafos + impedances) ----------
    b_all: list[float] = []
    A_all_row: list[int] = []
    A_all_col: list[int] = []
    A_all_data: list[float] = []
    undirected_edges: list[tuple[int, int]] = []

    def _add_branch(*, fpos: int, tpos: int, b_val: float) -> None:
        if not np.isfinite(b_val) or abs(float(b_val)) <= 0.0:
            return
        r = len(b_all)
        b_all.append(float(b_val))
        A_all_row.extend([r, r])
        A_all_col.extend([int(fpos), int(tpos)])
        A_all_data.extend([1.0, -1.0])
        undirected_edges.append((int(fpos), int(tpos)))

    added_lines_to_b = 0
    for lid in line_ids:
        row = net.line.loc[lid]
        if not _is_in_service(row):
            continue
        fb = int(row.get("from_bus", -1))
        tb = int(row.get("to_bus", -1))
        if fb not in bus_pos or tb not in bus_pos:
            continue

        try:
            b_i = float(_line_b_mw_from_row(net, row, strict_units=strict))
        except ValueError as e:
            logger.error(
                "Unit/parameter validation failed for line %d used in B matrix (strict_units=%s): %s",
                int(lid),
                bool(strict),
                str(e),
            )
            raise

        if not np.isfinite(b_i) or abs(b_i) <= 0.0:
            # Legacy permissive behavior: skip invalid branches if strict_units=False.
            continue

        _add_branch(fpos=int(bus_pos[fb]), tpos=int(bus_pos[tb]), b_val=b_i)
        added_lines_to_b += 1

    added_trafos_to_b = 0
    if hasattr(net, "trafo") and net.trafo is not None and len(net.trafo):
        for tid in [int(x) for x in sorted(net.trafo.index)]:
            row = net.trafo.loc[tid]
            if not _is_in_service(row):
                continue
            hv = int(row.get("hv_bus", -1))
            lv = int(row.get("lv_bus", -1))
            if hv not in bus_pos or lv not in bus_pos:
                continue

            # Phase shifting transformer is not modeled by this project's DC operator.
            try:
                shift_deg = float(row.get("shift_degree", 0.0))
            except (TypeError, ValueError):
                shift_deg = 0.0
            if not np.isfinite(shift_deg):
                if strict:
                    raise ValueError(
                        f"Trafo {tid}: shift_degree must be finite (strict_units); got {shift_deg!r}"
                    )
                shift_deg = 0.0

            if abs(float(shift_deg)) > _SHIFT_DEG_EPS:
                if not allow_shift:
                    raise ValueError(
                        f"Trafo {tid}: non-zero shift_degree={shift_deg} deg is not supported by the project's DC model. "
                        "Set allow_phase_shift=True only if you explicitly accept ignoring phase shifters."
                    )
                logger.warning(
                    "Trafo %d has non-zero shift_degree=%.6g deg; phase shift is ignored (allow_phase_shift=1).",
                    int(tid),
                    float(shift_deg),
                )

            try:
                b_raw = float(_trafo_b_mw_from_row(net, row, strict_units=strict))
            except ValueError as e:
                logger.error(
                    "Unit/parameter validation failed for trafo %d used in B matrix (strict_units=%s): %s",
                    int(tid),
                    bool(strict),
                    str(e),
                )
                raise

            if not np.isfinite(b_raw) or abs(b_raw) <= 0.0:
                continue

            try:
                tap = float(trafo_tap_ratio(row))
            except ValueError as e:
                raise ValueError(
                    f"Invalid tap ratio for pandapower trafo {tid}."
                ) from e

            b_eff = float(b_raw / tap)

            if tap != 1.0:
                logger.debug(
                    "Trafo %d: applying DC tap ratio (MATPOWER-style): tap=%.6g, b_raw=%.6g, b_eff=%.6g",
                    int(tid),
                    float(tap),
                    float(b_raw),
                    float(b_eff),
                )

            _add_branch(fpos=int(bus_pos[hv]), tpos=int(bus_pos[lv]), b_val=b_eff)
            added_trafos_to_b += 1

    added_imps_to_b = 0
    if hasattr(net, "impedance") and net.impedance is not None and len(net.impedance):
        for iid in [int(x) for x in sorted(net.impedance.index)]:
            row = net.impedance.loc[iid]
            if not _is_in_service(row):
                continue
            fb = int(row.get("from_bus", -1))
            tb = int(row.get("to_bus", -1))
            if fb not in bus_pos or tb not in bus_pos:
                continue

            try:
                b_i = float(_impedance_b_mw_from_row(net, row, strict_units=strict))
            except ValueError as e:
                logger.error(
                    "Unit/parameter validation failed for impedance %d used in B matrix (strict_units=%s): %s",
                    int(iid),
                    bool(strict),
                    str(e),
                )
                raise

            if not np.isfinite(b_i) or abs(b_i) <= 0.0:
                continue
            _add_branch(fpos=int(bus_pos[fb]), tpos=int(bus_pos[tb]), b_val=b_i)
            added_imps_to_b += 1

    if not b_all:
        raise RuntimeError(
            "Cannot build DC nodal matrix: no valid in-service branches (lines/trafo/impedance) "
            "with nonzero reactance/voltage were found."
        )

    logger.debug(
        "DCOperator summary: buses=%d, monitored_lines=%d (valid=%d), B-branches=%d "
        "[lines=%d, trafos=%d, impedances=%d] strict_units=%s allow_phase_shift=%s",
        n_bus,
        m_line,
        valid_monitored,
        len(b_all),
        added_lines_to_b,
        added_trafos_to_b,
        added_imps_to_b,
        bool(strict),
        bool(allow_shift),
    )

    _check_connected_to_slack(
        n_bus=n_bus, slack_pos=slack_pos, edges=undirected_edges, bus_ids=bus_ids
    )

    A_all = sp.csr_matrix(
        (A_all_data, (A_all_row, A_all_col)),
        shape=(len(b_all), n_bus),
        dtype=float,
    )
    b_all_arr = np.asarray(b_all, dtype=float)

    mask_non_slack = np.ones(n_bus, dtype=bool)
    mask_non_slack[slack_pos] = False

    red_pos_of_bus_pos = np.full(n_bus, -1, dtype=int)
    red_pos_of_bus_pos[np.where(mask_non_slack)[0]] = np.arange(n_bus - 1, dtype=int)

    A_all_red = A_all[:, mask_non_slack]  # (m_all, n-1)
    W_all = A_all_red.multiply(b_all_arr[:, None]).tocsr()
    Bred = (A_all_red.T @ W_all).tocsc()

    A_lines_red = A_lines[:, mask_non_slack]
    W = A_lines_red.multiply(b_lines[:, None]).tocsr()

    try:
        Bred_lu = spla.splu(Bred)
    except Exception as e:
        raise RuntimeError(
            "Reduced B matrix factorization failed (possibly singular/disconnected network "
            "or too many invalid branch reactances/voltages). "
            f"buses={n_bus}, monitored_lines={m_line}, B-branches={len(b_all)}"
        ) from e

    logger.debug(
        "Built DCOperator: n_bus=%d, n_line=%d, slack_pos=%d", n_bus, m_line, slack_pos
    )

    return DCOperator(
        bus_ids=tuple(bus_ids),
        line_ids=tuple(line_ids),
        slack_pos=int(slack_pos),
        from_bus_pos=from_bus_pos,
        to_bus_pos=to_bus_pos,
        b=b_lines,
        mask_non_slack=mask_non_slack,
        red_pos_of_bus_pos=red_pos_of_bus_pos,
        Bred_lu=Bred_lu,
        W=W,
    )


def build_dc_matrices(
    net,
    slack_bus: int = 0,
    *,
    strict_units: bool = True,
    allow_phase_shift: bool = False,
    chunk_size: int = 256,
    dtype: np.dtype = np.float64,
) -> Tuple[np.ndarray, object]:
    """
    Build a dense DC PTDF-like sensitivity matrix `H_full` for pandapower networks.

    This function always uses the SciPy-backed `DCOperator` to materialize H_full.

    Parameters
    ----------
    strict_units:
        See `build_dc_operator`.
    allow_phase_shift:
        See `build_dc_operator`.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if not _HAVE_SCIPY:
        raise ImportError(
            "SciPy is required for build_dc_matrices in this project. Install scipy."
        )

    op = build_dc_operator(
        net,
        slack_bus=int(slack_bus),
        strict_units=bool(strict_units),
        allow_phase_shift=bool(allow_phase_shift),
    )
    H_full = op.materialize_H_full(dtype=dtype, chunk_size=int(chunk_size))
    return H_full, op
