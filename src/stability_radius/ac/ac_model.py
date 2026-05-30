from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from stability_radius.dc.dc_model import trafo_tap_ratio, trafo_x_total_ohm
from stability_radius.pp_helpers import bus_vn_kv as _bus_vn_kv
from stability_radius.pp_helpers import is_in_service as _is_in_service
from stability_radius.pp_helpers import resolve_slack_pos as _resolve_slack_pos

logger = logging.getLogger(__name__)

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    sp = None  # type: ignore[assignment]
    spla = None  # type: ignore[assignment]
    _HAVE_SCIPY = False

_EPS_Z_PU = 1e-18
_SHIFT_DEG_EPS = 1e-9


def _detect_pv_buses(net: Any, bus_ids: list[int], slack_pos: int) -> np.ndarray:
    """Return boolean mask (n_bus,) where True = PV bus (generator with V control).

    PV buses are those controlled by in-service generators (``net.gen``) or
    extra grids (``net.ext_grid``) that are NOT the slack bus.  The slack bus
    is excluded because it is already removed from the reduced system.
    """
    pv_mask = np.zeros(len(bus_ids), dtype=bool)
    bus_pos = {bid: pos for pos, bid in enumerate(bus_ids)}

    # gen buses
    if hasattr(net, "gen") and net.gen is not None and len(net.gen):
        for gid in net.gen.index:
            row = net.gen.loc[gid]
            if not _is_in_service(row):
                continue
            gb = int(row.get("bus", -1))
            if gb in bus_pos:
                pv_mask[bus_pos[gb]] = True

    # ext_grid buses
    if hasattr(net, "ext_grid") and net.ext_grid is not None and len(net.ext_grid):
        for eid in net.ext_grid.index:
            row = net.ext_grid.loc[eid]
            if not _is_in_service(row):
                continue
            eb = int(row.get("bus", -1))
            if eb in bus_pos:
                pv_mask[bus_pos[eb]] = True

    # Slack bus is neither PV nor PQ in the reduced system.
    pv_mask[slack_pos] = False
    return pv_mask


def _line_z_total_ohm(line_row: Any, *, lossless: bool) -> complex:
    """
    Total line series impedance in Ohm.

    lossless=True enforces r=0 to keep AC linearization aligned with the project's DC convention.
    """
    x_ohm_per_km = float(line_row.get("x_ohm_per_km", np.nan))
    r_ohm_per_km = float(line_row.get("r_ohm_per_km", 0.0))
    length_km = float(line_row.get("length_km", np.nan))
    parallel = float(line_row.get("parallel", 1.0))

    if not math.isfinite(x_ohm_per_km):
        raise ValueError(f"Line: x_ohm_per_km must be finite; got {x_ohm_per_km!r}")
    if not math.isfinite(length_km):
        raise ValueError(f"Line: length_km must be finite; got {length_km!r}")
    if not math.isfinite(parallel) or parallel <= 0.0:
        raise ValueError(f"Line: parallel must be finite and >0; got {parallel!r}")

    x = float(x_ohm_per_km) * float(length_km) / float(parallel)
    if not math.isfinite(x) or abs(float(x)) <= 0.0:
        raise ValueError(f"Line: invalid x_total_ohm={x!r} (must be non-zero)")

    if bool(lossless):
        r = 0.0
    else:
        if not math.isfinite(r_ohm_per_km):
            raise ValueError(f"Line: r_ohm_per_km must be finite; got {r_ohm_per_km!r}")
        r = float(r_ohm_per_km) * float(length_km) / float(parallel)
        if not math.isfinite(r):
            raise ValueError(f"Line: invalid r_total_ohm={r!r}")

    z = complex(float(r), float(x))
    if abs(z) <= 0.0:
        raise ValueError("Line: invalid (r,x) -> zero impedance.")
    return z


def _z_base_ohm(*, vn_kv: float, sn_mva: float) -> float:
    """Internal helper for module-local processing."""
    v = float(vn_kv)
    s = float(sn_mva)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"vn_kv must be finite and >0, got {vn_kv!r}")
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError(f"sn_mva must be finite and >0, got {sn_mva!r}")
    return (v * v) / s


def _add_series_branch_ybus(
    *,
    row: list[int],
    col: list[int],
    data: list[complex],
    i: int,
    k: int,
    y: complex,
) -> None:
    """
    Add a simple series branch (no shunt, no tap) to Ybus contributions.

    Adds:
      Y_ii += y
      Y_kk += y
      Y_ik -= y
      Y_ki -= y
    """
    if i == k:
        return
    row.extend([i, k, i, k])
    col.extend([i, k, k, i])
    data.extend([y, y, -y, -y])


def _add_series_branch_ybus_with_tap(
    *,
    row: list[int],
    col: list[int],
    data: list[complex],
    i: int,
    k: int,
    y: complex,
    a: complex,
) -> None:
    """
    Add a series branch with complex tap ratio a on the i-side to Ybus.

    Standard Ybus stamping (series-only, no shunt):
      Y_ii += y / |a|^2
      Y_kk += y
      Y_ik -= y / conj(a)
      Y_ki -= y / a
    """
    if i == k:
        return

    aa = complex(a)
    if abs(aa) <= 0.0:
        raise ValueError("Invalid complex tap ratio a (|a|==0).")

    inv_a = 1.0 / aa
    inv_a_conj = 1.0 / np.conj(aa)
    y_ii = y * (inv_a * inv_a_conj)  # y / |a|^2
    y_ik = -y * inv_a_conj
    y_ki = -y * inv_a
    y_kk = y

    row.extend([i, k, i, k])
    col.extend([i, k, k, i])
    data.extend([y_ii, y_kk, y_ik, y_ki])


def _build_ybus_pu(
    *,
    net: Any,
    bus_ids: list[int],
    slack_pos: int,
    lossless: bool,
) -> Any:
    """
    Build sparse Ybus in per-unit using a minimal series-only model.

    Included elements (in service):
    - net.line as series branches (r,x from pandapower; r may be forced to 0 if lossless=True)
    - net.trafo as series branches with tap + phase shift (series-only t-model)
    - net.impedance as series branches (z_pu = rft_pu + j xft_pu on system base)
    """
    if not _HAVE_SCIPY:
        raise ImportError(
            "SciPy is required for ACOperator (sparse Jacobian). Install scipy."
        )

    n = int(len(bus_ids))
    bus_pos = {int(bid): pos for pos, bid in enumerate(bus_ids)}

    sn_mva = float(getattr(net, "sn_mva", np.nan))
    if not math.isfinite(sn_mva) or sn_mva <= 0.0:
        raise ValueError(f"pandapower net.sn_mva must be finite and >0; got {sn_mva!r}")

    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []

    # lines
    for lid in [int(x) for x in sorted(getattr(net, "line").index)]:
        row = net.line.loc[lid]
        if not _is_in_service(row):
            continue

        fb = int(row.get("from_bus", -1))
        tb = int(row.get("to_bus", -1))
        if fb not in bus_pos or tb not in bus_pos:
            raise ValueError(f"Line {lid} refers to missing buses {fb}->{tb}")

        z_ohm = _line_z_total_ohm(row, lossless=bool(lossless))
        vn_kv = _bus_vn_kv(net, fb)
        z_base = _z_base_ohm(vn_kv=float(vn_kv), sn_mva=sn_mva)
        z_pu = z_ohm / complex(z_base, 0.0)

        if abs(z_pu) <= _EPS_Z_PU:
            raise ValueError(f"Line {lid}: invalid z_pu ~ 0 (z_pu={z_pu}).")
        y = 1.0 / z_pu

        _add_series_branch_ybus(
            row=rows, col=cols, data=vals, i=int(bus_pos[fb]), k=int(bus_pos[tb]), y=y
        )

    # transformers (series-only t-model with complex tap)
    if hasattr(net, "trafo") and net.trafo is not None and len(net.trafo):
        for tid in [int(x) for x in sorted(net.trafo.index)]:
            row = net.trafo.loc[tid]
            if not _is_in_service(row):
                continue

            hv = int(row.get("hv_bus", -1))
            lv = int(row.get("lv_bus", -1))
            if hv not in bus_pos or lv not in bus_pos:
                raise ValueError(f"Trafo {tid} refers to missing buses {hv}->{lv}")

            x_ohm = float(trafo_x_total_ohm(net, row))

            vn_kv = float(row.get("vn_hv_kv", np.nan))
            if not math.isfinite(vn_kv) or vn_kv <= 0.0:
                vn_kv = _bus_vn_kv(net, hv)

            z_base = _z_base_ohm(vn_kv=float(vn_kv), sn_mva=sn_mva)

            # series-only, lossless by project policy in AC certificate
            z_pu = complex(0.0, float(x_ohm) / float(z_base))
            if abs(z_pu) <= _EPS_Z_PU:
                raise ValueError(f"Trafo {tid}: invalid z_pu ~ 0.")
            y = 1.0 / z_pu

            tap = float(trafo_tap_ratio(row))
            if not math.isfinite(tap) or tap <= 0.0:
                raise ValueError(f"Trafo {tid}: invalid tap_ratio={tap!r}")

            try:
                shift_deg = float(row.get("shift_degree", 0.0))
            except (TypeError, ValueError):
                shift_deg = 0.0
            if not math.isfinite(shift_deg):
                raise ValueError(
                    f"Trafo {tid}: shift_degree must be finite; got {shift_deg!r}"
                )
            if abs(float(shift_deg)) <= _SHIFT_DEG_EPS:
                shift_deg = 0.0

            phi = float(shift_deg) * math.pi / 180.0
            a = complex(tap * math.cos(phi), tap * math.sin(phi))

            _add_series_branch_ybus_with_tap(
                row=rows,
                col=cols,
                data=vals,
                i=int(bus_pos[hv]),
                k=int(bus_pos[lv]),
                y=y,
                a=a,
            )

    # impedances
    if hasattr(net, "impedance") and net.impedance is not None and len(net.impedance):
        for iid in [int(x) for x in sorted(net.impedance.index)]:
            row = net.impedance.loc[iid]
            if not _is_in_service(row):
                continue

            fb = int(row.get("from_bus", -1))
            tb = int(row.get("to_bus", -1))
            if fb not in bus_pos or tb not in bus_pos:
                raise ValueError(f"Impedance {iid} refers to missing buses {fb}->{tb}")

            r_pu = float(row.get("rft_pu", 0.0))
            x_pu = float(row.get("xft_pu", np.nan))
            if not math.isfinite(r_pu):
                raise ValueError(f"Impedance {iid}: invalid rft_pu={r_pu!r}")
            if not math.isfinite(x_pu) or abs(float(x_pu)) <= 0.0:
                raise ValueError(f"Impedance {iid}: invalid xft_pu={x_pu!r}")

            z_pu = complex(r_pu, x_pu)
            if abs(z_pu) <= _EPS_Z_PU:
                raise ValueError(f"Impedance {iid}: invalid z_pu ~ 0.")
            y = 1.0 / z_pu

            _add_series_branch_ybus(
                row=rows,
                col=cols,
                data=vals,
                i=int(bus_pos[fb]),
                k=int(bus_pos[tb]),
                y=y,
            )

    Y = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.complex128).tocsr()
    logger.debug(
        "Built Ybus (pu): n_bus=%d nnz=%d slack_pos=%d lossless=%s",
        int(n),
        int(Y.nnz),
        int(slack_pos),
        bool(lossless),
    )
    return Y


def _build_reduced_pf_jacobian_mw_per_unit(
    *,
    Ybus: Any,
    vm_pu: np.ndarray,
    va_rad: np.ndarray,
    slack_pos: int,
    sn_mva: float,
    pv_mask: np.ndarray | None = None,
) -> Any:
    """
    Build reduced PF Jacobian J with proper PV/PQ bus handling.

    Structure (with n_theta = n_bus-1 non-slack buses, n_pq = PQ-only bus count)::

        [dP/dθ  dP/dV_pq]   rows: P for all non-slack (n_theta)
        [dQ/dθ  dQ/dV_pq]   rows: Q for PQ buses only (n_pq)

    Variables: θ for all non-slack buses, V for PQ buses only.
    Equations: P for all non-slack buses, Q for PQ buses only.

    PV buses (generators) have fixed voltage magnitude, so V is not a free
    variable.  Their Q equation is also excluded because reactive power is
    determined by the generator (not a scheduled injection).

    Sign convention (critical for certificate soundness)
    ----------------------------------------------------
    Here P/Q are *net bus injections* computed from the network model:
        S_i = V_i * conj(I_i),  I = Ybus * V

    - Positive P means net active power injected *into the network* at bus i
      (i.e., generation - load in the usual convention).
    - Under the standard PF formulation P_calc(x) = P_spec, the first-order relation
      for perturbations is:
          J * dx = dP_spec   (and similarly for Q)
      i.e., there is NO extra sign flip when mapping specified injection changes to state changes.

    Units
    -----
    - P/Q are computed in per-unit and then scaled by sn_mva -> MW/MVAr
    - θ in rad
    - V in pu
    """
    if not _HAVE_SCIPY:
        raise ImportError(
            "SciPy is required for ACOperator (sparse Jacobian). Install scipy."
        )

    n_bus = int(vm_pu.size)
    if va_rad.shape != (n_bus,):
        raise ValueError("va_rad shape mismatch.")
    if n_bus <= 1:
        raise ValueError("ACOperator requires n_bus >= 2.")

    if not math.isfinite(float(sn_mva)) or float(sn_mva) <= 0.0:
        raise ValueError("sn_mva must be finite and >0.")

    # Basic sanity: expect angles in radians.
    if float(np.max(np.abs(va_rad))) > 10.0:
        raise ValueError(
            "Bus voltage angles look too large for radians (max|va|>10). "
            "Refuse to build Jacobian to avoid silent unit mismatch."
        )

    # ----- reduced indexing -----
    mask_non_slack = np.ones(n_bus, dtype=bool)
    mask_non_slack[int(slack_pos)] = False

    if pv_mask is None:
        pv_mask = np.zeros(n_bus, dtype=bool)

    pq_mask = mask_non_slack & ~pv_mask  # PQ buses: non-slack AND non-PV

    # theta_red_pos: maps bus pos -> theta variable index (all non-slack)
    theta_red_pos = np.full(n_bus, -1, dtype=int)
    theta_red_pos[np.where(mask_non_slack)[0]] = np.arange(
        int(np.sum(mask_non_slack)), dtype=int
    )
    n_theta = int(np.sum(mask_non_slack))  # = n_bus - 1

    # v_red_pos: maps bus pos -> V variable index (PQ buses only)
    v_red_pos = np.full(n_bus, -1, dtype=int)
    v_red_pos[np.where(pq_mask)[0]] = np.arange(int(np.sum(pq_mask)), dtype=int)
    n_pq = int(np.sum(pq_mask))

    n_vars = n_theta + n_pq

    # complex voltages
    V = vm_pu * np.exp(1j * va_rad)
    I = Ybus @ V
    S = V * np.conj(I)  # pu injection
    P = np.asarray(S.real, dtype=float)
    Q = np.asarray(S.imag, dtype=float)

    # iterate Ybus off-diagonal entries
    Ycoo = Ybus.tocoo()

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i, k, yik in zip(Ycoo.row, Ycoo.col, Ycoo.data):
        if i == k:
            continue
        ri_theta = int(theta_red_pos[int(i)])
        ck_theta = int(theta_red_pos[int(k)])
        if ri_theta < 0 or ck_theta < 0:
            continue  # one of them is slack

        Vi = float(vm_pu[int(i)])
        Vk = float(vm_pu[int(k)])
        if Vi <= 0.0 or Vk <= 0.0:
            raise ValueError(
                "Non-positive voltage magnitude encountered in base point."
            )

        theta = float(va_rad[int(i)] - va_rad[int(k)])
        s = math.sin(theta)
        c = math.cos(theta)

        G = float(np.real(yik))
        B = float(np.imag(yik))

        # per-unit partials
        dP_dtheta = Vi * Vk * (G * s - B * c) * float(sn_mva)
        dP_dV = Vi * (G * c + B * s) * float(sn_mva)

        dQ_dtheta = -Vi * Vk * (G * c + B * s) * float(sn_mva)
        dQ_dV = Vi * (G * s - B * c) * float(sn_mva)

        # P row for bus i, theta column for bus k (always present for non-slack)
        rows.append(ri_theta)
        cols.append(ck_theta)
        data.append(dP_dtheta)

        # P row for bus i, V column for bus k (only if k is PQ)
        ck_v = int(v_red_pos[int(k)])
        if ck_v >= 0:
            rows.append(ri_theta)
            cols.append(n_theta + ck_v)
            data.append(dP_dV)

        # Q row for bus i (only if i is PQ), theta column for bus k
        ri_v = int(v_red_pos[int(i)])
        if ri_v >= 0:
            rows.append(n_theta + ri_v)
            cols.append(ck_theta)
            data.append(dQ_dtheta)

            # Q row for bus i, V column for bus k (only if both i is PQ and k is PQ)
            if ck_v >= 0:
                rows.append(n_theta + ri_v)
                cols.append(n_theta + ck_v)
                data.append(dQ_dV)

    # diagonal terms for non-slack buses
    diag = Ybus.diagonal()
    for i in range(n_bus):
        ri_theta = int(theta_red_pos[int(i)])
        if ri_theta < 0:
            continue  # slack

        Vi = float(vm_pu[int(i)])
        if Vi <= 0.0:
            raise ValueError(
                "Non-positive voltage magnitude encountered in base point."
            )

        Yii = diag[int(i)]
        Gii = float(np.real(Yii))
        Bii = float(np.imag(Yii))

        # per-unit diagonal formulas, scaled
        dP_dtheta_ii = (-float(Q[int(i)]) - Bii * Vi * Vi) * float(sn_mva)
        dP_dV_ii = (float(P[int(i)]) / Vi + Gii * Vi) * float(sn_mva)

        dQ_dtheta_ii = (float(P[int(i)]) - Gii * Vi * Vi) * float(sn_mva)
        dQ_dV_ii = (float(Q[int(i)]) / Vi - Bii * Vi) * float(sn_mva)

        # P-theta diagonal (always present)
        rows.append(ri_theta)
        cols.append(ri_theta)
        data.append(dP_dtheta_ii)

        ri_v = int(v_red_pos[int(i)])

        # P-V diagonal (only if bus i is PQ)
        if ri_v >= 0:
            rows.append(ri_theta)
            cols.append(n_theta + ri_v)
            data.append(dP_dV_ii)

        # Q-theta diagonal (only if bus i is PQ)
        if ri_v >= 0:
            rows.append(n_theta + ri_v)
            cols.append(ri_theta)
            data.append(dQ_dtheta_ii)

            # Q-V diagonal (only if bus i is PQ)
            rows.append(n_theta + ri_v)
            cols.append(n_theta + ri_v)
            data.append(dQ_dV_ii)

    J = sp.coo_matrix((data, (rows, cols)), shape=(n_vars, n_vars), dtype=float).tocsc()

    # Debug-only: helps catch accidental sn_mva scaling changes.
    if J.nnz > 0:
        try:
            max_abs = float(np.max(np.abs(J.data)))
        except Exception:
            max_abs = float("nan")
    else:
        max_abs = 0.0

    logger.debug(
        "Built reduced AC PF Jacobian J: shape=%s nnz=%d sn_mva=%.6g max|J_ij|=%.6g "
        "n_theta=%d n_pq=%d n_pv=%d",
        J.shape,
        int(J.nnz),
        float(sn_mva),
        float(max_abs),
        int(n_theta),
        int(n_pq),
        int(np.sum(pv_mask)),
    )
    return J, mask_non_slack, theta_red_pos, v_red_pos, pq_mask, n_pq


@dataclass(frozen=True)
class ACOperator:
    """
    Sparse linear operator for AC PF sensitivities around a base point.

    Mathematical model (reduced, PV/PQ-aware)
    ------------------------------------------
      J * dx = du

    where:
      - x = [theta_non_slack; V_pq]         (n_theta + n_pq variables)
      - du = [dP_non_slack; dQ_pq]           (n_theta + n_pq equations)

    PV buses (generators with voltage control) have fixed V, so only theta
    appears as a variable.  Their Q equation is excluded because reactive
    power is determined by the generator.

    For adjoint sensitivities per constraint:
      a = J^{-T} b
    implemented via LU solve with transposed system.
    """

    bus_ids: tuple[int, ...]
    line_ids: tuple[int, ...]
    slack_pos: int
    sn_mva: float

    vm_pu: np.ndarray  # (n_bus,)
    va_rad: np.ndarray  # (n_bus,)

    mask_non_slack: np.ndarray  # (n_bus,), bool
    red_pos_of_bus_pos: (
        np.ndarray
    )  # (n_bus,), -1 for slack else 0..n-2 (= theta_red_pos)
    pv_mask: np.ndarray  # (n_bus,), True for PV buses (generators, excl. slack)
    pq_mask: np.ndarray  # (n_bus,), True for PQ buses (non-slack, non-PV)
    v_red_pos: np.ndarray  # (n_bus,), -1 for slack/PV, 0..n_pq-1 for PQ
    n_pq: int  # number of PQ buses

    from_bus_pos: np.ndarray  # (m_line,)
    to_bus_pos: np.ndarray  # (m_line,)
    y_series_pu: np.ndarray  # (m_line,), series admittance (pu); 0 for out-of-service

    Ybus: Any  # scipy.sparse.csr_matrix (complex)
    J: Any  # scipy.sparse.csc_matrix (float)
    J_lu: Any  # scipy.sparse.linalg.SuperLU

    @property
    def n_bus(self) -> int:
        """Execute the documented operation."""
        return int(len(self.bus_ids))

    @property
    def n_line(self) -> int:
        """Execute the documented operation."""
        return int(len(self.line_ids))

    @property
    def n_red(self) -> int:
        """Number of theta variables (= n_bus - 1, all non-slack)."""
        return int(self.n_bus - 1)

    @property
    def n_vars(self) -> int:
        """Total Jacobian dimension: n_theta + n_pq."""
        return int(self.n_red + self.n_pq)

    @property
    def theta_red_pos(self) -> np.ndarray:
        """Alias for red_pos_of_bus_pos (theta variable index for each bus)."""
        return self.red_pos_of_bus_pos

    def _solve_factorized_jacobian(self, rhs: np.ndarray, *, trans: str) -> np.ndarray:
        """Validate a right-hand side and solve the factorized AC Jacobian."""
        r = np.asarray(rhs, dtype=float)
        n = self.n_vars
        if r.ndim == 1:
            if r.shape != (n,):
                raise ValueError(f"rhs must have shape ({n},), got {r.shape}")
            return np.asarray(self.J_lu.solve(r, trans=trans), dtype=float)
        if r.ndim == 2:
            if r.shape[0] != n:
                raise ValueError(f"rhs must have shape ({n}, k), got {r.shape}")
            return np.asarray(self.J_lu.solve(r, trans=trans), dtype=float)
        raise ValueError("rhs must be 1D or 2D")

    def solve_J(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve J * x = rhs.

        Parameters
        ----------
        rhs:
            (n_vars,) or (n_vars, k) where n_vars = n_theta + n_pq.

        Returns
        -------
        np.ndarray
            Solution with same shape as rhs.
        """
        return self._solve_factorized_jacobian(rhs, trans="N")

    def solve_J_transpose(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve J^T * y = rhs.

        This is the core operation for adjoint sensitivities:
            y = J^{-T} rhs
        """
        return self._solve_factorized_jacobian(rhs, trans="T")


def build_ac_operator(
    *,
    net: Any,
    slack_bus: int,
    vm_pu: np.ndarray,
    va_rad: np.ndarray,
    line_indices: list[int] | None = None,
    lossless: bool = True,
) -> ACOperator:
    """
    Build ACOperator around a base AC PF point.

    Parameters
    ----------
    net:
        pandapower network (input data only).
    slack_bus:
        Slack bus id or position (in sorted bus order).
    vm_pu, va_rad:
        Base point voltages aligned with sorted(net.bus.index).
    line_indices:
        Optional explicit ordering of monitored net.line indices.
    lossless:
        If True, enforces r=0 in the internal Ybus (keeps closer to DC assumptions).

    Returns
    -------
    ACOperator
    """
    if not _HAVE_SCIPY:
        raise ImportError("SciPy is required for ACOperator. Install scipy.")

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = int(len(bus_ids))
    if n_bus <= 1:
        raise ValueError("ACOperator requires at least 2 buses.")

    slack_pos = _resolve_slack_pos(bus_ids, int(slack_bus))

    vm = np.asarray(vm_pu, dtype=float).reshape(-1)
    va = np.asarray(va_rad, dtype=float).reshape(-1)
    if vm.shape != (n_bus,) or va.shape != (n_bus,):
        raise ValueError(
            f"Base point voltage arrays must have shape ({n_bus},). "
            f"Got vm={vm.shape}, va={va.shape}."
        )

    sn_mva = float(getattr(net, "sn_mva", np.nan))
    if not math.isfinite(sn_mva) or sn_mva <= 0.0:
        raise ValueError(f"pandapower net.sn_mva must be finite and >0; got {sn_mva!r}")

    bus_pos = {bid: pos for pos, bid in enumerate(bus_ids)}

    # Monitored lines ordering
    line_ids = (
        [int(x) for x in sorted(net.line.index)]
        if line_indices is None
        else [int(x) for x in line_indices]
    )
    m_line = int(len(line_ids))
    if m_line == 0:
        raise ValueError("ACOperator requires at least 1 monitored line.")

    # Per-monitored-line endpoints + series admittance
    from_bus_pos = np.zeros(m_line, dtype=int)
    to_bus_pos = np.zeros(m_line, dtype=int)
    y_series_pu = np.zeros(m_line, dtype=np.complex128)

    for pos, lid in enumerate(line_ids):
        row = net.line.loc[int(lid)]
        fb = int(row.get("from_bus", -1))
        tb = int(row.get("to_bus", -1))
        if fb not in bus_pos or tb not in bus_pos:
            raise ValueError(f"Line {lid} refers to missing buses {fb}->{tb}")

        fpos = int(bus_pos[fb])
        tpos = int(bus_pos[tb])
        from_bus_pos[pos] = fpos
        to_bus_pos[pos] = tpos

        if not _is_in_service(row):
            y_series_pu[pos] = 0.0 + 0.0j
            continue

        z_ohm = _line_z_total_ohm(row, lossless=bool(lossless))
        vn_kv = _bus_vn_kv(net, fb)
        z_base = _z_base_ohm(vn_kv=float(vn_kv), sn_mva=sn_mva)
        z_pu = z_ohm / complex(z_base, 0.0)
        if abs(z_pu) <= _EPS_Z_PU:
            raise ValueError(f"Line {lid}: invalid z_pu ~ 0.")
        y_series_pu[pos] = 1.0 / z_pu

    Ybus = _build_ybus_pu(
        net=net, bus_ids=bus_ids, slack_pos=slack_pos, lossless=bool(lossless)
    )

    pv_mask = _detect_pv_buses(net, bus_ids, slack_pos)
    n_pv = int(np.sum(pv_mask))

    J, mask_non_slack, theta_red_pos, v_red_pos, pq_mask, n_pq = (
        _build_reduced_pf_jacobian_mw_per_unit(
            Ybus=Ybus,
            vm_pu=vm,
            va_rad=va,
            slack_pos=slack_pos,
            sn_mva=sn_mva,
            pv_mask=pv_mask,
        )
    )

    try:
        J_lu = spla.splu(J)
    except Exception as e:
        logger.exception("AC PF Jacobian factorization failed.")
        raise RuntimeError(
            "AC PF Jacobian factorization failed (singular / ill-conditioned). "
            "This typically indicates a non-solvable base point or disconnected network."
        ) from e

    logger.info(
        "Built ACOperator: n_bus=%d n_line=%d slack_pos=%d n_pv=%d n_pq=%d "
        "J_shape=%s J_nnz=%d lossless=%s",
        n_bus,
        m_line,
        int(slack_pos),
        int(n_pv),
        int(n_pq),
        J.shape,
        int(J.nnz),
        bool(lossless),
    )

    return ACOperator(
        bus_ids=tuple(bus_ids),
        line_ids=tuple(line_ids),
        slack_pos=int(slack_pos),
        sn_mva=float(sn_mva),
        vm_pu=vm,
        va_rad=va,
        mask_non_slack=mask_non_slack,
        red_pos_of_bus_pos=theta_red_pos,
        pv_mask=pv_mask,
        pq_mask=pq_mask,
        v_red_pos=v_red_pos,
        n_pq=int(n_pq),
        from_bus_pos=from_bus_pos,
        to_bus_pos=to_bus_pos,
        y_series_pu=y_series_pu,
        Ybus=Ybus,
        J=J,
        J_lu=J_lu,
    )
