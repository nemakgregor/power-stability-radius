from __future__ import annotations

"""
Worst-case perturbation verification via nonlinear AC power flow.

Given a worst-case perturbation vector Δu*_ℓ (from the AC L2 certificate),
verify by running a full nonlinear AC PF (pandapower) that the predicted
overload actually occurs.

Algorithm
---------
1. Deep-copy net; apply lossless policy (matching the certificate).
2. For each bus i, add sgen with p_mw = ΔP_i*, q_mvar = ΔQ_i*.
3. Run pandapower.runpp.
4. Extract |S_from|, |S_to| for the target line.
5. Compare against c_ℓ (thermal limit in MVA).

Key insight for the paper
-------------------------
If the linearized prediction says violation occurs at |S| = c + ε and the
nonlinear PF confirms |S_actual| > c, the certificate is validated.  If the
nonlinear PF shows no violation, the linearization error exceeds the margin —
which quantifies the conservatism.
"""

import copy
import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from stability_radius.base_point.pandapower_tools import (
    apply_lossless_policy_to_pandapower_net,
)
from stability_radius.radii.common import estimate_line_limit_mva_with_flag

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorstCaseVerificationResult:
    """
    Result of verifying a single worst-case perturbation via nonlinear AC PF.

    Fields
    ------
    line_id : int
        pandapower line index for the target line.
    predicted_s_mva : float
        Apparent power predicted by the linear model (|S0| + margin consumed).
    actual_s_mva : float
        Apparent power from the nonlinear AC PF (max of from/to ends).
    limit_mva : float
        Thermal limit for this line (MVA).
    violated : bool
        True if actual_s_mva > limit_mva.
    pf_converged : bool
        True if pandapower.runpp converged.
    relative_error : float
        |predicted - actual| / actual.  NaN if actual ≈ 0 or PF did not converge.
    """

    line_id: int
    predicted_s_mva: float
    actual_s_mva: float
    limit_mva: float
    violated: bool
    pf_converged: bool
    relative_error: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""
        return asdict(self)


def _compute_predicted_s_mva(
    *,
    s0_mva: float,
    h_vec: np.ndarray,
    delta_u: np.ndarray,
) -> float:
    """
    Predicted apparent power from the linear model.

    S_predicted = |S0| + h^T · Δu

    where h is the sensitivity (gradient of |S| w.r.t. injection vector u)
    at the binding end, and Δu is the perturbation vector.
    """
    return float(s0_mva) + float(np.dot(h_vec, delta_u))


def _build_worst_case_perturbation(
    *,
    h_vec: np.ndarray,
    radius: float,
    scale: float = 1.0,
    balance: bool = True,
) -> np.ndarray:
    """
    Construct the worst-case perturbation Δu* from the h-vector and radius.

    The worst-case direction is the (balance-projected) h-vector normalized
    to unit length, scaled by ``radius * scale``.

    Parameters
    ----------
    h_vec : (d,) array
        Sensitivity vector (full-dimension, 2*n_bus for AC: [ΔP block | ΔQ block]).
    radius : float
        AC L2 radius for this line/end.
    scale : float
        Scaling factor applied to the radius (1.0 = boundary of the ball).
    balance : bool
        If True, project h onto the balanced subspace (1^T ΔP = 0, 1^T ΔQ = 0)
        before normalizing.

    Returns
    -------
    delta_u : (d,) array
        Worst-case perturbation vector.
    """
    h = np.asarray(h_vec, dtype=float).copy()
    d = h.shape[0]
    n_bus = d // 2

    if balance and n_bus > 0:
        # Project each block onto the balanced subspace: x -= mean(x)
        h[:n_bus] -= np.mean(h[:n_bus])
        h[n_bus:] -= np.mean(h[n_bus:])

    norm_h = float(np.linalg.norm(h, ord=2))
    if norm_h < 1e-15:
        return np.zeros(d, dtype=float)

    direction = h / norm_h
    return direction * float(radius) * float(scale)


def verify_worst_case(
    *,
    net: Any,
    line_id: int,
    h_vec: np.ndarray,
    radius: float,
    s0_mva: float,
    limit_mva: float | None = None,
    scale: float = 1.0,
    balance: bool = True,
    lossless: bool = True,
    delta_u: np.ndarray | None = None,
    binding_end: str | None = None,
) -> WorstCaseVerificationResult:
    """
    Verify a worst-case perturbation by running a full nonlinear AC PF.

    Parameters
    ----------
    net : pandapower network
        The original (unperturbed) pandapower network.  A deep copy is made
        internally; the caller's network is never modified.
    line_id : int
        pandapower line index of the target line to check.
    h_vec : (2*n_bus,) array
        Full-dimension sensitivity vector for the binding end of this line.
        Convention: [ΔP_0, ..., ΔP_{n-1}, ΔQ_0, ..., ΔQ_{n-1}].
    radius : float
        AC L2 radius for this line (MVA per unit injection).
    s0_mva : float
        Base-point apparent power |S0| at the binding end (MVA).
    limit_mva : float or None
        Thermal limit for this line (MVA).  If None, extracted from ``net``.
    scale : float
        Scaling factor for the perturbation magnitude.  1.0 means the
        perturbation lies exactly on the boundary of the certified ball.
    balance : bool
        Whether the certificate used balanced projections (sum-to-zero).
    lossless : bool
        Whether to apply the lossless policy (r=0) to the network.
    delta_u : (2*n_bus,) array or None
        If provided, use this perturbation vector directly instead of
        constructing it from h_vec and radius.
    binding_end : str or None
        ``"from"`` or ``"to"``.  If provided, the actual |S| is read
        from this specific line end (matching the analytical certificate)
        instead of taking ``max(S_from, S_to)``.

    Returns
    -------
    WorstCaseVerificationResult
    """
    import pandapower as pp

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    n_bus = len(bus_ids)
    h = np.asarray(h_vec, dtype=float).reshape(-1)

    if h.shape[0] != 2 * n_bus:
        raise ValueError(
            f"h_vec dimension mismatch: expected 2*n_bus={2 * n_bus}, got {h.shape[0]}."
        )

    # Resolve thermal limit
    if limit_mva is None:
        if int(line_id) not in net.line.index:
            raise ValueError(f"line_id={line_id} not found in net.line.index.")
        lim, _is_uc = estimate_line_limit_mva_with_flag(net, net.line.loc[int(line_id)])
        limit_mva = float(lim)
    else:
        limit_mva = float(limit_mva)

    # Build perturbation vector
    if delta_u is not None:
        du = np.asarray(delta_u, dtype=float).reshape(-1)
        if du.shape[0] != 2 * n_bus:
            raise ValueError(
                f"delta_u dimension mismatch: expected 2*n_bus={2 * n_bus}, got {du.shape[0]}."
            )
    else:
        du = _build_worst_case_perturbation(
            h_vec=h, radius=radius, scale=scale, balance=balance
        )

    # Predicted apparent power from the linear model
    predicted_s = _compute_predicted_s_mva(s0_mva=s0_mva, h_vec=h, delta_u=du)

    # Deep-copy and optionally apply lossless policy
    nn = copy.deepcopy(net)
    if lossless:
        nn = apply_lossless_policy_to_pandapower_net(nn)

    # Apply perturbation as sgen injections
    dp = du[:n_bus]  # ΔP block (MW)
    dq = du[n_bus:]  # ΔQ block (MVar)

    for pos, bid in enumerate(bus_ids):
        pp.create_sgen(
            nn,
            bus=int(bid),
            p_mw=float(dp[pos]),
            q_mvar=float(dq[pos]),
            name=f"wc_delta_bus_{int(bid)}",
            in_service=True,
        )

    # Run nonlinear AC power flow
    pf_converged = False
    try:
        pp.runpp(nn, calculate_voltage_angles=True, enforce_q_lims=True, init="flat")
        pf_converged = bool(getattr(nn, "converged", True))
    except Exception:  # noqa: BLE001
        pf_converged = False

    if not pf_converged:
        return WorstCaseVerificationResult(
            line_id=int(line_id),
            predicted_s_mva=float(predicted_s),
            actual_s_mva=float("nan"),
            limit_mva=float(limit_mva),
            violated=False,
            pf_converged=False,
            relative_error=float("nan"),
        )

    # Extract actual apparent power at the binding line end
    lid = int(line_id)
    p_from = float(nn.res_line.loc[lid, "p_from_mw"])
    q_from = float(nn.res_line.loc[lid, "q_from_mvar"])
    p_to = float(nn.res_line.loc[lid, "p_to_mw"])
    q_to = float(nn.res_line.loc[lid, "q_to_mvar"])

    s_from = math.sqrt(p_from * p_from + q_from * q_from)
    s_to = math.sqrt(p_to * p_to + q_to * q_to)

    if binding_end == "from":
        actual_s = s_from
    elif binding_end == "to":
        actual_s = s_to
    else:
        # Fallback: max of both ends (legacy behaviour).
        actual_s = max(s_from, s_to)

    # Violated if actual flow exceeds limit
    violated = bool(actual_s > float(limit_mva))

    # Relative error between predicted and actual
    _EPS_S = 1e-12
    if actual_s > _EPS_S:
        relative_error = abs(predicted_s - actual_s) / actual_s
    else:
        relative_error = float("nan")

    result = WorstCaseVerificationResult(
        line_id=int(line_id),
        predicted_s_mva=float(predicted_s),
        actual_s_mva=float(actual_s),
        limit_mva=float(limit_mva),
        violated=bool(violated),
        pf_converged=True,
        relative_error=float(relative_error),
    )

    logger.info(
        "Worst-case verification line=%d: predicted=%.4f MVA, actual=%.4f MVA, "
        "limit=%.4f MVA, violated=%s, rel_error=%.6g",
        int(line_id),
        float(predicted_s),
        float(actual_s),
        float(limit_mva),
        bool(violated),
        float(relative_error),
    )

    return result
