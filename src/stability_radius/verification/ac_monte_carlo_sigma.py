from __future__ import annotations

"""
AC Monte Carlo verification with per-bus diagonal covariance N(0, Σ).

How this differs from the isotropic MC in ``monte_carlo.py``
------------------------------------------------------------
- ``monte_carlo.py`` uses scalar ``sigma_p_mw`` / ``sigma_q_mvar`` (uniform
  across buses).
- This module accepts **per-bus** sigma vectors ``sigma_p_mw`` and
  ``sigma_q_mvar`` (``np.ndarray`` of shape ``(n_bus,)``).

Sample generation
-----------------
Draw ``z ~ N(0, I_{2n})``, then scale element-wise::

    ΔP_i = σ_{P,i} · z_i
    ΔQ_i = σ_{Q,i} · z_{n+i}

Balance enforcement
-------------------
Project each sample onto balanced active and reactive subspaces with the
sigma-squared weighted conditional Gaussian projection. This keeps the
sampling model consistent with the AC sigma-radius denominator.

Soundness metric
----------------
``soundness_inside_sigma_ball`` is defined as the fraction of samples with
``‖Σ^{-1/2} Δu‖₂ ≤ r_σ`` that have **no** thermal violations on any line.
"""

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stability_radius.base_point.pandapower_tools import (
    apply_lossless_policy_to_pandapower_net,
)
from stability_radius.radii.common import estimate_line_limit_mva, line_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ACSigmaMCResult:
    """Result of AC Monte Carlo with per-bus diagonal covariance.

    Fields
    ------
    n_samples : int
        Total number of Monte Carlo samples drawn.
    n_violations : int
        Number of samples with at least one line thermal violation.
    n_pf_failures : int
        Number of samples where AC power flow did not converge.
    empirical_overload_probability : dict[str, float]
        Per-line empirical overload probability conditional on PF convergence,
        keyed by ``line_<id>``. PF failures are counted in
        ``bad_sample_probability`` but are not assigned to every line.
    empirical_overload_probability_conditional_on_pf_converged : dict[str, float]
        Explicit alias for the conditional per-line overload probability.
    pf_failure_probability : float
        Fraction of samples where AC PF did not converge.
    bad_sample_probability : float
        Fraction of samples with either a thermal overload or PF non-convergence.
    soundness_inside_sigma_ball : float
        Fraction of samples with ``‖Σ^{-1/2} Δu‖₂ ≤ r_σ`` that have
        no violations.  NaN if no samples fell inside the sigma ball.
    """

    n_samples: int
    n_violations: int
    n_pf_failures: int
    empirical_overload_probability: dict[str, float] = field(default_factory=dict)
    empirical_overload_probability_conditional_on_pf_converged: dict[str, float] = (
        field(default_factory=dict)
    )
    pf_failure_probability: float = 0.0
    bad_sample_probability: float = 0.0
    soundness_inside_sigma_ball: float = float("nan")


def _project_balance_sigma_weighted_inplace(
    dp: np.ndarray,
    dq: np.ndarray,
    sigma_p: np.ndarray,
    sigma_q: np.ndarray,
) -> None:
    """Project ΔP and ΔQ onto 1ᵀΔP = 0, 1ᵀΔQ = 0 using the σ²-weighted
    conditional projection.

    For ΔP ~ N(0, diag(σ²_P)), the correct conditional distribution given
    1ᵀΔP = 0 has covariance  Σ − Σ·1·(1ᵀΣ·1)⁻¹·1ᵀ·Σ, which is achieved by::

        ΔP_i  ←  ΔP_i − σ²_{P,i} · sum(ΔP) / sum(σ²_P)

    This preserves the correct anisotropic covariance structure and is
    consistent with the σ²-weighted h-projection used in ``compute_ac_sigma_radius``.
    """
    sigp2 = sigma_p * sigma_p  # (n_bus,)
    sigq2 = sigma_q * sigma_q

    sum_sigp2 = float(np.sum(sigp2))
    sum_sigq2 = float(np.sum(sigq2))

    if sum_sigp2 > 0.0:
        dp_sum = np.sum(dp, axis=1, keepdims=True)  # (n_samples, 1)
        dp -= sigp2[None, :] * dp_sum / sum_sigp2

    if sum_sigq2 > 0.0:
        dq_sum = np.sum(dq, axis=1, keepdims=True)
        dq -= sigq2[None, :] * dq_sum / sum_sigq2


def _sample_gaussian_sigma(
    *,
    rng: np.random.Generator,
    n: int,
    sigma_p: np.ndarray,
    sigma_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw *n* balanced Gaussian samples from N(0, Σ) | 1ᵀΔP=0, 1ᵀΔQ=0.

    z ~ N(0, I), then scale element-wise and apply the σ²-weighted
    conditional projection to enforce balance.
    """
    n_bus = int(sigma_p.shape[0])
    z = rng.standard_normal(size=(int(n), 2 * n_bus)).astype(float, copy=False)

    dp = sigma_p[None, :] * z[:, :n_bus]
    dq = sigma_q[None, :] * z[:, n_bus:]

    _project_balance_sigma_weighted_inplace(dp, dq, sigma_p, sigma_q)
    return dp, dq


def _sigma_inv_norm(
    dp: np.ndarray,
    dq: np.ndarray,
    inv_sigma_p: np.ndarray,
    inv_sigma_q: np.ndarray,
) -> np.ndarray:
    """Compute ‖Σ^{-1/2} Δu‖₂ for each sample row.

    Parameters
    ----------
    dp, dq : (n, n_bus) arrays
    inv_sigma_p, inv_sigma_q : (n_bus,) arrays  (1 / σ)

    Returns
    -------
    (n,) array of weighted norms.
    """
    scaled_p = dp * inv_sigma_p[None, :]
    scaled_q = dq * inv_sigma_q[None, :]
    return np.sqrt(
        np.sum(scaled_p * scaled_p, axis=1) + np.sum(scaled_q * scaled_q, axis=1)
    )


def _line_limits_mva_sorted(net: Any) -> tuple[list[int], np.ndarray]:
    """Sorted line indices and corresponding MVA thermal limits."""
    line_ids = [int(x) for x in sorted(net.line.index)]
    limits = np.empty(len(line_ids), dtype=float)
    for pos, lid in enumerate(line_ids):
        limits[pos] = float(estimate_line_limit_mva(net, net.line.loc[lid]))
    return line_ids, limits


def _check_sample_violations(
    net: Any,
    *,
    line_ids: list[int],
    limits_mva: np.ndarray,
    feas_tol_mva: float,
) -> tuple[bool, np.ndarray]:
    """Check per-line thermal violations after an AC PF solve.

    Returns
    -------
    (is_feasible, overloaded)
    where *overloaded* is a bool array of shape (n_lines,).
    """
    m = len(line_ids)
    overloaded = np.zeros(m, dtype=bool)

    if not hasattr(net, "res_line") or net.res_line is None or len(net.res_line) == 0:
        raise RuntimeError("pandapower did not produce res_line results.")

    worst = float("-inf")
    for pos, lid in enumerate(line_ids):
        row = net.line.loc[lid]
        if not bool(row.get("in_service", True)):
            continue

        p_from = float(net.res_line.loc[lid, "p_from_mw"])
        q_from = float(net.res_line.loc[lid, "q_from_mvar"])
        p_to = float(net.res_line.loc[lid, "p_to_mw"])
        q_to = float(net.res_line.loc[lid, "q_to_mvar"])

        s_from = math.sqrt(p_from * p_from + q_from * q_from)
        s_to = math.sqrt(p_to * p_to + q_to * q_to)
        s = max(s_from, s_to)

        viol = float(s - float(limits_mva[pos]))
        if viol > float(feas_tol_mva):
            overloaded[pos] = True
        if viol > worst:
            worst = viol

    feasible = bool(worst <= float(feas_tol_mva))
    return feasible, overloaded


def run_ac_monte_carlo_sigma(
    *,
    net: Any,
    sigma_p_mw: np.ndarray,
    sigma_q_mvar: np.ndarray,
    r_sigma: float,
    n_samples: int = 500,
    seed: int = 42,
    feas_tol_mva: float = 1e-3,
    lossless: bool = True,
) -> ACSigmaMCResult:
    """Run AC Monte Carlo with per-bus diagonal covariance N(0, Σ).

    Parameters
    ----------
    net : pandapower network
        The base-case network.  A deep copy is made internally.
    sigma_p_mw : (n_bus,) array
        Per-bus active-power injection standard deviation (MW).
    sigma_q_mvar : (n_bus,) array
        Per-bus reactive-power injection standard deviation (MVAr).
    r_sigma : float
        Sigma-radius threshold for the soundness check.  Samples with
        ``‖Σ^{-1/2} Δu‖₂ ≤ r_sigma`` are counted as "inside the sigma
        ball" for the soundness metric.
    n_samples : int
        Number of Monte Carlo samples.
    seed : int
        RNG seed for reproducibility.
    feas_tol_mva : float
        Feasibility tolerance for line thermal violations (MVA).
    lossless : bool
        Whether to apply the lossless policy (r=0) to the network.

    Returns
    -------
    ACSigmaMCResult
    """
    import pandapower as pp  # type: ignore

    sig_p = np.asarray(sigma_p_mw, dtype=float).reshape(-1)
    sig_q = np.asarray(sigma_q_mvar, dtype=float).reshape(-1)

    if np.any(~np.isfinite(sig_p)) or np.any(sig_p <= 0.0):
        raise ValueError("sigma_p_mw must be finite and >0 per bus.")
    if np.any(~np.isfinite(sig_q)) or np.any(sig_q <= 0.0):
        raise ValueError("sigma_q_mvar must be finite and >0 per bus.")
    if not math.isfinite(r_sigma) or r_sigma <= 0.0:
        raise ValueError("r_sigma must be finite and >0.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    # Deep-copy and apply lossless policy
    nn = copy.deepcopy(net)
    if lossless:
        nn = apply_lossless_policy_to_pandapower_net(nn)

    bus_ids = [int(x) for x in sorted(nn.bus.index)]
    n_bus = len(bus_ids)

    if sig_p.shape != (n_bus,):
        raise ValueError(f"sigma_p_mw must have shape ({n_bus},), got {sig_p.shape}")
    if sig_q.shape != (n_bus,):
        raise ValueError(f"sigma_q_mvar must have shape ({n_bus},), got {sig_q.shape}")

    # Attach per-bus perturbation sgen elements
    sgen_idx: list[int] = []
    for bid in bus_ids:
        idx = int(
            pp.create_sgen(
                nn,
                bus=int(bid),
                p_mw=0.0,
                q_mvar=0.0,
                name=f"mc_sigma_delta_bus_{int(bid)}",
                in_service=True,
            )
        )
        sgen_idx.append(idx)

    line_ids, limits_mva = _line_limits_mva_sorted(nn)
    m_line = len(line_ids)

    # Base PF (no perturbation)
    pp.runpp(nn, calculate_voltage_angles=True, enforce_q_lims=True, init="flat")
    if not bool(getattr(nn, "converged", True)):
        raise RuntimeError("AC MC sigma: base PF did not converge.")

    # Pre-compute inverse sigma (for weighted norm)
    inv_sig_p = 1.0 / sig_p
    inv_sig_q = 1.0 / sig_q

    # Generate all samples upfront
    rng = np.random.default_rng(int(seed))
    dp_all, dq_all = _sample_gaussian_sigma(
        rng=rng, n=n_samples, sigma_p=sig_p, sigma_q=sig_q
    )

    # Compute sigma-weighted norms for sigma-ball membership
    sigma_norms = _sigma_inv_norm(dp_all, dq_all, inv_sig_p, inv_sig_q)
    inside_ball = sigma_norms <= float(r_sigma)

    # Per-sample AC PF
    n_violations = 0
    n_pf_failures = 0
    per_line_overload_counts = np.zeros(m_line, dtype=np.int64)
    inside_ball_no_violation = 0
    n_inside_ball = int(np.sum(inside_ball))

    for j in range(int(n_samples)):
        nn.sgen.loc[sgen_idx, "p_mw"] = dp_all[j, :]
        nn.sgen.loc[sgen_idx, "q_mvar"] = dq_all[j, :]

        try:
            pp.runpp(
                nn,
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                init="results",
            )
            conv = bool(getattr(nn, "converged", True))
        except Exception:  # noqa: BLE001
            conv = False

        if not conv:
            n_pf_failures += 1
            n_violations += 1
            continue

        is_feas, overloaded = _check_sample_violations(
            nn,
            line_ids=line_ids,
            limits_mva=limits_mva,
            feas_tol_mva=float(feas_tol_mva),
        )
        per_line_overload_counts[overloaded] += 1

        if not is_feas:
            n_violations += 1
        elif bool(inside_ball[j]):
            inside_ball_no_violation += 1

    # Empirical overload probability per line
    empirical_overload_prob: dict[str, float] = {}
    empirical_overload_prob_cond: dict[str, float] = {}
    n_pf_converged = max(int(n_samples) - int(n_pf_failures), 0)
    for pos, lid in enumerate(line_ids):
        k = line_key(int(lid))
        empirical_overload_prob_cond[k] = float(per_line_overload_counts[pos]) / float(
            max(n_pf_converged, 1)
        )
        empirical_overload_prob[k] = empirical_overload_prob_cond[k]

    # Soundness inside sigma ball
    if n_inside_ball > 0:
        soundness = float(inside_ball_no_violation) / float(n_inside_ball)
    else:
        soundness = float("nan")

    logger.info(
        "AC MC sigma: n_samples=%d n_violations=%d n_pf_failures=%d "
        "n_inside_ball=%d soundness=%.6g r_sigma=%.6g",
        int(n_samples),
        int(n_violations),
        int(n_pf_failures),
        int(n_inside_ball),
        float(soundness),
        float(r_sigma),
    )

    return ACSigmaMCResult(
        n_samples=int(n_samples),
        n_violations=int(n_violations),
        n_pf_failures=int(n_pf_failures),
        empirical_overload_probability=empirical_overload_prob,
        empirical_overload_probability_conditional_on_pf_converged=empirical_overload_prob_cond,
        pf_failure_probability=float(n_pf_failures) / float(max(n_samples, 1)),
        bad_sample_probability=float(n_violations) / float(max(n_samples, 1)),
        soundness_inside_sigma_ball=float(soundness),
    )
