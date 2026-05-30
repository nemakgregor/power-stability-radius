from __future__ import annotations

"""
Monte Carlo verification for robustness certificates.

Modes
-----
- mode="dc": DC certificate verification (linear, fast).
- mode="ac": AC certificate verification (AC PF per sample).

Important correctness contract (AC)
-----------------------------------
AC MC must verify the SAME base regime that was used to compute the AC certificate.
Therefore we:
- require `__meta__.base_point_ac` with stored Vm/Va and solver/lossless info
- require solver/lossless consistency (explicit check; no heuristics)
- validate base PF |S| vs results.json AC base flows (tolerance in config)
"""

import copy
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from stability_radius.base_point.pandapower_tools import (
    apply_gen_dispatch_to_pandapower_net,
    apply_lossless_policy_to_pandapower_net,
    apply_opp_result_to_pandapower_net,
    ensure_ext_grid_at_slack,
    resolve_slack_bus_id,
)
from stability_radius.config import DEFAULT_MC
from stability_radius.dc.dc_model import build_dc_operator
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.common import line_key, sorted_line_limits_mva
from stability_radius.utils.json_utils import load_json_object, result_meta
from stability_radius.verification.sampling import (
    condition_diagonal_gaussian_balance_inplace,
)

from .types import (
    BASE_INFEASIBLE,
    BASE_OK,
    PROB_DEGENERATE_DIMENSION,
    PROB_OK,
    PROB_UNKNOWN,
    RADIUS_INVALID,
    RADIUS_OK,
    RADIUS_UNKNOWN,
    RADIUS_ZERO_BAD_LIMITS,
    RADIUS_ZERO_BINDING,
    SOUND_FAIL,
    SOUND_PASS,
    SOUND_SKIPPED_BASE_INFEASIBLE,
    SOUND_SKIPPED_INVALID_RADIUS,
    SOUND_SKIPPED_TRIVIAL_RADIUS,
    BasePointCheck,
    ProbabilisticCheck,
    RadiusCheck,
    SoundnessCheck,
    VerificationInputs,
    VerificationResult,
    overall_from_components,
)
from .verify_certificate import interpret_certificate_components

logger = logging.getLogger("stability_radius.verification.monte_carlo")

_Z_95 = 1.959963984540054  # 95% CI


def _wilson_ci95_percent(*, k: int, n: int) -> tuple[float, float]:
    """Internal helper for module-local processing."""
    if n <= 0:
        return float("nan"), float("nan")
    kk = int(k)
    nn = int(n)
    z = float(_Z_95)

    p = float(kk) / float(nn)
    denom = 1.0 + (z * z) / float(nn)
    center = (p + (z * z) / (2.0 * float(nn))) / denom
    half = (
        z
        * math.sqrt(
            max(p * (1.0 - p), 0.0) / float(nn)
            + (z * z) / (4.0 * float(nn) * float(nn))
        )
        / denom
    )
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return 100.0 * lo, 100.0 * hi


def _chi2_cdf(*, x: float, df: int) -> float:
    """Internal helper for module-local processing."""
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    xx = float(x)
    if not math.isfinite(xx) or xx <= 0.0:
        return 0.0
    from scipy.special import gammainc  # type: ignore

    return float(gammainc(float(df) / 2.0, xx / 2.0))


def _project_sum_zero_inplace(x: np.ndarray) -> np.ndarray:
    """Internal helper for module-local processing."""
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (k,n_bus), got {x.shape}")
    x -= np.mean(x, axis=1, keepdims=True)
    return x


def _sample_gaussian_balanced(
    *, rng: np.random.Generator, n: int, n_bus: int, sigma: float
) -> np.ndarray:
    """Internal helper for module-local processing."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if n_bus <= 1:
        raise ValueError("n_bus must be >= 2.")
    s = float(sigma)
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("sigma must be finite and positive.")
    z = (s * rng.standard_normal(size=(int(n), int(n_bus)))).astype(float, copy=False)
    return _project_sum_zero_inplace(z)


def _sample_uniform_l2_ball_balanced(
    *, rng: np.random.Generator, n: int, n_bus: int, radius: float
) -> np.ndarray:
    """Internal helper for module-local processing."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if n_bus <= 1:
        raise ValueError("n_bus must be >= 2.")
    r = float(radius)
    if not math.isfinite(r) or r < 0.0:
        raise ValueError("radius must be finite and non-negative.")
    if r == 0.0:
        return np.zeros((int(n), int(n_bus)), dtype=float)

    d = int(n_bus - 1)

    z = rng.standard_normal(size=(int(n), int(n_bus))).astype(float, copy=False)
    _project_sum_zero_inplace(z)
    norms = np.linalg.norm(z, ord=2, axis=1)

    bad = norms <= 1e-12
    while bool(np.any(bad)):
        k_bad = int(np.sum(bad))
        z_bad = rng.standard_normal(size=(k_bad, int(n_bus))).astype(float, copy=False)
        _project_sum_zero_inplace(z_bad)
        z[bad, :] = z_bad
        norms = np.linalg.norm(z, ord=2, axis=1)
        bad = norms <= 1e-12

    dirs = z / norms[:, None]
    u = rng.random(size=int(n)).astype(float, copy=False)
    rad = r * np.power(u, 1.0 / float(d))
    return dirs * rad[:, None]


def _extract_line_arrays_dc(
    *, results: dict[str, Any], net: Any
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Internal helper for module-local processing."""
    line_indices = [int(x) for x in sorted(net.line.index)]
    m = int(len(line_indices))

    f0 = np.empty(m, dtype=float)
    c = np.empty(m, dtype=float)
    r = np.empty(m, dtype=float)
    margin_raw = np.empty(m, dtype=float)
    norm_g = np.full(m, float("nan"), dtype=float)

    for pos, lid in enumerate(line_indices):
        k = line_key(int(lid))
        row = results.get(k)
        if not isinstance(row, dict):
            raise KeyError(f"results.json missing per-line entry: {k}")

        f0[pos] = float(row["flow0_mw"])
        c[pos] = float(row["p_limit_mw_est"])
        r[pos] = float(row["radius_l2"])
        margin_raw[pos] = float(c[pos] - abs(f0[pos]))

        if "norm_g" in row:
            try:
                norm_g[pos] = float(row.get("norm_g", float("nan")))
            except (TypeError, ValueError):
                norm_g[pos] = float("nan")

    return line_indices, f0, c, r, margin_raw, norm_g


def _compute_r_star_and_argmin(
    *,
    line_indices: list[int],
    radii: np.ndarray,
    margins_raw: np.ndarray,
    norm_g: np.ndarray,
    base_status: str,
) -> RadiusCheck:
    """Internal helper for module-local processing."""
    finite = np.isfinite(radii)
    if not bool(np.any(finite)):
        return RadiusCheck(
            status=RADIUS_INVALID,
            r_star=float("nan"),
            argmin_line_pos=-1,
            argmin_line_idx=-1,
            min_margin_mw=float(np.nanmin(margins_raw))
            if margins_raw.size
            else float("nan"),
            argmin_margin_mw=float("nan"),
            argmin_norm_g=float("nan"),
        )

    argmin_pos = int(np.argmin(np.where(finite, radii, float("inf"))))
    r_star = float(radii[argmin_pos])
    argmin_line_idx = int(line_indices[argmin_pos])
    argmin_margin = float(margins_raw[argmin_pos])
    argmin_norm = float(norm_g[argmin_pos])
    min_margin = float(np.min(margins_raw)) if margins_raw.size else float("nan")

    if not math.isfinite(r_star) or r_star < 0.0:
        status = RADIUS_INVALID
    elif r_star == 0.0:
        if (
            base_status == BASE_OK
            and math.isfinite(argmin_margin)
            and abs(argmin_margin) <= 1e-9
        ):
            status = RADIUS_ZERO_BINDING
        else:
            status = RADIUS_ZERO_BAD_LIMITS
    else:
        status = RADIUS_OK

    if not math.isfinite(min_margin):
        status = RADIUS_UNKNOWN

    return RadiusCheck(
        status=status,
        r_star=r_star,
        argmin_line_pos=argmin_pos,
        argmin_line_idx=argmin_line_idx,
        min_margin_mw=min_margin,
        argmin_margin_mw=argmin_margin,
        argmin_norm_g=argmin_norm,
    )


def _project_sum_zero_two_blocks_inplace(dp: np.ndarray, dq: np.ndarray) -> None:
    """Internal helper for module-local processing."""
    if dp.ndim != 2 or dq.ndim != 2:
        raise ValueError("dp and dq must be 2D")
    if dp.shape != dq.shape:
        raise ValueError("dp and dq shape mismatch")
    dp -= np.mean(dp, axis=1, keepdims=True)
    dq -= np.mean(dq, axis=1, keepdims=True)


def _sample_gaussian_ac(
    *,
    rng: np.random.Generator,
    n: int,
    n_bus: int,
    sigma_p_mw: "float | np.ndarray",
    sigma_q_mvar: "float | np.ndarray",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian injection perturbation samples.

    sigma_p_mw / sigma_q_mvar can be either:
    - a scalar float  → isotropic (same sigma for all buses)
    - a (n_bus,) array → heterogeneous (per-bus sigma)
    """
    sp = np.asarray(sigma_p_mw, dtype=float)
    sq = np.asarray(sigma_q_mvar, dtype=float)
    # broadcast scalar to per-bus array
    if sp.ndim == 0:
        sp = np.full(int(n_bus), float(sp))
    if sq.ndim == 0:
        sq = np.full(int(n_bus), float(sq))
    if np.any(~np.isfinite(sp)) or np.any(sp <= 0):
        raise ValueError("sigma_p_mw must be finite and >0 (all buses)")
    if np.any(~np.isfinite(sq)) or np.any(sq <= 0):
        raise ValueError("sigma_q_mvar must be finite and >0 (all buses)")

    z_p = rng.standard_normal(size=(int(n), int(n_bus)))
    z_q = rng.standard_normal(size=(int(n), int(n_bus)))
    dp = (sp[None, :] * z_p).astype(float, copy=False)
    dq = (sq[None, :] * z_q).astype(float, copy=False)
    condition_diagonal_gaussian_balance_inplace(dp, dq, sp, sq)
    return dp, dq


def _sample_uniform_l2_ball_ac(
    *,
    rng: np.random.Generator,
    n: int,
    n_bus: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Internal helper for module-local processing."""
    r = float(radius)
    if not math.isfinite(r) or r < 0:
        raise ValueError("radius must be finite and >=0")
    if r == 0.0:
        z = np.zeros((int(n), int(n_bus)), dtype=float)
        return z.copy(), z.copy()

    d = int(2 * (n_bus - 1))

    zP = rng.standard_normal(size=(int(n), int(n_bus))).astype(float, copy=False)
    zQ = rng.standard_normal(size=(int(n), int(n_bus))).astype(float, copy=False)
    _project_sum_zero_two_blocks_inplace(zP, zQ)

    norms = np.sqrt(np.sum(zP * zP, axis=1) + np.sum(zQ * zQ, axis=1))

    bad = norms <= 1e-12
    while bool(np.any(bad)):
        k_bad = int(np.sum(bad))
        zzP = rng.standard_normal(size=(k_bad, int(n_bus))).astype(float, copy=False)
        zzQ = rng.standard_normal(size=(k_bad, int(n_bus))).astype(float, copy=False)
        _project_sum_zero_two_blocks_inplace(zzP, zzQ)
        zP[bad, :] = zzP
        zQ[bad, :] = zzQ
        norms = np.sqrt(np.sum(zP * zP, axis=1) + np.sum(zQ * zQ, axis=1))
        bad = norms <= 1e-12

    zP /= norms[:, None]
    zQ /= norms[:, None]

    u = rng.random(size=int(n)).astype(float, copy=False)
    rad = r * np.power(u, 1.0 / float(d))

    return zP * rad[:, None], zQ * rad[:, None]


def _ac_pf_sample_violation_mva(
    net: Any,
    *,
    line_ids: list[int],
    limits_mva: np.ndarray,
    feas_tol_mva: float,
) -> tuple[bool, float, int]:
    """Internal helper for module-local processing."""
    if not hasattr(net, "res_line") or net.res_line is None or len(net.res_line) == 0:
        raise RuntimeError("pandapower did not produce res_line results.")

    worst = float("-inf")
    worst_pos = -1

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
        if viol > worst:
            worst = viol
            worst_pos = int(pos)

    feasible = bool(worst <= float(feas_tol_mva))
    return feasible, float(worst), int(worst_pos)


def _ac_pf_sample_per_line_violations_mva(
    net: Any,
    *,
    line_ids: list[int],
    limits_mva: np.ndarray,
    feas_tol_mva: float,
) -> tuple[bool, float, int, np.ndarray]:
    """Like _ac_pf_sample_violation_mva but also returns per-line overload flags.

    Returns
    -------
    (is_feasible, worst_violation, worst_line_pos, overloaded)
    where *overloaded* is a bool array of shape (n_lines,), True where
    max(|S_from|, |S_to|) > limit + feas_tol.
    """
    if not hasattr(net, "res_line") or net.res_line is None or len(net.res_line) == 0:
        raise RuntimeError("pandapower did not produce res_line results.")

    m = len(line_ids)
    overloaded = np.zeros(m, dtype=bool)
    worst = float("-inf")
    worst_pos = -1

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
            worst_pos = int(pos)

    feasible = bool(worst <= float(feas_tol_mva))
    return feasible, float(worst), int(worst_pos), overloaded


def _check_ac_base_point_matches_results(
    *,
    nn: Any,
    results: dict[str, Any],
    line_ids_sorted: list[int],
    tol_mva: float,
) -> dict[str, Any]:
    """Internal helper for module-local processing."""
    if not hasattr(nn, "res_line") or nn.res_line is None or len(nn.res_line) == 0:
        raise RuntimeError("pandapower did not produce res_line results (base point).")

    max_abs_diff = float("-inf")
    argmax_line_idx = -1

    for lid in line_ids_sorted:
        row = results.get(line_key(int(lid)))
        if not isinstance(row, dict):
            raise KeyError(f"results.json missing per-line entry: line_{lid}")
        if "ac_s0_from_mva" not in row or "ac_s0_to_mva" not in row:
            raise KeyError(
                "results.json missing AC base fields ac_s0_from_mva/ac_s0_to_mva (compute compute.ac.compute=1)."
            )

        p_from = float(nn.res_line.loc[lid, "p_from_mw"])
        q_from = float(nn.res_line.loc[lid, "q_from_mvar"])
        p_to = float(nn.res_line.loc[lid, "p_to_mw"])
        q_to = float(nn.res_line.loc[lid, "q_to_mvar"])

        s_from = math.sqrt(p_from * p_from + q_from * q_from)
        s_to = math.sqrt(p_to * p_to + q_to * q_to)

        s_from_ref = float(row["ac_s0_from_mva"])
        s_to_ref = float(row["ac_s0_to_mva"])

        d = max(abs(s_from - s_from_ref), abs(s_to - s_to_ref))
        if d > max_abs_diff:
            max_abs_diff = float(d)
            argmax_line_idx = int(lid)

    if math.isfinite(max_abs_diff) and max_abs_diff > float(tol_mva):
        raise ValueError(
            "AC MC base point mismatch vs results.json (different regime / dispatch / lossless policy). "
            f"max_abs_diff_S_mva={max_abs_diff:.6g} > tol={float(tol_mva):.6g}, argmax_line_idx={argmax_line_idx}. "
            "Fix by ensuring you run MC with the same ac.pf_solver / ac.lossless and same base dispatch."
        )

    return {
        "basepoint_ac_max_abs_diff_s_mva": float(max_abs_diff),
        "basepoint_ac_tol_mva": float(tol_mva),
        "basepoint_ac_argmax_line_idx": int(argmax_line_idx),
    }


def run_monte_carlo_verification(
    *,
    results_path: Path,
    input_case_path: Path,
    slack_bus: int,
    n_samples: int = DEFAULT_MC.n_samples,
    seed: int = DEFAULT_MC.seed,
    chunk_size: int = DEFAULT_MC.chunk_size,
    feas_tol: float = DEFAULT_MC.feas_tol_mw,
    cert_tol: float = DEFAULT_MC.cert_tol_mw,
    cert_max_samples: int = DEFAULT_MC.cert_max_samples,
    sigma_override_mw: float | None = None,
    mode: str = "dc",
    allow_download: bool = False,
    # AC-only parameters (explicit)
    # Can be a scalar float (uniform) or a (n_bus,) ndarray (per-bus heterogeneous)
    ac_sigma_p_mw: "float | np.ndarray | None" = None,
    ac_sigma_q_mvar: "float | np.ndarray | None" = None,
    ac_pf_solver: str = "pandapower",
    ac_lossless: bool = True,
    ac_basepoint_s_tol_mva: float = 1e-3,
    track_per_line_overloads: bool = False,
) -> VerificationResult:
    """Execute the documented operation."""
    mode_eff = str(mode).strip().lower()
    if mode_eff not in {"dc", "ac"}:
        raise ValueError("mode must be 'dc' or 'ac'")

    rp = Path(results_path).resolve()
    ip = Path(input_case_path).expanduser()
    ip = ip.resolve() if not ip.is_absolute() else ip

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    tol_feas = float(feas_tol)
    tol_cert = float(cert_tol)
    if not math.isfinite(tol_feas) or tol_feas < 0.0:
        raise ValueError("feas_tol must be finite and non-negative.")
    if not math.isfinite(tol_cert) or tol_cert < 0.0:
        raise ValueError("cert_tol must be finite and non-negative.")

    cert_max = int(cert_max_samples)
    if cert_max < 0:
        raise ValueError("cert_max_samples must be non-negative.")

    case_id = rp.stem
    results = load_json_object(rp)
    meta = result_meta(results)

    if not ip.exists():
        if not bool(allow_download):
            raise FileNotFoundError(
                f"Input case file not found: {ip}. Enable allow_download to download deterministically."
            )
        from stability_radius.utils.download import ensure_case_file

        ip = Path(ensure_case_file(str(ip))).resolve()

    net = load_network(ip)

    if mode_eff == "dc":
        dc_op = build_dc_operator(net, slack_bus=int(slack_bus))

        line_indices, f0, c, r, margin_raw, norm_g = _extract_line_arrays_dc(
            results=results, net=net
        )

        if sigma_override_mw is not None:
            sigma_mw = float(sigma_override_mw)
            sigma_source = "override"
        else:
            meta_dc = meta.get("dc", {}) if isinstance(meta.get("dc", {}), dict) else {}
            sigma_meta = meta_dc.get("inj_std_mw", None)
            if sigma_meta is None:
                raise ValueError(
                    "results.json missing __meta__.dc.inj_std_mw; pass --sigma-override-mw."
                )
            sigma_mw = float(sigma_meta)
            sigma_source = "results_meta"

        if not math.isfinite(sigma_mw) or sigma_mw <= 0.0:
            raise ValueError("sigma_mw must be finite and >0.")

        base_viol = np.where(np.abs(f0) > (c + tol_feas))[0]
        base_feasible = bool(base_viol.size == 0)
        base_max_violation = (
            float(np.max(np.abs(f0[base_viol]) - c[base_viol]))
            if base_viol.size
            else 0.0
        )

        base_status = BASE_OK if base_feasible else BASE_INFEASIBLE
        base_check = BasePointCheck(
            status=base_status,
            violated_lines=int(base_viol.size),
            max_violation_mw=float(base_max_violation),
        )

        radius_check = _compute_r_star_and_argmin(
            line_indices=line_indices,
            radii=r,
            margins_raw=margin_raw,
            norm_g=norm_g,
            base_status=base_status,
        )

        d = int(max(dc_op.n_bus - 1, 0))
        if d <= 0:
            raise ValueError("Invalid DCOperator dimension: n_bus must be >= 2.")

        p_ball_analytic = (
            100.0
            * _chi2_cdf(x=(float(radius_check.r_star) / float(sigma_mw)) ** 2, df=d)
            if math.isfinite(radius_check.r_star) and radius_check.r_star >= 0
            else float("nan")
        )

        rng = np.random.default_rng(int(seed))
        remaining = int(n_samples)

        feasible = 0
        in_ball = 0
        in_ball_and_feasible = 0

        worst_max_violation = float("-inf")
        worst_line_pos = -1
        worst_line_idx = -1
        worst_sample_l2 = float("nan")

        while remaining > 0:
            k = min(int(chunk_size), remaining)
            remaining -= k

            delta_full = _sample_gaussian_balanced(
                rng=rng, n=k, n_bus=int(dc_op.n_bus), sigma=float(sigma_mw)
            )
            norms = np.linalg.norm(delta_full, ord=2, axis=1)

            in_ball_mask = (
                norms <= float(radius_check.r_star)
                if math.isfinite(radius_check.r_star)
                else False
            )
            in_ball += int(np.sum(in_ball_mask))

            df = dc_op.flows_from_delta_injections(delta_full)
            viol = df + f0[None, :]
            np.abs(viol, out=viol)
            viol -= c[None, :]
            np.nan_to_num(
                viol,
                copy=False,
                nan=float("inf"),
                posinf=float("inf"),
                neginf=-float("inf"),
            )

            max_v = np.max(viol, axis=1)
            feasible_mask = max_v <= float(tol_feas)

            feasible += int(np.sum(feasible_mask))
            in_ball_and_feasible += int(np.sum(in_ball_mask & feasible_mask))

            batch_worst = float(np.max(max_v))
            if batch_worst > worst_max_violation:
                worst_max_violation = batch_worst
                j = int(np.argmax(max_v))
                worst_sample_l2 = float(norms[j])
                lp = int(np.argmax(viol[j, :]))
                worst_line_pos = lp
                worst_line_idx = (
                    int(dc_op.line_ids[lp]) if 0 <= lp < len(dc_op.line_ids) else -1
                )

        feas_ci = _wilson_ci95_percent(k=int(feasible), n=int(n_samples))
        ball_ci = _wilson_ci95_percent(k=int(in_ball), n=int(n_samples))

        if in_ball > 0:
            eta = 100.0 * float(in_ball_and_feasible) / float(in_ball)
            eta_ci = _wilson_ci95_percent(k=int(in_ball_and_feasible), n=int(in_ball))
        else:
            eta = float("nan")
            eta_ci = (float("nan"), float("nan"))

        denom = float(sigma_mw) * math.sqrt(float(d))
        rho = (
            float(radius_check.r_star) / denom
            if (math.isfinite(radius_check.r_star) and denom > 0.0)
            else float("nan")
        )

        p_safe = 100.0 * float(feasible) / float(n_samples)
        p_ball_mc = 100.0 * float(in_ball) / float(n_samples)

        n_ball_samples = min(int(n_samples), int(cert_max))
        if base_status != BASE_OK:
            soundness = SoundnessCheck(
                status=SOUND_SKIPPED_BASE_INFEASIBLE,
                n_ball_samples=0,
                violation_samples=0,
                max_violation_mw=float("nan"),
                max_violation_line_idx=-1,
                tol_mw=float(tol_cert),
            )
        elif radius_check.status == RADIUS_INVALID or not math.isfinite(
            float(radius_check.r_star)
        ):
            soundness = SoundnessCheck(
                status=SOUND_SKIPPED_INVALID_RADIUS,
                n_ball_samples=0,
                violation_samples=0,
                max_violation_mw=float("nan"),
                max_violation_line_idx=-1,
                tol_mw=float(tol_cert),
            )
        elif float(radius_check.r_star) <= 0.0:
            soundness = SoundnessCheck(
                status=SOUND_SKIPPED_TRIVIAL_RADIUS,
                n_ball_samples=int(n_ball_samples),
                violation_samples=0,
                max_violation_mw=float("-inf"),
                max_violation_line_idx=-1,
                tol_mw=float(tol_cert),
            )
        else:
            rng2 = np.random.default_rng(int(seed) + 1_000_003)
            remaining2 = int(n_ball_samples)
            violations = 0
            worst_v = float("-inf")
            worst_idx = -1
            total = 0

            while remaining2 > 0:
                k = min(int(chunk_size), remaining2)
                remaining2 -= k

                delta_full = _sample_uniform_l2_ball_balanced(
                    rng=rng2,
                    n=k,
                    n_bus=int(dc_op.n_bus),
                    radius=float(radius_check.r_star),
                )
                df = dc_op.flows_from_delta_injections(delta_full)
                viol = df + f0[None, :]
                np.abs(viol, out=viol)
                viol -= c[None, :]
                np.nan_to_num(
                    viol,
                    copy=False,
                    nan=float("inf"),
                    posinf=float("inf"),
                    neginf=-float("inf"),
                )

                max_v = np.max(viol, axis=1)
                ok = max_v <= float(tol_cert)

                violations += int(np.sum(~ok))
                total += int(k)

                batch_worst = float(np.max(max_v))
                if batch_worst > worst_v:
                    worst_v = batch_worst
                    j = int(np.argmax(max_v))
                    lp = int(np.argmax(viol[j, :]))
                    worst_idx = (
                        int(dc_op.line_ids[lp]) if 0 <= lp < len(dc_op.line_ids) else -1
                    )

            soundness = SoundnessCheck(
                status=SOUND_PASS if violations == 0 else SOUND_FAIL,
                n_ball_samples=int(total),
                violation_samples=int(violations),
                max_violation_mw=float(worst_v),
                max_violation_line_idx=int(worst_idx),
                tol_mw=float(tol_cert),
            )

        prob_status = PROB_OK
        if math.isfinite(float(p_ball_analytic)) and float(p_ball_analytic) <= 1e-12:
            prob_status = PROB_DEGENERATE_DIMENSION

        prob = ProbabilisticCheck(
            status=prob_status,
            p_safe_gaussian_percent=float(p_safe),
            p_safe_gaussian_ci95_low_percent=float(feas_ci[0]),
            p_safe_gaussian_ci95_high_percent=float(feas_ci[1]),
            p_ball_analytic_percent=float(p_ball_analytic),
            p_ball_mc_percent=float(p_ball_mc),
            p_ball_mc_ci95_low_percent=float(ball_ci[0]),
            p_ball_mc_ci95_high_percent=float(ball_ci[1]),
            eta_safe_given_in_ball_percent=float(eta),
            eta_ci95_low_percent=float(eta_ci[0]),
            eta_ci95_high_percent=float(eta_ci[1]),
            rho=float(rho),
        )

        overall = overall_from_components(
            base_status=str(base_check.status),
            radius_status=str(radius_check.status),
            soundness_status=str(soundness.status),
            probabilistic_status=str(prob.status),
        )

        inputs = VerificationInputs(
            case_id=str(case_id),
            results_path=str(rp),
            input_case_path=str(ip),
            slack_bus=int(slack_bus),
            n_bus=int(dc_op.n_bus),
            n_line=int(dc_op.n_line),
            dim_balance=int(d),
            n_samples=int(n_samples),
            seed=int(seed),
            chunk_size=int(chunk_size),
            sigma_mw=float(sigma_mw),
        )

        cert_interp = interpret_certificate_components(
            base=base_check, radius=radius_check, soundness=soundness
        )

        comparisons: dict[str, Any] = {
            "mode": "dc",
            "certificate_soundness": str(cert_interp.soundness),
            "certificate_usefulness": str(cert_interp.usefulness),
            "certificate_notes": list(cert_interp.notes),
            "gaussian_worst_max_violation_mw": float(worst_max_violation),
            "gaussian_worst_max_violation_line_pos": int(worst_line_pos),
            "gaussian_worst_max_violation_line_idx": int(worst_line_idx),
            "gaussian_worst_sample_l2": float(worst_sample_l2),
            "feas_tol_mw": float(tol_feas),
            "sigma_source": str(sigma_source),
        }

        return VerificationResult(
            schema_version=1,
            inputs=inputs,
            base_point=base_check,
            radius=radius_check,
            soundness=soundness,
            probabilistic=prob,
            comparisons=comparisons,
            overall=overall,
        )

    # ---------------- AC verification ----------------
    if ac_sigma_p_mw is None or ac_sigma_q_mvar is None:
        raise ValueError(
            "AC mode requires explicit ac_sigma_p_mw and ac_sigma_q_mvar (no hidden defaults)."
        )

    solver_eff = str(ac_pf_solver).strip().lower()
    if solver_eff != "pandapower":
        raise NotImplementedError(
            "AC Monte Carlo per-sample PF currently supports ONLY pandapower. "
            "Set ac.pf_solver=pandapower when generating results."
        )

    if not bool(ac_lossless):
        raise NotImplementedError(
            "AC MC with lossless=false is not supported. Set ac.lossless=true."
        )

    base_ac = meta.get("base_point_ac", None)
    if not isinstance(base_ac, dict):
        raise KeyError(
            "results.json missing __meta__.base_point_ac (re-run compute with compute.ac.compute=1)."
        )

    if str(base_ac.get("pf_solver", "")).strip().lower() not in {"pandapower", "pypsa"}:
        raise ValueError("Invalid __meta__.base_point_ac.pf_solver in results.json.")
    if str(base_ac.get("pf_solver", "")).strip().lower() != solver_eff:
        raise ValueError(
            "AC MC solver mismatch with certificate base point. "
            f"results.json pf_solver={base_ac.get('pf_solver')!r}, requested={solver_eff!r}."
        )
    if bool(base_ac.get("lossless", True)) != bool(ac_lossless):
        raise ValueError(
            "AC MC lossless mismatch with certificate base point. "
            f"results.json lossless={bool(base_ac.get('lossless'))}, requested={bool(ac_lossless)}."
        )

    # Support both scalar (uniform) and per-bus array sigma
    sigma_p = np.asarray(ac_sigma_p_mw, dtype=float)
    sigma_q = np.asarray(ac_sigma_q_mvar, dtype=float)
    if sigma_p.ndim == 0:
        sigma_p = float(sigma_p)
    if sigma_q.ndim == 0:
        sigma_q = float(sigma_q)

    # Extract AC radii from results
    line_ids_sorted = [int(x) for x in sorted(net.line.index)]
    m_line = int(len(line_ids_sorted))

    r_line = np.full(m_line, float("nan"), dtype=float)
    margin_line = np.full(m_line, float("nan"), dtype=float)
    h_norm = np.full(m_line, float("nan"), dtype=float)

    for pos, lid in enumerate(line_ids_sorted):
        row = results.get(line_key(int(lid)))
        if not isinstance(row, dict):
            raise KeyError(f"results.json missing per-line entry: line_{lid}")
        if "radius_ac_l2" not in row:
            raise KeyError(
                "results.json missing radius_ac_l2 (compute compute.ac.compute=1)."
            )
        r_line[pos] = float(row.get("radius_ac_l2", float("nan")))
        margin_line[pos] = float(row.get("margin_ac_mva", float("nan")))
        h_norm[pos] = float(row.get("||h||2", float("nan")))

    finite = np.isfinite(r_line)
    if not bool(np.any(finite)):
        radius_check = RadiusCheck(
            status=RADIUS_INVALID,
            r_star=float("nan"),
            argmin_line_pos=-1,
            argmin_line_idx=-1,
            min_margin_mw=float(np.nanmin(margin_line))
            if margin_line.size
            else float("nan"),
            argmin_margin_mw=float("nan"),
            argmin_norm_g=float("nan"),
        )
    else:
        argmin_pos = int(np.argmin(np.where(finite, r_line, float("inf"))))
        r_star = float(r_line[argmin_pos])
        argmin_idx = int(line_ids_sorted[argmin_pos])
        argmin_margin = float(margin_line[argmin_pos])
        argmin_h = float(h_norm[argmin_pos])
        min_margin = float(np.nanmin(margin_line))

        if not math.isfinite(r_star) or r_star < 0:
            status = RADIUS_INVALID
        elif r_star == 0.0:
            status = (
                RADIUS_ZERO_BINDING
                if math.isfinite(argmin_margin) and abs(argmin_margin) <= 1e-9
                else RADIUS_ZERO_BAD_LIMITS
            )
        else:
            status = RADIUS_OK

        radius_check = RadiusCheck(
            status=status,
            r_star=float(r_star),
            argmin_line_pos=int(argmin_pos),
            argmin_line_idx=int(argmin_idx),
            min_margin_mw=float(min_margin),
            argmin_margin_mw=float(argmin_margin),
            argmin_norm_g=float(argmin_h),
        )

    import pandapower as pp  # type: ignore

    nn = apply_lossless_policy_to_pandapower_net(net)
    slack_bus_id = resolve_slack_bus_id(nn, slack_bus)
    ensure_ext_grid_at_slack(nn, slack_bus_id)

    # Apply dispatch from results meta so MC uses the same base point as the
    # certificate.  For AC mode prefer base_point_ac, which stores the OPP gen
    # dispatch and vm_pu setpoints. base_point_dc is used only when no AC
    # dispatch metadata is present.
    dispatch_pairs = None
    vm_pu_setpoints: dict[int, float] | None = None
    if isinstance(base_ac, dict):
        dispatch_pairs = base_ac.get("gen_dispatch_mw_by_name", None)
        # Reconstruct vm_pu setpoints keyed by bus id
        bp_bus_ids = base_ac.get("bus_ids", [])
        bp_vm_pu = base_ac.get("vm_pu", [])
        if bp_bus_ids and bp_vm_pu and len(bp_bus_ids) == len(bp_vm_pu):
            vm_pu_setpoints = {
                int(bid): float(vm) for bid, vm in zip(bp_bus_ids, bp_vm_pu)
            }
    if not dispatch_pairs:
        bp_dc = meta.get("base_point_dc", None)
        if isinstance(bp_dc, dict):
            dispatch_pairs = bp_dc.get("gen_dispatch_mw_by_name", None)

    if vm_pu_setpoints:
        # Convert dispatch_pairs (list of [name, value] or dict) to dict
        gen_dispatch_dict: dict[str, float] | None = None
        if dispatch_pairs:
            if isinstance(dispatch_pairs, dict):
                gen_dispatch_dict = dispatch_pairs
            else:
                gen_dispatch_dict = {str(k): float(v) for k, v in dispatch_pairs}
        apply_opp_result_to_pandapower_net(
            nn,
            opp_gen_dispatch=gen_dispatch_dict,
            opp_vm_pu=vm_pu_setpoints,
        )
    else:
        apply_gen_dispatch_to_pandapower_net(nn, dispatch_pairs)

    # Attach deterministic per-bus perturbation elements
    if not hasattr(nn, "sgen") or nn.sgen is None:
        raise RuntimeError("pandapower net has no sgen table (unexpected).")

    bus_ids = [int(x) for x in sorted(nn.bus.index)]
    sgen_idx: list[int] = []
    for bid in bus_ids:
        idx = int(
            pp.create_sgen(
                nn,
                bus=int(bid),
                p_mw=0.0,
                q_mvar=0.0,
                name=f"mc_delta_bus_{int(bid)}",
                in_service=True,
            )
        )
        sgen_idx.append(idx)

    line_ids, limits_mva = sorted_line_limits_mva(nn)

    # Base PF (no perturbation) must match results.json regime.
    try:
        pp.runpp(
            nn,
            calculate_voltage_angles=True,
            enforce_q_lims=True,
            init="flat",
            max_iter=50,
            numba=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("AC MC: base PF failed in primary solve.") from exc
    if not bool(getattr(nn, "converged", True)):
        raise RuntimeError("AC MC: base PF did not converge (net.converged=False).")

    base_diag = _check_ac_base_point_matches_results(
        nn=nn,
        results=results,
        line_ids_sorted=line_ids_sorted,
        tol_mva=float(ac_basepoint_s_tol_mva),
    )

    base_ok, base_worst, _ = _ac_pf_sample_violation_mva(
        nn, line_ids=line_ids, limits_mva=limits_mva, feas_tol_mva=float(tol_feas)
    )
    base_check = BasePointCheck(
        status=BASE_OK if base_ok else BASE_INFEASIBLE,
        violated_lines=0 if base_ok else 1,
        max_violation_mw=float(base_worst),
    )

    n_bus = int(len(bus_ids))
    d = int(2 * (n_bus - 1))
    if d <= 0:
        raise ValueError("AC MC: invalid dimension (n_bus must be >=2).")

    _sigma_p_scalar = (
        float(np.mean(sigma_p)) if isinstance(sigma_p, np.ndarray) else float(sigma_p)
    )
    _sigma_q_scalar = (
        float(np.mean(sigma_q)) if isinstance(sigma_q, np.ndarray) else float(sigma_q)
    )
    _sigma_uniform = (
        isinstance(sigma_p, float)
        and isinstance(sigma_q, float)
        and abs(sigma_p - sigma_q) <= 1e-15
    )
    if (
        _sigma_uniform
        and math.isfinite(radius_check.r_star)
        and radius_check.r_star >= 0
    ):
        p_ball_analytic = 100.0 * _chi2_cdf(
            x=(float(radius_check.r_star) / _sigma_p_scalar) ** 2, df=d
        )
        prob_status = PROB_OK
    else:
        p_ball_analytic = float("nan")
        prob_status = PROB_UNKNOWN

    rng = np.random.default_rng(int(seed))
    feasible = 0
    in_ball = 0
    in_ball_and_feasible = 0
    pf_failures = 0

    worst_max_violation = float("-inf")
    worst_line_pos = -1
    worst_line_idx = -1
    worst_sample_l2 = float("nan")

    track_pl = bool(track_per_line_overloads) and mode_eff == "ac"
    per_line_overload_counts: np.ndarray | None = None
    if track_pl:
        per_line_overload_counts = np.zeros(len(line_ids), dtype=np.int64)

    remaining = int(n_samples)
    while remaining > 0:
        k = min(int(chunk_size), remaining)
        remaining -= k

        dp, dq = _sample_gaussian_ac(
            rng=rng, n=k, n_bus=n_bus, sigma_p_mw=sigma_p, sigma_q_mvar=sigma_q
        )
        norms = np.sqrt(np.sum(dp * dp, axis=1) + np.sum(dq * dq, axis=1))

        if math.isfinite(radius_check.r_star):
            in_ball_mask = norms <= float(radius_check.r_star)
            in_ball += int(np.sum(in_ball_mask))
        else:
            in_ball_mask = np.zeros(k, dtype=bool)

        for j in range(k):
            nn.sgen.loc[sgen_idx, "p_mw"] = dp[j, :]
            nn.sgen.loc[sgen_idx, "q_mvar"] = dq[j, :]

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
                pf_failures += 1
                is_feas = False
                worst = float("inf")
                wpos = -1
            elif track_pl and per_line_overload_counts is not None:
                is_feas, worst, wpos, overloaded = (
                    _ac_pf_sample_per_line_violations_mva(
                        nn,
                        line_ids=line_ids,
                        limits_mva=limits_mva,
                        feas_tol_mva=float(tol_feas),
                    )
                )
                per_line_overload_counts[overloaded] += 1
            else:
                is_feas, worst, wpos = _ac_pf_sample_violation_mva(
                    nn,
                    line_ids=line_ids,
                    limits_mva=limits_mva,
                    feas_tol_mva=float(tol_feas),
                )

            if is_feas:
                feasible += 1
            if bool(in_ball_mask[j]) and is_feas:
                in_ball_and_feasible += 1

            if worst > worst_max_violation:
                worst_max_violation = float(worst)
                worst_sample_l2 = float(norms[j])
                worst_line_pos = int(wpos)
                worst_line_idx = (
                    int(line_ids[wpos]) if 0 <= wpos < len(line_ids) else -1
                )

    feas_ci = _wilson_ci95_percent(k=int(feasible), n=int(n_samples))
    ball_ci = _wilson_ci95_percent(k=int(in_ball), n=int(n_samples))

    if in_ball > 0:
        eta = 100.0 * float(in_ball_and_feasible) / float(in_ball)
        eta_ci = _wilson_ci95_percent(k=int(in_ball_and_feasible), n=int(in_ball))
    else:
        eta = float("nan")
        eta_ci = (float("nan"), float("nan"))

    denom = (
        _sigma_p_scalar * math.sqrt(float(d))
        if math.isfinite(_sigma_p_scalar) and _sigma_p_scalar > 0
        else float("nan")
    )
    rho = (
        float(radius_check.r_star) / denom
        if (math.isfinite(radius_check.r_star) and math.isfinite(denom) and denom > 0.0)
        else float("nan")
    )

    p_safe = 100.0 * float(feasible) / float(n_samples)
    p_ball_mc = 100.0 * float(in_ball) / float(n_samples)

    n_ball_samples = min(int(n_samples), int(cert_max))
    if base_check.status != BASE_OK:
        soundness = SoundnessCheck(
            status=SOUND_SKIPPED_BASE_INFEASIBLE,
            n_ball_samples=0,
            violation_samples=0,
            max_violation_mw=float("nan"),
            max_violation_line_idx=-1,
            tol_mw=float(tol_cert),
        )
    elif radius_check.status == RADIUS_INVALID or not math.isfinite(
        float(radius_check.r_star)
    ):
        soundness = SoundnessCheck(
            status=SOUND_SKIPPED_INVALID_RADIUS,
            n_ball_samples=0,
            violation_samples=0,
            max_violation_mw=float("nan"),
            max_violation_line_idx=-1,
            tol_mw=float(tol_cert),
        )
    elif float(radius_check.r_star) <= 0.0:
        soundness = SoundnessCheck(
            status=SOUND_SKIPPED_TRIVIAL_RADIUS,
            n_ball_samples=int(n_ball_samples),
            violation_samples=0,
            max_violation_mw=float("-inf"),
            max_violation_line_idx=-1,
            tol_mw=float(tol_cert),
        )
    else:
        rng2 = np.random.default_rng(int(seed) + 1_000_003)
        violations = 0
        worst_v = float("-inf")
        worst_idx = -1

        dpb, dqb = _sample_uniform_l2_ball_ac(
            rng=rng2,
            n=int(n_ball_samples),
            n_bus=n_bus,
            radius=float(radius_check.r_star),
        )
        for j in range(int(n_ball_samples)):
            nn.sgen.loc[sgen_idx, "p_mw"] = dpb[j, :]
            nn.sgen.loc[sgen_idx, "q_mvar"] = dqb[j, :]

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
                violations += 1
                worst = float("inf")
                wpos = -1
            else:
                ok, worst, wpos = _ac_pf_sample_violation_mva(
                    nn,
                    line_ids=line_ids,
                    limits_mva=limits_mva,
                    feas_tol_mva=float(tol_cert),
                )
                if not ok:
                    violations += 1

            if worst > worst_v:
                worst_v = float(worst)
                worst_idx = int(line_ids[wpos]) if 0 <= wpos < len(line_ids) else -1

        soundness = SoundnessCheck(
            status=SOUND_PASS if violations == 0 else SOUND_FAIL,
            n_ball_samples=int(n_ball_samples),
            violation_samples=int(violations),
            max_violation_mw=float(worst_v),
            max_violation_line_idx=int(worst_idx),
            tol_mw=float(tol_cert),
        )

    prob = ProbabilisticCheck(
        status=str(prob_status),
        p_safe_gaussian_percent=float(p_safe),
        p_safe_gaussian_ci95_low_percent=float(feas_ci[0]),
        p_safe_gaussian_ci95_high_percent=float(feas_ci[1]),
        p_ball_analytic_percent=float(p_ball_analytic),
        p_ball_mc_percent=float(p_ball_mc),
        p_ball_mc_ci95_low_percent=float(ball_ci[0]),
        p_ball_mc_ci95_high_percent=float(ball_ci[1]),
        eta_safe_given_in_ball_percent=float(eta),
        eta_ci95_low_percent=float(eta_ci[0]),
        eta_ci95_high_percent=float(eta_ci[1]),
        rho=float(rho),
    )

    overall = overall_from_components(
        base_status=str(base_check.status),
        radius_status=str(radius_check.status),
        soundness_status=str(soundness.status),
        probabilistic_status=str(prob.status),
    )

    inputs = VerificationInputs(
        case_id=str(case_id),
        results_path=str(rp),
        input_case_path=str(ip),
        slack_bus=int(slack_bus),
        n_bus=int(n_bus),
        n_line=int(len(line_ids)),
        dim_balance=int(d),
        n_samples=int(n_samples),
        seed=int(seed),
        chunk_size=int(chunk_size),
        sigma_mw=_sigma_p_scalar,
    )

    cert_interp = interpret_certificate_components(
        base=base_check, radius=radius_check, soundness=soundness
    )

    comparisons = {
        "mode": "ac",
        "units": "MVA for feasibility/violations; MW/MVAr for injections",
        "certificate_soundness": str(cert_interp.soundness),
        "certificate_usefulness": str(cert_interp.usefulness),
        "certificate_notes": list(cert_interp.notes),
        "ac_sigma_p_mw": _sigma_p_scalar,
        "ac_sigma_q_mvar": _sigma_q_scalar,
        "ac_lossless": bool(ac_lossless),
        "ac_pf_solver": str(solver_eff),
        "ac_basepoint_s_tol_mva": float(ac_basepoint_s_tol_mva),
        **base_diag,
        "pf_failures_gaussian": int(pf_failures),
        "gaussian_worst_max_violation_mva": float(worst_max_violation),
        "gaussian_worst_max_violation_line_pos": int(worst_line_pos),
        "gaussian_worst_max_violation_line_idx": int(worst_line_idx),
        "gaussian_worst_sample_l2": float(worst_sample_l2),
        "feas_tol_mva": float(tol_feas),
    }

    if track_pl and per_line_overload_counts is not None:
        n_pf_converged = max(int(n_samples) - int(pf_failures), 0)
        per_line_fracs = per_line_overload_counts.astype(float) / float(
            max(n_pf_converged, 1)
        )
        comparisons["per_line_overload_counts"] = {
            line_key(int(line_ids[pos])): int(per_line_overload_counts[pos])
            for pos in range(len(line_ids))
        }
        comparisons["per_line_overload_fractions_conditional_on_pf_converged"] = {
            line_key(int(line_ids[pos])): float(per_line_fracs[pos])
            for pos in range(len(line_ids))
        }
        comparisons["per_line_overload_fraction_denominator"] = int(n_pf_converged)
        comparisons["bad_sample_probability"] = float(
            1.0 - (float(feasible) / float(max(int(n_samples), 1)))
        )
        comparisons["pf_failure_probability"] = float(
            float(pf_failures) / float(max(int(n_samples), 1))
        )

    return VerificationResult(
        schema_version=1,
        inputs=inputs,
        base_point=base_check,
        radius=radius_check,
        soundness=soundness,
        probabilistic=prob,
        comparisons=comparisons,
        overall=overall,
    )
