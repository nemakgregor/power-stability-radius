from __future__ import annotations

"""
Monte Carlo verification for the DC L2-ball robustness certificate.

Distribution used (slack-invariant)
-----------------------------------
We work in the balanced injection subspace of dimension:

  d = n_bus - 1

defined as:
  {Δp in R^{n_bus} : 1^T Δp = 0}.

Gaussian:
  - Sample z ~ N(0, σ^2 I_{n_bus})
  - Project: Δp = z - mean(z) * 1
This is equivalent to Δp ~ N(0, σ^2 I_d) in any orthonormal basis of the balanced subspace.

Certified ball:
  {Δp : ||Δp||_2 <= r*  and 1^T Δp = 0}

Determinism
-----------
- All randomness is controlled by `seed`.
- Chunking is deterministic (fixed ordering).
"""

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from stability_radius.config import DEFAULT_MC
from stability_radius.dc.dc_model import build_dc_operator
from stability_radius.parsers.matpower import load_network
from stability_radius.radii.common import line_key
from stability_radius.utils import log_stage
from stability_radius.utils.download import ensure_case_file

from .types import (
    BASE_INFEASIBLE,
    BASE_OK,
    PROB_DEGENERATE_DIMENSION,
    PROB_OK,
    RADIUS_INVALID,
    RADIUS_OK,
    RADIUS_UNKNOWN,
    RADIUS_ZERO_BAD_LIMITS,
    RADIUS_ZERO_BINDING,
    SOUND_FAIL,
    SOUND_PASS,
    SOUND_SKIPPED_BASE_INFEASIBLE,
    SOUND_SKIPPED_INVALID_RADIUS,
    SOUND_SKIPPED_NO_SAMPLES,
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


def _load_results(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj)}")
    return obj


def _get_meta(results: dict[str, Any]) -> dict[str, Any]:
    meta = results.get("__meta__")
    return meta if isinstance(meta, dict) else {}


def _wilson_ci95_percent(*, k: int, n: int) -> tuple[float, float]:
    """
    95% Wilson CI for a binomial proportion, returned in percent units.

    This is more stable than Wald for extreme probabilities (near 0 or 1).
    """
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
    """
    Chi-square CDF using SciPy special function:
        F_{χ²(df)}(x) = gammainc(df/2, x/2)
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    xx = float(x)
    if not math.isfinite(xx) or xx <= 0.0:
        return 0.0

    try:
        from scipy.special import gammainc  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError("SciPy is required to compute chi-square CDF.") from e

    return float(gammainc(float(df) / 2.0, xx / 2.0))


def _project_sum_zero_inplace(x: np.ndarray) -> np.ndarray:
    """
    Project rows of x onto the hyperplane sum(row)=0 in-place.

    Parameters
    ----------
    x:
        Array (k, n_bus).

    Returns
    -------
    np.ndarray
        Same array (x) after in-place projection.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (k,n_bus), got {x.shape}")
    x -= np.mean(x, axis=1, keepdims=True)
    return x


def _sample_gaussian(
    *, rng: np.random.Generator, n: int, n_bus: int, sigma_mw: float
) -> np.ndarray:
    """
    Sample balanced Δp in full bus coordinates.

    Sampling scheme
    ---------------
      z ~ N(0, σ^2 I_n)
      Δp = z - mean(z) * 1

    Returns
    -------
    np.ndarray
        Shape (n, n_bus), each row sums to ~0.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if n_bus <= 1:
        raise ValueError("n_bus must be >= 2.")
    s = float(sigma_mw)
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("sigma_mw must be finite and positive.")

    z = (s * rng.standard_normal(size=(int(n), int(n_bus)))).astype(float, copy=False)
    return _project_sum_zero_inplace(z)


def _sample_uniform_l2_ball(
    *, rng: np.random.Generator, n: int, n_bus: int, radius: float
) -> np.ndarray:
    """
    Sample uniformly in the balanced L2 ball:

        {x in R^{n_bus} : sum(x)=0 and ||x||_2 <= radius}

    Returns
    -------
    np.ndarray
        Shape (n, n_bus).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if n_bus <= 1:
        raise ValueError("n_bus must be >= 2.")
    r = float(radius)
    if not math.isfinite(r) or r < 0.0:
        raise ValueError("radius must be finite and non-negative.")
    if r == 0.0:
        return np.zeros((int(n), int(n_bus)), dtype=float)

    d = int(n_bus - 1)  # intrinsic dimension of the balanced subspace

    z = rng.standard_normal(size=(int(n), int(n_bus))).astype(float, copy=False)
    _project_sum_zero_inplace(z)
    norms = np.linalg.norm(z, ord=2, axis=1)

    # Re-draw degenerate directions deterministically.
    bad = norms <= 1e-12
    while bool(np.any(bad)):
        k_bad = int(np.sum(bad))
        z_bad = rng.standard_normal(size=(k_bad, int(n_bus))).astype(float, copy=False)
        _project_sum_zero_inplace(z_bad)
        z[bad, :] = z_bad
        norms = np.linalg.norm(z, ord=2, axis=1)
        bad = norms <= 1e-12

    dirs = z / norms[:, None]

    # Radius distribution for uniform-in-ball: U^(1/d)
    u = rng.random(size=int(n)).astype(float, copy=False)
    rad = r * np.power(u, 1.0 / float(d))
    return dirs * rad[:, None]


def _flows_from_delta_injections_reduced(*, dc_op, delta_red: np.ndarray) -> np.ndarray:
    """
    Compute Δf for reduced injections Δp_red (non-slack buses).

    This avoids allocating a full (n_bus) delta array.
    """
    dp = np.asarray(delta_red, dtype=float)
    if dp.ndim != 2:
        raise ValueError(f"delta_red must be 2D (k,d), got {dp.shape}")

    rhs = dp.T  # (d, k) = (n_bus-1, k)
    theta = dc_op.solve_Bred(rhs)  # (d, k)
    flow = (dc_op.W @ theta).T  # (k, m_line)
    return np.asarray(flow, dtype=float)


def _flows_from_delta_injections_balanced_full(
    *, dc_op, delta_full: np.ndarray
) -> np.ndarray:
    """
    Compute Δf for balanced full-bus injections Δp (k, n_bus).

    For DCOperator we drop the slack component (it is redundant for balanced Δp).
    """
    dp = np.asarray(delta_full, dtype=float)
    if dp.ndim != 2:
        raise ValueError(f"delta_full must be 2D (k,n_bus), got {dp.shape}")
    if dp.shape[1] != int(dc_op.n_bus):
        raise ValueError(
            f"delta_full must have n_bus={int(dc_op.n_bus)} columns; got {dp.shape}"
        )

    dp_red = dp[:, dc_op.mask_non_slack]
    return _flows_from_delta_injections_reduced(dc_op=dc_op, delta_red=dp_red)


def _extract_line_arrays(
    *,
    results: dict[str, Any],
    net,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract per-line arrays aligned with sorted net.line.index order.

    Returns
    -------
    (line_indices, f0, c, radius_l2, margin_raw, norm_g)
    """
    line_indices = [int(x) for x in sorted(net.line.index)]
    m = int(len(line_indices))

    f0 = np.empty(m, dtype=float)
    c = np.empty(m, dtype=float)
    r = np.empty(m, dtype=float)
    margin_raw = np.empty(m, dtype=float)
    norm_g = np.full(m, float("nan"), dtype=float)

    missing: list[str] = []
    for pos, lid in enumerate(line_indices):
        k = line_key(int(lid))
        row = results.get(k)
        if not isinstance(row, dict):
            missing.append(k)
            continue

        try:
            f0[pos] = float(row["flow0_mw"])
            c[pos] = float(row["p_limit_mw_est"])
            r[pos] = float(row["radius_l2"])
        except KeyError as e:
            raise KeyError(f"Missing required field {e} in {k}") from e

        margin_raw[pos] = float(c[pos] - abs(f0[pos]))

        if "norm_g" in row:
            try:
                norm_g[pos] = float(row.get("norm_g", float("nan")))
            except (TypeError, ValueError):
                norm_g[pos] = float("nan")

    if missing:
        raise KeyError(
            f"results.json does not contain required line keys (first 10): {missing[:10]}"
        )

    if np.isnan(c).any():
        bad = np.where(np.isnan(c))[0]
        raise ValueError(
            "results.json contains NaN in p_limit_mw_est; verification is undefined. "
            f"Bad line positions count={int(bad.size)} (first 10: {bad[:10].tolist()})."
        )

    return line_indices, f0, c, r, margin_raw, norm_g


def _compute_r_star_and_argmin(
    *,
    line_indices: list[int],
    radii: np.ndarray,
    margins_raw: np.ndarray,
    norm_g: np.ndarray,
    base_status: str,
) -> RadiusCheck:
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
    argmin_line_idx = (
        int(line_indices[argmin_pos]) if 0 <= argmin_pos < len(line_indices) else -1
    )
    argmin_margin = (
        float(margins_raw[argmin_pos])
        if 0 <= argmin_pos < margins_raw.size
        else float("nan")
    )
    argmin_norm = (
        float(norm_g[argmin_pos]) if 0 <= argmin_pos < norm_g.size else float("nan")
    )
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


def _run_gaussian_stage(
    *,
    case_id: str,
    dc_op,
    f0: np.ndarray,
    c: np.ndarray,
    r_star: float,
    sigma_mw: float,
    n_samples: int,
    seed: int,
    chunk_size: int,
    feas_tol_mw: float,
) -> tuple[
    int,  # feasible_samples
    tuple[float, float],  # feasible_ci
    int,  # in_ball_samples
    tuple[float, float],  # in_ball_ci
    int,  # in_ball_and_feasible_samples
    float,  # worst_max_violation
    int,  # worst_line_pos
    int,  # worst_line_idx
    float,  # worst_sample_l2
]:
    rng = np.random.default_rng(int(seed))

    feasible = 0
    in_ball = 0
    in_ball_and_feasible = 0

    worst_max_violation = float("-inf")
    worst_line_pos = -1
    worst_line_idx = -1
    worst_sample_l2 = float("nan")

    n_bus = int(dc_op.n_bus)
    d = int(max(n_bus - 1, 0))
    if d <= 0:
        raise ValueError("Invalid DCOperator dimension: n_bus must be >= 2.")

    remaining = int(n_samples)
    while remaining > 0:
        k = min(int(chunk_size), remaining)
        remaining -= k

        delta_full = _sample_gaussian(
            rng=rng, n=k, n_bus=n_bus, sigma_mw=float(sigma_mw)
        )
        norms = np.linalg.norm(delta_full, ord=2, axis=1)
        in_ball_mask = norms <= float(r_star)
        in_ball += int(np.sum(in_ball_mask))

        df = _flows_from_delta_injections_balanced_full(
            dc_op=dc_op, delta_full=delta_full
        )
        viol = df + f0[None, :]
        np.abs(viol, out=viol)
        viol -= c[None, :]  # abs(flow) - limit

        np.nan_to_num(
            viol,
            copy=False,
            nan=float("inf"),
            posinf=float("inf"),
            neginf=-float("inf"),
        )

        max_v = np.max(viol, axis=1)
        feasible_mask = max_v <= float(feas_tol_mw)

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

    p_feas = 100.0 * float(feasible) / float(n_samples)
    logger.info(
        "case=%s stage=gaussian_feasibility n=%d sigma=%.6g d=%d p_safe=%.6g ci95=[%.6g,%.6g]",
        str(case_id),
        int(n_samples),
        float(sigma_mw),
        int(d),
        float(p_feas),
        float(feas_ci[0]),
        float(feas_ci[1]),
    )

    return (
        int(feasible),
        feas_ci,
        int(in_ball),
        ball_ci,
        int(in_ball_and_feasible),
        float(worst_max_violation),
        int(worst_line_pos),
        int(worst_line_idx),
        float(worst_sample_l2),
    )


def _run_soundness_stage(
    *,
    case_id: str,
    dc_op,
    f0: np.ndarray,
    c: np.ndarray,
    r_star: float,
    n_ball_samples: int,
    seed: int,
    chunk_size: int,
    tol_mw: float,
) -> SoundnessCheck:
    if n_ball_samples <= 0:
        return SoundnessCheck(
            status=SOUND_SKIPPED_NO_SAMPLES,
            n_ball_samples=0,
            violation_samples=0,
            max_violation_mw=float("nan"),
            max_violation_line_idx=-1,
            tol_mw=float(tol_mw),
        )

    if not math.isfinite(float(r_star)):
        return SoundnessCheck(
            status=SOUND_SKIPPED_INVALID_RADIUS,
            n_ball_samples=int(n_ball_samples),
            violation_samples=int(n_ball_samples),
            max_violation_mw=float("nan"),
            max_violation_line_idx=-1,
            tol_mw=float(tol_mw),
        )

    if float(r_star) <= 0.0:
        return SoundnessCheck(
            status=SOUND_SKIPPED_TRIVIAL_RADIUS,
            n_ball_samples=int(n_ball_samples),
            violation_samples=0,
            max_violation_mw=float("-inf"),
            max_violation_line_idx=-1,
            tol_mw=float(tol_mw),
        )

    rng = np.random.default_rng(int(seed))

    n_bus = int(dc_op.n_bus)
    remaining = int(n_ball_samples)

    violations = 0
    worst_max_violation = float("-inf")
    worst_line_idx = -1

    total = 0

    while remaining > 0:
        k = min(int(chunk_size), remaining)
        remaining -= k

        delta_full = _sample_uniform_l2_ball(
            rng=rng, n=k, n_bus=n_bus, radius=float(r_star)
        )

        df = _flows_from_delta_injections_balanced_full(
            dc_op=dc_op, delta_full=delta_full
        )
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
        ok = max_v <= float(tol_mw)

        violations += int(np.sum(~ok))
        total += int(k)

        batch_worst = float(np.max(max_v))
        if batch_worst > worst_max_violation:
            worst_max_violation = batch_worst
            j = int(np.argmax(max_v))
            lp = int(np.argmax(viol[j, :]))
            worst_line_idx = (
                int(dc_op.line_ids[lp]) if 0 <= lp < len(dc_op.line_ids) else -1
            )

    status = SOUND_PASS if violations == 0 else SOUND_FAIL
    logger.info(
        "case=%s stage=ball_soundness r*=%.6g n=%d violations=%d status=%s",
        str(case_id),
        float(r_star),
        int(total),
        int(violations),
        str(status),
    )

    return SoundnessCheck(
        status=status,
        n_ball_samples=int(total),
        violation_samples=int(violations),
        max_violation_mw=float(worst_max_violation),
        max_violation_line_idx=int(worst_line_idx),
        tol_mw=float(tol_mw),
    )


def run_monte_carlo_verification(
    *,
    results_path: Path,
    input_case_path: Path,
    slack_bus: int,
    n_samples: int = DEFAULT_MC.n_samples,
    seed: int = DEFAULT_MC.seed,
    chunk_size: int = DEFAULT_MC.chunk_size,
    feas_tol_mw: float = DEFAULT_MC.feas_tol_mw,
    cert_tol_mw: float = DEFAULT_MC.cert_tol_mw,
    cert_max_samples: int = DEFAULT_MC.cert_max_samples,
    sigma_override_mw: float | None = None,
) -> VerificationResult:
    """
    Run verification for a single case.

    Notes
    -----
    - `sigma_override_mw` (if provided) overrides results.json meta inj_std_mw for MC evaluation,
      enabling cross-case comparable experiments (e.g., fixed rho targets).
    """
    rp = Path(results_path).resolve()
    ip = Path(input_case_path).resolve()

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    tol_feas = float(feas_tol_mw)
    if not math.isfinite(tol_feas) or tol_feas < 0.0:
        raise ValueError("feas_tol_mw must be finite and non-negative.")

    tol_cert = float(cert_tol_mw)
    if not math.isfinite(tol_cert) or tol_cert < 0.0:
        raise ValueError("cert_tol_mw must be finite and non-negative.")

    cert_max = int(cert_max_samples)
    if cert_max < 0:
        raise ValueError("cert_max_samples must be non-negative.")

    case_id = rp.stem

    with log_stage(logger, "Read Results (results.json)"):
        results = _load_results(rp)
        meta = _get_meta(results)

    with log_stage(logger, "Ensure input case file (download if missing)"):
        ensured = ensure_case_file(str(ip))
        ip_eff = Path(ensured).resolve()

    with log_stage(logger, "Read Data (MATPOWER/PGLib -> pandapower)"):
        net = load_network(ip_eff)

    with log_stage(
        logger,
        f"Build DC Model (DCOperator, slack_bus={slack_bus})",
    ):
        dc_op = build_dc_operator(net, slack_bus=int(slack_bus))

    line_indices, f0, c, r, margin_raw, norm_g = _extract_line_arrays(
        results=results, net=net
    )

    # sigma is taken from results meta by default (must match how radii/MC are interpreted)
    sigma_source = "results_meta"
    if sigma_override_mw is not None:
        sigma_source = "override"
        try:
            sigma_mw = float(sigma_override_mw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid sigma_override_mw={sigma_override_mw!r}") from e
    else:
        sigma_meta = meta.get("inj_std_mw", None)
        if sigma_meta is None:
            raise ValueError(
                "results.json is missing __meta__.inj_std_mw which is required for Gaussian verification. "
                "Regenerate results with the current pipeline or pass sigma_override_mw."
            )
        try:
            sigma_mw = float(sigma_meta)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid __meta__.inj_std_mw={sigma_meta!r}") from e

    if not math.isfinite(sigma_mw) or sigma_mw <= 0.0:
        raise ValueError(f"sigma_mw must be finite and >0, got {sigma_mw!r}")

    logger.info(
        "case=%s sigma_mw=%.6g (source=%s)",
        str(case_id),
        float(sigma_mw),
        sigma_source,
    )

    base_viol = np.where(np.abs(f0) > (c + tol_feas))[0]
    base_feasible = bool(base_viol.size == 0)
    base_max_violation = (
        float(np.max(np.abs(f0[base_viol]) - c[base_viol])) if base_viol.size else 0.0
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

    # Analytic ball mass (chi-square) for isotropic Gaussian in the balanced subspace.
    d = int(max(dc_op.n_bus - 1, 0))
    if d <= 0:
        raise ValueError("Invalid DCOperator dimension: n_bus must be >= 2.")

    if math.isfinite(float(radius_check.r_star)) and float(radius_check.r_star) >= 0.0:
        x = (float(radius_check.r_star) / float(sigma_mw)) ** 2
        p_ball_analytic = 100.0 * _chi2_cdf(x=x, df=d)
    else:
        p_ball_analytic = float("nan")

    # Gaussian stage
    with log_stage(logger, "Monte Carlo: Gaussian safety (balanced subspace)"):
        (
            gauss_feasible,
            gauss_feas_ci,
            gauss_in_ball,
            gauss_ball_ci,
            gauss_in_ball_and_feasible,
            gauss_worst_max_violation,
            gauss_worst_line_pos,
            gauss_worst_line_idx,
            gauss_worst_sample_l2,
        ) = _run_gaussian_stage(
            case_id=case_id,
            dc_op=dc_op,
            f0=f0,
            c=c,
            r_star=float(radius_check.r_star)
            if math.isfinite(radius_check.r_star)
            else 0.0,
            sigma_mw=float(sigma_mw),
            n_samples=int(n_samples),
            seed=int(seed),
            chunk_size=int(chunk_size),
            feas_tol_mw=float(tol_feas),
        )

    # Conditional eta (safe | in_ball)
    if gauss_in_ball > 0:
        eta = 100.0 * float(gauss_in_ball_and_feasible) / float(gauss_in_ball)
        eta_ci = _wilson_ci95_percent(
            k=int(gauss_in_ball_and_feasible), n=int(gauss_in_ball)
        )
    else:
        eta = float("nan")
        eta_ci = (float("nan"), float("nan"))

    # rho = r* / (sigma * sqrt(d))
    if math.isfinite(float(radius_check.r_star)) and float(radius_check.r_star) >= 0.0:
        denom = float(sigma_mw) * math.sqrt(float(d))
        rho = float(radius_check.r_star) / denom if denom > 0.0 else float("nan")
    else:
        rho = float("nan")

    p_safe = 100.0 * float(gauss_feasible) / float(n_samples)
    p_ball_mc = 100.0 * float(gauss_in_ball) / float(n_samples)

    # Soundness stage (hard correctness check)
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
        with log_stage(logger, "Monte Carlo: L2-ball soundness (uniform in ball)"):
            soundness = _run_soundness_stage(
                case_id=case_id,
                dc_op=dc_op,
                f0=f0,
                c=c,
                r_star=float(radius_check.r_star),
                n_ball_samples=int(n_ball_samples),
                seed=int(seed) + 1_000_003,
                chunk_size=int(chunk_size),
                tol_mw=float(tol_cert),
            )

    cert_interp = interpret_certificate_components(
        base=base_check, radius=radius_check, soundness=soundness
    )

    # Probabilistic status (soft)
    prob_status = PROB_OK
    if math.isfinite(float(p_ball_analytic)) and float(p_ball_analytic) <= 1e-12:
        prob_status = PROB_DEGENERATE_DIMENSION

    prob = ProbabilisticCheck(
        status=prob_status,
        p_safe_gaussian_percent=float(p_safe),
        p_safe_gaussian_ci95_low_percent=float(gauss_feas_ci[0]),
        p_safe_gaussian_ci95_high_percent=float(gauss_feas_ci[1]),
        p_ball_analytic_percent=float(p_ball_analytic),
        p_ball_mc_percent=float(p_ball_mc),
        p_ball_mc_ci95_low_percent=float(gauss_ball_ci[0]),
        p_ball_mc_ci95_high_percent=float(gauss_ball_ci[1]),
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
        input_case_path=str(ip_eff),
        slack_bus=int(slack_bus),
        n_bus=int(dc_op.n_bus),
        n_line=int(dc_op.n_line),
        dim_balance=int(d),
        n_samples=int(n_samples),
        seed=int(seed),
        chunk_size=int(chunk_size),
        sigma_mw=float(sigma_mw),
    )

    comparisons: dict[str, Any] = {
        "certificate_soundness": str(cert_interp.soundness),
        "certificate_usefulness": str(cert_interp.usefulness),
        "certificate_notes": list(cert_interp.notes),
        "gaussian_worst_max_violation_mw": float(gauss_worst_max_violation),
        "gaussian_worst_max_violation_line_pos": int(gauss_worst_line_pos),
        "gaussian_worst_max_violation_line_idx": int(gauss_worst_line_idx),
        "gaussian_worst_sample_l2": float(gauss_worst_sample_l2),
        "feas_tol_mw": float(tol_feas),
        "sigma_source": str(sigma_source),
    }

    logger.info(
        "case=%s | base=%s | r*=%.6g | d=%d | cert_soundness=%s | usefulness=%s | "
        "p_safe=%.3f%% | p_ball_analytic=%.3f%% | rho=%.6g | overall=%s",
        str(case_id),
        str(base_check.status),
        float(radius_check.r_star),
        int(d),
        str(cert_interp.soundness),
        str(cert_interp.usefulness),
        float(p_safe),
        float(p_ball_analytic),
        float(rho),
        str(overall.status),
    )

    logger.info(
        "case=%s stage=summary overall=%s reasons=%s",
        str(case_id),
        str(overall.status),
        list(overall.reasons),
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
