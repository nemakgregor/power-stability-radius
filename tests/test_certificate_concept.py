from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")

from stability_radius.dc.dc_model import build_dc_operator
from stability_radius.radii.common import LineBaseQuantities
from stability_radius.radii.core_l2 import row_l2_norms_projected_ones_complement
from stability_radius.radii.l2 import compute_l2_radius


def _make_meshed_net() -> Tuple[object, tuple[int, int, int, int]]:
    """
    Small deterministic meshed network (4 buses, 5 lines) for certificate concept tests.

    We intentionally keep it tiny and dependency-light:
    - no OPF
    - only DCOperator + linear algebra
    """
    net = pp.create_empty_network(sn_mva=100.0)

    b0 = int(pp.create_bus(net, vn_kv=110.0))
    b1 = int(pp.create_bus(net, vn_kv=110.0))
    b2 = int(pp.create_bus(net, vn_kv=110.0))
    b3 = int(pp.create_bus(net, vn_kv=110.0))

    common = dict(
        length_km=1.0,
        r_ohm_per_km=0.01,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
        max_loading_percent=100.0,
    )

    # Square + diagonal
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b1, x_ohm_per_km=0.10, **common
    )
    pp.create_line_from_parameters(
        net, from_bus=b1, to_bus=b2, x_ohm_per_km=0.13, **common
    )
    pp.create_line_from_parameters(
        net, from_bus=b2, to_bus=b3, x_ohm_per_km=0.11, **common
    )
    pp.create_line_from_parameters(
        net, from_bus=b3, to_bus=b0, x_ohm_per_km=0.09, **common
    )
    pp.create_line_from_parameters(
        net, from_bus=b0, to_bus=b2, x_ohm_per_km=0.20, **common
    )

    return net, (b0, b1, b2, b3)


def _base_quantities_from_linear_model(
    *, net: object, slack_bus: int, p_base: np.ndarray, limit_margin_mw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LineBaseQuantities, np.ndarray]:
    """
    Construct a synthetic "base point" and limits for testing.

    Returns
    -------
    H_full, f0, limits, base, projected_norms
    """
    dc_op = build_dc_operator(net, slack_bus=int(slack_bus))
    H_full = dc_op.materialize_H_full(dtype=np.float64, chunk_size=256)

    # Base flows for a balanced injection vector (sum=0).
    f0 = dc_op.flows_from_delta_injections(p_base).reshape(-1)
    limits = np.abs(f0) + np.asarray(limit_margin_mw, dtype=float).reshape(-1)

    idx = [int(x) for x in sorted(net.line.index)]
    base = LineBaseQuantities(
        line_indices=idx,
        flow0_mw=f0.copy(),
        p0_abs_mw=np.abs(f0),
        limit_mva_assumed_mw=limits.copy(),
        margin_mw=np.maximum(limits - np.abs(f0), 0.0),
        opf_status="test",
        opf_objective=0.0,
    )

    proj_norms = row_l2_norms_projected_ones_complement(H_full)
    return H_full, f0, limits, base, proj_norms


def test_closed_form_adversarial_boundary_test() -> None:
    """
    Test A (iron): closed-form boundary / adversarial direction test.

    We verify that:
    - r* computed from margins and projected row norms is tight
    - inside-ball perturbation stays feasible
    - slightly outside-ball perturbation violates the limiting line
    """
    net, (b0, b1, b2, b3) = _make_meshed_net()

    # Balanced base injections across all buses (sum=0).
    p_base = np.array([15.0, -5.0, -7.0, -3.0], dtype=float)
    assert p_base.shape == (4,)
    assert abs(float(np.sum(p_base))) <= 1e-12

    slack = b0

    # Make margins positive and not tiny.
    limit_margin = np.array([30.0] * len(net.line), dtype=float)

    H, f0, c, base, norms = _base_quantities_from_linear_model(
        net=net, slack_bus=slack, p_base=p_base, limit_margin_mw=limit_margin
    )

    res = compute_l2_radius(net, H, limit_factor=1.0, base=base)

    # Extract radii in stable per-line order.
    idx = base.line_indices
    radii = np.array(
        [float(res[f"line_{lid}"]["radius_l2"]) for lid in idx], dtype=float
    )
    assert radii.shape == (len(idx),)

    # Must match the certificate formula: r_i = margin_i / ||Proj(g_i)||
    margins_raw = c - np.abs(f0)
    expected_radii = np.divide(
        margins_raw,
        norms,
        out=np.full_like(margins_raw, float("inf")),
        where=norms > 1e-12,
    )
    assert np.allclose(radii, expected_radii, rtol=0.0, atol=1e-10)

    finite_mask = np.isfinite(radii)
    assert bool(np.any(finite_mask))

    argmin_pos = int(np.argmin(np.where(finite_mask, radii, float("inf"))))
    r_star = float(radii[argmin_pos])
    assert math.isfinite(r_star) and r_star > 0.0

    # Construct tight adversarial direction in the balanced subspace.
    g_full = np.asarray(H[argmin_pos, :], dtype=float).reshape(-1)
    g_proj = g_full - float(np.mean(g_full))
    norm_g = float(np.linalg.norm(g_proj, ord=2))
    assert norm_g == pytest.approx(float(norms[argmin_pos]))

    # Ensure the direction is balanced.
    assert abs(float(np.sum(g_proj))) <= 1e-9

    eps = 1e-3
    s = float(np.sign(f0[argmin_pos])) if abs(float(f0[argmin_pos])) > 1e-12 else 1.0

    dp_in = s * (r_star - eps) * (g_proj / norm_g)
    dp_out = s * (r_star + eps) * (g_proj / norm_g)

    assert abs(float(np.sum(dp_in))) <= 1e-9
    assert abs(float(np.sum(dp_out))) <= 1e-9

    flows_in = f0 + (H @ dp_in)
    assert np.all(np.abs(flows_in) <= c + 1e-8)

    flows_out = f0 + (H @ dp_out)
    assert float(np.abs(flows_out[argmin_pos])) > float(c[argmin_pos]) + 1e-6


def test_slack_invariance_for_balanced_certificate() -> None:
    """
    Test B (iron): slack invariance.

    For balanced disturbances (sum(dp)=0) and the projected-norm certificate,
    r* and the limiting lines must be independent of the chosen slack bus.
    """
    net, (b0, b1, b2, b3) = _make_meshed_net()

    p_base = np.array([15.0, -5.0, -7.0, -3.0], dtype=float)
    limit_margin = np.array([25.0] * len(net.line), dtype=float)

    # Two different slack choices.
    H0, f0_0, c0, base0, norms0 = _base_quantities_from_linear_model(
        net=net, slack_bus=b0, p_base=p_base, limit_margin_mw=limit_margin
    )
    H2, f0_2, c2, base2, norms2 = _base_quantities_from_linear_model(
        net=net, slack_bus=b2, p_base=p_base, limit_margin_mw=limit_margin
    )

    # Base flows are a physical quantity: must match.
    assert np.allclose(f0_0, f0_2, atol=1e-10, rtol=0.0)
    assert np.allclose(c0, c2, atol=1e-10, rtol=0.0)

    res0 = compute_l2_radius(net, H0, limit_factor=1.0, base=base0)
    res2 = compute_l2_radius(net, H2, limit_factor=1.0, base=base2)

    idx = base0.line_indices
    r0 = np.array([float(res0[f"line_{lid}"]["radius_l2"]) for lid in idx], dtype=float)
    r2 = np.array([float(res2[f"line_{lid}"]["radius_l2"]) for lid in idx], dtype=float)

    assert np.allclose(r0, r2, atol=1e-10, rtol=0.0)

    # Top-K limiting lines (by radius) must match in positions.
    k = min(10, len(idx))
    top0 = np.argsort(r0)[:k].tolist()
    top2 = np.argsort(r2)[:k].tolist()
    assert top0 == top2

    # H matrices must map balanced injections consistently.
    rng = np.random.default_rng(0)
    dp = rng.standard_normal(size=(H0.shape[1],)).astype(float, copy=False)
    dp -= float(np.mean(dp))  # enforce sum(dp)=0
    assert abs(float(np.sum(dp))) <= 1e-9

    df0 = H0 @ dp
    df2 = H2 @ dp
    assert np.allclose(df0, df2, atol=1e-9, rtol=0.0)


def test_scaling_monotonicity_for_limits() -> None:
    """
    Test C (iron): scaling monotonicity.

    Increasing limits must not decrease r*. Decreasing limits must not increase r*,
    as long as the base point remains feasible.
    """
    net, (b0, b1, b2, b3) = _make_meshed_net()

    p_base = np.array([15.0, -5.0, -7.0, -3.0], dtype=float)
    slack = b0

    # Make limits generous so scaling down by 0.9 remains feasible.
    limit_margin = np.array([100.0] * len(net.line), dtype=float)

    H, f0, c, base, norms = _base_quantities_from_linear_model(
        net=net, slack_bus=slack, p_base=p_base, limit_margin_mw=limit_margin
    )

    margins_raw = c - np.abs(f0)
    assert np.all(margins_raw > 0.0)

    def r_star_for_limits(alpha: float) -> float:
        c_new = float(alpha) * c
        margins_new = c_new - np.abs(f0)
        radii_new = np.divide(
            margins_new,
            norms,
            out=np.full_like(margins_new, float("inf")),
            where=norms > 1e-12,
        )
        finite = np.isfinite(radii_new)
        assert bool(np.any(finite))
        return float(np.min(radii_new[finite]))

    r0 = r_star_for_limits(1.0)
    r_up = r_star_for_limits(1.1)
    r_dn = r_star_for_limits(0.9)

    assert r_up >= r0 - 1e-12
    assert r_dn <= r0 + 1e-12
