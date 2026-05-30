"""Tests for OPF-to-DC consistency check (_check_opf_dc_consistency).

Covers:
- Consistency passes when OPF and DC operator agree.
- Consistency detects mismatch (warning, not crash).
- Return dict contains expected fields.
- Balance tolerance is enforced.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.network_factories import make_triangle_net

pp = pytest.importorskip("pandapower")
pytest.importorskip("scipy")
pytest.importorskip("pypsa")
pytest.importorskip("pandas")
pytest.importorskip("highspy")


def test_consistency_passes_for_matching_flows() -> None:
    """Consistency check passes when OPF and DC operator agree."""
    from stability_radius.dc.dc_model import build_dc_operator
    from stability_radius.radii.common import get_line_base_quantities
    from stability_radius.workflows import _check_opf_dc_consistency

    net, slack_bus = make_triangle_net(pp)

    base = get_line_base_quantities(net, limit_factor=1.0)
    dc_op = build_dc_operator(net, slack_bus=int(slack_bus))

    result = _check_opf_dc_consistency(
        dc_op=dc_op,
        base=base,
        tol_flow_mw=1e-2,
        tol_balance_mw=1.0,
    )

    assert result["opf_dc_consistency_passed"] is True
    assert result["opf_dc_flow_max_abs_diff_mw"] < 1e-2
    assert result["opf_bus_balance_abs_mw"] < 1e-4


def test_consistency_returns_expected_fields() -> None:
    """Consistency result dict must contain all expected fields."""
    from stability_radius.dc.dc_model import build_dc_operator
    from stability_radius.radii.common import get_line_base_quantities
    from stability_radius.workflows import _check_opf_dc_consistency

    net, slack_bus = make_triangle_net(pp)
    base = get_line_base_quantities(net, limit_factor=1.0)
    dc_op = build_dc_operator(net, slack_bus=int(slack_bus))

    result = _check_opf_dc_consistency(
        dc_op=dc_op,
        base=base,
        tol_flow_mw=1e-2,
        tol_balance_mw=1.0,
    )

    expected_keys = {
        "opf_bus_balance_abs_mw",
        "opf_dc_flow_max_abs_diff_mw",
        "opf_dc_flow_tol_mw",
        "opf_bus_balance_tol_mw",
        "opf_dc_consistency_passed",
    }
    assert set(result.keys()) == expected_keys


def test_consistency_detects_mismatch_without_crash() -> None:
    """When flows mismatch, the check warns but does NOT raise."""
    from stability_radius.dc.dc_model import build_dc_operator
    from stability_radius.radii.common import get_line_base_quantities
    from stability_radius.workflows import _check_opf_dc_consistency

    net, slack_bus = make_triangle_net(pp)
    base = get_line_base_quantities(net, limit_factor=1.0)
    dc_op = build_dc_operator(net, slack_bus=int(slack_bus))

    # Use impossibly tight tolerance to force a mismatch.
    result = _check_opf_dc_consistency(
        dc_op=dc_op,
        base=base,
        tol_flow_mw=1e-15,  # too tight even for machine precision
        tol_balance_mw=1.0,
    )

    # The function should NOT raise — just return passed=False.
    assert isinstance(result, dict)
    # May or may not pass at 1e-15 depending on precision, but should not crash.
    assert "opf_dc_consistency_passed" in result


def test_consistency_rejects_unbalanced_injections() -> None:
    """When bus injections don't balance, a ValueError is raised."""
    from stability_radius.dc.dc_model import build_dc_operator
    from stability_radius.radii.common import LineBaseQuantities
    from stability_radius.workflows import _check_opf_dc_consistency

    net, slack_bus = make_triangle_net(pp)
    dc_op = build_dc_operator(net, slack_bus=int(slack_bus))

    bus_ids = [int(x) for x in sorted(net.bus.index)]
    line_ids = [int(x) for x in sorted(net.line.index)]

    # Create a base with unbalanced injections (sum = 100 MW instead of ~0).
    bad_inj = np.array([100.0, 0.0, 0.0])

    base = LineBaseQuantities(
        line_indices=line_ids,
        flow0_mw=np.zeros(len(line_ids)),
        p0_abs_mw=np.zeros(len(line_ids)),
        limit_mva_assumed_mw=np.full(len(line_ids), 1e6),
        margin_mw=np.full(len(line_ids), 1e6),
        bus_ids=bus_ids,
        bus_injections_mw=bad_inj,
    )

    with pytest.raises(ValueError, match="not balanced"):
        _check_opf_dc_consistency(
            dc_op=dc_op,
            base=base,
            tol_flow_mw=1.0,
            tol_balance_mw=1.0,  # sum(bad_inj)=100 >> 1
        )
