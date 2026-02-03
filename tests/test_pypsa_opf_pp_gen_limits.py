from __future__ import annotations

import pytest

from stability_radius.opf.pypsa_opf import _pp_gen_p_bounds_to_pypsa


def test_pp_gen_p_bounds_to_pypsa_skips_nonpositive_pmax():
    assert _pp_gen_p_bounds_to_pypsa(gid=1, p_min_mw=0.0, p_max_mw=0.0) is None
    assert _pp_gen_p_bounds_to_pypsa(gid=2, p_min_mw=-10.0, p_max_mw=0.0) is None
    assert _pp_gen_p_bounds_to_pypsa(gid=3, p_min_mw=-10.0, p_max_mw=-5.0) is None


def test_pp_gen_p_bounds_to_pypsa_returns_params_for_positive_pmax():
    p_nom, p_min_pu = _pp_gen_p_bounds_to_pypsa(gid=1, p_min_mw=-5.0, p_max_mw=10.0)
    assert p_nom == pytest.approx(10.0)
    assert p_min_pu == pytest.approx(-0.5)

    p_nom2, p_min_pu2 = _pp_gen_p_bounds_to_pypsa(gid=2, p_min_mw=5.0, p_max_mw=10.0)
    assert p_nom2 == pytest.approx(10.0)
    assert p_min_pu2 == pytest.approx(0.5)


def test_pp_gen_p_bounds_to_pypsa_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        _pp_gen_p_bounds_to_pypsa(gid=1, p_min_mw=float("nan"), p_max_mw=10.0)

    with pytest.raises(ValueError):
        _pp_gen_p_bounds_to_pypsa(gid=2, p_min_mw=10.0, p_max_mw=float("nan"))

    with pytest.raises(ValueError):
        _pp_gen_p_bounds_to_pypsa(gid=3, p_min_mw=10.0, p_max_mw=5.0)
