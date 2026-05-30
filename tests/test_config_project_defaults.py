from __future__ import annotations

import pytest
from tests.config_assertions import (
    assert_float_default_matches,
    assert_int_default_matches,
    assert_str_default_matches,
    load_root_config,
)


def test_project_yaml_opf_unconstrained_line_nom_matches_python_default() -> None:
    """
    Reproducibility regression test.

    The same default value must be used:
    - when running via YAML (CLI / experiments), and
    - when running programmatically (DEFAULT_OPF in Python).
    """
    pytest.importorskip("omegaconf")

    from stability_radius.config import DEFAULT_OPF

    assert_float_default_matches(
        section="opf",
        key="unconstrained_line_nom_mw",
        expected=DEFAULT_OPF.unconstrained_line_nom_mw,
    )


def test_project_yaml_opf_headroom_factor_matches_python_default() -> None:
    """
    Reproducibility regression test for OPF security margin.
    """
    pytest.importorskip("omegaconf")

    from stability_radius.config import DEFAULT_OPF

    assert_float_default_matches(
        section="opf", key="headroom_factor", expected=DEFAULT_OPF.headroom_factor
    )


def test_project_yaml_highs_defaults_match_python_defaults() -> None:
    """
    Reproducibility regression test for deterministic HiGHS defaults.
    """
    pytest.importorskip("omegaconf")

    from stability_radius.config import DEFAULT_OPF

    cfg = load_root_config()

    yaml_threads = int(cfg["opf"]["threads"])
    py_threads = int(DEFAULT_OPF.highs.threads)
    yaml_seed = int(cfg["opf"]["random_seed"])
    py_seed = int(DEFAULT_OPF.highs.random_seed)

    assert yaml_threads == py_threads
    assert yaml_seed == py_seed


def test_project_yaml_monte_carlo_seed_matches_python_default() -> None:
    """
    Reproducibility regression test for Monte Carlo seed default.
    """
    pytest.importorskip("omegaconf")

    from stability_radius.config import DEFAULT_MC

    cfg = load_root_config()
    assert int(cfg["monte_carlo"]["sampling"]["seed"]) == int(DEFAULT_MC.seed)


def test_project_yaml_logging_runs_dir_matches_python_default() -> None:
    pytest.importorskip("omegaconf")

    from stability_radius.config import DEFAULT_LOGGING

    assert_str_default_matches(
        section="logging", key="runs_dir", expected=DEFAULT_LOGGING.runs_dir
    )
