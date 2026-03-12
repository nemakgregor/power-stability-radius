from __future__ import annotations

from pathlib import Path

import pytest


def test_project_yaml_opf_unconstrained_line_nom_matches_python_default() -> None:
    """
    Reproducibility regression test.

    The same default value must be used:
    - when running via YAML (CLI / experiments), and
    - when running programmatically (DEFAULT_OPF in Python).
    """
    pytest.importorskip("omegaconf")

    from stability_radius.config import DEFAULT_OPF, load_project_config

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "conf" / "config.yaml"

    cfg = load_project_config(cfg_path, allow_missing=False)
    assert cfg is not None

    yaml_val = float(cfg["opf"]["unconstrained_line_nom_mw"])
    py_val = float(DEFAULT_OPF.unconstrained_line_nom_mw)

    assert yaml_val == pytest.approx(py_val)


def test_project_yaml_opf_headroom_factor_matches_python_default() -> None:
    """
    Reproducibility regression test for OPF security margin.
    """
    pytest.importorskip("omegaconf")

    from stability_radius.config import DEFAULT_OPF, load_project_config

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "conf" / "config.yaml"

    cfg = load_project_config(cfg_path, allow_missing=False)
    assert cfg is not None

    yaml_val = float(cfg["opf"]["headroom_factor"])
    py_val = float(DEFAULT_OPF.headroom_factor)

    assert yaml_val == pytest.approx(py_val)


def test_project_yaml_highs_defaults_match_python_defaults() -> None:
    """
    Reproducibility regression test for deterministic HiGHS defaults.
    """
    pytest.importorskip("omegaconf")

    from stability_radius.config import DEFAULT_OPF, load_project_config

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "conf" / "config.yaml"

    cfg = load_project_config(cfg_path, allow_missing=False)
    assert cfg is not None

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

    from stability_radius.config import DEFAULT_MC, load_project_config

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "conf" / "config.yaml"

    cfg = load_project_config(cfg_path, allow_missing=False)
    assert cfg is not None

    yaml_val = int(cfg["monte_carlo"]["sampling"]["seed"])
    py_val = int(DEFAULT_MC.seed)

    assert yaml_val == py_val


def test_project_yaml_logging_runs_dir_matches_python_default() -> None:
    pytest.importorskip("omegaconf")

    from stability_radius.config import DEFAULT_LOGGING, load_project_config

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "conf" / "config.yaml"

    cfg = load_project_config(cfg_path, allow_missing=False)
    assert cfg is not None

    yaml_val = str(cfg["logging"]["runs_dir"])
    py_val = str(DEFAULT_LOGGING.runs_dir)

    assert yaml_val == py_val
