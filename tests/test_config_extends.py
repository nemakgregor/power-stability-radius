from __future__ import annotations

from pathlib import Path

import pytest


def test_load_project_config_supports_extends(tmp_path: Path) -> None:
    """
    Ensure minimal `extends:` inheritance works and is resolved relative to the child file.
    """
    pytest.importorskip("omegaconf")

    from stability_radius.config import load_project_config

    base = tmp_path / "base.yaml"
    base.write_text(
        "\n".join(
            [
                "a: 1",
                "b:",
                "  c: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    child_dir = tmp_path / "child"
    child_dir.mkdir(parents=True, exist_ok=True)

    child = child_dir / "child.yaml"
    child.write_text(
        "\n".join(
            [
                "extends: ../base.yaml",
                "b:",
                "  c: 3",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_project_config(child, allow_missing=False)

    assert cfg is not None
    assert int(cfg["a"]) == 1
    assert int(cfg["b"]["c"]) == 3


def test_full_yaml_chain_opf_unconstrained_line_nom_matches_python_default() -> None:
    """
    Reproducibility regression test (YAML chain vs Python defaults).

    Contract
    --------
    The composed YAML chain (conf/config.yaml + all extends) must produce the same
    default value for opf.unconstrained_line_nom_mw as stability_radius.config.DEFAULT_OPF.

    Rationale
    ---------
    Otherwise:
    - CLI runs (YAML) and
    - programmatic runs/tests (Python defaults)
    can silently use different surrogate thermal limits for "unconstrained" lines.
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
