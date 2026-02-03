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
