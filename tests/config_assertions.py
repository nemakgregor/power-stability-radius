from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_root_config() -> Any:
    """Load the composed project config from conf/config.yaml."""
    pytest.importorskip("omegaconf")

    from stability_radius.config import load_project_config

    cfg = load_project_config(ROOT / "conf" / "config.yaml", allow_missing=False)
    assert cfg is not None
    return cfg


def assert_float_default_matches(
    *, section: str, key: str, expected: float, rel: float | None = None
) -> None:
    """Assert a float YAML default matches the Python default."""
    cfg = load_root_config()
    actual = float(cfg[section][key])
    assert actual == pytest.approx(float(expected), rel=rel)


def assert_int_default_matches(*, section: str, key: str, expected: int) -> None:
    """Assert an integer YAML default matches the Python default."""
    cfg = load_root_config()
    assert int(cfg[section][key]) == int(expected)


def assert_str_default_matches(*, section: str, key: str, expected: str) -> None:
    """Assert a string YAML default matches the Python default."""
    cfg = load_root_config()
    assert str(cfg[section][key]) == str(expected)
