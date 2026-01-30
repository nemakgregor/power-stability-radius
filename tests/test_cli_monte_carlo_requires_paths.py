from __future__ import annotations

from pathlib import Path

import pytest


def test_monte_carlo_parser_allows_defaults_but_runtime_validates(
    tmp_path: Path, monkeypatch
) -> None:
    """
    CLI regression: monte-carlo should allow config-driven defaults (not argparse `required=True`),
    but must fail fast with a clear error if both CLI and config are empty.
    """
    from stability_radius.cli import build_parser, run_monte_carlo

    monkeypatch.chdir(tmp_path)

    parser = build_parser(cfg=None)
    args = parser.parse_args(["--runs-dir", "runs", "--run-tests", "0", "monte-carlo"])

    with pytest.raises(ValueError, match=r"monte-carlo requires --results"):
        run_monte_carlo(
            args, cfg_loaded=None, cfg_path=tmp_path / "cfg.yaml", argv=["monte-carlo"]
        )

    # Must not create run artifacts on invalid input (fail fast).
    assert not (tmp_path / "runs").exists()
