from __future__ import annotations

from pathlib import Path

import pytest


def test_parse_report_cases_accepts_omegaconf_listconfig(tmp_path: Path) -> None:
    """
    Regression test for CLI config parsing:

    OmegaConf uses ListConfig/DictConfig (not builtin list/dict). The CLI must accept
    such containers for `report.cases`, otherwise `report` fails with:
        report.cases must be a non-empty list ...
    """
    pytest.importorskip("omegaconf")

    from omegaconf import OmegaConf

    from stability_radius.cli import _parse_report_cases_from_cfg

    cfg = OmegaConf.create(
        {
            "report": {
                "cases": [
                    {
                        "id": "caseA",
                        "input": "data/input/caseA.m",
                        "results": "caseA.json",
                        "known_critical_pairs": [[1, "2"]],
                    }
                ]
            }
        }
    )

    results_dir_abs = (tmp_path / "verification" / "results").resolve()
    base_dir = tmp_path.resolve()

    cases = _parse_report_cases_from_cfg(
        cfg_loaded=cfg, results_dir_abs=results_dir_abs, base_dir=base_dir
    )

    assert len(cases) == 1
    c0 = cases[0]

    assert c0["id"] == "caseA"
    assert Path(c0["results"]).parent == results_dir_abs
    assert Path(c0["results"]).name == "caseA.json"
    assert str(Path(c0["input"])).endswith(str(Path("data/input/caseA.m")))
    assert c0["known_critical_pairs"] == [[1, 2]]
