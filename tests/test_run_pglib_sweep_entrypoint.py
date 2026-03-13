from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from entry_points import run_pglib_sweep as mod
from stability_radius.config import OPFConfig


def test_detect_slack_bus_prefers_in_service_ext_grid() -> None:
    net = SimpleNamespace(
        bus=pd.DataFrame(index=[0, 1, 2]),
        ext_grid=pd.DataFrame(
            {
                "bus": [2, 1],
                "in_service": [False, True],
            }
        ),
    )

    assert mod._detect_slack_bus(net) == 1


def test_detect_slack_bus_falls_back_to_smallest_bus_index() -> None:
    net = SimpleNamespace(
        bus=pd.DataFrame(index=[5, 3, 7]),
        ext_grid=pd.DataFrame(columns=["bus", "in_service"]),
    )

    assert mod._detect_slack_bus(net) == 3


def test_compute_case_enables_shared_dc_and_ac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_compute_results_for_case(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"__meta__": {"schema_version": 3}}

    monkeypatch.setattr(mod, "compute_results_for_case", _fake_compute_results_for_case)

    result = mod._compute_case(
        input_path="data/input/pglib_opf_case30_ieee.m",
        slack_bus=0,
        base_dispatch="case",
        dc_cfg={
            "mode": "operator",
            "chunk_size": 16,
            "dtype": "float64",
            "inj_std_mw": 1.0,
        },
        ac_cfg={
            "chunk_size": 32,
            "balance": True,
            "pf_init": "flat",
            "pf_solver": "pandapower",
            "lossless": True,
        },
        ac_fpf_cfg={},
        opf_cfg=OPFConfig(),
        allow_download=False,
        opf_dc_flow_consistency_tol_mw=1e-3,
    )

    assert result["__meta__"]["schema_version"] == 3
    assert captured["compute_dc"] is True
    assert captured["compute_ac"] is True
    assert captured["input_path"] == "data/input/pglib_opf_case30_ieee.m"
    assert captured["base_dispatch"] == "case"
    assert captured["allow_download"] is False
    dc_ext = captured["dc_extensions"]
    assert dc_ext.probabilistic_enabled is True
