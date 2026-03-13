from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from entry_points import run_scalability as mod


def test_run_scalability_writes_summary_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "pglib_sweep.yaml"
    config_path.write_text(
        """
cases:
  - name: case30
    file: case30.m
    slack_bus: 0
compute:
  base_dispatch: case
  dc:
    mode: operator
    chunk_size: 32
    dtype: float64
    inj_std_mw: 1.0
  ac:
    chunk_size: 16
    balance: true
    pf_init: flat
    pf_solver: pandapower
    lossless: true
artifacts_root: run_artifacts
scalability_output_dir: run_scalability_test
allow_download: false
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        mod,
        "load_network",
        lambda _: SimpleNamespace(
            bus=pd.DataFrame(index=[0, 1, 2]),
            line=pd.DataFrame(index=[0, 1]),
        ),
    )

    calls: list[dict[str, object]] = []

    def _fake_compute_results_for_case(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"__meta__": {"schema_version": 3}}

    monkeypatch.setattr(mod, "compute_results_for_case", _fake_compute_results_for_case)

    perf_counter_values = iter([0.0, 1.0, 10.0, 13.0, 20.0, 22.0, 30.0, 34.0])
    monkeypatch.setattr(mod.time, "perf_counter", lambda: next(perf_counter_values))

    mod.run(config_path, repeats=2)

    output_path = (
        tmp_path
        / "run_artifacts"
        / "run_scalability"
        / "run_scalability_test"
        / "scalability.json"
    )
    records = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(records) == 1
    record = records[0]
    assert record["case"] == "case30"
    assert record["n_bus"] == 3
    assert record["n_line"] == 2
    assert record["dc_time_sec_mean"] == pytest.approx(1.5)
    assert record["ac_time_sec_mean"] == pytest.approx(3.5)

    assert len(calls) == 4
    assert calls[0]["compute_dc"] is True
    assert calls[0]["compute_ac"] is False
    assert calls[1]["compute_dc"] is False
    assert calls[1]["compute_ac"] is True
