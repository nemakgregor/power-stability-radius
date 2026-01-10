from __future__ import annotations

import pytest

pp = pytest.importorskip("pandapower")


def test_load_network_fallback_parses_m_file(tmp_path, monkeypatch):
    """
    Ensure we can load a MATPOWER .m file even when pandapower's from_mpc path is unavailable.

    We force the fallback by monkeypatching stability_radius.parsers.matpower.from_mpc
    to raise NotImplementedError (similar to pandapower behavior when matpowercaseframes
    isn't installed).
    """
    from stability_radius.parsers import matpower as mp

    def _raise_not_implemented(*args, **kwargs):
        raise NotImplementedError(
            "matpowercaseframes is used to convert .m file. Please install that python package."
        )

    monkeypatch.setattr(mp, "from_mpc", _raise_not_implemented)

    case_text = """function mpc = case2
mpc.version = '2';
mpc.baseMVA = 100;

mpc.bus = [
1 3 0 0 0 0 1 1.00 0 110 1 1.05 0.95;
2 1 10 5 0 0 1 1.00 0 110 1 1.05 0.95;
];

mpc.gen = [
1 0 0 10 -10 1.00 100 1 50 0;
];

mpc.branch = [
1 2 0.01 0.05 0 100 100 100 0 0 1 -360 360;
];
"""
    path = tmp_path / "case2.m"
    path.write_text(case_text, encoding="utf-8")

    net = mp.load_network(path)
    assert len(net.bus) == 2
    # Converter should create a single branch element (usually a line).
    assert len(net.line) + len(getattr(net, "impedance", [])) >= 1
