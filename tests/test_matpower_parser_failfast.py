from __future__ import annotations

import pytest

pp = pytest.importorskip("pandapower")


def test_load_network_parses_m_file(tmp_path):
    """
    Ensure we can load a MATPOWER .m file using the internal deterministic parser.
    Also ensure MATPOWER branch rateA is propagated for downstream thermal limit extraction.
    """
    from stability_radius.parsers import matpower as mp

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

    if len(net.line) >= 1:
        assert "rateA" in net.line.columns
        lid = int(sorted(net.line.index)[0])
        assert float(net.line.loc[lid, "rateA"]) == pytest.approx(100.0)
