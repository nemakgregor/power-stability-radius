"""Tests for the UnitCommitment.jl JSON parser."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from stability_radius.parsers import uc_jl


# ---------------------------------------------------------------------------
# Fixture: minimal UC.jl instance (3 buses, 24 hours)
# ---------------------------------------------------------------------------


def _make_uc_fixture() -> dict:
    """Build a minimal UC.jl JSON dict with 3 buses and 24 timesteps."""
    np.random.seed(42)
    hours = 24

    # Deterministic load profiles (sinusoidal + noise-free for reproducibility)
    load_b1 = [100 + 20 * math.sin(2 * math.pi * t / 24) for t in range(hours)]
    load_b2 = [50 + 10 * math.sin(2 * math.pi * t / 24) for t in range(hours)]
    load_b3 = [30.0] * hours  # constant load -> σ = 0

    # Time-varying generator capacity on bus b1
    gen1_cap = [200 + 15 * math.cos(2 * math.pi * t / 24) for t in range(hours)]

    # Scalar (constant) generator on bus b2
    gen2_cap = 80.0

    return {
        "Buses": {
            "b1": {"Load (MW)": load_b1},
            "b2": {"Load (MW)": load_b2},
            "b3": {"Load (MW)": load_b3},
        },
        "Generators": {
            "gen1": {
                "Bus": "b1",
                "Max power (MW)": gen1_cap,
                "Min power (MW)": [50.0] * hours,
            },
            "gen2": {
                "Bus": "b2",
                "Max power (MW)": gen2_cap,
                "Min power (MW)": 20.0,
            },
        },
        "Transmission lines": {
            "l1": {
                "Source bus": "b1",
                "Target bus": "b2",
                "Reactance (ohms)": 0.05,
            },
            "l2": {
                "Source bus": "b2",
                "Target bus": "b3",
                "Reactance (ohms)": 0.08,
            },
        },
    }


@pytest.fixture()
def uc_json_path(tmp_path):
    """Write the UC.jl fixture to a temporary JSON file and return the path."""
    data = _make_uc_fixture()
    path = tmp_path / "uc_instance.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadSigma:
    """Core extraction behaviour."""

    def test_returns_expected_keys(self, uc_json_path):
        result = uc_jl.load_sigma(uc_json_path)
        assert set(result.keys()) == {
            "sigma_p_mw",
            "sigma_q_mvar",
            "n_timesteps",
            "bus_mapping",
            "metadata",
        }

    def test_array_shapes(self, uc_json_path):
        result = uc_jl.load_sigma(uc_json_path)
        assert result["sigma_p_mw"].shape == (3,)
        assert result["sigma_q_mvar"].shape == (3,)

    def test_n_timesteps(self, uc_json_path):
        result = uc_jl.load_sigma(uc_json_path)
        assert result["n_timesteps"] == 24

    def test_sigma_load_values(self, uc_json_path):
        """Verify σ_load is the population std of the load time series."""
        data = _make_uc_fixture()
        result = uc_jl.load_sigma(uc_json_path)
        mapping = result["bus_mapping"]

        expected_b1 = float(np.std(data["Buses"]["b1"]["Load (MW)"], ddof=0))
        expected_b2 = float(np.std(data["Buses"]["b2"]["Load (MW)"], ddof=0))
        expected_b3 = 0.0  # constant

        # b1 also has generator variance, so check σ_load component separately
        # by creating an instance with no generators
        data_no_gen = _make_uc_fixture()
        data_no_gen["Generators"] = {}
        path_no_gen = uc_json_path.parent / "no_gen.json"
        path_no_gen.write_text(json.dumps(data_no_gen), encoding="utf-8")
        result_no_gen = uc_jl.load_sigma(path_no_gen)
        m2 = result_no_gen["bus_mapping"]

        assert result_no_gen["sigma_p_mw"][m2["b1"]] == pytest.approx(
            expected_b1, rel=1e-6
        )
        assert result_no_gen["sigma_p_mw"][m2["b2"]] == pytest.approx(
            expected_b2, rel=1e-6
        )
        assert result_no_gen["sigma_p_mw"][m2["b3"]] == pytest.approx(
            expected_b3, abs=1e-12
        )

    def test_sigma_gen_accumulation(self, uc_json_path):
        """σ_P should combine load σ and generator capacity σ via RSS."""
        data = _make_uc_fixture()
        result = uc_jl.load_sigma(uc_json_path)
        mapping = result["bus_mapping"]

        sigma_load_b1 = float(np.std(data["Buses"]["b1"]["Load (MW)"], ddof=0))
        sigma_gen_b1 = float(
            np.std(data["Generators"]["gen1"]["Max power (MW)"], ddof=0)
        )
        expected_p_b1 = math.sqrt(sigma_load_b1**2 + sigma_gen_b1**2)

        assert result["sigma_p_mw"][mapping["b1"]] == pytest.approx(
            expected_p_b1, rel=1e-6
        )

    def test_constant_load_yields_zero_sigma(self, uc_json_path):
        result = uc_jl.load_sigma(uc_json_path)
        mapping = result["bus_mapping"]
        # b3 has constant load and no generator -> σ = 0
        assert result["sigma_p_mw"][mapping["b3"]] == pytest.approx(0.0, abs=1e-12)
        assert result["sigma_q_mvar"][mapping["b3"]] == pytest.approx(0.0, abs=1e-12)

    def test_scalar_gen_no_contribution(self, uc_json_path):
        """A scalar (constant) generator capacity should not add variance."""
        data = _make_uc_fixture()
        result = uc_jl.load_sigma(uc_json_path)
        mapping = result["bus_mapping"]

        # b2 only has a constant-capacity generator, so σ_P == σ_load only
        sigma_load_b2 = float(np.std(data["Buses"]["b2"]["Load (MW)"], ddof=0))
        assert result["sigma_p_mw"][mapping["b2"]] == pytest.approx(
            sigma_load_b2, rel=1e-6
        )

    def test_sigma_q_power_factor(self, uc_json_path):
        """σ_Q = σ_P * tan(arccos(pf))."""
        pf = 0.9
        result = uc_jl.load_sigma(uc_json_path, power_factor=pf)
        tan_phi = math.tan(math.acos(pf))
        np.testing.assert_allclose(
            result["sigma_q_mvar"],
            result["sigma_p_mw"] * tan_phi,
            atol=1e-12,
        )

    def test_custom_power_factor(self, uc_json_path):
        """Changing the power factor should change σ_Q proportionally."""
        r1 = uc_jl.load_sigma(uc_json_path, power_factor=0.85)
        r2 = uc_jl.load_sigma(uc_json_path, power_factor=0.95)
        # pf=0.85 -> larger tan(phi) -> larger σ_Q
        assert np.all(r1["sigma_q_mvar"] >= r2["sigma_q_mvar"])


class TestBusMapping:
    """Bus-name-to-index mapping behaviour."""

    def test_inferred_mapping_is_sorted(self, uc_json_path):
        result = uc_jl.load_sigma(uc_json_path)
        mapping = result["bus_mapping"]
        assert mapping == {"b1": 0, "b2": 1, "b3": 2}

    def test_explicit_mapping(self, uc_json_path):
        custom = {"b1": 2, "b2": 0, "b3": 1}
        result = uc_jl.load_sigma(uc_json_path, bus_mapping=custom)
        assert result["bus_mapping"] == custom
        # b3 (constant load, no gen) is now at index 1
        assert result["sigma_p_mw"][1] == pytest.approx(0.0, abs=1e-12)

    def test_incomplete_explicit_mapping_raises(self, uc_json_path):
        with pytest.raises(ValueError, match="missing entries"):
            uc_jl.load_sigma(uc_json_path, bus_mapping={"b1": 0})


class TestErrorHandling:
    """File-level error paths."""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            uc_jl.load_sigma(tmp_path / "nonexistent.json")

    def test_wrong_extension_raises(self, tmp_path):
        path = tmp_path / "case.csv"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match=r"\.json"):
            uc_jl.load_sigma(path)

    def test_no_buses_raises(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"Generators": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="No 'Buses'"):
            uc_jl.load_sigma(path)


class TestMetadata:
    """Verify metadata dict content."""

    def test_metadata_fields(self, uc_json_path):
        result = uc_jl.load_sigma(uc_json_path)
        meta = result["metadata"]
        assert meta["n_buses"] == 3
        assert meta["n_generators"] == 2
        assert meta["power_factor"] == 0.9
        assert "source" in meta
