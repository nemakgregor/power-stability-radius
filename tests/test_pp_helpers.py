"""Tests for stability_radius.pp_helpers — shared pandapower-like table helpers."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

from stability_radius.pp_helpers import bus_vn_kv, is_in_service, resolve_slack_pos


# ---------------------------------------------------------------------------
# is_in_service
# ---------------------------------------------------------------------------


class TestIsInService:
    """Contract: returns bool from row["in_service"] via .get() or []."""

    def test_dict_row_true(self):
        assert is_in_service({"in_service": True}) is True

    def test_dict_row_false(self):
        assert is_in_service({"in_service": False}) is False

    def test_dict_row_missing_uses_default_true(self):
        assert is_in_service({}) is True

    def test_dict_row_missing_explicit_default_false(self):
        assert is_in_service({}, default=False) is False

    def test_pandas_series_row(self):
        row = pd.Series({"in_service": True, "p_mw": 10.0})
        assert is_in_service(row) is True

    def test_pandas_series_false(self):
        row = pd.Series({"in_service": False, "p_mw": 10.0})
        assert is_in_service(row) is False

    def test_object_without_get_falls_back_to_getitem(self):
        """Row that has no .get() but supports []."""

        class DictLike:
            def __getitem__(self, key):
                if key == "in_service":
                    return False
                raise KeyError(key)

        assert is_in_service(DictLike()) is False

    def test_object_with_neither_get_nor_getitem_uses_default(self):
        """Row that supports neither .get() nor [] returns the default."""
        assert is_in_service(42) is True
        assert is_in_service(42, default=False) is False


# ---------------------------------------------------------------------------
# bus_vn_kv
# ---------------------------------------------------------------------------


class TestBusVnKv:
    """Contract: returns float vn_kv or NaN on any failure."""

    @staticmethod
    def _make_net(bus_data: dict | None = None) -> SimpleNamespace:
        if bus_data is None:
            return SimpleNamespace(bus=None)
        df = pd.DataFrame(bus_data)
        if "bus_id" in bus_data:
            df = df.set_index("bus_id")
        return SimpleNamespace(bus=df)

    def test_returns_vn_kv_for_valid_bus(self):
        net = self._make_net({"bus_id": [0, 1], "vn_kv": [110.0, 220.0]})
        assert bus_vn_kv(net, 0) == 110.0
        assert bus_vn_kv(net, 1) == 220.0

    def test_returns_nan_for_missing_bus_id(self):
        net = self._make_net({"bus_id": [0], "vn_kv": [110.0]})
        assert math.isnan(bus_vn_kv(net, 99))

    def test_returns_nan_for_missing_vn_kv_column(self):
        net = self._make_net({"bus_id": [0], "name": ["bus0"]})
        assert math.isnan(bus_vn_kv(net, 0))

    def test_returns_nan_for_empty_bus_table(self):
        net = SimpleNamespace(bus=pd.DataFrame(columns=["vn_kv"]))
        assert math.isnan(bus_vn_kv(net, 0))

    def test_returns_nan_for_none_bus_table(self):
        net = SimpleNamespace(bus=None)
        assert math.isnan(bus_vn_kv(net, 0))

    def test_returns_nan_for_net_without_bus_attribute(self):
        net = SimpleNamespace()
        assert math.isnan(bus_vn_kv(net, 0))


# ---------------------------------------------------------------------------
# resolve_slack_pos
# ---------------------------------------------------------------------------


class TestResolveSlackPos:
    """Contract: resolves bus id or positional index; raises ValueError otherwise."""

    def test_resolves_bus_id_to_position(self):
        bus_ids = [10, 20, 30]
        assert resolve_slack_pos(bus_ids, 20) == 1

    def test_resolves_first_bus_id(self):
        bus_ids = [10, 20, 30]
        assert resolve_slack_pos(bus_ids, 10) == 0

    def test_resolves_positional_index_when_not_a_bus_id(self):
        # bus_ids = [100, 200, 300] — positions 0,1,2 are not valid bus ids
        bus_ids = [100, 200, 300]
        assert resolve_slack_pos(bus_ids, 1) == 1

    def test_bus_id_takes_precedence_over_position(self):
        # bus_ids = [0, 1, 2] — slack_bus=1 is both a bus id and a position
        # By contract, bus id match takes precedence.
        bus_ids = [0, 1, 2]
        assert resolve_slack_pos(bus_ids, 1) == 1

    def test_raises_for_invalid_id_and_out_of_range_position(self):
        bus_ids = [10, 20, 30]
        with pytest.raises(ValueError, match="slack_bus must be"):
            resolve_slack_pos(bus_ids, 999)

    def test_raises_for_negative_position(self):
        bus_ids = [10, 20, 30]
        with pytest.raises(ValueError, match="slack_bus must be"):
            resolve_slack_pos(bus_ids, -1)
