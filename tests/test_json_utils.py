from __future__ import annotations

import json

import numpy as np
import pytest

from stability_radius.utils import NumpyJSONEncoder, numpy_to_builtin


def test_numpy_to_builtin_converts_scalars_and_arrays() -> None:
    assert numpy_to_builtin(np.int64(7)) == 7
    assert numpy_to_builtin(np.float64(1.5)) == pytest.approx(1.5)
    assert numpy_to_builtin(np.array([1, 2, 3], dtype=np.int64)) == [1, 2, 3]


def test_numpy_to_builtin_raises_for_unsupported_types() -> None:
    with pytest.raises(TypeError):
        numpy_to_builtin({"k": "v"})


def test_numpy_json_encoder_serializes_nested_numpy_values() -> None:
    payload = {
        "i": np.int64(3),
        "x": np.array([0.5, 1.5], dtype=np.float64),
    }
    text = json.dumps(payload, cls=NumpyJSONEncoder)
    assert '"i": 3' in text
    assert '"x": [0.5, 1.5]' in text
