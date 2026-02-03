from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pytest


@dataclass
class _FakeResponse:
    text: str
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.calls: List[Tuple[str, float]] = []

    def get(self, url: str, timeout: float):
        self.calls.append((url, timeout))
        return _FakeResponse(self._response_text)


class _FailFirstThenSucceedSession:
    """
    Fake session used to validate deterministic URL fallback:
    - First URL that contains "raw.githubusercontent.com" fails.
    - Second candidate succeeds.
    """

    def __init__(self, response_text: str):
        self._response_text = response_text
        self.calls: List[Tuple[str, float]] = []

    def get(self, url: str, timeout: float):
        self.calls.append((url, timeout))
        if "raw.githubusercontent.com" in url:
            raise RuntimeError("Simulated DNS failure for raw.githubusercontent.com")
        return _FakeResponse(self._response_text)


def test_download_ieee_case_validates_case_number():
    from stability_radius.utils.download import download_ieee_case

    with pytest.raises(ValueError):
        download_ieee_case("not-a-number", target_path="x.m")

    with pytest.raises(ValueError):
        download_ieee_case(0, target_path="x.m")


def test_download_ieee_case_writes_file(tmp_path):
    from stability_radius.utils.download import download_ieee_case

    target = tmp_path / "ieee14.m"
    fake = _FakeSession("case14-content")

    path = download_ieee_case(
        14, target_path=str(target), session=fake, base_url="http://example.com"
    )
    assert path == str(target)
    assert target.read_text(encoding="utf-8") == "case14-content"
    assert fake.calls == [("http://example.com/case14.m", 15.0)]


def test_download_ieee_case_skips_when_exists(tmp_path):
    from stability_radius.utils.download import download_ieee_case

    target = tmp_path / "ieee30.m"
    target.write_text("existing", encoding="utf-8")

    fake = _FakeSession("new")
    path = download_ieee_case(30, target_path=str(target), session=fake)
    assert path == str(target)
    assert target.read_text(encoding="utf-8") == "existing"
    assert fake.calls == []


def test_download_pglib_opf_case_falls_back_from_raw_to_github(tmp_path):
    from stability_radius.utils.download import download_pglib_opf_case

    target = tmp_path / "pglib_opf_case30_ieee.m"
    fake = _FailFirstThenSucceedSession("pglib-case-content")

    path = download_pglib_opf_case(
        case_filename="pglib_opf_case30_ieee.m",
        target_path=str(target),
        session=fake,
        base_url="https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master",
    )
    assert path == str(target)
    assert target.read_text(encoding="utf-8") == "pglib-case-content"

    # Must attempt raw first, then github raw.
    assert len(fake.calls) == 2
    assert "raw.githubusercontent.com" in fake.calls[0][0]
    assert "github.com" in fake.calls[1][0]
