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
