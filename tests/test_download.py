from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
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
    Fake session used to validate deterministic URL candidate order:
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


# ---------------------------------------------------------------------------
# UC.jl download tests
# ---------------------------------------------------------------------------


@dataclass
class _FakeBinaryResponse:
    """Fake response carrying raw bytes (for gzip-compressed payloads)."""

    content: bytes
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeBinarySession:
    """Fake session that returns gzip-compressed bytes."""

    def __init__(self, payload_bytes: bytes):
        self._payload = payload_bytes
        self.calls: List[Tuple[str, float]] = []

    def get(self, url: str, timeout: float):
        self.calls.append((url, timeout))
        return _FakeBinaryResponse(content=self._payload)


class _FailingBinarySession:
    """Fake session that always raises on GET."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, float]] = []

    def get(self, url: str, timeout: float):
        self.calls.append((url, timeout))
        raise RuntimeError("Simulated network failure")


def _make_gzipped_json(data: dict) -> bytes:
    """Produce gzip-compressed JSON bytes from a dict."""
    return gzip.compress(json.dumps(data).encode("utf-8"))


def test_download_uc_jl_instance_writes_decompressed_json(tmp_path):
    from stability_radius.utils.download import download_uc_jl_instance

    payload = {"Buses": {"b1": {"Load (MW)": [100, 110]}}}
    gz_bytes = _make_gzipped_json(payload)
    fake = _FakeBinarySession(gz_bytes)

    result = download_uc_jl_instance(
        "case14",
        tmp_path,
        session=fake,
        base_url="https://example.com/uc",
    )

    assert result == tmp_path / "case14.json"
    assert result.exists()

    written = json.loads(result.read_text(encoding="utf-8"))
    assert written == payload

    assert len(fake.calls) == 1
    url, _ = fake.calls[0]
    assert url == "https://example.com/uc/case14/2017-01-01.json.gz"


def test_download_uc_jl_instance_custom_date(tmp_path):
    from stability_radius.utils.download import download_uc_jl_instance

    payload = {"Buses": {}}
    gz_bytes = _make_gzipped_json(payload)
    fake = _FakeBinarySession(gz_bytes)

    download_uc_jl_instance(
        "case30",
        tmp_path,
        date="2018-06-15",
        session=fake,
        base_url="https://example.com/uc",
    )

    url, _ = fake.calls[0]
    assert "2018-06-15.json.gz" in url


def test_download_uc_jl_instance_skips_when_exists(tmp_path):
    from stability_radius.utils.download import download_uc_jl_instance

    target = tmp_path / "case14.json"
    target.write_text('{"existing": true}', encoding="utf-8")

    fake = _FakeBinarySession(b"should-not-be-used")
    result = download_uc_jl_instance("case14", tmp_path, session=fake)

    assert result == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"existing": True}
    assert fake.calls == []


def test_download_uc_jl_instance_overwrites_when_requested(tmp_path):
    from stability_radius.utils.download import download_uc_jl_instance

    target = tmp_path / "case14.json"
    target.write_text('{"old": true}', encoding="utf-8")

    new_payload = {"new": True}
    fake = _FakeBinarySession(_make_gzipped_json(new_payload))

    result = download_uc_jl_instance(
        "case14",
        tmp_path,
        overwrite=True,
        session=fake,
        base_url="https://example.com/uc",
    )

    assert result == target
    assert json.loads(target.read_text(encoding="utf-8")) == new_payload
    assert len(fake.calls) == 1


def test_download_uc_jl_instance_raises_on_failure(tmp_path):
    from stability_radius.utils.download import DownloadError, download_uc_jl_instance

    fake = _FailingBinarySession()

    with pytest.raises(DownloadError, match="case14"):
        download_uc_jl_instance(
            "case14",
            tmp_path,
            session=fake,
            base_url="https://example.com/uc",
        )

    assert len(fake.calls) == 1


def test_download_uc_jl_instance_validates_case_name():
    from stability_radius.utils.download import download_uc_jl_instance

    with pytest.raises(ValueError, match="non-empty"):
        download_uc_jl_instance("", Path("/tmp"))


def test_download_uc_jl_instance_creates_dest_dir(tmp_path):
    from stability_radius.utils.download import download_uc_jl_instance

    nested = tmp_path / "a" / "b" / "c"
    payload = {"Buses": {"b1": {}}}
    fake = _FakeBinarySession(_make_gzipped_json(payload))

    result = download_uc_jl_instance(
        "case14",
        nested,
        session=fake,
        base_url="https://example.com/uc",
    )

    assert result.exists()
    assert nested.is_dir()
