from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_license_file_and_pyproject_metadata_present() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "MIT License" in license_text
    assert 'license = "MIT"' in pyproject_text
    assert 'readme = "README.md"' in pyproject_text


def test_citation_file_present() -> None:
    citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    assert "cff-version: 1.2.0" in citation_text
    assert 'title: "Power Stability Radius"' in citation_text
    assert "type: software" in citation_text
    assert 'license: "MIT"' in citation_text
