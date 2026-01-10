from __future__ import annotations

import logging
import os
from typing import Optional, Union

import requests

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://raw.githubusercontent.com/MATPOWER/matpower/master/data"


def download_ieee_case(
    case_number: Union[int, str],
    target_path: Optional[str] = None,
    *,
    overwrite: bool = False,
    timeout: float = 15.0,
    base_url: str = _DEFAULT_BASE_URL,
    session: Optional[requests.Session] = None,
) -> str:
    """
    Download a MATPOWER IEEE case file (e.g., case14.m, case30.m, case118.m) by case number.

    Parameters
    ----------
    case_number:
        IEEE/MATPOWER case number (e.g., 14, 30, 57, 118, 300).
    target_path:
        Where to save the downloaded file. If None, defaults to "data/input/ieee{N}.m".
    overwrite:
        If True, re-download even if the file already exists.
    timeout:
        Network timeout in seconds for the HTTP request.
    base_url:
        Base URL where MATPOWER case files are hosted.
    session:
        Optional `requests.Session` to reuse connections.

    Returns
    -------
    str
        Path to the existing or downloaded file.

    Raises
    ------
    ValueError
        If `case_number` is not a positive integer.
    RuntimeError
        If download fails.
    """
    try:
        n = int(case_number)
    except (TypeError, ValueError) as e:
        raise ValueError(f"case_number must be an integer, got {case_number!r}") from e
    if n <= 0:
        raise ValueError(f"case_number must be positive, got {n}")

    if target_path is None:
        target_path = f"data/input/ieee{n}.m"

    if os.path.exists(target_path) and not overwrite:
        logger.info("File already exists: %s", target_path)
        return target_path

    url = f"{base_url}/case{n}.m"
    logger.info("Downloading MATPOWER case%d from %s", n, url)

    http = session or requests
    try:
        response = http.get(url, timeout=timeout)
        response.raise_for_status()

        dir_name = os.path.dirname(target_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(target_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(response.text)

        logger.info("Downloaded and saved: %s", target_path)
        return target_path
    except Exception as e:
        logger.error("Failed to download case%d.m: %s", n, e)
        raise RuntimeError(
            f"Failed to download case{n}.m. Check network connectivity and the case number."
        ) from e


def download_ieee30(target_path: str = "data/input/ieee30.m") -> str:
    """
    Backward-compatible wrapper for downloading IEEE 30-bus case.

    Prefer `download_ieee_case(30, target_path=...)` in new code.
    """
    return download_ieee_case(30, target_path=target_path)
