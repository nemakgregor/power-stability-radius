from __future__ import annotations

"""
Download helpers for test cases (MATPOWER IEEE and PGLib-OPF).

Design goals
------------
- Deterministic behavior:
  * stable candidate URL ordering
  * no implicit/randomized fallbacks
  * controlled retry policy (default: 1 try per candidate URL)
- Clear error messages and logging for debugging connectivity issues.
- Testability:
  * supports injecting a custom `session` object with a `.get(url, timeout=...)` method.

Environment overrides
---------------------
You can fully override the base URL candidate list via environment variables:

- SR_MATPOWER_BASE_URLS="https://...,https://..."
- SR_PGLIB_OPF_BASE_URLS="https://...,https://..."

If an env var is set (non-empty), it is used *as-is* (no additional defaults).
"""

import logging
import os
import time
from typing import Optional, Sequence, Union

import requests

logger = logging.getLogger(__name__)

_DEFAULT_MATPOWER_BASE_URL_RAW = (
    "https://raw.githubusercontent.com/MATPOWER/matpower/master/data"
)
_DEFAULT_MATPOWER_BASE_URL_GITHUB = (
    "https://github.com/MATPOWER/matpower/raw/master/data"
)

_PGLIB_OPF_BASE_URL_RAW = (
    "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master"
)
_PGLIB_OPF_BASE_URL_GITHUB = "https://github.com/power-grid-lib/pglib-opf/raw/master"

_ENV_MATPOWER_BASE_URLS = "SR_MATPOWER_BASE_URLS"
_ENV_PGLIB_OPF_BASE_URLS = "SR_PGLIB_OPF_BASE_URLS"


class DownloadError(RuntimeError):
    """
    Raised when downloading from one or more candidate URLs fails.

    Attributes
    ----------
    urls:
        Candidate URLs attempted (or intended to be attempted). May be empty.
    """

    def __init__(self, message: str, *, urls: Sequence[str] | None = None) -> None:
        super().__init__(str(message))
        self.urls: tuple[str, ...] = tuple(str(u) for u in (urls or ()))


def _unique_preserve(values: Sequence[str]) -> list[str]:
    """Return unique items preserving the first occurrence order."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        vv = str(v).strip()
        if not vv or vv in seen:
            continue
        seen.add(vv)
        out.append(vv)
    return out


def _parse_env_base_urls(env_var: str) -> list[str]:
    """
    Parse comma-separated base URLs from environment.

    If the variable is not set or empty, returns an empty list.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return []
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    return _unique_preserve(urls)


def _candidate_base_urls(
    *, explicit_base_url: str, defaults: Sequence[str], env_var: str
) -> list[str]:
    """
    Build deterministic list of candidate base URLs.

    Priority
    --------
    1) If env var is set -> use it (exactly, no implicit fallbacks).
    2) Otherwise -> [explicit_base_url] + defaults (deduplicated).
    """
    env_urls = _parse_env_base_urls(env_var)
    if env_urls:
        logger.info("Using base URLs from %s: %s", env_var, env_urls)
        return env_urls

    explicit = str(explicit_base_url).strip().rstrip("/")
    candidates = _unique_preserve(
        [explicit, *[d.strip().rstrip("/") for d in defaults]]
    )
    return candidates


def _response_text(response: object) -> str:
    """
    Extract response payload as text.

    Supports:
    - real `requests.Response` (has `.text`)
    - minimal fake responses used in tests.
    """
    if hasattr(response, "text"):
        return str(getattr(response, "text"))
    if hasattr(response, "content"):
        try:
            return bytes(getattr(response, "content")).decode("utf-8", errors="replace")
        except Exception:
            return str(getattr(response, "content"))
    raise TypeError("Unsupported response type: expected .text or .content attribute")


def _download_text_file(
    *,
    url: str,
    target_path: str,
    overwrite: bool,
    timeout: float,
    session: Optional[requests.Session],
    retries: int = 1,
    backoff_sec: float = 0.0,
) -> str:
    """
    Download a text file from `url` into `target_path`.

    Notes
    -----
    - Retries are deterministic: the same URL is retried `retries` times before failing.
    - Raises DownloadError on failure so callers can try alternate candidate URLs.
    """
    if os.path.exists(target_path) and not overwrite:
        logger.info("File already exists, skipping download: %s", target_path)
        return target_path

    if retries <= 0:
        raise ValueError("retries must be positive")

    http = session or requests
    last_exc: Exception | None = None

    for attempt in range(1, int(retries) + 1):
        try:
            logger.debug("Downloading (attempt %d/%d): %s", attempt, retries, url)
            response = http.get(url, timeout=timeout)
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()

            dir_name = os.path.dirname(target_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            text = _response_text(response)
            with open(target_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)

            logger.info("Downloaded and saved: %s", target_path)
            return target_path
        except Exception as e:  # noqa: BLE001 - network + IO errors
            last_exc = e
            logger.warning(
                "Download failed (attempt %d/%d): %s (%s)",
                attempt,
                retries,
                url,
                e,
            )
            if attempt < retries and backoff_sec > 0:
                time.sleep(float(backoff_sec))

    logger.error("Failed to download %s after %d attempt(s).", url, retries)
    raise DownloadError(
        f"Failed to download after {retries} attempt(s): {url}", urls=(url,)
    ) from last_exc


def _download_from_candidates(
    *,
    urls: Sequence[str],
    target_path: str,
    overwrite: bool,
    timeout: float,
    session: Optional[requests.Session],
    retries_per_url: int,
    backoff_sec: float,
) -> str:
    """
    Try downloading from candidate URLs in order.

    Raises
    ------
    DownloadError
        If all candidates fail.
    """
    if not urls:
        raise ValueError("urls must be non-empty")

    errors: list[str] = []
    for i, url in enumerate(urls, start=1):
        logger.info("Download candidate %d/%d: %s", i, len(urls), str(url))
        try:
            return _download_text_file(
                url=str(url),
                target_path=target_path,
                overwrite=overwrite,
                timeout=timeout,
                session=session,
                retries=int(retries_per_url),
                backoff_sec=float(backoff_sec),
            )
        except DownloadError as e:
            errors.append(f"{url}: {e}")
            logger.error("Candidate URL failed: %s", url)

    msg = "All candidate URLs failed:\n" + "\n".join(errors)
    raise DownloadError(msg, urls=tuple(str(u) for u in urls))


def download_ieee_case(
    case_number: Union[int, str],
    target_path: Optional[str] = None,
    *,
    overwrite: bool = False,
    timeout: float = 15.0,
    base_url: str = _DEFAULT_MATPOWER_BASE_URL_RAW,
    session: Optional[requests.Session] = None,
    retries_per_url: int = 1,
    backoff_sec: float = 0.0,
) -> str:
    """
    Download a MATPOWER IEEE case file (e.g., case14.m, case30.m, case118.m).

    Candidate URLs (default)
    ------------------------
    1) raw.githubusercontent.com (configurable via `base_url`)
    2) github.com/.../raw/...

    Determinism note
    ----------------
    By default, each candidate URL is attempted exactly once. This is important for
    reproducible behavior and for tests that validate fallback ordering.

    You can override the base URL candidate list explicitly via environment variable:
      - SR_MATPOWER_BASE_URLS="https://...,https://..."

    Returns
    -------
    str
        Path to the existing or downloaded file.
    """
    try:
        n = int(case_number)
    except (TypeError, ValueError) as e:
        raise ValueError(f"case_number must be an integer, got {case_number!r}") from e
    if n <= 0:
        raise ValueError(f"case_number must be positive, got {n}")

    if target_path is None:
        target_path = f"data/input/ieee{n}.m"

    base_urls = _candidate_base_urls(
        explicit_base_url=str(base_url),
        defaults=[_DEFAULT_MATPOWER_BASE_URL_GITHUB],
        env_var=_ENV_MATPOWER_BASE_URLS,
    )
    urls = [f"{u.rstrip('/')}/case{n}.m" for u in base_urls]
    logger.info("Downloading MATPOWER case%d (candidates=%d)", n, len(urls))

    try:
        return _download_from_candidates(
            urls=urls,
            target_path=str(target_path),
            overwrite=bool(overwrite),
            timeout=float(timeout),
            session=session,
            retries_per_url=int(retries_per_url),
            backoff_sec=float(backoff_sec),
        )
    except DownloadError as e:
        raise RuntimeError(
            f"Failed to download case{n}.m. Check network connectivity and the case number."
        ) from e


def download_pglib_opf_case(
    *,
    case_filename: str,
    target_path: Optional[str] = None,
    overwrite: bool = False,
    timeout: float = 30.0,
    base_url: str = _PGLIB_OPF_BASE_URL_RAW,
    session: Optional[requests.Session] = None,
    retries_per_url: int = 1,
    backoff_sec: float = 0.0,
) -> str:
    """
    Download a PGLib-OPF case file.

    Candidate URLs (default)
    ------------------------
    1) raw.githubusercontent.com (configurable via `base_url`)
    2) github.com/.../raw/...

    Determinism note
    ----------------
    By default, each candidate URL is attempted exactly once. This is intentional:
    when the first candidate fails (e.g., DNS issue for raw.githubusercontent.com),
    we immediately fall back to the next mirror, which improves robustness in CI and
    matches expected deterministic fallback semantics.

    You can override the candidate list explicitly via environment variable:
      - SR_PGLIB_OPF_BASE_URLS="https://...,https://..."

    Returns
    -------
    str
        Path to the existing or downloaded file.
    """
    if not case_filename or not case_filename.endswith(".m"):
        raise ValueError(
            f"case_filename must be a non-empty .m filename, got {case_filename!r}"
        )

    if target_path is None:
        target_path = os.path.join("data", "input", case_filename)

    base_urls = _candidate_base_urls(
        explicit_base_url=str(base_url),
        defaults=[_PGLIB_OPF_BASE_URL_GITHUB],
        env_var=_ENV_PGLIB_OPF_BASE_URLS,
    )
    urls = [f"{u.rstrip('/')}/{case_filename}" for u in base_urls]
    logger.info(
        "Downloading PGLib-OPF case %s (candidates=%d)", case_filename, len(urls)
    )

    return _download_from_candidates(
        urls=urls,
        target_path=str(target_path),
        overwrite=bool(overwrite),
        timeout=float(timeout),
        session=session,
        retries_per_url=int(retries_per_url),
        backoff_sec=float(backoff_sec),
    )


def download_ieee30(target_path: str = "data/input/ieee30.m") -> str:
    """
    Backward-compatible wrapper for downloading IEEE 30-bus case.

    Prefer `download_ieee_case(30, target_path=...)` in new code.
    """
    return download_ieee_case(30, target_path=target_path)
