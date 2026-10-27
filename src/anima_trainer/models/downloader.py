"""HTTP download with resume, replacing Citron's `wget -c` (also works on Windows).

Data goes to "<name>.part" and is renamed only when complete (and, when the expected file is
known, only when its size and SHA-256 match), so an interrupted, cancelled or damaged download
never leaves a wrong model in place; an interrupted one can be resumed later.
Every failure surfaces as DownloadError (or DownloadCancelled), never as a raw exception.
"""

import http.client
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from anima_trainer import __version__
from anima_trainer.models.catalog import ExpectedFile
from anima_trainer.models.file_check import CheckCancelled, file_sha256

CHUNK_SIZE = 1 << 20
TIMEOUT_S = 30
USER_AGENT = f"anima-trainer/{__version__}"
ALLOWED_SCHEMES = frozenset({"http", "https"})
HTTP_PARTIAL_CONTENT = 206
HTTP_RANGE_NOT_SATISFIABLE = 416
_CONTENT_RANGE_TOTAL = re.compile(r"/(\d+)\s*$")

ProgressCallback = Callable[["DownloadProgress"], None]


class DownloadError(Exception):
    """The download failed; a partial file may remain for a later resume."""


class DownloadCancelled(Exception):
    """The download was cancelled; the partial file is kept for a later resume."""


@dataclass(frozen=True)
class DownloadProgress:
    downloaded: int
    total: int | None


@dataclass(frozen=True)
class _Transfer:
    on_progress: ProgressCallback
    is_cancelled: Callable[[], bool]
    chunk_size: int
    timeout_s: float
    expected: ExpectedFile | None


def partial_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".part")


def download_file(
    url: str,
    destination: Path,
    on_progress: ProgressCallback = lambda _: None,
    is_cancelled: Callable[[], bool] = lambda: False,
    chunk_size: int = CHUNK_SIZE,
    timeout_s: float = TIMEOUT_S,
    expected: ExpectedFile | None = None,
) -> Path:
    if urlsplit(url).scheme not in ALLOWED_SCHEMES:
        raise DownloadError(f"Unsupported URL scheme in {url}: only http and https are allowed")
    transfer = _Transfer(on_progress, is_cancelled, chunk_size, timeout_s, expected)
    try:
        return _download(url, destination, transfer)
    except (DownloadError, DownloadCancelled):
        raise
    except HTTPError as error:
        raise DownloadError(f"HTTP {error.code} downloading {url}") from error
    except (URLError, OSError, http.client.HTTPException) as error:
        raise DownloadError(f"Download of {url} failed: {error}") from error


def _download(url: str, destination: Path, transfer: _Transfer) -> Path:
    part = partial_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT, **({"Range": f"bytes={offset}-"} if offset else {})}
    try:
        request = Request(url, headers=headers)
        # The scheme is restricted to http/https in download_file().
        response_cm = urlopen(request, timeout=transfer.timeout_s)  # nosec B310
    except HTTPError as error:
        error.close()  # releases the socket; code and headers stay readable
        if error.code != HTTP_RANGE_NOT_SATISFIABLE or not offset:
            raise
        return _handle_unsatisfiable_range(url, destination, transfer, error.headers, offset)
    with response_cm as response:
        resumed = offset > 0 and response.status == HTTP_PARTIAL_CONTENT
        start = offset if resumed else 0
        total = _total_size(response.headers, start)
        with open(part, "ab" if resumed else "wb") as handle:
            written = _copy(response, handle, start, total, transfer)
    if total is not None and written != total:
        raise DownloadError(f"Download of {url} incomplete: {written} of {total} bytes")
    return _finalize(part, destination, transfer)


def _handle_unsatisfiable_range(
    url: str, destination: Path, transfer: _Transfer, headers: Message, offset: int
) -> Path:
    """416 means the .part is complete only if its size equals the remote size."""
    part = partial_path(destination)
    if _total_size(headers, 0) == offset:
        return _finalize(part, destination, transfer)
    # Stale leftover from a different (larger) file: discard it and start over.
    part.unlink()
    return _download(url, destination, transfer)


def _finalize(part: Path, destination: Path, transfer: _Transfer) -> Path:
    """Rename the complete .part, after checking it against the expected file if known.

    A mismatch deletes the .part: resuming a damaged file would only damage it again.
    """
    expected = transfer.expected
    if expected is not None:
        size = part.stat().st_size
        if size != expected.size:
            part.unlink()
            raise DownloadError(
                f"Downloaded file has the wrong size ({size} bytes, expected {expected.size}); "
                "it was deleted, try again"
            )
        try:
            digest = file_sha256(part, is_cancelled=transfer.is_cancelled)
        except CheckCancelled as error:
            raise DownloadCancelled("Cancelled while checking the download") from error
        if digest != expected.sha256:
            part.unlink()
            raise DownloadError(
                "Downloaded file SHA-256 does not match the expected model; it was deleted, "
                "try again"
            )
    os.replace(part, destination)
    return destination


def _copy(
    response: BinaryIO, handle: BinaryIO, start: int, total: int | None, transfer: _Transfer
) -> int:
    written = start
    while True:
        if transfer.is_cancelled():
            raise DownloadCancelled(f"Cancelled after {written} bytes")
        chunk = response.read(transfer.chunk_size)
        if not chunk:
            return written
        handle.write(chunk)
        written += len(chunk)
        transfer.on_progress(DownloadProgress(written, total))


def _total_size(headers: Message, start: int) -> int | None:
    content_range = headers.get("Content-Range")
    match = _CONTENT_RANGE_TOTAL.search(content_range) if content_range else None
    if match:
        return int(match[1])
    length = headers.get("Content-Length")
    return start + int(length) if length and length.isdigit() else None
