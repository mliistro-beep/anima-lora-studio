"""HTTP download with resume, replacing Citron's `wget -c` (also works on Windows).

Data goes to "<name>.part" and is renamed only when complete, so an interrupted or cancelled
download never leaves a truncated model in place and can be resumed later.
"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from anima_trainer import __version__

CHUNK_SIZE = 1 << 20
TIMEOUT_S = 30
USER_AGENT = f"anima-trainer/{__version__}"
HTTP_PARTIAL_CONTENT = 206
HTTP_RANGE_NOT_SATISFIABLE = 416
_CONTENT_RANGE_TOTAL = re.compile(r"/(\d+)\s*$")


class DownloadError(Exception):
    """The download failed; a partial file may remain for a later resume."""


class DownloadCancelled(Exception):
    """The download was cancelled; the partial file is kept for a later resume."""


@dataclass(frozen=True)
class DownloadProgress:
    downloaded: int
    total: int | None


def partial_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".part")


def download_file(
    url: str,
    destination: Path,
    on_progress: Callable[[DownloadProgress], None] = lambda _: None,
    is_cancelled: Callable[[], bool] = lambda: False,
    chunk_size: int = CHUNK_SIZE,
    timeout_s: float = TIMEOUT_S,
) -> Path:
    part = partial_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT, **({"Range": f"bytes={offset}-"} if offset else {})}
    try:
        with urlopen(Request(url, headers=headers), timeout=timeout_s) as response:
            resumed = offset > 0 and response.status == HTTP_PARTIAL_CONTENT
            start = offset if resumed else 0
            total = _total_size(response.headers, start)
            with open(part, "ab" if resumed else "wb") as handle:
                written = _copy(
                    response, handle, start, total, on_progress, is_cancelled, chunk_size
                )
    except HTTPError as error:
        if error.code == HTTP_RANGE_NOT_SATISFIABLE and offset:
            os.replace(part, destination)
            return destination
        raise DownloadError(f"HTTP {error.code} downloading {url}") from error
    except (URLError, OSError) as error:
        raise DownloadError(f"Download of {url} failed: {error}") from error
    if total is not None and written != total:
        raise DownloadError(f"Download of {url} incomplete: {written} of {total} bytes")
    os.replace(part, destination)
    return destination


def _copy(
    response: BinaryIO,
    handle: BinaryIO,
    start: int,
    total: int | None,
    on_progress: Callable[[DownloadProgress], None],
    is_cancelled: Callable[[], bool],
    chunk_size: int,
) -> int:
    written = start
    while True:
        if is_cancelled():
            raise DownloadCancelled(f"Cancelled after {written} bytes")
        chunk = response.read(chunk_size)
        if not chunk:
            return written
        handle.write(chunk)
        written += len(chunk)
        on_progress(DownloadProgress(written, total))


def _total_size(headers: object, start: int) -> int | None:
    content_range = headers.get("Content-Range")  # type: ignore[attr-defined]
    match = _CONTENT_RANGE_TOTAL.search(content_range) if content_range else None
    if match:
        return int(match[1])
    length = headers.get("Content-Length")  # type: ignore[attr-defined]
    return start + int(length) if length and length.isdigit() else None
