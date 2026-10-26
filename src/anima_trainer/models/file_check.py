"""Checks that a model file the user points to is exactly the one the app would download.

The size is compared first, so a wrong multi-GB file is rejected at once; then the SHA-256 is
computed in chunks with progress and cancel. The file is only read, never written.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

from anima_trainer.models.catalog import ExpectedFile
from anima_trainer.models.linked_models import LinkedModel, is_unchanged, link_for

CHUNK_SIZE = 8 << 20

CheckProgress = Callable[[int, int], None]


class ModelFileRejected(Exception):
    """The file is not the expected model (or cannot be read); the message says why."""


class CheckCancelled(Exception):
    """The check was cancelled before it finished."""


def check_model_file(
    path: Path,
    expected: ExpectedFile,
    on_progress: CheckProgress = lambda _done, _total: None,
    is_cancelled: Callable[[], bool] = lambda: False,
    chunk_size: int = CHUNK_SIZE,
) -> LinkedModel:
    if not path.is_file():
        raise ModelFileRejected(f"{path} is not a file")
    try:
        link = link_for(path)
        if link.size != expected.size:
            raise ModelFileRejected(
                f"wrong size ({link.size} bytes, expected {expected.size}): "
                "this is not the expected model file"
            )
        digest = _sha256(path, link.size, on_progress, is_cancelled, chunk_size)
    except OSError as error:
        raise ModelFileRejected(f"cannot read {path}: {error}") from error
    if digest != expected.sha256:
        raise ModelFileRejected(
            "SHA-256 does not match: the file is a different version or is damaged"
        )
    if not is_unchanged(link):
        raise ModelFileRejected(f"{path} changed while it was being checked")
    return link


def _sha256(
    path: Path,
    size: int,
    on_progress: CheckProgress,
    is_cancelled: Callable[[], bool],
    chunk_size: int,
) -> str:
    digest = hashlib.sha256()
    done = 0
    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            if is_cancelled():
                raise CheckCancelled
            digest.update(chunk)
            done += len(chunk)
            on_progress(done, size)
    return digest.hexdigest()
