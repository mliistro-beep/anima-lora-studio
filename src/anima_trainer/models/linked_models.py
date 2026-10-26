"""Model files the user already has, used from where they are instead of being downloaded.

A link is trusted only while the file keeps the size and modification time it had when its
SHA-256 was checked; a moved, replaced or edited file counts as missing again.
"""

import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LinkedModel:
    path: Path
    size: int
    mtime_ns: int


def link_for(path: Path) -> LinkedModel:
    """The link for `path` as it is now; raises OSError when it cannot be read."""
    info = path.stat()
    return LinkedModel(path, info.st_size, info.st_mtime_ns)


def is_unchanged(link: LinkedModel) -> bool:
    try:
        info = link.path.stat()
    except OSError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_size == link.size
        and info.st_mtime_ns == link.mtime_ns
    )
