"""Facts about the training venv that change how sd-scripts must be configured.

Python 3.14 made `forkserver` the default start method on POSIX. sd-scripts then pickles
the dataset (with its in-memory latent cache) for every DataLoader worker, and torch shares
each cached tensor through a file descriptor; past 256 fds the start fails with
"ValueError: too many fds". Running the loader in the main process avoids it.
"""

import os
from pathlib import Path

FORKSERVER_DEFAULT_VERSION = (3, 14)
PYVENV_CFG = "pyvenv.cfg"
VERSION_KEYS = ("version", "version_info")


def read_venv_version(venv_dir: Path) -> tuple[int, int] | None:
    """Major and minor version of the venv's Python from pyvenv.cfg, None if unknown."""
    try:
        text = (venv_dir / PYVENV_CFG).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() in VERSION_KEYS:
            return _major_minor(value.strip())
    return None


def needs_single_process_loader(version: tuple[int, int] | None, platform: str = os.name) -> bool:
    """True when DataLoader workers would start through the forkserver and fail."""
    return platform != "nt" and version is not None and version >= FORKSERVER_DEFAULT_VERSION


def _major_minor(value: str) -> tuple[int, int] | None:
    major, _, rest = value.partition(".")
    minor = rest.split(".", 1)[0]
    if not (major.isdigit() and minor.isdigit()):
        return None
    return int(major), int(minor)
