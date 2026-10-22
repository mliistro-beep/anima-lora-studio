"""Finds a system Python able to create the training venv.

sd-scripts is tested on Python 3.10 (3.11 and 3.12 "should work"); newer versions come
after them because they are less proven. Python 3.14 and later are only a fallback: their
forkserver default breaks DataLoader workers (see venv_info), which then must be disabled.
"""

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from anima_trainer.runtime.venv_info import FORKSERVER_DEFAULT_VERSION

CANDIDATES = (
    "python3.12",
    "python3.11",
    "python3.10",
    "python3.13",
    "python3.14",
    "python3",
    "python",
)
MIN_VERSION = (3, 10)
PROBE_TIMEOUT_S = 15
# ensurepip is what `python -m venv` needs to put pip in the venv; Debian ships it apart.
VERSION_PROBE = "import sys, venv, ensurepip; print('%d.%d' % sys.version_info[:2])"
NO_PYTHON_MESSAGE = (
    "No usable Python found. Install Python 3.10 or newer with venv support "
    "(on Debian/Ubuntu: sudo apt install python3-venv), then try again."
)

Which = Callable[[str], str | None]
Runner = Callable[..., subprocess.CompletedProcess[Any]]


@dataclass(frozen=True)
class PythonInterpreter:
    path: str
    version: tuple[int, int]


def find_python(
    which: Which = shutil.which, run: Runner = subprocess.run
) -> PythonInterpreter | None:
    """The first usable candidate older than 3.14, else the first usable one, else None.

    Usable means on PATH, new enough and able to create a venv.
    """
    fallback = None
    for name in CANDIDATES:
        path = which(name)
        if path is None:
            continue
        version = _probe_version(path, run)
        if version is None or version < MIN_VERSION:
            continue
        if version < FORKSERVER_DEFAULT_VERSION:
            return PythonInterpreter(path, version)
        fallback = fallback or PythonInterpreter(path, version)
    return fallback


def _probe_version(path: str, run: Runner) -> tuple[int, int] | None:
    try:
        result = run(
            [path, "-c", VERSION_PROBE],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    major, _, minor = result.stdout.strip().partition(".")
    if not (major.isdigit() and minor.isdigit()):
        return None
    return int(major), int(minor)
