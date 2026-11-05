"""A lock held by another real process, as another window or terminal command would hold it."""

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

HOLDER = """
import sys
from PySide6.QtCore import QLockFile
lock = QLockFile(sys.argv[1])
lock.setStaleLockTime(0)
print("held" if lock.tryLock(0) else "refused", flush=True)
sys.stdin.read()
"""


def start_holder(lock_file: Path) -> subprocess.Popen[str]:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(lock_file)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "held"
    return process


@contextmanager
def held_by_another_process(lock_file: Path) -> Iterator[int]:
    """Yields the pid of the process holding `lock_file`."""
    process = start_holder(lock_file)
    try:
        yield process.pid
    finally:
        process.kill()
        process.communicate()  # also closes the pipes
