"""One process at a time uses the GPU of a data folder: a window or a terminal command.

Inside a process several jobs may share the lock (a generation during a training), so it is held
until the last job gives it back. QLockFile removes a lock left by a dead process; its age limit
is turned off, or a training of some hours would look abandoned.
"""

from pathlib import Path

from PySide6.QtCore import QLockFile

TRAINING_JOB = "training"
GENERATOR_JOB = "generator"
INSTALL_JOB = "install"
TERMINAL_JOB = "terminal"

NO_AGE_LIMIT = 0
NO_WAIT = 0


class GpuLock:
    def __init__(self) -> None:
        self._file: Path | None = None
        self._lock: QLockFile | None = None
        self._jobs: set[str] = set()

    def acquire(self, job: str, file: Path) -> str | None:
        """Take the lock for `job`; returns why it cannot be taken, or None."""
        if self._jobs:
            if file != self._file:
                return f"The GPU is in use with another data folder ({self._file})."
            self._jobs.add(job)
            return None
        try:
            file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            return f"Could not create the lock file {file}: {error}"
        lock = QLockFile(str(file))
        lock.setStaleLockTime(NO_AGE_LIMIT)
        if not lock.tryLock(NO_WAIT):
            return _refusal(lock, file)
        self._file, self._lock = file, lock
        self._jobs.add(job)
        return None

    def release(self, job: str) -> None:
        self._jobs.discard(job)
        if not self._jobs and self._lock is not None:
            self._lock.unlock()
            self._file, self._lock = None, None


def _refusal(lock: QLockFile, file: Path) -> str:
    if lock.error() != QLockFile.LockError.LockFailedError:
        return f"Could not create the lock file {file}: {lock.error().name}"
    pid = lock.getLockInfo()[0]
    who = f"Another Anima LoRA Studio process (pid {pid})" if pid else "Another process"
    return f"{who} is using the GPU with this data folder: wait for it to end, or stop it."
