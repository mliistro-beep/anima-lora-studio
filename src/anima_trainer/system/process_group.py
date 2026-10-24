"""Process groups for QProcess children that start children of their own.

`accelerate launch` and `pip` spawn subprocesses and do not forward SIGTERM to them, so on
POSIX the process gets its own session and signals go to the whole group. Elsewhere (or
before the process has a pid) QProcess terminate()/kill() are used instead.
"""

import contextlib
import os
import signal

from PySide6.QtCore import QProcess

_POSIX = os.name == "posix"


def use_own_session(process: QProcess) -> None:
    """Start the process as leader of a new session, so its children share its group."""
    if _POSIX:
        process.setUnixProcessParameters(QProcess.UnixProcessFlag.CreateNewSession)


def terminate_group(process: QProcess) -> None:
    """Ask the process and its children to exit (SIGTERM to the group on POSIX)."""
    if _has_group(process):
        _signal_group(process, signal.SIGTERM)
    else:
        process.terminate()


def kill_group(process: QProcess) -> None:
    """Kill the process and its children (SIGKILL to the group on POSIX)."""
    if _has_group(process):
        _signal_group(process, signal.SIGKILL)
    else:
        process.kill()


def _has_group(process: QProcess) -> bool:
    return _POSIX and process.processId() > 0


def _signal_group(process: QProcess, signum: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):  # the group has already exited
        os.killpg(process.processId(), signum)
