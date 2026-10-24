"""Plans and runs the install of the training runtime for the Settings tab.

Checks that the plan can run (Python and git found) before touching the data folder,
writes everything the installer reports to <data>/logs/install_<timestamp>.log and turns
its signals into log lines, the current step and the final result.
"""

import contextlib
import shutil
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Protocol, TextIO

from PySide6.QtCore import QObject, Signal, SignalInstance

from anima_trainer.runtime.install_plan import InstallPlan, InstallStep, plan_install
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.runtime.python_finder import find_python
from anima_trainer.training.runner import log_file_path

INSTALL_LOG_NAME = "install"


class Installer(Protocol):
    """The part of RuntimeInstaller the session uses."""

    step_started: SignalInstance
    line_received: SignalInstance
    finished: SignalInstance

    def start(self, steps: Iterable[InstallStep]) -> None: ...
    def cancel(self) -> None: ...
    def shutdown(self) -> None: ...
    def is_running(self) -> bool: ...


def plan_for(paths: RuntimePaths) -> InstallPlan:
    """The install plan with the Python and git found on this machine."""
    return plan_install(paths, find_python(), shutil.which("git"))


class InstallSession(QObject):
    """One install at a time into the data folder given to start()."""

    log_line = Signal(str)
    step_changed = Signal(str)  # "Step 2/7: <title>"
    finished = Signal(bool, str)  # success, message

    def __init__(
        self,
        installer: Installer,
        plan: Callable[[RuntimePaths], InstallPlan] = plan_for,
        clock: Callable[[], datetime] = datetime.now,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._installer = installer
        self._plan = plan
        self._clock = clock
        self._log: TextIO | None = None
        self._log_path: Path | None = None
        installer.step_started.connect(self._on_step_started)
        installer.line_received.connect(self._emit_line)
        installer.finished.connect(self._on_finished)

    def is_running(self) -> bool:
        """True from a successful start() until finished has been emitted."""
        return self._installer.is_running()

    def start(self, paths: RuntimePaths) -> tuple[str, ...]:
        """Install into `paths`; returns the problems that prevent it (nothing runs then)."""
        if self.is_running():
            raise RuntimeError("An installation is already active")
        plan = self._plan(paths)
        if not plan.ready:
            return plan.problems
        now = self._clock()
        log_path = log_file_path(paths.logs_dir, INSTALL_LOG_NAME, now)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Kept open until the install finishes, so no context manager.
            self._log = open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
            self._log_path = log_path
        except OSError as error:
            return (f"Could not write the install log {log_path}: {error}",)
        self.log_line.emit(f"Install log: {log_path}")
        self._write(f"Started: {now.isoformat()}")
        try:
            self._installer.start(plan.steps)
        except Exception:
            self._close_log()  # no finished will come to close it
            raise
        return ()

    def cancel(self) -> None:
        """Stop the install; finished reports it as cancelled."""
        self._installer.cancel()

    def shutdown(self) -> None:
        """Block until the running install command has exited."""
        self._installer.shutdown()

    def _on_step_started(self, position: int, total: int, step: InstallStep) -> None:
        text = f"Step {position}/{total}: {step.title}"
        self.step_changed.emit(text)
        self._emit_line(text)

    def _on_finished(self, ok: bool, message: str) -> None:
        self._emit_line(f"{'OK' if ok else 'ERROR'}: {message}")
        self._close_log()
        self.finished.emit(ok, message)

    def _emit_line(self, line: str) -> None:
        self._write(line)
        self.log_line.emit(line)

    def _write(self, line: str) -> None:
        if self._log is None:
            return
        try:
            self._log.write(line + "\n")
            self._log.flush()  # install output is small; keep the file current for a crash
        except OSError as error:
            self._close_log()
            # Shown once: the log stays closed for the rest of the install.
            self.log_line.emit(f"WARN: Install log {self._log_path} could not be written: {error}")

    def _close_log(self) -> None:
        log, self._log = self._log, None
        if log is not None:
            with contextlib.suppress(OSError):
                log.close()
