"""Runs the steps of an InstallPlan one after the other and reports their output.

Commands run with QProcess in their own process group (pip and venv start children), and
in-process actions run on the GUI thread between them. A step's marker is written only
after it succeeded, so a failed or cancelled install resumes from that step with a new
plan. The next step always starts from the event loop, which keeps the window responsive
and lets cancel() land between steps.
"""

import codecs
import logging
from collections.abc import Iterable

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from anima_trainer.runtime.install_plan import InstallStep
from anima_trainer.system.process_group import (
    KILL_GRACE_MS,
    SHUTDOWN_KILL_WAIT_MS,
    kill_group,
    terminate_group,
    use_own_session,
)
from anima_trainer.system.process_output import split_lines

DONE_MESSAGE = "Installation completed."
CANCELLED_MESSAGE = "Installation cancelled."

_LOG = logging.getLogger(__name__)


class RuntimeInstaller(QObject):
    step_started = Signal(int, int, object)  # position from 1, number of steps, InstallStep
    line_received = Signal(str)
    finished = Signal(bool, str)  # success, message to show

    def __init__(self, kill_grace_ms: int = KILL_GRACE_MS, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._kill_grace_ms = kill_grace_ms
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill)
        self._steps: tuple[InstallStep, ...] = ()
        self._running = False
        # Numbers each start(): a deferred next step from an earlier run is ignored.
        self._run = 0
        self._index = -1
        self._process: QProcess | None = None
        self._pending = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._cancelled = False

    def is_running(self) -> bool:
        """True from start() until finished has been emitted."""
        return self._running

    def has_pending_kill(self) -> bool:
        """True while a cancelled command is still within its grace period."""
        return self._kill_timer.isActive()

    def start(self, steps: Iterable[InstallStep]) -> None:
        """Run the steps in order; raises RuntimeError if an install is already active."""
        if self._running:
            raise RuntimeError("An installation is already active")
        self._steps = tuple(steps)
        self._running = True
        self._run += 1
        self._index = -1
        self._cancelled = False
        self._schedule_next_step()

    def cancel(self) -> None:
        """Stop the install: SIGTERM to the running command, SIGKILL after the grace period."""
        if not self._running or self._cancelled:
            return
        self._cancelled = True
        if self._process is not None:
            terminate_group(self._process)
            self._kill_timer.start(self._kill_grace_ms)

    def shutdown(self) -> None:
        """Cancel and block until the running command has exited; call before quitting."""
        process = self._process
        self.cancel()
        if process is None:
            if self._running:
                self._finish(False, CANCELLED_MESSAGE)
            return
        if not process.waitForFinished(self._kill_grace_ms):
            self._force_kill()
            process.waitForFinished(SHUTDOWN_KILL_WAIT_MS)

    def _schedule_next_step(self) -> None:
        # With self as context the call is dropped if the installer is deleted first.
        QTimer.singleShot(0, self, lambda run=self._run: self._next_step(run))

    def _next_step(self, run: int) -> None:
        if self._stopped(run):
            return
        self._index += 1
        if self._index == len(self._steps):
            self._finish(True, DONE_MESSAGE)
            return
        step = self._steps[self._index]
        self.step_started.emit(self._index + 1, len(self._steps), step)
        if self._stopped(run):
            return  # a step_started slot called cancel() or shutdown()
        if step.command:
            self._start_process(step)
        else:
            self._run_action(step)

    def _stopped(self, run: int) -> bool:
        """True when this run must not go on; a cancelled run is finished here.

        A method rather than inline tests, because slots can change the state during emit().
        """
        if not self._running or run != self._run:
            return True  # finished by shutdown(), possibly before this deferred call ran
        if self._cancelled:
            self._finish(False, CANCELLED_MESSAGE)
            return True
        return False

    def _run_action(self, step: InstallStep) -> None:
        try:
            step.run_action()
        except Exception as error:  # reported to the user; the install stops here
            _LOG.warning("Install step %s failed", step.key, exc_info=True)
            self._finish(False, f"{step.title} failed: {error}")
            return
        self._complete(step)

    def _start_process(self, step: InstallStep) -> None:
        process = QProcess(self)
        process.setProgram(step.command[0])
        process.setArguments(list(step.command[1:]))
        if step.working_dir is not None:
            process.setWorkingDirectory(str(step.working_dir))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")  # pip and the GPU check print as they go
        process.setProcessEnvironment(environment)
        use_own_session(process)
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._pending = ""
        self._decoder.reset()
        self._process = process
        process.start()

    def _on_output(self) -> None:
        if self._process is None:
            return
        data = self._decoder.decode(self._process.readAllStandardOutput().data())
        lines, self._pending = split_lines(self._pending, data)
        for line in lines:
            self.line_received.emit(line)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._on_output()
        remainder = (self._pending + self._decoder.decode(b"", final=True)).rstrip("\r")
        if remainder:
            self.line_received.emit(remainder)
        self._release_process()
        step = self._steps[self._index]
        if self._cancelled:
            self._finish(False, CANCELLED_MESSAGE)
        elif exit_status == QProcess.ExitStatus.CrashExit:
            self._finish(False, f"{step.title} stopped unexpectedly.")
        elif exit_code != 0:
            self._finish(False, f"{step.title} failed (exit code {exit_code}).")
        else:
            self._complete(step)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart or self._process is None:
            return
        reason = f"could not start {self._process.program()}: {self._process.errorString()}"
        self._release_process()
        if self._cancelled:
            self._finish(False, CANCELLED_MESSAGE)
            return
        self._finish(False, f"{self._steps[self._index].title} failed: {reason}")

    def _complete(self, step: InstallStep) -> None:
        if step.marker is not None:
            try:
                step.marker.parent.mkdir(parents=True, exist_ok=True)
                step.marker.write_text(step.stamp + "\n", encoding="utf-8")
            except OSError as error:
                self._finish(False, f"{step.title}: could not write {step.marker}: {error}")
                return
        self._schedule_next_step()

    def _force_kill(self) -> None:
        if self._process is not None and self._cancelled:
            kill_group(self._process)

    def _release_process(self) -> None:
        self._kill_timer.stop()
        if self._process is not None:
            self._process.deleteLater()
        self._process = None

    def _finish(self, ok: bool, message: str) -> None:
        if not self._running:
            return  # this run has already reported its result
        self._running = False
        self.finished.emit(ok, message)
