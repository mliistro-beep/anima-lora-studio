"""Runs the training process with QProcess: streams lines, parsed events and a log file.

Unlike Citron, a run can be stopped. On POSIX the process gets its own session so that
stop() reaches the whole process group: `accelerate launch` starts the real training as a
child and does not forward SIGTERM to it.
"""

import codecs
import contextlib
import os
import re
import signal
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from anima_trainer.training.command_builder import LaunchSpec
from anima_trainer.training.log_parser import parse_line
from anima_trainer.training.toml_builder import FILE_TIMESTAMP_FORMAT

LOG_TAIL_LINES = 200
KILL_GRACE_MS = 10_000
SHUTDOWN_KILL_WAIT_MS = 5_000
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


@dataclass(frozen=True)
class RunOutcome:
    exit_code: int
    crashed: bool
    stopped: bool
    log_path: Path


def log_file_path(logs_dir: Path, project_name: str, now: datetime) -> Path:
    return logs_dir / f"{project_name}_{now.strftime(FILE_TIMESTAMP_FORMAT)}.log"


def split_lines(pending: str, data: str) -> tuple[tuple[str, ...], str]:
    """Split on \\n, \\r\\n and \\r (tqdm redraws); a trailing \\r waits for a possible \\n."""
    text = pending + data
    held = "\r" if text.endswith("\r") else ""
    parts = _LINE_BREAK.split(text[: len(text) - len(held)])
    return tuple(parts[:-1]), parts[-1] + held


class TrainingRunner(QObject):
    line_received = Signal(str)
    event_received = Signal(object)
    finished = Signal(object)  # RunOutcome
    failed_to_start = Signal(str)
    warning = Signal(str)  # problems that do not stop the run, e.g. the log file failing

    def __init__(self, kill_grace_ms: int = KILL_GRACE_MS, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._kill_grace_ms = kill_grace_ms
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill)
        self._process: QProcess | None = None
        self._log: TextIO | None = None
        self._log_path: Path | None = None
        self._pending = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        self._tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        self._stopping = False

    def is_running(self) -> bool:
        return self._process is not None

    def log_tail(self) -> tuple[str, ...]:
        return tuple(self._tail)

    def has_pending_kill(self) -> bool:
        return self._kill_timer.isActive()

    def start(self, spec: LaunchSpec, log_path: Path) -> None:
        """Start a run; raises RuntimeError if one is active and OSError if the log can't open."""
        if self._process is not None:
            raise RuntimeError("A training run is already active")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Kept open for the whole run and closed in _cleanup(), so no context manager.
        self._log = open(log_path, "w", encoding="utf-8", errors="ignore")  # noqa: SIM115
        self._log.write(f"Command: {spec.display()}\nStarted: {datetime.now().isoformat()}\n\n")
        self._log_path = log_path
        self._reset_stream_state()
        self._process = self._create_process(spec)
        self._process.start()

    def stop(self) -> None:
        if self._process is None or self._stopping:
            return
        self._stopping = True
        self._signal_group("SIGTERM", self._process.terminate)
        self._kill_timer.start(self._kill_grace_ms)

    def shutdown(self) -> None:
        """Stop the run and block until it has exited; call before the application quits.

        Without this the QProcess destructor would kill only `accelerate`, leaving the
        training child (in its own session) running on the GPU.
        """
        process = self._process
        if process is None:
            return
        self.stop()
        if not process.waitForFinished(self._kill_grace_ms):
            self._force_kill()
            process.waitForFinished(SHUTDOWN_KILL_WAIT_MS)

    def _create_process(self, spec: LaunchSpec) -> QProcess:
        process = QProcess(self)
        process.setProgram(str(spec.program))
        process.setArguments(list(spec.arguments))
        process.setWorkingDirectory(str(spec.working_dir))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        for name, value in spec.env.items():
            environment.insert(name, value)
        process.setProcessEnvironment(environment)
        if os.name == "posix":
            process.setUnixProcessParameters(QProcess.UnixProcessFlag.CreateNewSession)
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        return process

    def _reset_stream_state(self) -> None:
        self._pending = ""
        self._decoder.reset()
        self._tail.clear()
        self._stopping = False

    def _on_output(self) -> None:
        if self._process is None:
            return
        data = self._decoder.decode(self._process.readAllStandardOutput().data())
        lines, self._pending = split_lines(self._pending, data)
        for line in lines:
            self._emit_line(line)
        self._flush_log()

    def _emit_line(self, line: str) -> None:
        self._write_log(line + "\n")
        self._tail.append(line)
        self.line_received.emit(line)
        event = parse_line(line)
        if event is not None:
            self.event_received.emit(event)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._on_output()
        remainder = (self._pending + self._decoder.decode(b"", final=True)).rstrip("\r")
        if remainder:
            self._emit_line(remainder)
        self._flush_log()
        crashed = exit_status == QProcess.ExitStatus.CrashExit
        outcome = RunOutcome(exit_code, crashed, self._stopping, self._log_path or Path())
        self._cleanup()
        self.finished.emit(outcome)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart or self._process is None:
            return
        message = f"Could not start {self._process.program()}: {self._process.errorString()}"
        self._cleanup()
        self.failed_to_start.emit(message)

    def _force_kill(self) -> None:
        if self._process is not None and self._stopping:
            self._signal_group("SIGKILL", self._process.kill)

    def _signal_group(self, signal_name: str, fallback: Callable[[], None]) -> None:
        """Signal the process group on POSIX; elsewhere fall back to QProcess terminate/kill."""
        pid = self._process.processId() if self._process is not None else 0
        signum = getattr(signal, signal_name, None)
        if os.name != "posix" or pid <= 0 or signum is None:
            fallback()
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signum)

    def _write_log(self, text: str) -> None:
        if self._log is None:
            return
        try:
            self._log.write(text)
        except OSError as error:
            self._abandon_log(error)

    def _flush_log(self) -> None:
        # Once per output chunk rather than per line: tqdm produces thousands of lines.
        if self._log is None:
            return
        try:
            self._log.flush()
        except OSError as error:
            self._abandon_log(error)

    def _abandon_log(self, error: OSError) -> None:
        """Stop writing the log after the first failure but keep streaming to the UI."""
        log, self._log = self._log, None
        with contextlib.suppress(OSError):
            if log is not None:
                log.close()
        self.warning.emit(f"Log file {self._log_path} could not be written: {error}")

    def _cleanup(self) -> None:
        self._kill_timer.stop()
        if self._log is not None:
            with contextlib.suppress(OSError):
                self._log.close()
        if self._process is not None:
            self._process.deleteLater()
        self._log = None
        self._process = None
