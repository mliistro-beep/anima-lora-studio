"""Samples RAM, swap and GPU usage at once and then every minute while a training runs.

nvidia-smi runs as an asynchronous QProcess with a timeout, so a slow or hung driver never
blocks the UI and never stops the samples (after the timeout the line has RAM and swap only);
/proc/meminfo is read on the UI thread, which is instant.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from anima_trainer.system.gpu_info import NVIDIA_SMI_TIMEOUT_S
from anima_trainer.system.resource_usage import (
    NVIDIA_SMI_USAGE_ARGUMENTS,
    GpuUsage,
    MemoryUsage,
    parse_gpu_usage,
    parse_meminfo,
    resource_line,
)

SAMPLE_INTERVAL_MS = 60_000
SAMPLE_TIMEOUT_MS = NVIDIA_SMI_TIMEOUT_S * 1000
MEMINFO = Path("/proc/meminfo")


class ResourceMonitor(QObject):
    line_ready = Signal(str)

    def __init__(
        self,
        interval_ms: int = SAMPLE_INTERVAL_MS,
        program: str = "nvidia-smi",
        arguments: tuple[str, ...] = NVIDIA_SMI_USAGE_ARGUMENTS,
        meminfo: Path = MEMINFO,
        clock: Callable[[], datetime] = datetime.now,
        timeout_ms: int = SAMPLE_TIMEOUT_MS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._program = program
        self._arguments = arguments
        self._meminfo = meminfo
        self._clock = clock
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._sample)
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.setInterval(timeout_ms)
        self._timeout.timeout.connect(self._on_timeout)
        self._process: QProcess | None = None

    def is_sampling(self) -> bool:
        return self._process is not None

    def start(self) -> None:
        if self._timer.isActive():
            return
        self._timer.start()
        self._sample()

    def stop(self) -> None:
        self._timer.stop()
        self._timeout.stop()
        process, self._process = self._process, None
        if process is not None:
            _abandon(process)

    def _sample(self) -> None:
        if self._process is not None:  # the previous sample is still waiting for nvidia-smi
            return
        process = QProcess(self)
        process.setProgram(self._program)
        process.setArguments(list(self._arguments))
        # The slots check the sender, so a signal from an abandoned sample is ignored.
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        self._timeout.start()
        process.start()

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        process = self._process
        if process is None or self.sender() is not process:
            return
        ok = exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0
        stdout = bytes(process.readAllStandardOutput().data()).decode("utf-8", errors="replace")
        self._emit(parse_gpu_usage(stdout) if ok else ())

    def _on_error(self, error: QProcess.ProcessError) -> None:
        # Only a failed start ends a sample here: after a crash or a kill, finished follows.
        process = self._process
        if (
            error == QProcess.ProcessError.FailedToStart
            and process is not None
            and self.sender() is process
        ):
            self._emit(())

    def _on_timeout(self) -> None:
        # The RAM-only line is written now rather than on finished: a driver hang can leave
        # nvidia-smi in uninterruptible sleep, where the kill never ends it.
        self._emit(())

    def _emit(self, gpus: tuple[GpuUsage, ...]) -> None:
        self._timeout.stop()
        process, self._process = self._process, None
        if process is not None:
            _abandon(process)
        self.line_ready.emit(resource_line(self._clock(), self._read_memory(), gpus))

    def _read_memory(self) -> MemoryUsage | None:
        try:
            text = self._meminfo.read_text(encoding="ascii", errors="replace")
        except OSError:
            return None
        return parse_meminfo(text)


def _abandon(process: QProcess) -> None:
    """Kill a sample without waiting on the UI thread; it is deleted once it has ended."""
    process.kill()
    if process.state() == QProcess.ProcessState.NotRunning:
        process.deleteLater()
    else:
        process.finished.connect(process.deleteLater)
