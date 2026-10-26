"""Runs one long job at a time (a download, a file check) on a QThread with Qt signals.

The job gets a progress callback and an is_cancelled() check. Its known errors become a
`failed` message, `cancel_error` becomes `cancelled`, anything else a generic failure.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, QThread, Signal

Report = Callable[[object, object], None]
Job = Callable[[Report, Callable[[], bool]], object]


@dataclass(frozen=True)
class _Result:
    kind: str  # "finished", "failed" or "cancelled"
    value: object = None  # the job's return value, or the failure message


class _TaskThread(QThread):
    # object instead of int: model files exceed the 32-bit range of int signals.
    progress = Signal(object, object)

    def __init__(
        self,
        job: Job,
        known_errors: tuple[type[Exception], ...],
        cancel_error: type[Exception],
    ) -> None:
        super().__init__()
        self._job = job
        self._known_errors = known_errors
        self._cancel_error = cancel_error
        self._cancel = threading.Event()
        # Written once by run() before the thread ends, read after QThread.finished.
        self.result = _Result("failed", "The task did not run")

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            value = self._job(self.progress.emit, self._cancel.is_set)
        except self._cancel_error:
            self.result = _Result("cancelled")
        except self._known_errors as error:
            self.result = _Result("failed", str(error))
        except Exception as error:  # noqa: BLE001 - a thread must never die silently
            self.result = _Result("failed", f"Unexpected error: {error!r}")
        else:
            self.result = _Result("finished", value)


class BackgroundTask(QObject):
    """progress(done, total or None), then exactly one of finished/failed/cancelled.

    The terminal signal is emitted after the worker thread has fully stopped, so
    is_running() is already False and a slot may start a new job right away.
    Call shutdown() before the application exits.
    """

    progress = Signal(object, object)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        known_errors: tuple[type[Exception], ...],
        cancel_error: type[Exception],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._known_errors = known_errors
        self._cancel_error = cancel_error
        self._thread: _TaskThread | None = None

    def is_running(self) -> bool:
        return self._thread is not None

    def cancel(self) -> None:
        if self._thread is not None:
            self._thread.cancel()

    def shutdown(self) -> None:
        """Cancel any running job and block until its thread has exited."""
        thread = self._thread
        if thread is None:
            return
        thread.cancel()
        thread.wait()
        self._on_thread_done(thread)

    def _run(self, job: Job) -> None:
        if self._thread is not None:
            raise RuntimeError("A task is already running")
        thread = _TaskThread(job, self._known_errors, self._cancel_error)
        # Bind each event to its own thread: events still queued from a thread that was
        # already shut down must not reach the caller or act on a newer job.
        thread.progress.connect(lambda done, total: self._forward_progress(thread, done, total))
        thread.finished.connect(lambda: self._on_thread_done(thread))
        self._thread = thread
        thread.start()

    def _forward_progress(self, thread: _TaskThread, done: object, total: object) -> None:
        if thread is self._thread:
            self.progress.emit(done, total)

    def _on_thread_done(self, thread: _TaskThread) -> None:
        if thread is not self._thread:
            return
        thread.wait()
        self._thread = None
        result = thread.result
        thread.deleteLater()
        if result.kind == "finished":
            self.finished.emit(result.value)
        elif result.kind == "cancelled":
            self.cancelled.emit()
        else:
            self.failed.emit(str(result.value))
