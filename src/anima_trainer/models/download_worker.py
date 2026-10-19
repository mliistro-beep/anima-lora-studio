"""Runs download_file() on a background QThread and reports through Qt signals."""

import threading
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from anima_trainer.models.downloader import (
    CHUNK_SIZE,
    DownloadCancelled,
    DownloadError,
    DownloadProgress,
    download_file,
)


@dataclass(frozen=True)
class _Result:
    kind: str  # "finished", "failed" or "cancelled"
    detail: str = ""


class _DownloadThread(QThread):
    # object instead of int: model files exceed the 32-bit range of int signals.
    progress = Signal(object, object)

    def __init__(self, url: str, destination: Path, chunk_size: int) -> None:
        super().__init__()
        self._url = url
        self._destination = destination
        self._chunk_size = chunk_size
        self._cancel = threading.Event()
        # Written once by run() before the thread ends, read after QThread.finished.
        self.result = _Result("failed", "Download did not run")

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            path = download_file(
                self._url,
                self._destination,
                on_progress=self._report,
                is_cancelled=self._cancel.is_set,
                chunk_size=self._chunk_size,
            )
        except DownloadCancelled:
            self.result = _Result("cancelled")
        except DownloadError as error:
            self.result = _Result("failed", str(error))
        except Exception as error:  # noqa: BLE001 - a thread must never die silently
            self.result = _Result("failed", f"Unexpected download error: {error!r}")
        else:
            self.result = _Result("finished", str(path))

    def _report(self, progress: DownloadProgress) -> None:
        self.progress.emit(progress.downloaded, progress.total)


class ModelDownloader(QObject):
    """progress(downloaded, total or None), then exactly one of finished/failed/cancelled.

    The terminal signal is emitted after the worker thread has fully stopped, so
    is_running() is already False and a slot may start a new download right away.
    Call shutdown() before the application exits.
    """

    progress = Signal(object, object)
    finished = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, chunk_size: int = CHUNK_SIZE, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._chunk_size = chunk_size
        self._thread: _DownloadThread | None = None

    def is_running(self) -> bool:
        return self._thread is not None

    def start(self, url: str, destination: Path) -> None:
        if self._thread is not None:
            raise RuntimeError("A download is already running")
        thread = _DownloadThread(url, destination, self._chunk_size)
        thread.progress.connect(self.progress)
        thread.finished.connect(self._on_thread_done)
        self._thread = thread
        thread.start()

    def cancel(self) -> None:
        if self._thread is not None:
            self._thread.cancel()

    def shutdown(self) -> None:
        """Cancel any running download and block until its thread has exited."""
        thread = self._thread
        if thread is None:
            return
        thread.cancel()
        thread.wait()
        self._on_thread_done()

    def _on_thread_done(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.wait()
        self._thread = None
        result = thread.result
        thread.deleteLater()
        if result.kind == "finished":
            self.finished.emit(result.detail)
        elif result.kind == "cancelled":
            self.cancelled.emit()
        else:
            self.failed.emit(result.detail)
