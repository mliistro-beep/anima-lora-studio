"""Runs detect_gpus() off the UI thread: the torch probe can take up to a minute."""

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, QThread, Signal

from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.system.gpu_info import FALLBACK_CHOICES, detect_gpus

Detect = Callable[[RuntimePaths], tuple[str, ...]]

_LOG = logging.getLogger(__name__)


class _DetectThread(QThread):
    def __init__(self, paths: RuntimePaths, detect: Detect) -> None:
        super().__init__()
        self._paths = paths
        self._detect = detect
        # Written by run() before the thread ends, read after QThread.finished.
        self.choices: tuple[str, ...] = FALLBACK_CHOICES
        self.error: str | None = None

    def run(self) -> None:
        try:
            self.choices = self._detect(self._paths)
        except Exception as error:  # noqa: BLE001 - a thread must never die silently
            _LOG.exception("GPU detection failed for %s", self._paths.root)
            self.choices = FALLBACK_CHOICES
            self.error = f"GPU detection failed ({error!r}), using {', '.join(FALLBACK_CHOICES)}."


class GpuDetector(QObject):
    """detected(choices) for the latest start(); results of superseded runs are dropped.

    When detection fails, warning(message) comes first and detected() gets the fallback.
    """

    detected = Signal(object)  # tuple[str, ...]
    warning = Signal(str)

    def __init__(self, detect: Detect = detect_gpus, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._detect = detect
        self._latest: _DetectThread | None = None
        self._threads: tuple[_DetectThread, ...] = ()

    def is_running(self) -> bool:
        return bool(self._threads)

    def start(self, paths: RuntimePaths) -> None:
        thread = _DetectThread(paths, self._detect)
        thread.finished.connect(lambda: self._on_done(thread))
        self._latest = thread
        self._threads = (*self._threads, thread)
        thread.start()

    def shutdown(self) -> None:
        """Wait for every detection thread; call before the application exits."""
        self._latest = None
        for thread in self._threads:
            thread.wait()
            thread.deleteLater()
        self._threads = ()

    def _on_done(self, thread: _DetectThread) -> None:
        if thread not in self._threads:
            return
        thread.wait()
        self._threads = tuple(t for t in self._threads if t is not thread)
        thread.deleteLater()
        if thread is self._latest:
            self._latest = None
            if thread.error is not None:
                self.warning.emit(thread.error)
            self.detected.emit(thread.choices)
