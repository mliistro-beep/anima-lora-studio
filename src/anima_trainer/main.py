"""Application entry point."""

import signal
import sys
from collections.abc import Callable
from types import FrameType

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication

from anima_trainer.models.download_worker import ModelDownloader
from anima_trainer.settings.citron_root import (
    DEFAULT_ROOT_CANDIDATES,
    load_citron_root,
    open_citron_root,
)
from anima_trainer.system.gpu_info import detect_gpus
from anima_trainer.system.gpu_worker import Detect, GpuDetector
from anima_trainer.training.runner import TrainingRunner
from anima_trainer.ui.main_window import APP_NAME, MainWindow
from anima_trainer.workflow.training_session import TrainingSession

# QSettings file: ~/.config/anima-trainer/anima-trainer.conf on Linux.
SETTINGS_ORGANIZATION = "anima-trainer"
SETTINGS_APPLICATION = "anima-trainer"
QUIT_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGINT", "SIGTERM") if hasattr(signal, name)
)
# Python runs signal handlers only between bytecodes; this wakes it inside Qt's event loop.
SIGNAL_POLL_MS = 200

# APP_NAME is re-exported: it was importable from here before the window had its own module.
__all__ = ["APP_NAME", "create_window", "install_quit_signals", "main"]


def create_window(store: QSettings, detect: Detect = detect_gpus) -> MainWindow:
    # Without a saved or found root, show the usual location so Settings can correct it.
    root = load_citron_root(store) or DEFAULT_ROOT_CANDIDATES[0]
    opened = open_citron_root(root)
    session = TrainingSession(opened.paths, opened.settings, TrainingRunner(), ModelDownloader())
    return MainWindow(session, GpuDetector(detect), store, opened)


def install_quit_signals(quit_app: Callable[[], None]) -> Callable[[], None]:
    """Make Ctrl+C and SIGTERM call `quit_app`; returns a function restoring the old handlers."""
    previous = {signum: signal.getsignal(signum) for signum in QUIT_SIGNALS}

    def handle(_signum: int, _frame: FrameType | None) -> None:
        quit_app()

    for signum in QUIT_SIGNALS:
        signal.signal(signum, handle)

    def restore() -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    return restore


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName(SETTINGS_ORGANIZATION)
    app.setApplicationName(SETTINGS_APPLICATION)
    app.setApplicationDisplayName(APP_NAME)
    window = create_window(QSettings())
    app.aboutToQuit.connect(window.shutdown)
    install_quit_signals(app.quit)
    wake = QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(SIGNAL_POLL_MS)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
