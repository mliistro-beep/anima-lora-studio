"""Application entry point."""

import sys

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication

from anima_trainer.models.download_worker import ModelDownloader
from anima_trainer.runtime.data_dir import (
    SETTINGS_APPLICATION,
    SETTINGS_ORGANIZATION,
    default_data_dir,
    load_data_dir,
    open_data_dir,
)
from anima_trainer.runtime.installer import RuntimeInstaller
from anima_trainer.system.gpu_info import detect_gpus
from anima_trainer.system.gpu_lock import GpuLock
from anima_trainer.system.gpu_memory import read_all_gpu_memory
from anima_trainer.system.gpu_worker import Detect, GpuDetector
from anima_trainer.system.memory_safety import read_memory_safety
from anima_trainer.system.quit_signals import SIGNAL_POLL_MS, install_quit_signals
from anima_trainer.system.resource_monitor import ResourceMonitor
from anima_trainer.training.runner import TrainingRunner
from anima_trainer.ui.app_icon import DESKTOP_FILE_NAME, app_icon
from anima_trainer.ui.main_window import APP_NAME, MainWindow
from anima_trainer.workflow.generator_session import GeneratorSession
from anima_trainer.workflow.install_session import InstallSession
from anima_trainer.workflow.training_session import TrainingSession

# APP_NAME is re-exported: it was importable from here before the window had its own module.
__all__ = ["APP_NAME", "create_window", "install_quit_signals", "main"]


def create_window(store: QSettings, detect: Detect = detect_gpus) -> MainWindow:
    opened = open_data_dir(load_data_dir(store, default_data_dir()))
    runner = TrainingRunner(monitor=ResourceMonitor())
    lock = GpuLock()  # one per process: a generation may join a training of this window
    session = TrainingSession(opened.paths, opened.settings, runner, ModelDownloader(), lock=lock)
    install = InstallSession(RuntimeInstaller(), lock=lock)
    # The memory check reads /proc each time the data folder is shown (start and Apply); the
    # name is looked up at call time so tests can replace read_memory_safety.
    return MainWindow(
        session,
        GpuDetector(detect, read_all_gpu_memory),
        store,
        opened,
        install,
        generator=GeneratorSession(TrainingRunner(), lock=lock),
        memory_check=lambda: read_memory_safety(),
    )


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName(SETTINGS_ORGANIZATION)
    app.setApplicationName(SETTINGS_APPLICATION)
    app.setApplicationDisplayName(APP_NAME)
    app.setDesktopFileName(DESKTOP_FILE_NAME)
    app.setWindowIcon(app_icon())
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
