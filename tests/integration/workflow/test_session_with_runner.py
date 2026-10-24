"""TrainingSession and MainWindow with the real TrainingRunner and a fake accelerate."""

import shlex
import sys
from datetime import datetime
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox

from anima_trainer.models.download_worker import ModelDownloader
from anima_trainer.runtime.data_dir import open_data_dir
from anima_trainer.runtime.installer import RuntimeInstaller
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.settings.store import save_settings
from anima_trainer.system.gpu_worker import GpuDetector
from anima_trainer.training.runner import TrainingRunner
from anima_trainer.training.start_service import StartRequest
from anima_trainer.ui.main_window import MainWindow
from anima_trainer.workflow.install_session import InstallSession
from anima_trainer.workflow.training_session import SessionState, TrainingSession
from tests.process_helpers import wait_until_dead

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="the fake accelerate is a shell script"),
]
FAKE = Path(__file__).parents[2] / "fixtures" / "fake_accelerate.py"
LOG_FIXTURE = Path(__file__).parents[2] / "fixtures" / "citron" / "training_log_excerpt.log"
TIMEOUT_MS = 15000
REQUEST = StartRequest(gpu_index="0", cpu_threads=1, base_model="anima-base-v1.0")


def _citron(tmp_path, *fake_args):
    """A data folder whose venv `accelerate` runs fake_accelerate.py with fake_args."""
    paths = open_data_dir(tmp_path / "data").paths
    for file in (
        paths.train_script,
        paths.accelerate_config_file,
        paths.qwen3_model,
        paths.vae_model,
        paths.dit_model("anima-base-v1.0"),
    ):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"")
    command = " ".join(shlex.quote(part) for part in (sys.executable, str(FAKE), *fake_args))
    paths.venv_accelerate.parent.mkdir(parents=True, exist_ok=True)
    paths.venv_accelerate.write_text(f"#!/bin/sh\nexec {command}\n")
    paths.venv_accelerate.chmod(0o755)
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.png").write_bytes(b"")
    (images / "a.txt").write_text("caption")
    settings = TrainingSettings(
        project_name="proj", image_directory=str(images), output_directory=str(tmp_path / "out")
    )
    save_settings(paths.config_file, settings)
    return paths, settings


def test_real_run_streams_progress_and_reports_success(qtbot, tmp_path):
    paths, settings = _citron(tmp_path, "replay", str(LOG_FIXTURE), "0")
    session = TrainingSession(paths, settings, TrainingRunner(), ModelDownloader())
    log, progress = [], []
    session.log_line.connect(log.append)
    session.progress_changed.connect(progress.append)
    assert session.configure(settings).success

    session.start(REQUEST)
    qtbot.waitUntil(lambda: session.state is SessionState.IDLE, timeout=TIMEOUT_MS)

    assert "OK: Training completed successfully!" in log
    assert progress[-1].epoch is not None
    assert list(paths.logs_dir.glob("proj_*.log"))


def test_closing_the_window_kills_the_training_child(qtbot, tmp_path, monkeypatch):
    pid_file = tmp_path / "child.pid"
    paths, settings = _citron(tmp_path, "sleep", str(pid_file))
    session = TrainingSession(
        paths, settings, TrainingRunner(), ModelDownloader(), clock=lambda: datetime(2026, 1, 1)
    )
    store = QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat)
    install = InstallSession(RuntimeInstaller())
    detector = GpuDetector(lambda _p: ("0",))
    window = MainWindow(session, detector, store, open_data_dir(paths.root), install)
    qtbot.addWidget(window)
    window.training_tab.configure_button.click()
    window.training_tab.start_button.click()
    qtbot.waitUntil(pid_file.exists, timeout=TIMEOUT_MS)
    qtbot.waitUntil(lambda: pid_file.read_text().strip() != "", timeout=TIMEOUT_MS)
    assert window.session.state is SessionState.TRAINING
    child_pid = int(pid_file.read_text())
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    assert window.close()

    assert wait_until_dead(child_pid)
