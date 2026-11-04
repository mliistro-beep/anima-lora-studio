"""A whole training run in the real MainWindow: the real TrainingRunner starts a fake
accelerate that reads the training TOML the app wrote, prints sd-scripts' progress lines and
writes the checkpoints and preview images where sd-scripts would."""

import shlex
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from anima_trainer.runtime.data_dir import open_data_dir
from anima_trainer.runtime.install_plan import InstallPlan
from anima_trainer.system.gpu_memory import GpuMemory
from anima_trainer.system.gpu_worker import GpuDetector
from anima_trainer.training.runner import TrainingRunner
from anima_trainer.ui.main_window import MainWindow
from anima_trainer.workflow.generator_session import GeneratorSession
from anima_trainer.workflow.install_session import InstallSession
from anima_trainer.workflow.training_session import SessionState, TrainingSession
from tests.e2e.helpers import GPUS, INSTALL_STEPS, NOW, make_runtime
from tests.fakes import FakeDownloader, FakeInstaller, FakeRunner

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(sys.platform == "win32", reason="the fake accelerate is a shell script"),
]
FAKE = Path(__file__).parents[1] / "fixtures" / "fake_accelerate.py"
PROJECT = "e2e_run"
EPOCHS = 2
TOTAL_STEPS = 4  # fake_accelerate.py STEPS_PER_EPOCH x EPOCHS
RUN_TIMEOUT_MS = 20000
GPU_MEMORY = (GpuMemory(0, 12288, 11000), GpuMemory(1, 12288, 11000))
PROMPTS = ("a girl", "a cat")


def _fake_accelerate(path, exit_code):
    """The venv's accelerate: fake_accelerate.py in train mode, given the app's arguments."""
    command = " ".join(shlex.quote(part) for part in (sys.executable, str(FAKE), "train"))
    path.write_text(f'#!/bin/sh\nexec {command} {exit_code} "$@"\n')
    path.chmod(0o755)


@pytest.fixture
def make_app(qtbot, tmp_path, dataset):
    """Builds the window on a runtime whose fake accelerate exits with `exit_code`."""

    def build(exit_code=0):
        config = {"project_name": PROJECT, "max_train_epochs": EPOCHS, "save_every_n_epochs": 1}
        paths = make_runtime(tmp_path / "data", dataset, config)
        _fake_accelerate(paths.venv_accelerate, exit_code)
        store = QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat)
        opened = open_data_dir(paths.root)
        session = TrainingSession(
            opened.paths, opened.settings, TrainingRunner(), FakeDownloader(), clock=lambda: NOW,
            kernel_log=lambda: None,
        )  # fmt: skip
        detector = GpuDetector(lambda _paths: GPUS, lambda: GPU_MEMORY)
        install = InstallSession(
            FakeInstaller(), plan=lambda _paths: InstallPlan(INSTALL_STEPS, ()), clock=lambda: NOW
        )
        generator = GeneratorSession(FakeRunner(), clock=lambda: NOW)
        with qtbot.waitSignal(detector.memory_detected, timeout=RUN_TIMEOUT_MS):
            window = MainWindow(session, detector, store, opened, install, generator=generator)
        qtbot.addWidget(window)
        return window, paths

    return build


def _turn_previews_on(window):
    window.previews_tab.form.field_widget("enabled").setChecked(True)
    window.previews_tab.prompts.setPlainText("\n".join(PROMPTS))


def _train(qtbot, window):
    """Configure and Start, then wait for the run to end (states recorded from before the
    click, so a run shorter than a poll cannot slip past the wait)."""
    window.training_tab.configure_button.click()
    assert "OK: Configuration complete" in window.training_tab.status.text()
    states = []
    window.session.state_changed.connect(states.append)
    window.training_tab.start_button.click()
    qtbot.waitUntil(
        lambda: SessionState.TRAINING in states and states[-1] is SessionState.IDLE,
        timeout=RUN_TIMEOUT_MS,
    )


def _open_generator_checkpoints(window):
    """Opens the Generator tab (which re-reads the output folder) and returns its labels."""
    window.tabs.setCurrentWidget(window.generator_tab)
    return [checkpoint.label for checkpoint in window.generator_tab.checkpoints()]


def test_a_run_with_previews_shows_progress_checkpoints_and_the_gallery(qtbot, make_app):
    window, paths = make_app()
    stale = paths.root / "out" / "sample" / f"{PROJECT}_e000003_00_20200101000000_1.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"from an earlier run")
    _turn_previews_on(window)

    _train(qtbot, window)

    tab = window.training_tab
    assert "OK: Training completed successfully!" in tab.log.text()
    progress = tab.progress
    assert progress.epoch_label.text() == f"Epoch {EPOCHS} / {EPOCHS}"
    assert (progress.bar.value(), progress.bar.maximum()) == (TOTAL_STEPS, TOTAL_STEPS)
    assert progress.loss_label.text() == "Loss: 0.1200"
    assert progress.checkpoint_label.text() == f"Last checkpoint: {PROJECT}.safetensors"
    assert _open_generator_checkpoints(window) == ["Epoch 1", "Final"]
    window.tabs.setCurrentWidget(window.previews_tab)
    gallery = window.previews_tab.gallery
    rows = [gallery.verticalHeaderItem(row).text() for row in range(gallery.rowCount())]
    columns = [gallery.horizontalHeaderItem(col).text() for col in range(gallery.columnCount())]
    assert rows == ["Epoch 1", "Epoch 2"]  # not the stale epoch 3 of an earlier run
    assert columns == ["1: a girl", "2: a cat"]
    for row in range(EPOCHS):
        for column in range(len(PROMPTS)):
            assert gallery.image_at(row, column).name.startswith(
                f"{PROJECT}_e{row + 1:06d}_{column:02d}_"
            )
            assert not gallery.item(row, column).icon().isNull()
    assert list(paths.logs_dir.glob(f"{PROJECT}_*.log"))


def test_a_run_without_previews_writes_no_preview_images(qtbot, make_app):
    window, paths = make_app()

    _train(qtbot, window)

    assert "OK: Training completed successfully!" in window.training_tab.log.text()
    assert _open_generator_checkpoints(window) == ["Epoch 1", "Final"]
    assert not (paths.root / "out" / "sample").exists()
    assert window.previews_tab.gallery.rowCount() == 0


def test_a_failed_run_reports_the_exit_code_and_no_success(qtbot, make_app):
    window, _paths = make_app(exit_code=1)

    _train(qtbot, window)

    log = window.training_tab.log.text()
    assert "ERROR: Training failed (exit code: 1)" in log
    assert "OK: Training completed successfully!" not in log
    assert window.training_tab.start_button.isEnabled()
