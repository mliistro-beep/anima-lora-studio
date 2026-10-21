import json
from datetime import datetime
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox

from anima_trainer.runtime.data_dir import load_data_dir, open_data_dir
from anima_trainer.settings.store import load_settings
from anima_trainer.system.gpu_worker import GpuDetector
from anima_trainer.training.log_parser import EpochStarted
from anima_trainer.training.runner import RunOutcome
from anima_trainer.ui.main_window import MainWindow
from anima_trainer.workflow.training_session import SessionState, TrainingSession
from tests.fakes import FakeDownloader, FakeRunner

pytestmark = pytest.mark.e2e
NOW = datetime(2026, 10, 20, 9, 0, 0)
TIMEOUT_MS = 5000
GPUS = ("0: NVIDIA A", "1: NVIDIA B")


def _make_runtime(root, dataset, config=None):
    opened = open_data_dir(root)
    paths = opened.paths
    for file in (
        paths.train_script,
        paths.venv_accelerate,
        paths.accelerate_config_file,
        paths.qwen3_model,
        paths.vae_model,
        paths.dit_model("anima-base-v1.0"),
    ):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"")
    saved = {
        "project_name": "saved_proj",
        "image_directory": str(dataset),
        "output_directory": str(root / "out"),
        "network_dim": 16,
        "gpu_index": "1",
        "cache_latents": False,
        **(config or {}),
    }
    paths.config_file.write_text(json.dumps(saved))
    return paths


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"")
    (folder / "a.txt").write_text("caption")
    return folder


@pytest.fixture
def make_window(qtbot, tmp_path):
    def build(root, detect=lambda _paths: GPUS):
        store = QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat)
        opened = open_data_dir(root)
        runner, downloader = FakeRunner(), FakeDownloader()
        session = TrainingSession(
            opened.paths,
            opened.settings,
            runner,
            downloader,
            clock=lambda: NOW,
            kernel_log=lambda: None,
        )
        detector = GpuDetector(detect)
        with qtbot.waitSignal(detector.detected, timeout=TIMEOUT_MS):
            window = MainWindow(session, detector, store, opened)
        qtbot.addWidget(window)
        return window, runner, downloader, store

    return build


def test_window_has_the_three_tabs_and_saved_values(make_window, tmp_path, dataset):
    _make_runtime(tmp_path / "data", dataset)

    window, *_ = make_window(tmp_path / "data")

    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert titles == ["Training", "Advanced Settings", "Settings"]
    basic = window.training_tab.form.values()
    assert (basic["project_name"], basic["network_dim"]) == ("saved_proj", 16)
    assert window.advanced_tab.form.values()["cache_latents"] is False


def test_detected_gpus_select_the_saved_index(make_window, tmp_path, dataset):
    _make_runtime(tmp_path / "data", dataset)

    window, *_ = make_window(tmp_path / "data")

    assert window.training_tab.form.field_widget("gpu_index").currentText() == "1: NVIDIA B"


def test_missing_data_dir_opens_settings_and_blocks_training(make_window, tmp_path):
    window, *_ = make_window(tmp_path / "missing")

    assert window.tabs.currentWidget() is window.settings_tab
    assert not window.training_tab.configure_button.isEnabled()
    assert not window.training_tab.start_button.isEnabled()
    assert "not installed" in window.settings_tab.problems_label.text()


def test_configure_writes_tomls_and_saves_form_values(make_window, tmp_path, dataset):
    paths = _make_runtime(tmp_path / "data", dataset)
    window, *_ = make_window(paths.root)
    window.training_tab.form.field_widget("network_dim").setValue(8)

    window.training_tab.configure_button.click()

    assert "OK: Configuration complete, ready to train." in window.training_tab.status.text()
    saved = load_settings(paths.config_file).settings
    assert saved.network_dim == 8
    assert Path(saved.last_train_config).exists()


def test_invalid_form_value_is_reported_without_configuring(make_window, tmp_path, dataset):
    paths = _make_runtime(tmp_path / "data", dataset)
    window, *_ = make_window(paths.root)
    window.training_tab.form.field_widget("learning_rate").setText("abc")

    window.training_tab.configure_button.click()

    assert window.training_tab.status.text().startswith("ERROR: Learning Rate: ")
    assert load_settings(paths.config_file).settings.last_train_config == ""


def _configured_window(make_window, tmp_path, dataset):
    paths = _make_runtime(tmp_path / "data", dataset)
    window, runner, downloader, store = make_window(paths.root)
    window.training_tab.configure_button.click()
    return window, runner, downloader, paths


def test_start_streams_log_and_progress_then_finishes(make_window, tmp_path, dataset):
    window, runner, _, _ = _configured_window(make_window, tmp_path, dataset)
    tab = window.training_tab

    tab.start_button.click()

    assert len(runner.started) == 1
    spec, _log_path = runner.started[0]
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "1"
    assert not tab.start_button.isEnabled()
    assert tab.stop_button.isEnabled()
    assert not window.settings_tab.apply_button.isEnabled()

    runner.line_received.emit("epoch 1/10")
    runner.event_received.emit(EpochStarted(1, 10))
    runner.finished.emit(RunOutcome(0, False, False, Path("/log")))

    assert "epoch 1/10" in tab.log.text()
    assert tab.progress.epoch_label.text() == "Epoch 1 / 10"
    assert "OK: Training completed successfully!" in tab.log.text()
    assert tab.start_button.isEnabled()
    assert not tab.stop_button.isEnabled()


def test_stop_button_stops_the_run(make_window, tmp_path, dataset):
    window, runner, _, _ = _configured_window(make_window, tmp_path, dataset)
    window.training_tab.start_button.click()

    window.training_tab.stop_button.click()

    assert runner.stopped == 1
    assert window.session.state is SessionState.STOPPING


def test_apply_data_dir_saves_it_and_reloads_settings(make_window, tmp_path, dataset):
    _make_runtime(tmp_path / "data", dataset)
    other = _make_runtime(tmp_path / "other", dataset, {"project_name": "other_proj"})
    window, _, _, store = make_window(tmp_path / "data")

    window.settings_tab.root_picker.set_path(str(other.root))
    window.settings_tab.apply_button.click()

    assert load_data_dir(store, default=Path("unused")) == other.root
    assert window.session.paths == other
    assert window.training_tab.form.values()["project_name"] == "other_proj"


def test_download_button_downloads_missing_dit(make_window, tmp_path, dataset):
    paths = _make_runtime(tmp_path / "data", dataset)
    window, _, downloader, _ = make_window(paths.root)

    window.settings_tab.download_buttons["anima-preview"].click()

    assert downloader.started[0][1] == paths.dit_model("anima-preview")
    assert window.settings_tab.cancel_button.isEnabled()
    window.settings_tab.cancel_button.click()
    assert downloader.cancels == 1


def test_present_dit_cannot_be_downloaded_again(make_window, tmp_path, dataset):
    _make_runtime(tmp_path / "data", dataset)

    window, *_ = make_window(tmp_path / "data")

    assert not window.settings_tab.download_buttons["anima-base-v1.0"].isEnabled()


def test_close_while_training_asks_and_can_be_refused(make_window, tmp_path, dataset, monkeypatch):
    window, runner, _, _ = _configured_window(make_window, tmp_path, dataset)
    window.training_tab.start_button.click()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)

    assert not window.close()
    assert runner.shut_down == 0


def test_close_while_training_shuts_down_when_confirmed(
    make_window, tmp_path, dataset, monkeypatch
):
    window, runner, downloader, _ = _configured_window(make_window, tmp_path, dataset)
    window.training_tab.start_button.click()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    assert window.close()
    assert (runner.shut_down, downloader.shut_down) == (1, 1)


def test_close_when_idle_does_not_ask(make_window, tmp_path, dataset, monkeypatch):
    _make_runtime(tmp_path / "data", dataset)
    window, runner, _, _ = make_window(tmp_path / "data")

    def fail(*args, **kwargs):
        raise AssertionError("should not ask")

    monkeypatch.setattr(QMessageBox, "question", fail)

    assert window.close()
    assert runner.shut_down == 1


def test_start_ignores_invalid_fields_it_does_not_use(make_window, tmp_path, dataset):
    # Like Citron, Start only reads GPU, CPU threads and base model besides the saved TOMLs.
    window, runner, _, _ = _configured_window(make_window, tmp_path, dataset)
    window.training_tab.form.field_widget("learning_rate").setText("abc")

    window.training_tab.start_button.click()

    assert len(runner.started) == 1


def test_progress_panel_is_reset_when_a_download_ends(make_window, tmp_path, dataset):
    _make_runtime(tmp_path / "data", dataset)
    window, _, downloader, _ = make_window(tmp_path / "data")
    window.settings_tab.download_buttons["anima-preview"].click()
    downloader.progress.emit(10, None)

    downloader.cancelled.emit()

    panel = window.training_tab.progress
    assert panel.epoch_label.text() == "Epoch - / -"
    assert panel.bar.maximum() > 0


def test_downloads_are_disabled_without_a_data_folder(make_window, tmp_path):
    window, *_ = make_window(tmp_path / "missing")

    assert not any(button.isEnabled() for button in window.settings_tab.download_buttons.values())


def test_gpu_detection_failure_is_shown(make_window, tmp_path, dataset):
    _make_runtime(tmp_path / "data", dataset)

    def broken(_paths):
        raise ValueError("no torch")

    window, *_ = make_window(tmp_path / "data", detect=broken)

    assert "WARN: GPU detection failed" in window.training_tab.status.text()


def test_shutdown_runs_once(make_window, tmp_path, dataset):
    _make_runtime(tmp_path / "data", dataset)
    window, runner, downloader, _ = make_window(tmp_path / "data")

    window.shutdown()
    window.shutdown()

    assert (runner.shut_down, downloader.shut_down) == (1, 1)
