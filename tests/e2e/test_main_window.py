import hashlib
import tomllib
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QMessageBox

from anima_trainer.models.catalog import SUPPORT_MODEL_URLS, VAE_FILENAME, ExpectedFile
from anima_trainer.models.check_worker import ModelFileChecker
from anima_trainer.models.linked_store import load_links
from anima_trainer.runtime.data_dir import load_data_dir, open_data_dir
from anima_trainer.runtime.install_plan import InstallPlan, verified_marker
from anima_trainer.settings.store import load_settings
from anima_trainer.system.gpu_worker import GpuDetector
from anima_trainer.training.log_parser import EpochStarted
from anima_trainer.training.runner import RunOutcome
from anima_trainer.ui.main_window import MainWindow
from anima_trainer.workflow.install_session import InstallSession
from anima_trainer.workflow.link_session import LinkSession
from anima_trainer.workflow.training_session import SessionState, TrainingSession
from tests.e2e.helpers import GPUS, INSTALL_STEPS, NOW, TIMEOUT_MS, make_runtime
from tests.fakes import FakeDownloader, FakeInstaller, FakeRunner

pytestmark = pytest.mark.e2e


@pytest.fixture
def installer():
    return FakeInstaller()


@pytest.fixture
def install_plan():
    """Mutable holder for the plan the install session gets: tests may replace it.

    "paths" records the folder of every plan request.
    """
    return {"plan": InstallPlan(INSTALL_STEPS, ()), "paths": []}


@pytest.fixture
def make_window(qtbot, tmp_path, installer, install_plan):
    def build(root, detect=lambda _paths: GPUS, expected=None):
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

        def plan(paths):
            install_plan["paths"].append(paths)
            return install_plan["plan"]

        install = InstallSession(installer, plan=plan, clock=lambda: NOW)
        links = None if expected is None else LinkSession(store, ModelFileChecker(), expected)
        with qtbot.waitSignal(detector.detected, timeout=TIMEOUT_MS):
            window = MainWindow(session, detector, store, opened, install, links=links)
        qtbot.addWidget(window)
        return window, runner, downloader, store

    return build


def test_window_has_the_tabs_and_saved_values(make_window, tmp_path, dataset):
    make_runtime(tmp_path / "data", dataset)

    window, *_ = make_window(tmp_path / "data")

    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert titles == ["Training", "Advanced Settings", "Previews", "Generator", "Settings"]
    basic = window.training_tab.form.values()
    assert (basic["project_name"], basic["network_dim"]) == ("saved_proj", 16)
    assert window.advanced_tab.form.values()["cache_latents"] is False


def test_detected_gpus_select_the_saved_index(make_window, tmp_path, dataset):
    make_runtime(tmp_path / "data", dataset)

    window, *_ = make_window(tmp_path / "data")

    assert window.training_tab.form.field_widget("gpu_index").currentText() == "1: NVIDIA B"


def test_missing_data_dir_opens_settings_and_blocks_training(make_window, tmp_path):
    window, *_ = make_window(tmp_path / "missing")

    assert window.tabs.currentWidget() is window.settings_tab
    assert not window.training_tab.configure_button.isEnabled()
    assert not window.training_tab.start_button.isEnabled()
    assert "not installed" in window.settings_tab.problems_label.text()


def test_configure_writes_tomls_and_saves_form_values(make_window, tmp_path, dataset):
    paths = make_runtime(tmp_path / "data", dataset)
    window, *_ = make_window(paths.root)
    window.training_tab.form.field_widget("network_dim").setValue(8)

    window.training_tab.configure_button.click()

    assert "OK: Configuration complete, ready to train." in window.training_tab.status.text()
    saved = load_settings(paths.config_file).settings
    assert saved.network_dim == 8
    assert Path(saved.last_train_config).exists()


def test_invalid_form_value_is_reported_without_configuring(make_window, tmp_path, dataset):
    paths = make_runtime(tmp_path / "data", dataset)
    window, *_ = make_window(paths.root)
    window.training_tab.form.field_widget("learning_rate").setText("abc")

    window.training_tab.configure_button.click()

    assert window.training_tab.status.text().startswith("ERROR: Learning Rate: ")
    assert load_settings(paths.config_file).settings.last_train_config == ""


def _configured_window(make_window, tmp_path, dataset):
    paths = make_runtime(tmp_path / "data", dataset)
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
    make_runtime(tmp_path / "data", dataset)
    other = make_runtime(tmp_path / "other", dataset, {"project_name": "other_proj"})
    window, _, _, store = make_window(tmp_path / "data")

    window.settings_tab.root_picker.set_path(str(other.root))
    window.settings_tab.apply_button.click()

    assert load_data_dir(store, default=Path("unused")) == other.root
    assert window.session.paths == other
    assert window.training_tab.form.values()["project_name"] == "other_proj"


def test_download_button_downloads_missing_dit(make_window, tmp_path, dataset):
    paths = make_runtime(tmp_path / "data", dataset)
    window, _, downloader, _ = make_window(paths.root)

    window.settings_tab.download_buttons["anima-preview"].click()

    assert downloader.started[0][1] == paths.dit_model("anima-preview")
    assert window.settings_tab.cancel_button.isEnabled()
    window.settings_tab.cancel_button.click()
    assert downloader.cancels == 1


def test_present_dit_cannot_be_downloaded_again(make_window, tmp_path, dataset):
    make_runtime(tmp_path / "data", dataset)

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
    make_runtime(tmp_path / "data", dataset)
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
    make_runtime(tmp_path / "data", dataset)
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
    make_runtime(tmp_path / "data", dataset)

    def broken(_paths):
        raise ValueError("no torch")

    window, *_ = make_window(tmp_path / "data", detect=broken)

    assert "WARN: GPU detection failed" in window.training_tab.status.text()


def test_shutdown_runs_once(make_window, tmp_path, dataset):
    make_runtime(tmp_path / "data", dataset)
    window, runner, downloader, _ = make_window(tmp_path / "data")

    window.shutdown()
    window.shutdown()

    assert (runner.shut_down, downloader.shut_down) == (1, 1)


def test_download_button_downloads_a_missing_vae(make_window, tmp_path, dataset):
    paths = make_runtime(tmp_path / "data", dataset)
    paths.vae_model.unlink()
    window, _, downloader, _ = make_window(paths.root)
    button = window.settings_tab.download_buttons[VAE_FILENAME]

    button.click()

    assert downloader.started == [(SUPPORT_MODEL_URLS[VAE_FILENAME], paths.vae_model)]
    paths.vae_model.write_bytes(b"")
    downloader.finished.emit(str(paths.vae_model))
    assert not window.settings_tab.download_buttons[VAE_FILENAME].isEnabled()


def test_fresh_folder_offers_install_and_runs_the_plan(make_window, tmp_path, installer):
    window, *_ = make_window(tmp_path / "data")
    tab = window.settings_tab

    assert tab.install_button.text() == "Install"
    assert tab.install_button.isEnabled()
    tab.install_button.click()

    assert installer.started == [INSTALL_STEPS]
    assert not tab.install_button.isEnabled()
    assert tab.cancel_install_button.isEnabled()
    assert not tab.apply_button.isEnabled()
    assert not any(button.isEnabled() for button in tab.download_buttons.values())
    assert not window.training_tab.configure_button.isEnabled()
    assert "Install log:" in tab.install_log.text()


def test_install_progress_is_shown_in_the_settings_tab(make_window, tmp_path, installer):
    window, *_ = make_window(tmp_path / "data")
    window.settings_tab.install_button.click()

    installer.step_started.emit(1, 1, INSTALL_STEPS[0])
    installer.line_received.emit("created venv")

    assert window.settings_tab.install_step_label.text() == "Step 1/1: Create the training venv"
    assert "created venv" in window.settings_tab.install_log.text()


def test_successful_install_enables_training_and_saves_the_folder(
    qtbot, make_window, tmp_path, dataset, installer
):
    root = tmp_path / "data"
    detected = []

    def detect(paths):
        detected.append(paths.root)
        return GPUS

    window, _, _, store = make_window(root, detect=detect)
    window.settings_tab.install_button.click()
    make_runtime(root, dataset)  # what a real install leaves behind

    installer.finish(True, "Installation completed.")

    qtbot.waitUntil(lambda: len(detected) == 2, timeout=TIMEOUT_MS)  # torch is there now

    tab = window.settings_tab
    assert tab.problems_label.text() == "OK: Training runtime installed."
    assert tab.install_button.text() == "Repair"
    assert tab.install_button.isEnabled()
    assert not tab.cancel_install_button.isEnabled()
    assert window.training_tab.configure_button.isEnabled()
    assert load_data_dir(store, tmp_path / "default") == root
    assert "OK: Training runtime installed." in window.training_tab.status.text()


def test_failed_install_keeps_training_blocked(make_window, tmp_path, installer):
    window, *_ = make_window(tmp_path / "data")
    window.settings_tab.install_button.click()

    installer.finish(False, "Check the GPU failed (exit code 1).")

    assert "ERROR: Check the GPU failed" in window.settings_tab.install_log.text()
    # The install log created the data folder; the runtime itself is still missing.
    assert "Training script not found" in window.settings_tab.problems_label.text()
    assert window.settings_tab.install_button.text() == "Install"
    assert window.settings_tab.install_button.isEnabled()
    assert not window.training_tab.configure_button.isEnabled()


def test_install_problems_are_shown_without_starting(
    make_window, tmp_path, installer, install_plan
):
    install_plan["plan"] = InstallPlan(INSTALL_STEPS, ("git not found.",))
    window, *_ = make_window(tmp_path / "data")

    window.settings_tab.install_button.click()

    assert installer.started == []
    assert "ERROR: git not found." in window.settings_tab.install_log.text()
    assert window.settings_tab.install_button.isEnabled()


def test_cancel_install_reaches_the_installer(make_window, tmp_path, installer):
    window, *_ = make_window(tmp_path / "data")
    window.settings_tab.install_button.click()

    window.settings_tab.cancel_install_button.click()

    assert installer.cancels == 1


def test_install_is_disabled_while_training(make_window, tmp_path, dataset):
    window, *_ = _configured_window(make_window, tmp_path, dataset)

    window.training_tab.start_button.click()

    assert window.session.state is SessionState.TRAINING
    assert not window.settings_tab.install_button.isEnabled()


def test_close_while_installing_asks_and_shuts_the_install_down(
    make_window, tmp_path, installer, monkeypatch
):
    window, *_ = make_window(tmp_path / "data")
    window.settings_tab.install_button.click()
    asked = []

    def answer(*_args):
        asked.append(1)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", answer)

    assert window.close()
    assert not window.isVisible()
    assert asked == [1]
    assert installer.shut_down == 1


def test_close_while_installing_can_be_refused(make_window, tmp_path, installer, monkeypatch):
    window, *_ = make_window(tmp_path / "data")
    window.show()
    window.settings_tab.install_button.click()
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.No)

    assert not window.close()
    assert window.isVisible()
    assert installer.shut_down == 0
    assert window.settings_tab.cancel_install_button.isEnabled()


def test_quit_during_install_does_not_restart_gpu_detection(make_window, tmp_path, installer):
    detected = []

    def detect(paths):
        detected.append(paths.root)
        return GPUS

    window, *_ = make_window(tmp_path / "data", detect=detect)
    window.settings_tab.install_button.click()

    window.shutdown()  # the installer reports "cancelled" from inside shutdown()

    assert installer.shut_down == 1
    assert len(detected) == 1  # only the detection at startup


def test_install_uses_the_folder_typed_in_the_picker(
    make_window, tmp_path, installer, install_plan
):
    window, *_ = make_window(tmp_path / "data")
    other = tmp_path / "other"
    window.settings_tab.root_picker.set_path(str(other))

    window.settings_tab.install_button.click()

    assert [paths.root for paths in install_plan["paths"]] == [other]
    assert window.session.paths.root == other
    assert installer.started == [INSTALL_STEPS]


def test_install_with_an_empty_folder_field_asks_for_a_folder(make_window, tmp_path, installer):
    window, *_ = make_window(tmp_path / "data")
    window.settings_tab.root_picker.set_path("  ")

    window.settings_tab.install_button.click()

    assert installer.started == []
    assert window.session.paths.root == tmp_path / "data"
    assert "Choose the data folder first." in window.settings_tab.problems_label.text()


def test_install_into_a_file_is_refused_without_switching_folder(make_window, tmp_path, installer):
    window, *_ = make_window(tmp_path / "data")
    a_file = tmp_path / "a_file"
    a_file.write_text("")
    window.settings_tab.root_picker.set_path(str(a_file / "sub"))

    window.settings_tab.install_button.click()

    assert installer.started == []
    assert window.session.paths.root == tmp_path / "data"
    assert "ERROR: Not a folder:" in window.settings_tab.install_log.text()


def test_failed_install_over_a_complete_runtime_is_not_reported_as_installed(
    make_window, tmp_path, dataset, installer
):
    # The files of an earlier install are there; only the last step failed.
    root = tmp_path / "data"
    paths = make_runtime(root, dataset)
    window, *_ = make_window(root)
    window.settings_tab.install_button.click()
    verified_marker(paths).unlink()

    installer.finish(False, "Check that torch sees the GPU failed (exit code 1).")

    status = window.training_tab.status.text()
    assert "OK: Training runtime installed." not in status
    assert "Repair" in window.settings_tab.problems_label.text()
    assert not window.training_tab.configure_button.isEnabled()


def test_install_end_keeps_the_earlier_status_lines(make_window, tmp_path, installer):
    window, *_ = make_window(tmp_path / "data")
    window.training_tab.status.set_lines(("ERROR: Learning rate: invalid",))
    window.settings_tab.install_button.click()

    installer.finish(False, "Cancelled.")

    status = window.training_tab.status.text()
    assert "ERROR: Learning rate: invalid" in status
    assert "Training script not found" in status


DIT_WEIGHTS = b"anima preview weights" * 50
PREVIEW_EXPECTED = {
    "anima-preview": ExpectedFile(hashlib.sha256(DIT_WEIGHTS).hexdigest(), len(DIT_WEIGHTS))
}


def _own_dit(tmp_path, content=DIT_WEIGHTS):
    path = tmp_path / "my-models" / "anima-preview.safetensors"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content)
    return path


def _use_file(qtbot, window, monkeypatch, key, path):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a: (str(path), ""))
    window.settings_tab.link_buttons[key].click()
    qtbot.waitUntil(
        lambda: not window.settings_tab.cancel_check_button.isEnabled(), timeout=TIMEOUT_MS
    )
    return window.settings_tab.check_label.text()


def _row_status(window, key):
    row = list(window.settings_tab.link_buttons).index(key)
    return window.settings_tab.models_table.item(row, 1).text()


def test_linked_dit_with_the_right_hash_is_used_in_the_toml(
    qtbot, make_window, tmp_path, dataset, monkeypatch
):
    paths = make_runtime(tmp_path / "data", dataset, {"base_model": "anima-preview"})
    window, _, _, store = make_window(paths.root, expected=PREVIEW_EXPECTED)
    own = _own_dit(tmp_path)

    message = _use_file(qtbot, window, monkeypatch, "anima-preview", own)

    assert message.startswith("OK: DiT anima-preview file accepted")
    assert load_links(store)["anima-preview"].path == own
    assert _row_status(window, "anima-preview") == "OK (your file)"
    assert window.settings_tab.link_buttons["anima-preview"].text() == "Unlink"
    assert not window.settings_tab.download_buttons["anima-preview"].isEnabled()
    window.training_tab.configure_button.click()
    toml = load_settings(paths.config_file).settings.last_train_config
    with open(toml, "rb") as file:
        assert tomllib.load(file)["pretrained_model_name_or_path"] == str(own)
    assert not (paths.models_dir / "dit" / "anima-preview.safetensors").exists()


def test_file_with_a_different_hash_is_rejected(qtbot, make_window, tmp_path, dataset, monkeypatch):
    paths = make_runtime(tmp_path / "data", dataset)
    window, _, _, store = make_window(paths.root, expected=PREVIEW_EXPECTED)
    own = _own_dit(tmp_path, bytes(reversed(DIT_WEIGHTS)))

    message = _use_file(qtbot, window, monkeypatch, "anima-preview", own)

    assert "is not a compatible DiT anima-preview file: SHA-256" in message
    assert dict(load_links(store)) == {}
    assert _row_status(window, "anima-preview") == "Missing"
    assert window.settings_tab.download_buttons["anima-preview"].isEnabled()


def test_linked_file_survives_a_restart_until_it_changes(
    qtbot, make_window, tmp_path, dataset, monkeypatch
):
    paths = make_runtime(tmp_path / "data", dataset)
    window, *_ = make_window(paths.root, expected=PREVIEW_EXPECTED)
    own = _own_dit(tmp_path)
    _use_file(qtbot, window, monkeypatch, "anima-preview", own)
    window.shutdown()

    reopened, _, downloader, _ = make_window(paths.root, expected=PREVIEW_EXPECTED)
    assert reopened.session.paths.dit_model("anima-preview") == own

    own.write_bytes(DIT_WEIGHTS + b"edited")
    reopened.tabs.setCurrentWidget(reopened.training_tab)
    reopened.tabs.setCurrentWidget(reopened.settings_tab)  # the table is re-read when shown
    assert _row_status(reopened, "anima-preview") == "Missing (your file was moved or changed)"
    reopened.settings_tab.download_buttons["anima-preview"].click()
    assert downloader.started[0][1] == paths.models_dir / "dit" / "anima-preview.safetensors"


def test_unlink_goes_back_to_the_data_folder(qtbot, make_window, tmp_path, dataset, monkeypatch):
    paths = make_runtime(tmp_path / "data", dataset)
    window, _, _, store = make_window(paths.root, expected=PREVIEW_EXPECTED)
    _use_file(qtbot, window, monkeypatch, "anima-preview", _own_dit(tmp_path))

    window.settings_tab.link_buttons["anima-preview"].click()

    assert dict(load_links(store)) == {}
    assert _row_status(window, "anima-preview") == "Missing"
    assert window.session.paths.dit_model("anima-preview").is_relative_to(paths.root)


def test_training_is_blocked_while_a_file_is_checked(qtbot, make_window, tmp_path, dataset):
    paths = make_runtime(tmp_path / "data", dataset)
    window, *_ = make_window(paths.root, expected=PREVIEW_EXPECTED)

    window._set_checking(True)

    assert not window.training_tab.start_button.isEnabled()
    assert not window.settings_tab.install_button.isEnabled()
    assert not window.settings_tab.link_buttons[VAE_FILENAME].isEnabled()
    assert window.settings_tab.cancel_check_button.isEnabled()
    window._set_checking(False)


def test_unknown_model_check_error_is_shown(qtbot, make_window, tmp_path, dataset):
    paths = make_runtime(tmp_path / "data", dataset)
    window, *_ = make_window(paths.root, expected=PREVIEW_EXPECTED)

    window.settings_tab.link_requested.emit(VAE_FILENAME, str(tmp_path / "x"))

    assert window.settings_tab.check_label.text() == f"ERROR: Unknown model: {VAE_FILENAME}"
