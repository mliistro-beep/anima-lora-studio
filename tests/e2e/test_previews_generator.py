"""The Previews and Generator tabs in the real MainWindow, with fake runners and a known GPU."""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import toml
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QMessageBox

from anima_trainer.models.catalog import VAE_FILENAME, ExpectedFile
from anima_trainer.models.check_worker import ModelFileChecker
from anima_trainer.preview.settings import load_preview_settings
from anima_trainer.preview.vram_profile import IDLE_MARGIN_MIB, TRAINING_MARGIN_MIB
from anima_trainer.runtime.data_dir import open_data_dir
from anima_trainer.runtime.install_plan import InstallPlan
from anima_trainer.runtime_scripts import TRAIN_WITH_PREVIEWS
from anima_trainer.system.gpu_memory import GpuMemory
from anima_trainer.system.gpu_worker import GpuDetector
from anima_trainer.training.runner import RunOutcome
from anima_trainer.ui.main_window import MainWindow
from anima_trainer.ui.preview_controller import TRAINING_NOTE
from anima_trainer.workflow.generator_session import GeneratorSession
from anima_trainer.workflow.install_session import InstallSession
from anima_trainer.workflow.link_session import LinkSession
from anima_trainer.workflow.training_session import TrainingSession
from tests.e2e.helpers import GPUS, INSTALL_STEPS, NOW, PROJECT, TIMEOUT_MS, make_runtime
from tests.fakes import FakeDownloader, FakeInstaller, FakeRunner

pytestmark = pytest.mark.e2e
GPU_MEMORY = (GpuMemory(0, 12288, 11000), GpuMemory(1, 12288, 11000))
OK = RunOutcome(0, crashed=False, stopped=False, log_path=None)
VAE_WEIGHTS = b"qwen image vae weights" * 50
VAE_EXPECTED = {
    VAE_FILENAME: ExpectedFile(hashlib.sha256(VAE_WEIGHTS).hexdigest(), len(VAE_WEIGHTS))
}


@pytest.fixture
def app(qtbot, tmp_path, dataset):
    """The window on an installed runtime, its training runner and its generator runner."""
    paths = make_runtime(tmp_path / "data", dataset)
    store = QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat)
    opened = open_data_dir(paths.root)
    runner, generator_runner = FakeRunner(), FakeRunner()
    session = TrainingSession(
        opened.paths, opened.settings, runner, FakeDownloader(), clock=lambda: NOW,
        kernel_log=lambda: None,
    )  # fmt: skip
    detector = GpuDetector(lambda _paths: GPUS, lambda: GPU_MEMORY)
    install = InstallSession(
        FakeInstaller(), plan=lambda _paths: InstallPlan(INSTALL_STEPS, ()), clock=lambda: NOW
    )
    generator = GeneratorSession(generator_runner, clock=lambda: NOW)
    links = LinkSession(store, ModelFileChecker(), VAE_EXPECTED)
    with qtbot.waitSignal(detector.memory_detected, timeout=TIMEOUT_MS):
        window = MainWindow(
            session, detector, store, opened, install, links=links, generator=generator
        )
    qtbot.addWidget(window)
    return window, runner, generator_runner, paths


def _turn_previews_on(window, prompts="a girl\na cat"):
    window.previews_tab.form.field_widget("enabled").setChecked(True)
    window.previews_tab.prompts.setPlainText(prompts)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")
    return path


def _option(arguments, name):
    return arguments[arguments.index(name) + 1]


def test_configure_with_previews_writes_the_prompts_and_saves_the_settings(app):
    window, _, _, paths = app
    _turn_previews_on(window)

    window.training_tab.configure_button.click()

    status = window.training_tab.status.text()
    assert "Preview size 1024x1024 chosen for 12288 MiB of GPU memory." in status
    samples = sorted(paths.configs_dir.glob(f"{PROJECT}_samples_*.json"))
    assert len(samples) == 1
    assert [entry["prompt"] for entry in json.loads(samples[0].read_text())] == ["a girl", "a cat"]
    training = toml.loads(Path(window.session.settings.last_train_config).read_text())
    assert training["sample_prompts"] == str(samples[0])
    assert load_preview_settings(paths.preview_settings_file).settings.prompts == (
        "a girl",
        "a cat",
    )


def test_previews_without_prompts_or_with_a_bad_value_do_not_configure(app):
    window, _, _, paths = app
    _turn_previews_on(window, prompts="")

    window.training_tab.configure_button.click()

    assert (
        "ERROR: Previews are on but there is no preview prompt."
        in window.training_tab.status.text()
    )
    window.previews_tab.prompts.setPlainText("a girl")
    window.previews_tab.form.field_widget("cfg_scale").setText("high")
    window.training_tab.configure_button.click()

    status = window.training_tab.status.text()
    assert "ERROR: Previews tab:" in status and "CFG Scale" in status
    assert not list(paths.configs_dir.glob("*_samples_*.json"))


def test_previews_off_keep_the_training_launch_of_sd_scripts(app):
    window, runner, _, paths = app
    window.training_tab.configure_button.click()

    window.training_tab.start_button.click()

    assert str(paths.train_script) in runner.started[0][0].arguments


def test_training_with_previews_uses_the_preview_launcher_and_fills_the_gallery(app):
    window, runner, _, paths = app
    _turn_previews_on(window)
    window.training_tab.configure_button.click()

    window.training_tab.start_button.click()

    assert str(TRAIN_WITH_PREVIEWS) in runner.started[0][0].arguments
    assert window.previews.watcher.is_watching()
    assert not window.previews_tab.form.isEnabled()
    stamp = (datetime.now() + timedelta(minutes=1)).strftime("%Y%m%d%H%M%S")
    image = _touch(paths.root / "out" / "sample" / f"{PROJECT}_e000001_01_{stamp}_42.png")
    window.previews.watcher.refresh()
    gallery = window.previews_tab.gallery
    assert (gallery.rowCount(), gallery.columnCount()) == (1, 2)
    assert gallery.image_at(0, 1) == image
    assert gallery.horizontalHeaderItem(1).text() == "2: a cat"

    runner.finished.emit(OK)

    assert not window.previews.watcher.is_watching()
    assert window.previews_tab.form.isEnabled()


def test_previews_of_an_earlier_run_show_when_the_tab_opens(app):
    window, _, _, paths = app
    image = _touch(paths.root / "out" / "sample" / f"{PROJECT}_e000002_00_20261026090000.png")

    window.tabs.setCurrentWidget(window.previews_tab)

    assert window.previews_tab.gallery.image_at(0, 0) == image
    assert window.previews_tab.gallery.verticalHeaderItem(0).text() == "Epoch 2"


def _checkpoints(paths):
    out = paths.root / "out"
    return _touch(out / f"{PROJECT}-000001.safetensors"), _touch(out / f"{PROJECT}.safetensors")


def test_generator_lists_the_checkpoints_and_generates_with_the_chosen_one(app):
    window, _, generator_runner, paths = app
    _, final = _checkpoints(paths)
    window.tabs.setCurrentWidget(window.generator_tab)
    tab = window.generator_tab
    combo_items = [tab.checkpoint.itemText(i) for i in range(tab.checkpoint.count())]
    assert combo_items == ["No checkpoint", "Epoch 1", "Final"]
    assert tab.checkpoint.currentText() == "Final"
    tab.prompt.setPlainText("a girl")

    tab.generate_button.click()

    spec = generator_runner.started[0][0]
    assert _option(spec.arguments, "--lora") == str(final)
    assert "--prompt=a girl" in spec.arguments
    assert _option(spec.arguments, "--margin_mib") == str(IDLE_MARGIN_MIB)
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "1"  # the GPU chosen in the Training tab
    assert not window.training_tab.start_button.isEnabled()
    assert tab.stop_button.isEnabled() and not tab.generate_button.isEnabled()

    output = _option(spec.arguments, "--output")
    generator_runner.line_received.emit("STEP: 25/25")
    generator_runner.line_received.emit(f"IMAGE: {output}")
    generator_runner.finished.emit(OK)

    assert "OK: 1 image generated." in tab.log.text()
    assert tab.progress.value() == 25
    assert window.training_tab.start_button.isEnabled()
    assert tab.generate_button.isEnabled()
    assert (paths.root / "generated").as_posix() in output  # the generator's default folder


def test_generator_uses_a_model_file_linked_while_the_app_is_open(app, qtbot, monkeypatch):
    window, _, generator_runner, paths = app
    _checkpoints(paths)
    paths.vae_model.unlink()
    own = _touch(paths.root.parent / "my-models" / VAE_FILENAME)
    own.write_bytes(VAE_WEIGHTS)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a: (str(own), ""))
    window.tabs.setCurrentWidget(window.settings_tab)  # the model table is re-read when shown
    window.settings_tab.link_buttons[VAE_FILENAME].click()
    qtbot.waitUntil(
        lambda: not window.settings_tab.cancel_check_button.isEnabled(), timeout=TIMEOUT_MS
    )
    assert window.settings_tab.check_label.text().startswith("OK:")
    window.tabs.setCurrentWidget(window.generator_tab)
    window.generator_tab.prompt.setPlainText("a girl")

    window.generator_tab.generate_button.click()

    assert "Missing models" not in window.generator_tab.log.text()
    assert _option(generator_runner.started[0][0].arguments, "--vae") == str(own)


def test_compare_epochs_generates_one_image_per_checkpoint(app):
    window, _, generator_runner, paths = app
    first, final = _checkpoints(paths)
    window.tabs.setCurrentWidget(window.generator_tab)
    window.generator_tab.prompt.setPlainText("a girl")

    window.generator_tab.compare_button.click()
    output = _option(generator_runner.started[0][0].arguments, "--output")
    generator_runner.line_received.emit(f"IMAGE: {output}")
    generator_runner.finished.emit(OK)

    loras = [_option(spec.arguments, "--lora") for spec, _log in generator_runner.started]
    seeds = {_option(spec.arguments, "--seed") for spec, _log in generator_runner.started}
    assert loras == [str(first), str(final)]
    assert len(seeds) == 1


def test_generating_during_a_training_keeps_the_training_margin(app):
    window, runner, generator_runner, _ = app
    window.training_tab.configure_button.click()
    window.training_tab.start_button.click()
    tab = window.generator_tab
    tab.prompt.setPlainText("a girl")

    assert tab.state_note.text() == TRAINING_NOTE
    tab.generate_button.click()

    assert _option(generator_runner.started[0][0].arguments, "--margin_mib") == str(
        TRAINING_MARGIN_MIB
    )
    runner.finished.emit(OK)
    assert not tab.state_note.isVisibleTo(window)


def test_generate_without_a_prompt_explains_it(app):
    window, _, generator_runner, _ = app

    window.generator_tab.generate_button.click()

    assert "ERROR: Write a prompt first." in window.generator_tab.log.text()
    assert generator_runner.started == []


def test_closing_while_generating_stops_the_generator(app, monkeypatch):
    window, _, generator_runner, _ = app
    window.generator_tab.prompt.setPlainText("a girl")
    window.generator_tab.generate_button.click()
    asked = []
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: asked.append(1) or QMessageBox.StandardButton.Yes
    )

    assert window.close()

    assert asked == [1]
    assert generator_runner.shut_down == 1


def test_bad_preview_values_do_not_block_configure_while_previews_are_off(app):
    window, _, _, paths = app
    _turn_previews_on(window)
    window.training_tab.configure_button.click()
    window.previews_tab.form.field_widget("enabled").setChecked(False)
    window.previews_tab.form.field_widget("cfg_scale").setText("high")

    window.training_tab.configure_button.click()

    status = window.training_tab.status.text()
    assert "WARN: Previews are off; these Previews tab values were not saved:" in status
    assert "WARN: CFG Scale:" in status
    assert "ERROR:" not in status
    assert "sample_prompts" not in toml.loads(
        Path(window.session.settings.last_train_config).read_text()
    )
    saved = load_preview_settings(paths.preview_settings_file).settings
    assert not saved.enabled  # a later Configure must not turn them back on
    assert saved.prompts == ("a girl", "a cat")


def test_preview_settings_are_kept_when_configure_fails(app):
    window, _, _, paths = app
    _turn_previews_on(window)
    window.training_tab.form.field_widget("image_directory").set_path("")

    window.training_tab.configure_button.click()

    assert not window.session.settings.last_train_config
    saved = load_preview_settings(paths.preview_settings_file).settings
    assert saved.prompts == ("a girl", "a cat")


def test_opening_a_project_shows_only_the_previews_of_its_last_run(app):
    window, _, _, paths = app
    sample = paths.root / "out" / "sample"
    _touch(sample / f"{PROJECT}_e000005_00_20261026090000.png")
    newest = _touch(sample / f"{PROJECT}_e000001_00_20261026110000.png")

    window.training_tab.configure_button.click()

    gallery = window.previews_tab.gallery
    assert gallery.rowCount() == 1
    assert gallery.image_at(0, 0) == newest


def test_the_data_folder_and_models_are_locked_while_generating(app, tmp_path):
    window, _, generator_runner, paths = app
    tab = window.settings_tab
    window.generator_tab.prompt.setPlainText("a girl")

    window.generator_tab.generate_button.click()

    assert not tab.apply_button.isEnabled()
    assert not tab.root_picker.isEnabled()
    assert not tab.install_button.isEnabled()
    assert not any(button.isEnabled() for button in tab.link_buttons.values())
    tab.root_applied.emit(str(tmp_path / "other"))
    assert window.session.paths.root == paths.root

    generator_runner.finished.emit(RunOutcome(1, crashed=False, stopped=False, log_path=None))

    assert tab.apply_button.isEnabled()
    assert tab.install_button.isEnabled()
