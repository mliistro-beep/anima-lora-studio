from datetime import datetime
from pathlib import Path

import pytest

from anima_trainer.models.catalog import QWEN3_FILENAME, SUPPORT_MODEL_URLS
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.settings.store import load_settings
from anima_trainer.training.log_parser import EpochStarted, StepProgress
from anima_trainer.training.oom_hint import GPU_OOM_HINT
from anima_trainer.training.progress import ProgressState
from anima_trainer.training.runner import RunOutcome
from anima_trainer.training.start_service import StartRequest
from anima_trainer.workflow.training_session import SessionState, TrainingSession
from tests.fakes import FakeDownloader, FakeRunner

NOW = datetime(2026, 10, 20, 9, 0, 0)


@pytest.fixture
def citron(tmp_path):
    paths = RuntimePaths(tmp_path / "citron")
    for file in (paths.train_script, paths.qwen3_model, paths.vae_model):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"")
    return paths


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"")
    (folder / "a.txt").write_text("caption")
    return folder


@pytest.fixture
def parts(qtbot, citron):
    runner, downloader = FakeRunner(), FakeDownloader()
    kernel_calls = []

    def kernel_log():
        kernel_calls.append(1)
        return None

    session = TrainingSession(
        citron, TrainingSettings(), runner, downloader, clock=lambda: NOW, kernel_log=kernel_log
    )
    log, states, progress = [], [], []
    session.log_line.connect(log.append)
    session.state_changed.connect(states.append)
    session.progress_changed.connect(progress.append)
    return {
        "session": session,
        "runner": runner,
        "downloader": downloader,
        "log": log,
        "states": states,
        "progress": progress,
        "kernel_calls": kernel_calls,
    }


def _settings(dataset, tmp_path):
    return TrainingSettings(
        project_name="proj", image_directory=str(dataset), output_directory=str(tmp_path / "out")
    )


def _install_dit(paths, base_model="anima-base-v1.0"):
    dit = paths.dit_model(base_model)
    dit.parent.mkdir(parents=True, exist_ok=True)
    dit.write_bytes(b"")


def _configured(parts, dataset, tmp_path):
    result = parts["session"].configure(_settings(dataset, tmp_path))
    assert result.success
    return result


REQUEST = StartRequest(gpu_index="0", cpu_threads=1, base_model="anima-base-v1.0")


def test_starts_idle(parts):
    assert parts["session"].state is SessionState.IDLE


def test_configure_success_updates_and_saves_settings(parts, dataset, tmp_path, citron):
    changed = []
    parts["session"].settings_changed.connect(changed.append)

    result = _configured(parts, dataset, tmp_path)

    assert parts["session"].settings == result.settings
    assert changed == [result.settings]
    assert load_settings(citron.config_file).settings.last_train_config


def test_configure_failure_keeps_previous_settings(parts):
    result = parts["session"].configure(TrainingSettings(image_directory=""))

    assert not result.success
    assert parts["session"].settings == TrainingSettings()


def test_start_launches_runner_when_dit_present(parts, dataset, tmp_path, citron):
    _install_dit(citron)
    _configured(parts, dataset, tmp_path)

    parts["session"].start(REQUEST)

    assert len(parts["runner"].started) == 1
    assert parts["session"].state is SessionState.TRAINING
    assert parts["log"][0] == "Using GPU index: 0"
    assert parts["progress"][-1] == ProgressState()


def test_start_without_configure_reports_error_and_stays_idle(parts, citron):
    _install_dit(citron)

    parts["session"].start(REQUEST)

    assert parts["runner"].started == []
    assert parts["session"].state is SessionState.IDLE
    assert parts["log"][-1].startswith("ERROR: No training config found")


def test_log_open_failure_is_reported(parts, dataset, tmp_path, citron):
    _install_dit(citron)
    _configured(parts, dataset, tmp_path)
    parts["runner"].start_error = OSError("disk full")

    parts["session"].start(REQUEST)

    assert parts["session"].state is SessionState.IDLE
    assert "disk full" in parts["log"][-1]


def test_start_downloads_missing_dit_then_launches(parts, dataset, tmp_path, citron):
    _configured(parts, dataset, tmp_path)
    models_changed = []
    parts["session"].models_changed.connect(lambda: models_changed.append(1))

    parts["session"].start(REQUEST)

    assert parts["session"].state is SessionState.DOWNLOADING
    assert parts["downloader"].started[0][1] == citron.dit_model("anima-base-v1.0")
    assert parts["runner"].started == []

    parts["downloader"].finished.emit(str(citron.dit_model("anima-base-v1.0")))

    assert models_changed == [1]
    assert len(parts["runner"].started) == 1
    assert parts["session"].state is SessionState.TRAINING


def test_failed_download_does_not_launch(parts, dataset, tmp_path):
    _configured(parts, dataset, tmp_path)
    parts["session"].start(REQUEST)

    parts["downloader"].failed.emit("HTTP 404")

    assert parts["runner"].started == []
    assert parts["session"].state is SessionState.IDLE
    assert parts["log"][-1] == "ERROR: Download failed: HTTP 404"


def test_stop_during_download_cancels_it(parts, dataset, tmp_path):
    _configured(parts, dataset, tmp_path)
    parts["session"].start(REQUEST)

    parts["session"].stop()
    parts["downloader"].cancelled.emit()

    assert parts["downloader"].cancels == 1
    assert parts["runner"].started == []
    assert parts["session"].state is SessionState.IDLE


def test_download_model_without_training(parts, citron):
    parts["session"].download_model("anima-preview")

    assert parts["downloader"].started[0][1] == citron.dit_model("anima-preview")
    parts["downloader"].finished.emit("done")
    assert parts["runner"].started == []
    assert parts["session"].state is SessionState.IDLE


def test_download_progress_is_forwarded(parts):
    received = []
    parts["session"].download_progress.connect(lambda done, total: received.append((done, total)))
    parts["session"].download_model("anima-preview")

    parts["downloader"].progress.emit(10, 100)

    assert received == [(10, 100)]


def _training(parts, dataset, tmp_path, citron):
    _install_dit(citron)
    _configured(parts, dataset, tmp_path)
    parts["session"].start(REQUEST)
    parts["log"].clear()
    parts["states"].clear()


def test_runner_lines_and_events_are_forwarded(parts, dataset, tmp_path, citron):
    _training(parts, dataset, tmp_path, citron)

    parts["runner"].line_received.emit("hello")
    parts["runner"].event_received.emit(EpochStarted(1, 10))
    parts["runner"].event_received.emit(StepProgress(5, 100, 3, 60, 0.2))

    assert parts["log"] == ["hello"]
    assert parts["progress"][-1].epoch == 1
    assert parts["progress"][-1].step == 5


def test_runner_warning_is_logged(parts, dataset, tmp_path, citron):
    _training(parts, dataset, tmp_path, citron)

    parts["runner"].warning.emit("log broken")

    assert parts["log"] == ["WARN: log broken"]


def test_stop_during_training_stops_runner(parts, dataset, tmp_path, citron):
    _training(parts, dataset, tmp_path, citron)

    parts["session"].stop()
    parts["session"].stop()

    assert parts["runner"].stopped == 1
    assert parts["session"].state is SessionState.STOPPING


def test_successful_run_reports_and_returns_to_idle(parts, dataset, tmp_path, citron):
    _training(parts, dataset, tmp_path, citron)

    parts["runner"].finished.emit(RunOutcome(0, False, False, Path("/log")))

    assert "OK: Training completed successfully!" in parts["log"]
    assert parts["session"].state is SessionState.IDLE
    assert parts["kernel_calls"] == []


def test_failed_run_adds_oom_hint(parts, dataset, tmp_path, citron):
    _training(parts, dataset, tmp_path, citron)
    parts["runner"].tail = ("torch.OutOfMemoryError: CUDA out of memory",)

    parts["runner"].finished.emit(RunOutcome(1, False, False, Path("/log")))

    assert parts["log"][-1] == f"HINT: {GPU_OOM_HINT}"
    assert parts["kernel_calls"] == [1]


def test_failed_to_start_returns_to_idle(parts, dataset, tmp_path, citron):
    _training(parts, dataset, tmp_path, citron)

    parts["runner"].failed_to_start.emit("no accelerate")

    assert parts["log"] == ["ERROR: no accelerate"]
    assert parts["session"].state is SessionState.IDLE


def test_busy_session_refuses_new_work(parts, dataset, tmp_path, citron):
    _training(parts, dataset, tmp_path, citron)

    with pytest.raises(RuntimeError):
        parts["session"].start(REQUEST)
    with pytest.raises(RuntimeError):
        parts["session"].configure(TrainingSettings())
    with pytest.raises(RuntimeError):
        parts["session"].download_model("anima-preview")
    with pytest.raises(RuntimeError):
        parts["session"].set_runtime(citron, TrainingSettings())


def test_set_citron_replaces_paths_and_settings(parts, tmp_path):
    other = RuntimePaths(tmp_path / "other")
    settings = TrainingSettings(project_name="other")

    parts["session"].set_runtime(other, settings)

    assert parts["session"].paths == other
    assert parts["session"].settings == settings


def test_shutdown_stops_runner_and_downloader(parts):
    parts["session"].shutdown()

    assert (parts["runner"].shut_down, parts["downloader"].shut_down) == (1, 1)


def test_states_are_announced_once_per_change(parts, dataset, tmp_path, citron):
    _training(parts, dataset, tmp_path, citron)

    parts["session"].stop()
    parts["runner"].finished.emit(RunOutcome(15, True, True, Path("/log")))

    assert parts["states"] == [SessionState.STOPPING, SessionState.IDLE]


def test_runner_refusing_a_second_run_is_not_hidden(parts, dataset, tmp_path, citron):
    # RuntimeError means a run is already active: resetting to IDLE would re-enable Start.
    _install_dit(citron)
    _configured(parts, dataset, tmp_path)
    parts["runner"].start_error = RuntimeError("A training run is already active")

    with pytest.raises(RuntimeError):
        parts["session"].start(REQUEST)


@pytest.mark.parametrize("end", ["finished", "failed", "cancelled"])
def test_download_end_resets_progress(parts, end):
    parts["session"].download_model("anima-preview")
    parts["progress"].clear()

    signal = getattr(parts["downloader"], end)
    signal.emit() if end == "cancelled" else signal.emit("x")

    assert parts["progress"] == [ProgressState()]


def test_refused_start_resets_progress(parts, citron):
    _install_dit(citron)

    parts["session"].start(REQUEST)

    assert parts["progress"] == [ProgressState()]


def test_stop_racing_a_normal_exit_is_reported_as_completed(parts, dataset, tmp_path, citron):
    _training(parts, dataset, tmp_path, citron)
    parts["session"].stop()

    parts["runner"].finished.emit(RunOutcome(0, False, True, Path("/log")))

    assert "OK: Training completed successfully!" in parts["log"]
    assert parts["kernel_calls"] == []


def test_download_model_fetches_qwen3_into_the_models_folder(parts, citron):
    citron.qwen3_model.unlink()

    parts["session"].download_model(QWEN3_FILENAME)

    assert parts["downloader"].started == [(SUPPORT_MODEL_URLS[QWEN3_FILENAME], citron.qwen3_model)]
    assert parts["log"][0] == "INFO: Downloading Qwen3..."
    assert parts["session"].state is SessionState.DOWNLOADING


def test_download_model_keeps_the_citron_message_for_a_dit(parts):
    parts["session"].download_model("anima-preview")

    assert parts["log"][0] == "INFO: Downloading base model 'anima-preview'..."


def test_download_model_skips_a_present_model(parts):
    parts["session"].download_model(QWEN3_FILENAME)  # the fixture created it

    assert parts["downloader"].started == []
    assert parts["session"].state is SessionState.IDLE
