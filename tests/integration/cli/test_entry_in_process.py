"""The command's own code in this process, so the coverage measure sees it; the behaviour as a
user runs it, in a real process, is in test_train_command.py."""

import json
import shlex
import signal
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QTimer

from anima_trainer.cli import entry, train_command
from anima_trainer.cli.train_plan import EXIT_NOT_STARTED, EXIT_OK, EXIT_STOPPED
from anima_trainer.workflow.training_session import TrainingSession
from tests.e2e.helpers import make_runtime

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="the fake accelerate is a shell script"),
]
FAKE = Path(__file__).parents[2] / "fixtures" / "fake_accelerate.py"


@pytest.fixture
def runtime(qapp, tmp_path, monkeypatch):
    """An installed data folder used in place of the user's own folder and links."""
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.png").write_bytes(b"")
    (images / "a.txt").write_text("caption")
    settings = {"project_name": "in_process", "max_train_epochs": 2, "gpu_index": "0"}
    paths = make_runtime(tmp_path / "data", images, settings)
    monkeypatch.setattr(train_command, "load_data_dir", lambda _store, _default: paths.root)
    monkeypatch.setattr(train_command, "load_links", lambda _store: {})
    monkeypatch.setattr(train_command, "read_all_gpu_memory", lambda: ())
    names = qapp.organizationName(), qapp.applicationName()
    yield paths
    qapp.setOrganizationName(names[0])
    qapp.setApplicationName(names[1])


def _accelerate(paths, *mode):
    command = " ".join(shlex.quote(part) for part in (sys.executable, str(FAKE), *mode))
    paths.venv_accelerate.write_text(f'#!/bin/sh\nexec {command} "$@"\n')
    paths.venv_accelerate.chmod(0o755)


def test_help_is_written_without_opening_the_app(capsys):
    assert entry.main(["--help"]) == 0
    assert "train" in capsys.readouterr().out


def test_no_command_opens_the_graphical_app(monkeypatch):
    monkeypatch.setattr("anima_trainer.main.main", lambda: 42)
    assert entry.main([]) == 42


def test_unknown_arguments_go_to_the_graphical_app_for_qt(monkeypatch):
    monkeypatch.setattr("anima_trainer.main.main", lambda: 42)
    assert entry.main(["-platform", "offscreen"]) == 42


def test_the_arguments_come_from_the_command_line_by_default(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["anima-trainer", "--help"])
    assert entry.main() == 0
    assert "usage" in capsys.readouterr().out


def test_ctrl_c_before_the_training_starts_is_a_stop(monkeypatch):
    def interrupted(_arguments):
        raise KeyboardInterrupt

    monkeypatch.setattr(train_command, "run_train", interrupted)
    assert entry.main(["train"]) == EXIT_STOPPED


def test_train_runs_to_the_end(runtime, capsys):
    _accelerate(runtime, "train", "0")
    assert entry.main(["train"]) == EXIT_OK
    assert "OK: Training completed successfully!" in capsys.readouterr().out


def test_a_blocked_training_is_reported(runtime, capsys):
    runtime.vae_model.unlink()
    assert entry.main(["train"]) == EXIT_NOT_STARTED
    assert "ERROR: Missing models: VAE" in capsys.readouterr().out


def test_an_incomplete_runtime_is_reported_before_the_models(runtime, capsys):
    runtime.vae_model.unlink()
    runtime.train_script.unlink()
    assert entry.main(["train"]) == EXIT_NOT_STARTED
    output = capsys.readouterr().out
    assert "ERROR: Training script not found" in output
    assert "graphical app" in output
    assert "Missing models" not in output


def test_a_signal_stops_the_run_and_restores_the_handlers(runtime, tmp_path, capsys):
    _accelerate(runtime, "sleep", str(tmp_path / "child.pid"))
    before = signal.getsignal(signal.SIGTERM)
    QTimer.singleShot(500, lambda: signal.raise_signal(signal.SIGTERM))
    assert entry.main(["train"]) == EXIT_STOPPED
    assert "INFO: Training stopped." in capsys.readouterr().out
    assert signal.getsignal(signal.SIGTERM) == before
    assert QCoreApplication.instance() is not None


def test_a_signal_during_the_start_is_not_lost(runtime, tmp_path, capsys, monkeypatch):
    """The session is still idle, where stop() does nothing; the request must wait for the run."""
    _accelerate(runtime, "sleep", str(tmp_path / "child.pid"))
    original_start = TrainingSession.start

    def start_with_a_signal(session, request):
        signal.raise_signal(signal.SIGTERM)
        original_start(session, request)

    monkeypatch.setattr(TrainingSession, "start", start_with_a_signal)
    assert entry.main(["train"]) == EXIT_STOPPED
    output = capsys.readouterr().out
    assert "INFO: Stopping the training" in output
    assert "INFO: Training stopped." in output


def test_invalid_saved_settings_do_not_train_or_overwrite_the_file(runtime, capsys):
    saved = json.loads(runtime.config_file.read_text()) | {"learning_rate": "broken"}
    runtime.config_file.write_text(json.dumps(saved))
    before = runtime.config_file.read_text()
    assert entry.main(["train"]) == EXIT_NOT_STARTED
    output = capsys.readouterr().out
    assert "learning_rate" in output
    assert "ERROR: " in output
    assert runtime.config_file.read_text() == before
    assert list(runtime.configs_dir.glob("*.toml")) == []


def test_invalid_preview_settings_do_not_train_or_overwrite_the_file(runtime, capsys):
    runtime.preview_settings_file.write_text('{"enabled": "maybe"}')
    before = runtime.preview_settings_file.read_text()
    assert entry.main(["train"]) == EXIT_NOT_STARTED
    assert "ERROR: " in capsys.readouterr().out
    assert runtime.preview_settings_file.read_text() == before
