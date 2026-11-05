"""The command's own code in this process, so the coverage measure sees it; the behaviour as a
user runs it, in a real process, is in test_train_command.py."""

import json
import shlex
import signal
import sys
from pathlib import Path
from typing import get_args

import pytest
from PySide6.QtCore import QCoreApplication, QTimer

from anima_trainer.cli import entry, terminal_run, train_command, train_preparation
from anima_trainer.cli.train_plan import EXIT_NOT_STARTED, EXIT_OK, EXIT_STOPPED
from anima_trainer.settings.choices import BaseModelName
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
    monkeypatch.setattr(train_preparation, "load_data_dir", lambda _store, _default: paths.root)
    monkeypatch.setattr(train_preparation, "load_links", lambda _store: {})
    monkeypatch.setattr(train_preparation, "read_all_gpu_memory", lambda: ())
    monkeypatch.setattr(terminal_run, "read_all_gpu_memory", lambda: ())
    names = qapp.organizationName(), qapp.applicationName()
    yield paths
    qapp.setOrganizationName(names[0])
    qapp.setApplicationName(names[1])


def _train(paths, *extra):
    return [
        "train", "--preset", "Person", "--project", "in_process",
        "--dataset", str(paths.root.parent / "images"), "--output", str(paths.root / "out"),
        "--epochs", "2", "--repeats", "1", "--yes", *extra,
    ]  # fmt: skip


def _accelerate(paths, *mode):
    command = " ".join(shlex.quote(part) for part in (sys.executable, str(FAKE), *mode))
    paths.venv_accelerate.write_text(f'#!/bin/sh\nexec {command} "$@"\n')
    paths.venv_accelerate.chmod(0o755)


def test_help_is_written_without_opening_the_app(capsys):
    assert entry.main(["--help"]) == 0
    assert "train" in capsys.readouterr().out


def test_the_tui_command_is_started(monkeypatch):
    monkeypatch.setattr("anima_trainer.cli.tui.command.run_tui", lambda arguments: 7)
    assert entry.main(["tui"]) == 7


def test_ctrl_c_before_the_tui_opens_is_a_stop(monkeypatch):
    def interrupted(_arguments):
        raise KeyboardInterrupt

    monkeypatch.setattr("anima_trainer.cli.tui.command.run_tui", interrupted)
    assert entry.main(["tui"]) == EXIT_STOPPED


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
    assert entry.main(_train(runtime)) == EXIT_OK
    assert "OK: Training completed successfully!" in capsys.readouterr().out


def test_a_blocked_training_is_reported(runtime, capsys):
    runtime.vae_model.unlink()
    assert entry.main(_train(runtime)) == EXIT_NOT_STARTED
    assert "ERROR: Missing models: VAE" in capsys.readouterr().out


def test_an_incomplete_runtime_is_reported_before_the_models(runtime, capsys):
    runtime.vae_model.unlink()
    runtime.train_script.unlink()
    assert entry.main(_train(runtime)) == EXIT_NOT_STARTED
    output = capsys.readouterr().out
    assert "ERROR: Training script not found" in output
    assert "graphical app" in output
    assert "Missing models" not in output


def test_a_signal_stops_the_run_and_restores_the_handlers(runtime, tmp_path, capsys):
    _accelerate(runtime, "sleep", str(tmp_path / "child.pid"))
    before = signal.getsignal(signal.SIGTERM)
    QTimer.singleShot(500, lambda: signal.raise_signal(signal.SIGTERM))
    assert entry.main(_train(runtime)) == EXIT_STOPPED
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
    assert entry.main(_train(runtime)) == EXIT_STOPPED
    output = capsys.readouterr().out
    assert "INFO: Stopping the training" in output
    assert "INFO: Training stopped." in output


def test_invalid_saved_settings_do_not_train_or_overwrite_the_file(runtime, capsys):
    saved = json.loads(runtime.config_file.read_text()) | {"learning_rate": "broken"}
    runtime.config_file.write_text(json.dumps(saved))
    before = runtime.config_file.read_text()
    assert entry.main(_train(runtime)) == EXIT_NOT_STARTED
    output = capsys.readouterr().out
    assert "learning_rate" in output
    assert "ERROR: " in output
    assert runtime.config_file.read_text() == before
    assert list(runtime.configs_dir.glob("*.toml")) == []


def test_invalid_preview_settings_do_not_train_or_overwrite_the_file(runtime, capsys):
    runtime.preview_settings_file.write_text('{"enabled": "maybe"}')
    before = runtime.preview_settings_file.read_text()
    assert entry.main(_train(runtime)) == EXIT_NOT_STARTED
    assert "ERROR: " in capsys.readouterr().out
    assert runtime.preview_settings_file.read_text() == before


def test_the_presets_can_be_listed_without_a_runtime(runtime, capsys):
    runtime.train_script.unlink()
    assert entry.main(["train", "--list-presets"]) == EXIT_OK
    output = capsys.readouterr().out
    for name in ("Style", "Character", "Concept", "Person"):
        assert name in output


def test_the_user_presets_are_listed_and_a_bad_one_is_reported(runtime, capsys):
    runtime.presets_dir.mkdir()
    (runtime.presets_dir / "mine.toml").write_text('name = "Mine"')
    (runtime.presets_dir / "bad.toml").write_text("name = ")
    assert entry.main(["train", "--list-presets"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "Mine" in output
    assert "WARN: Preset bad.toml ignored" in output


def test_a_user_preset_trains_like_a_built_in_one(runtime, capsys):
    runtime.presets_dir.mkdir()
    (runtime.presets_dir / "mine.toml").write_text('name = "Mine"\n[settings]\nnetwork_dim = 8\n')
    _accelerate(runtime, "train", "0")
    arguments = _train(runtime)
    arguments[arguments.index("--preset") + 1] = "mine"
    assert entry.main(arguments) == EXIT_OK
    assert "OK: Training completed successfully!" in capsys.readouterr().out


def test_an_unknown_preset_names_the_available_ones(runtime, capsys):
    arguments = _train(runtime)
    arguments[arguments.index("--preset") + 1] = "nope"
    assert entry.main(arguments) == EXIT_NOT_STARTED
    output = capsys.readouterr().out
    assert "ERROR: Unknown preset 'nope'" in output
    assert "Person" in output


def test_no_preset_points_to_the_list(capsys, runtime):
    assert entry.main(["train"]) == EXIT_NOT_STARTED
    output = capsys.readouterr().out
    assert "--preset" in output
    assert "--list-presets" in output


def test_a_missing_option_stops_before_anything_else(runtime, capsys):
    arguments = _train(runtime)
    position = arguments.index("--epochs")
    del arguments[position : position + 2]
    assert entry.main(arguments) == EXIT_NOT_STARTED
    assert "--epochs is required (preset Person suggests 10)" in capsys.readouterr().out
    assert list(runtime.configs_dir.glob("*.toml")) == []


def test_a_value_the_settings_refuse_is_reported(runtime, capsys):
    arguments = _train(runtime)
    arguments[arguments.index("--project") + 1] = "a/b"
    assert entry.main(arguments) == EXIT_NOT_STARTED
    assert "ERROR: " in capsys.readouterr().out
    assert list(runtime.configs_dir.glob("*.toml")) == []


def test_a_preset_with_another_base_model_is_refused(runtime, capsys):
    runtime.presets_dir.mkdir()
    other = next(name for name in get_args(BaseModelName) if name != "anima-base-v1.0")
    (runtime.presets_dir / "other.toml").write_text(
        f'name = "Other"\n[settings]\nbase_model = "{other}"\n'
    )
    arguments = _train(runtime)
    arguments[arguments.index("--preset") + 1] = "Other"
    assert entry.main(arguments) == EXIT_NOT_STARTED
    assert "only anima-base-v1.0" in capsys.readouterr().out


def test_a_missing_dataset_is_reported_before_anything_is_written(runtime, capsys, tmp_path):
    arguments = _train(runtime)
    arguments[arguments.index("--dataset") + 1] = str(tmp_path / "missing")
    assert entry.main(arguments) == EXIT_NOT_STARTED
    assert "ERROR: Directory not found" in capsys.readouterr().out
    assert list(runtime.configs_dir.glob("*.toml")) == []


def test_a_dataset_without_images_is_reported_before_anything_is_written(runtime, capsys, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    arguments = _train(runtime)
    arguments[arguments.index("--dataset") + 1] = str(empty)
    assert entry.main(arguments) == EXIT_NOT_STARTED
    assert "ERROR: " in capsys.readouterr().out
    assert list(runtime.configs_dir.glob("*.toml")) == []


def test_the_gpu_memory_is_read_again_after_the_confirmation(runtime, monkeypatch):
    """Time can pass at the question, and previews are sized from what is free when it starts."""
    reads = []
    monkeypatch.setattr(train_preparation, "read_all_gpu_memory", lambda: reads.append(1) or ())
    monkeypatch.setattr(terminal_run, "read_all_gpu_memory", lambda: reads.append(2) or ())
    _accelerate(runtime, "train", "0")
    assert entry.main(_train(runtime)) == EXIT_OK
    assert reads == [1, 2]


def test_the_tui_needs_a_terminal(capsys):
    assert entry.main(["tui"]) == EXIT_NOT_STARTED
    assert "anima-trainer train" in capsys.readouterr().out
