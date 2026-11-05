"""The TUI, driven in this process with Textual's test pilot and a fake accelerate."""

import asyncio
import json
import shlex
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication
from rich.text import Text
from textual.widgets import Button, Input, RichLog, Select, Static

from anima_trainer.cli.train_plan import EXIT_FAILED
from anima_trainer.cli.train_preparation import terminal_context
from anima_trainer.cli.tui.app import TrainerApp
from anima_trainer.cli.tui.folder_picker import FolderPicker, FoldersOnly, start_folder
from anima_trainer.cli.tui.screens import FormScreen, QuitDialog, SummaryScreen, TrainingScreen
from anima_trainer.runtime.data_dir import open_data_dir
from tests.e2e.helpers import make_runtime
from tests.lock_helpers import held_by_another_process
from tests.process_helpers import wait_until_dead

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="the fake accelerate is a shell script"),
]
FAKE = Path(__file__).parents[2] / "fixtures" / "fake_accelerate.py"
SIZE = (120, 50)
TIMEOUT_S = 20


@pytest.fixture
def paths(qapp, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.png").write_bytes(b"")
    (images / "a.txt").write_text("caption")
    return make_runtime(tmp_path / "data", images, {"gpu_index": "0"})


def _accelerate(paths, *mode):
    command = " ".join(shlex.quote(part) for part in (sys.executable, str(FAKE), *mode))
    paths.venv_accelerate.write_text(f'#!/bin/sh\nexec {command} "$@"\n')
    paths.venv_accelerate.chmod(0o755)


def _app(paths, forced=None):
    forced = [] if forced is None else forced
    context = terminal_context(open_data_dir(paths.root), {})
    return TrainerApp(context, QCoreApplication.instance(), force_exit=forced.append)


def _play(app, scenario):
    """Run `scenario(pilot)` on `app`, and always stop what it started."""

    async def main():
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            await scenario(pilot)

    try:
        asyncio.run(main())
    finally:
        app.shutdown()


async def _until(pilot, condition):
    deadline = time.monotonic() + TIMEOUT_S
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await pilot.pause(0.05)


def _input(app, name):
    return app.screen.query_one(f"#{name}", Input)


def _text(app, name):
    return str(app.screen.query_one(f"#{name}", Static).content)


def _log(app):
    return "\n".join(strip.text for strip in app.screen.query_one("#log", RichLog).lines)


def _fill(app, paths, preset="Person", epochs="2", repeats="1"):
    _input(app, "project").value = "tui_run"
    _input(app, "dataset").value = str(paths.root.parent / "images")
    _input(app, "output").value = str(paths.root / "out")
    app.screen.query_one("#preset", Select).value = preset
    return epochs, repeats


async def _to_summary(pilot, paths):
    app = pilot.app
    epochs, repeats = _fill(app, paths)
    await pilot.pause()
    _input(app, "epochs").value = epochs
    _input(app, "repeats").value = repeats
    await pilot.click("#continue")
    await _until(pilot, lambda: isinstance(app.screen, SummaryScreen))


async def _to_training(pilot, paths):
    await _to_summary(pilot, paths)
    await pilot.click("#start")
    await _until(pilot, lambda: isinstance(pilot.app.screen, TrainingScreen))


def test_choosing_a_preset_proposes_its_epochs_but_no_repeats(paths):
    app = _app(paths)

    async def scenario(pilot):
        assert isinstance(app.screen, FormScreen)
        app.screen.query_one("#preset", Select).value = "Style"
        await pilot.pause()
        assert _input(app, "epochs").value == "16"
        assert _input(app, "repeats").value == ""
        assert _input(app, "resolution").value == "768"
        assert "200-300+" in _text(app, "preset-about")

    _play(app, scenario)


def test_the_last_values_are_proposed_but_not_epochs_from_last_time_or_repeats(paths):
    paths.tui_values_file.write_text(
        json.dumps({"project": "p", "dataset": "/d", "output": "/o", "preset": "Concept"})
    )
    app = _app(paths)

    async def scenario(pilot):
        assert _input(app, "project").value == "p"
        assert _input(app, "dataset").value == "/d"
        assert _input(app, "output").value == "/o"
        assert app.screen.query_one("#preset", Select).value == "Concept"
        assert _input(app, "epochs").value == "10"
        assert _input(app, "repeats").value == ""
        assert app.focused is _input(app, "repeats")

    _play(app, scenario)


def test_a_remembered_preset_that_is_gone_is_left_to_choose(paths):
    paths.tui_values_file.write_text(json.dumps({"preset": "Gone"}))
    app = _app(paths)

    async def scenario(pilot):
        assert app.screen.query_one("#preset", Select).is_blank()
        assert "WARN: Preset 'Gone'" in _text(app, "errors")

    _play(app, scenario)


def test_missing_values_are_named_and_the_form_stays(paths):
    app = _app(paths)

    async def scenario(pilot):
        await pilot.click("#continue")
        await pilot.pause()
        assert isinstance(app.screen, FormScreen)
        assert "preset is required" in _text(app, "errors")
        app.screen.query_one("#preset", Select).value = "Person"
        await pilot.pause()
        app.screen.query_one("#continue", Button).press()
        await pilot.pause()
        errors = _text(app, "errors")
        assert "project is required" in errors
        assert "repeats is required" in errors
        assert "--" not in errors

    _play(app, scenario)


def test_a_number_that_is_only_a_sign_is_an_error_not_a_crash(paths):
    app = _app(paths)

    async def scenario(pilot):
        _fill(app, paths)
        await pilot.pause()
        _input(app, "repeats").value = "-"
        await pilot.click("#continue")
        await pilot.pause()
        assert "repeats must be a whole number" in _text(app, "errors")

    _play(app, scenario)


def test_a_path_with_brackets_is_shown_as_it_is(paths):
    app = _app(paths)

    async def scenario(pilot):
        _fill(app, paths)
        await pilot.pause()
        _input(app, "dataset").value = "/missing/[red]x--y"
        _input(app, "repeats").value = "1"
        await pilot.click("#continue")
        await pilot.pause()
        assert "/missing/[red]x--y" in _text(app, "errors")

    _play(app, scenario)


def test_the_summary_shows_the_steps_and_edit_goes_back_with_the_values(paths):
    app = _app(paths)

    async def scenario(pilot):
        await _to_summary(pilot, paths)
        summary = _text(app, "summary")
        assert "1 image x 1 repeat x 2 epochs = 2 samples" in summary
        assert "Total:     2 steps" in summary
        await pilot.click("#edit")
        await _until(pilot, lambda: isinstance(app.screen, FormScreen))
        assert _input(app, "repeats").value == "1"
        assert _input(app, "project").value == "tui_run"

    _play(app, scenario)


def test_a_whole_training_ends_well_and_leaves_the_window_settings_alone(paths):
    _accelerate(paths, "train", "0")
    before = paths.config_file.read_text()
    app = _app(paths)

    async def scenario(pilot):
        await _to_training(pilot, paths)
        await _until(pilot, lambda: "completed" in _text(app, "status"))
        assert "OK: Training completed successfully!" in _log(app)
        assert app.screen.query_one("#stop", Button).disabled
        await pilot.click("#back")
        await _until(pilot, lambda: isinstance(app.screen, FormScreen))
        await pilot.click("#quit-form")

    _play(app, scenario)
    assert app.exit_code == 0
    assert (paths.root / "out" / "tui_run.safetensors").exists()
    assert paths.config_file.read_text() == before
    remembered = json.loads(paths.tui_values_file.read_text())
    assert remembered["project"] == "tui_run"
    assert remembered["preset"] == "Person"
    assert "repeats" not in remembered


def test_a_failed_training_says_so(paths):
    _accelerate(paths, "train", "3")
    app = _app(paths)

    async def scenario(pilot):
        await _to_training(pilot, paths)
        await _until(pilot, lambda: "failed" in _text(app, "status"))

    _play(app, scenario)
    assert app.exit_code == 1


def test_another_process_on_the_gpu_is_reported_and_nothing_runs(paths):
    _accelerate(paths, "train", "0")
    app = _app(paths)

    async def scenario(pilot):
        with held_by_another_process(paths.gpu_lock_file) as pid:
            await _to_training(pilot, paths)
            await _until(pilot, lambda: str(pid) in _log(app))
        assert "not started" in _text(app, "status")
        assert list(paths.configs_dir.glob("*.toml")) == []

    _play(app, scenario)
    assert app.exit_code == 2


def test_stop_ends_the_training_and_its_children(paths, tmp_path):
    pid_file = tmp_path / "child.pid"
    _accelerate(paths, "sleep", str(pid_file))
    app = _app(paths)

    async def scenario(pilot):
        await _to_training(pilot, paths)
        await _until(pilot, pid_file.exists)
        await _until(pilot, lambda: pid_file.read_text().strip())
        await pilot.click("#stop")
        await _until(pilot, lambda: "stopped" in _text(app, "status"))
        assert wait_until_dead(int(pid_file.read_text()))

    _play(app, scenario)
    assert app.exit_code == 130


def test_quitting_during_a_training_asks_first_and_no_keeps_it_running(paths, tmp_path):
    pid_file = tmp_path / "child.pid"
    _accelerate(paths, "sleep", str(pid_file))
    app = _app(paths)

    async def scenario(pilot):
        await _to_training(pilot, paths)
        await _until(pilot, lambda: pid_file.exists() and pid_file.read_text().strip())
        await pilot.press("ctrl+q")
        await _until(pilot, lambda: isinstance(app.screen, QuitDialog))
        await pilot.click("#quit-no")
        await _until(pilot, lambda: isinstance(app.screen, TrainingScreen))
        assert app.run_active
        await pilot.press("ctrl+q")
        await _until(pilot, lambda: isinstance(app.screen, QuitDialog))
        await pilot.click("#quit-yes")
        await _until(pilot, lambda: not app.run_active)

    _play(app, scenario)
    assert app.exit_code == 130
    assert wait_until_dead(int(pid_file.read_text()))


def test_quitting_at_the_form_does_not_ask(paths):
    app = _app(paths)

    async def scenario(pilot):
        await pilot.press("ctrl+q")
        await pilot.pause()
        assert not isinstance(app.screen, QuitDialog)

    _play(app, scenario)
    assert app.exit_code == 0


def test_a_quit_signal_during_a_training_stops_it_without_asking(paths, tmp_path):
    pid_file = tmp_path / "child.pid"
    _accelerate(paths, "sleep", str(pid_file))
    app = _app(paths)

    async def scenario(pilot):
        await _to_training(pilot, paths)
        await _until(pilot, lambda: pid_file.exists() and pid_file.read_text().strip())
        app.request_quit()
        await _until(pilot, lambda: not app.run_active)

    _play(app, scenario)
    assert app.exit_code == 130
    assert wait_until_dead(int(pid_file.read_text()))


def test_values_that_cannot_be_remembered_do_not_stop_the_training(paths):
    _accelerate(paths, "train", "0")
    paths.tui_values_file.mkdir()  # a folder in place of the file
    app = _app(paths)

    async def scenario(pilot):
        await _to_training(pilot, paths)
        await _until(pilot, lambda: "completed" in _text(app, "status"))
        assert "WARN: " in _log(app)

    _play(app, scenario)


def test_continue_can_be_clicked_again_after_an_error(paths):
    app = _app(paths)

    async def scenario(pilot):
        await pilot.click("#continue")
        await pilot.pause()
        assert "preset is required" in _text(app, "errors")
        app.screen.query_one("#preset", Select).value = "Person"
        await pilot.pause(0.5)
        await pilot.click("#continue")
        await pilot.pause()
        assert "project is required" in _text(app, "errors")

    _play(app, scenario)


def test_after_a_signal_a_hanging_close_cannot_keep_the_process_alive(paths, monkeypatch):
    monkeypatch.setattr("anima_trainer.cli.tui.app.FORCED_EXIT_DELAY_S", 0.05)
    forced = []
    app = _app(paths, forced)

    async def scenario(pilot):
        app.exit = lambda: None  # Textual stuck while closing on a lost terminal
        app.request_quit()
        await _until(pilot, lambda: forced)

    try:
        _play(app, scenario)
    finally:
        del app.exit
    assert forced == [0]


def test_a_normal_quit_never_forces_the_exit(paths, monkeypatch):
    monkeypatch.setattr("anima_trainer.cli.tui.app.FORCED_EXIT_DELAY_S", 0.05)
    forced = []
    app = _app(paths, forced)

    async def scenario(pilot):
        await pilot.press("ctrl+q")
        await pilot.pause(0.2)

    _play(app, scenario)
    time.sleep(0.1)
    assert forced == []


def test_a_crash_of_the_interface_is_not_reported_as_success(paths):
    app = _app(paths)

    async def scenario(pilot):
        app.exit(return_code=1)  # what Textual does after an exception in a screen
        await pilot.pause()

    _play(app, scenario)
    assert app.final_exit_code == EXIT_FAILED


def test_a_failed_training_keeps_its_code_at_the_end(paths):
    _accelerate(paths, "train", "3")
    app = _app(paths)

    async def scenario(pilot):
        await _to_training(pilot, paths)
        await _until(pilot, lambda: "failed" in _text(app, "status"))

    _play(app, scenario)
    assert app.final_exit_code == EXIT_FAILED


def test_two_signals_arm_one_forced_exit(paths, monkeypatch):
    monkeypatch.setattr("anima_trainer.cli.tui.app.FORCED_EXIT_DELAY_S", 0.2)
    forced = []
    app = _app(paths, forced)

    async def scenario(pilot):
        app.exit = lambda: None
        app.request_quit()
        await pilot.pause(0.08)
        app.request_quit()
        await _until(pilot, lambda: forced)
        await pilot.pause(0.3)

    try:
        _play(app, scenario)
    finally:
        del app.exit
    assert forced == [0]


def test_the_end_of_a_run_is_shown_once(paths, monkeypatch):
    _accelerate(paths, "train", "0")
    shown = []
    original = TrainingScreen.show_end

    def counting(screen, code):
        shown.append(code)
        return original(screen, code)

    monkeypatch.setattr(TrainingScreen, "show_end", counting)
    app = _app(paths)

    async def scenario(pilot):
        await _to_training(pilot, paths)
        await _until(pilot, lambda: shown)
        app._on_ended()

    _play(app, scenario)
    assert shown == [0]


def test_a_resolution_of_zero_is_an_error_and_a_blank_one_is_the_default(paths):
    app = _app(paths)

    async def scenario(pilot):
        _fill(app, paths)
        await pilot.pause()
        _input(app, "repeats").value = "1"
        _input(app, "resolution").value = "0"
        await pilot.click("#continue")
        await pilot.pause()
        assert "resolution" in _text(app, "errors")
        assert isinstance(app.screen, FormScreen)
        _input(app, "resolution").value = ""
        await pilot.pause(0.5)
        await pilot.click("#continue")
        await _until(pilot, lambda: isinstance(app.screen, SummaryScreen))
        assert "Size:      768" in _text(app, "summary")

    _play(app, scenario)


def test_a_preset_name_with_brackets_is_plain_text_in_the_list(paths):
    paths.presets_dir.mkdir()
    (paths.presets_dir / "wip.toml").write_text('name = "[wip] x"')
    app = _app(paths)

    async def scenario(pilot):
        labels = [label for label, _value in app.screen.query_one("#preset", Select)._options[1:]]
        assert all(isinstance(label, Text) for label in labels)
        assert "[wip] x" in [label.plain for label in labels]

    _play(app, scenario)


def _tree_child(tree, name):
    return next((n for n in tree.root.children if str(n.label) == name), None)


def test_a_folder_is_chosen_by_browsing_and_fills_the_field(paths):
    app = _app(paths)
    home = paths.root.parent

    async def scenario(pilot):
        _input(app, "dataset").value = str(home)
        await pilot.click("#browse-dataset")
        await _until(pilot, lambda: isinstance(app.screen, FolderPicker))
        tree = app.screen.query_one(FoldersOnly)
        await _until(pilot, lambda: _tree_child(tree, "images") is not None)
        tree.select_node(_tree_child(tree, "images"))
        await _until(pilot, lambda: _text(app, "folder-current") == str(home / "images"))
        await pilot.click("#folder-use")
        await _until(pilot, lambda: isinstance(app.screen, FormScreen))
        assert _input(app, "dataset").value == str(home / "images")

    _play(app, scenario)


def test_files_and_hidden_folders_are_not_listed(paths):
    home = paths.root.parent
    (home / ".hidden").mkdir()
    (home / "file.txt").write_text("")
    app = _app(paths)

    async def scenario(pilot):
        _input(app, "output").value = str(home)
        await pilot.click("#browse-output")
        await _until(pilot, lambda: isinstance(app.screen, FolderPicker))
        tree = app.screen.query_one(FoldersOnly)
        await _until(pilot, lambda: _tree_child(tree, "images") is not None)
        assert _tree_child(tree, ".hidden") is None
        assert _tree_child(tree, "file.txt") is None

    _play(app, scenario)


def test_up_moves_to_the_parent_folder(paths):
    app = _app(paths)
    home = paths.root.parent

    async def scenario(pilot):
        _input(app, "dataset").value = str(home / "images")
        await pilot.click("#browse-dataset")
        await _until(pilot, lambda: isinstance(app.screen, FolderPicker))
        await pilot.click("#folder-up")
        await _until(pilot, lambda: _text(app, "folder-current") == str(home))
        tree = app.screen.query_one(FoldersOnly)
        await _until(pilot, lambda: _tree_child(tree, "images") is not None)

    _play(app, scenario)


def test_cancel_leaves_the_field_as_it_was(paths):
    app = _app(paths)

    async def scenario(pilot):
        _input(app, "dataset").value = "/typed/by/hand"
        await pilot.click("#browse-dataset")
        await _until(pilot, lambda: isinstance(app.screen, FolderPicker))
        await pilot.press("escape")
        await _until(pilot, lambda: isinstance(app.screen, FormScreen))
        assert _input(app, "dataset").value == "/typed/by/hand"

    _play(app, scenario)


def test_browsing_starts_from_the_nearest_folder_that_exists(tmp_path):
    (tmp_path / "a").mkdir()
    assert start_folder(str(tmp_path / "a"), tmp_path / "home") == tmp_path / "a"
    assert start_folder(str(tmp_path / "a" / "b" / "c"), tmp_path / "home") == tmp_path / "a"


def test_browsing_a_blank_field_starts_from_home(tmp_path):
    assert start_folder("  ", tmp_path) == tmp_path
