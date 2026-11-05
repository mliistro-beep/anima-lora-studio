"""`anima-trainer train` as a user runs it: a real process, no display, a fake accelerate."""

import contextlib
import os
import pty
import select
import shlex
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

from tests.e2e.helpers import make_runtime
from tests.lock_helpers import held_by_another_process
from tests.process_helpers import is_alive, wait_until_dead

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="the fake accelerate is a shell script"),
]
FAKE = Path(__file__).parents[2] / "fixtures" / "fake_accelerate.py"
TIMEOUT_S = 30
PROJECT = "cli_run"
ENTRY = ("-m", "anima_trainer.cli.entry", "train")
PRESET = "Person"


def _options(paths, *extra):
    """A complete command line for the runtime `setup` made."""
    return (
        "--preset", PRESET, "--project", PROJECT,
        "--dataset", str(paths.root.parents[1] / "images"),
        "--output", str(paths.root / "out"), "--epochs", "2", "--repeats", "1", *extra,
    )  # fmt: skip


def _train(paths, *extra):
    return (*ENTRY, *_options(paths, "--yes", *extra))


@pytest.fixture
def setup(tmp_path):
    """Scratch XDG folders holding an installed runtime and a saved configuration."""
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.png").write_bytes(b"")
    (images / "a.txt").write_text("caption")
    data_home, config_home = tmp_path / "data", tmp_path / "config"
    config_home.mkdir()
    settings = {"project_name": PROJECT, "max_train_epochs": 2, "gpu_index": "0"}
    paths = make_runtime(data_home / "anima-trainer", images, settings)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"}
    }
    env |= {"XDG_DATA_HOME": str(data_home), "XDG_CONFIG_HOME": str(config_home)}
    return paths, env


def _accelerate(paths, *mode):
    command = " ".join(shlex.quote(part) for part in (sys.executable, str(FAKE), *mode))
    paths.venv_accelerate.write_text(f'#!/bin/sh\nexec {command} "$@"\n')
    paths.venv_accelerate.chmod(0o755)


def _run(env, *arguments):
    return subprocess.run(
        [sys.executable, *arguments], env=env, capture_output=True, text=True, timeout=TIMEOUT_S
    )


def _wait_for_file(path):
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"{path} was not written")


def _kill_group_of(pid_file):
    """After a failed test: the fake training and its child share a group, but not the command."""
    if pid_file.exists() and pid_file.read_text().strip():
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(int(pid_file.read_text())), signal.SIGKILL)


def test_a_saved_configuration_trains_to_the_end_without_a_display(setup):
    paths, env = setup
    _accelerate(paths, "train", "0")
    result = _run(env, *_train(paths))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: Training completed successfully!" in result.stdout
    assert (paths.root / "out" / f"{PROJECT}.safetensors").exists()
    assert "%|" not in result.stdout  # tqdm lines become the progress display
    assert "/4" in result.stdout


def test_no_graphical_qt_module_is_loaded(setup):
    paths, env = setup
    _accelerate(paths, "train", "0")
    code = (
        "import json, sys\n"
        "from anima_trainer.cli.entry import main\n"
        f"code = main({['train', *_options(paths, '--yes')]!r})\n"
        "loaded = [m for m in sys.modules if m in ('PySide6.QtGui', 'PySide6.QtWidgets')]\n"
        "sys.stderr.write('LOADED=' + json.dumps(loaded))\n"
        "sys.exit(code)\n"
    )
    result = _run(env, "-c", code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LOADED=[]" in result.stderr


def test_a_failed_run_exits_with_failure_and_reports_it(setup):
    paths, env = setup
    _accelerate(paths, "train", "3")
    result = _run(env, *_train(paths))
    assert result.returncode == 1
    assert "exit code: 3" in result.stdout


def test_a_missing_model_stops_before_writing_any_config(setup):
    paths, env = setup
    paths.vae_model.unlink()
    result = _run(env, *_train(paths))
    assert result.returncode == 2
    assert "ERROR: Missing models: VAE" in result.stdout
    assert "graphical app" in result.stdout
    assert list(paths.configs_dir.glob("*.toml")) == []


def test_a_missing_base_model_is_not_downloaded(setup):
    paths, env = setup
    paths.dit_model("anima-base-v1.0").unlink()
    result = _run(env, *_train(paths))
    assert result.returncode == 2
    assert "anima-base-v1.0" in result.stdout
    assert "Downloading" not in result.stdout


def test_a_runtime_that_is_not_installed_is_reported(setup):
    paths, env = setup
    paths.train_script.unlink()
    result = _run(env, *_train(paths))
    assert result.returncode == 2
    assert "Training script not found" in result.stdout


def test_ctrl_c_stops_the_training_and_its_children(setup, tmp_path):
    paths, env = setup
    pid_file = tmp_path / "child.pid"
    _accelerate(paths, "sleep", str(pid_file))
    process = subprocess.Popen(
        [sys.executable, *_train(paths)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        child = _wait_for_file(pid_file)
        process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=TIMEOUT_S)
        assert process.returncode == 130, output
        assert "INFO: Training stopped." in output
        assert wait_until_dead(child)
    finally:
        process.kill()
        process.wait()
        _kill_group_of(pid_file)


def test_a_lost_terminal_stops_the_training_and_leaves_no_process(setup, tmp_path):
    """SSH dropped without tmux: nothing may stay on the GPU, in its own session."""
    paths, env = setup
    pid_file = tmp_path / "child.pid"
    _accelerate(paths, "sleep", str(pid_file))
    pid, terminal = _fork_on_a_terminal(env, *_train(paths))
    finished = 0
    terminal_open = True
    try:
        child = _wait_for_file(pid_file)
        assert is_alive(child)
        os.close(terminal)  # hangs up: the kernel sends SIGHUP
        terminal_open = False
        assert wait_until_dead(child, TIMEOUT_S)
        deadline = time.monotonic() + TIMEOUT_S
        while not finished and time.monotonic() < deadline:
            finished, status = os.waitpid(pid, os.WNOHANG)
            time.sleep(0.05)
        assert finished, "the command did not exit"
        assert os.waitstatus_to_exitcode(status) == 130
    finally:
        if terminal_open:
            os.close(terminal)
        if not finished:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        _kill_group_of(pid_file)


def test_help_lists_the_commands(setup):
    _, env = setup
    result = _run(env, "-m", "anima_trainer.cli.entry", "--help")
    assert result.returncode == 0
    assert "train" in result.stdout


def test_a_training_of_another_process_stops_the_command_before_writing_anything(setup):
    paths, env = setup
    _accelerate(paths, "train", "0")
    before = paths.config_file.read_text()
    with held_by_another_process(paths.gpu_lock_file) as pid:
        result = _run(env, *_train(paths))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "ERROR: " in result.stdout
    assert str(pid) in result.stdout
    assert paths.config_file.read_text() == before
    assert list(paths.configs_dir.glob("*.toml")) == []


def test_a_running_command_stops_another_command(setup, tmp_path):
    paths, env = setup
    pid_file = tmp_path / "child.pid"
    _accelerate(paths, "sleep", str(pid_file))
    first = subprocess.Popen(
        [sys.executable, *_train(paths)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_file(pid_file)
        second = _run(env, *_train(paths))
        assert second.returncode == 2, second.stdout + second.stderr
        assert str(first.pid) in second.stdout
    finally:
        first.send_signal(signal.SIGINT)
        first.wait(timeout=TIMEOUT_S)
        _kill_group_of(pid_file)


def test_the_settings_of_the_window_are_left_as_they_were(setup):
    paths, env = setup
    _accelerate(paths, "train", "0")
    paths.preview_settings_file.write_text('{"enabled": false}')
    before = paths.config_file.read_text(), paths.preview_settings_file.read_text()
    result = _run(env, *_train(paths))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (paths.config_file.read_text(), paths.preview_settings_file.read_text()) == before


def test_the_training_config_has_the_preset_and_the_options(setup):
    paths, env = setup
    _accelerate(paths, "train", "0")
    result = _run(env, *_train(paths, "--resolution", "1024"))
    assert result.returncode == 0, result.stdout + result.stderr
    training = tomllib.loads(next(paths.configs_dir.glob("*_training_*.toml")).read_text())
    dataset = next(paths.configs_dir.glob("*_dataset_*.toml")).read_text()
    assert training["network_dim"] == 32
    assert training["network_alpha"] == 16
    assert training["max_train_epochs"] == 2
    assert "resolution = 1024" in dataset
    assert "num_repeats = 1" in dataset
    assert "caption_dropout_rate = 0.05" in dataset


def test_a_relative_dataset_is_found_from_where_the_command_runs(setup):
    paths, env = setup
    _accelerate(paths, "train", "0")
    arguments = list(_train(paths))
    arguments[arguments.index("--dataset") + 1] = "images"
    result = subprocess.run(
        [sys.executable, *arguments],
        env=env,
        cwd=paths.root.parents[1],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_without_a_terminal_and_without_yes_nothing_starts(setup):
    paths, env = setup
    _accelerate(paths, "train", "0")
    result = subprocess.run(
        [sys.executable, *ENTRY, *_options(paths)],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 2
    assert "--yes" in result.stdout
    assert list(paths.configs_dir.glob("*.toml")) == []


def _fork_on_a_terminal(env, *arguments):
    """Start the command on a new terminal; returns its pid and the terminal's other end."""
    pid, terminal = pty.fork()
    if pid == 0:  # the child must never return into the test run, even if exec fails
        try:
            os.execve(sys.executable, [sys.executable, *arguments], env)
        finally:
            os._exit(127)
    return pid, terminal


def _reap(pid):
    """The exit code of `pid`, killing it if it has not ended within the time limit."""
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        finished, status = os.waitpid(pid, os.WNOHANG)
        if finished:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    raise AssertionError("the command did not exit")


def _answer_at_the_prompt(paths, env, answer):
    """Run the command on a real terminal and answer its question; returns its output and code."""
    pid, terminal = _fork_on_a_terminal(env, *ENTRY, *_options(paths))
    output = b""
    answered = False
    deadline = time.monotonic() + TIMEOUT_S
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            readable, _, _ = select.select([terminal], [], [], remaining)
            if not readable:
                break
            try:
                chunk = os.read(terminal, 4096)
            except OSError:  # the command has ended
                break
            if not chunk:
                break
            output += chunk
            if not answered and b"[y/N]" in output:
                os.write(terminal, answer)
                answered = True
    finally:
        if time.monotonic() >= deadline:
            os.kill(pid, signal.SIGKILL)
        code = _reap(pid)
        os.close(terminal)
    return output.decode(errors="replace"), code


def test_answering_yes_on_a_terminal_starts_the_training(setup):
    paths, env = setup
    _accelerate(paths, "train", "0")
    output, code = _answer_at_the_prompt(paths, env, b"y\n")
    assert code == 0, output
    assert "OK: Training completed successfully!" in output


@pytest.mark.parametrize("answer", [b"n\n", b"\n", b"maybe\n", b"\x04"])
def test_any_other_answer_does_not_start_it(setup, answer):
    paths, env = setup
    _accelerate(paths, "train", "0")
    output, code = _answer_at_the_prompt(paths, env, answer)
    assert code == 2, output
    assert "INFO: Training not started." in output
    assert list(paths.configs_dir.glob("*.toml")) == []
    assert not paths.gpu_lock_file.exists()


def test_ctrl_c_at_the_question_starts_nothing(setup):
    paths, env = setup
    _accelerate(paths, "train", "0")
    output, code = _answer_at_the_prompt(paths, env, b"\x03")
    assert code == 130, output
    assert "INFO: Training not started." in output
    assert list(paths.configs_dir.glob("*.toml")) == []


def test_output_sent_to_a_file_does_not_hide_the_question(setup):
    """stdin is a terminal but the question would go into the file: refuse, do not wait."""
    paths, env = setup
    _accelerate(paths, "train", "0")
    controller, terminal = pty.openpty()
    try:
        result = subprocess.run(
            [sys.executable, *ENTRY, *_options(paths)],
            env=env,
            stdin=terminal,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    finally:
        os.close(terminal)
        os.close(controller)
    assert result.returncode == 2
    assert "--yes" in result.stdout
    assert "[y/N]" not in result.stdout
