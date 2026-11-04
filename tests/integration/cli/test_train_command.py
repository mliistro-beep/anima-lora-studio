"""`anima-trainer train` as a user runs it: a real process, no display, a fake accelerate."""

import contextlib
import json
import os
import pty
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.e2e.helpers import make_runtime
from tests.process_helpers import is_alive, wait_until_dead

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="the fake accelerate is a shell script"),
]
FAKE = Path(__file__).parents[2] / "fixtures" / "fake_accelerate.py"
TIMEOUT_S = 30
PROJECT = "cli_run"
ENTRY = ("-m", "anima_trainer.cli.entry", "train")


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
    result = _run(env, *ENTRY)
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
        "code = main(['train'])\n"
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
    result = _run(env, *ENTRY)
    assert result.returncode == 1
    assert "exit code: 3" in result.stdout


def test_a_missing_model_stops_before_writing_any_config(setup):
    paths, env = setup
    paths.vae_model.unlink()
    result = _run(env, *ENTRY)
    assert result.returncode == 2
    assert "ERROR: Missing models: VAE" in result.stdout
    assert "graphical app" in result.stdout
    assert list(paths.configs_dir.glob("*.toml")) == []


def test_a_missing_base_model_is_not_downloaded(setup):
    paths, env = setup
    paths.dit_model("anima-base-v1.0").unlink()
    result = _run(env, *ENTRY)
    assert result.returncode == 2
    assert "anima-base-v1.0" in result.stdout
    assert "Downloading" not in result.stdout


def test_a_runtime_that_is_not_installed_is_reported(setup):
    paths, env = setup
    paths.train_script.unlink()
    result = _run(env, *ENTRY)
    assert result.returncode == 2
    assert "Training script not found" in result.stdout


def test_settings_never_configured_are_reported(setup):
    paths, env = setup
    paths.config_file.write_text(json.dumps({"project_name": PROJECT}))
    result = _run(env, *ENTRY)
    assert result.returncode == 2
    assert "ERROR: Image directory cannot be empty." in result.stdout


def test_ctrl_c_stops_the_training_and_its_children(setup, tmp_path):
    paths, env = setup
    pid_file = tmp_path / "child.pid"
    _accelerate(paths, "sleep", str(pid_file))
    process = subprocess.Popen(
        [sys.executable, *ENTRY],
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
    pid, terminal = pty.fork()
    if pid == 0:  # the child: becomes the command on the new terminal
        os.execve(sys.executable, [sys.executable, *ENTRY], env)
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
