"""`anima-trainer tui` as a user runs it: a real process on a real terminal."""

import contextlib
import fcntl
import json
import os
import pty
import select
import shlex
import signal
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

import pytest

from tests.e2e.helpers import make_runtime
from tests.process_helpers import wait_until_dead

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX terminal"),
]
FAKE = Path(__file__).parents[2] / "fixtures" / "fake_accelerate.py"
ENTRY = ("-m", "anima_trainer.cli.entry", "tui")
TIMEOUT_S = 30


@pytest.fixture
def setup(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.png").write_bytes(b"")
    (images / "a.txt").write_text("caption")
    data_home, config_home = tmp_path / "data", tmp_path / "config"
    config_home.mkdir()
    paths = make_runtime(data_home / "anima-trainer", images, {"gpu_index": "0"})
    paths.tui_values_file.write_text(
        json.dumps(
            {
                "project": "pty_run",
                "dataset": str(images),
                "output": str(paths.root / "out"),
                "preset": "Person",
            }
        )
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"}
    }
    env |= {
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "TERM": "xterm-256color",
    }
    pid_file = tmp_path / "child.pid"
    command = " ".join(
        shlex.quote(part) for part in (sys.executable, str(FAKE), "sleep", str(pid_file))
    )
    paths.venv_accelerate.write_text(f'#!/bin/sh\nexec {command} "$@"\n')
    paths.venv_accelerate.chmod(0o755)
    return paths, env, pid_file


def _fork(env):
    pid, terminal = pty.fork()
    if pid == 0:
        try:
            os.execve(sys.executable, [sys.executable, *ENTRY], env)
        finally:
            os._exit(127)
    fcntl.ioctl(terminal, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 120, 0, 0))
    return pid, terminal


def _drain_until(terminal, condition, send=()):
    """Read the screen, typing `send` one piece each second, until `condition()` holds."""
    pending = list(send)
    next_send = time.monotonic() + 2
    deadline = time.monotonic() + TIMEOUT_S
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        readable, _, _ = select.select([terminal], [], [], 0.1)
        if readable:
            os.read(terminal, 65536)
        if pending and time.monotonic() >= next_send:
            os.write(terminal, pending.pop(0))
            next_send = time.monotonic() + 1


def _reap(pid):
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        finished, status = os.waitpid(pid, os.WNOHANG)
        if finished:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    raise AssertionError("the command did not exit")


def _kill_group_of(pid_file):
    if pid_file.exists() and pid_file.read_text().strip():
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(int(pid_file.read_text())), signal.SIGKILL)


def test_a_lost_terminal_stops_the_training_and_leaves_no_process(setup):
    _paths, env, pid_file = setup
    pid, terminal = _fork(env)
    closed = False
    try:
        # Enter on Train in the menu, repeats, then Enter on the form and on Start.
        _drain_until(
            terminal,
            lambda: pid_file.exists() and pid_file.read_text().strip(),
            send=(b"\r", b"1", b"\r", b"\r"),
        )
        os.close(terminal)
        closed = True
        assert wait_until_dead(int(pid_file.read_text()), TIMEOUT_S)
        assert _reap(pid) == 130
    finally:
        if not closed:
            os.close(terminal)
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass
        _kill_group_of(pid_file)


def test_without_a_terminal_it_points_to_the_train_command(setup):
    _paths, env, _pid_file = setup
    result = subprocess.run(
        [sys.executable, *ENTRY],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 2
    assert "anima-trainer train" in result.stdout


def test_no_graphical_qt_module_is_loaded(setup):
    _paths, env, _pid_file = setup
    code = (
        "import json, sys\n"
        "from anima_trainer.cli.entry import main\n"
        "code = main(['tui'])\n"
        "loaded = [m for m in sys.modules if m in ('PySide6.QtGui', 'PySide6.QtWidgets')]\n"
        "sys.stderr.write('LOADED=' + json.dumps(loaded))\n"
        "sys.exit(code)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 2
    assert "LOADED=[]" in result.stderr
