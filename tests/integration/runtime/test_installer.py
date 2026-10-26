import sys
import time
from pathlib import Path

import pytest
import shiboken6

from anima_trainer.runtime.install_plan import SD_SCRIPTS_COMMIT, InstallStep
from anima_trainer.runtime.installer import CANCELLED_MESSAGE, RuntimeInstaller
from tests.process_helpers import wait_until_dead

pytestmark = pytest.mark.integration

FAKE = Path(__file__).parents[2] / "fixtures" / "fake_accelerate.py"
TIMEOUT_MS = 15000


def _python(key, code, working_dir, marker=None):
    return InstallStep(key, f"Step {key}", (sys.executable, "-c", code), working_dir, marker)


def _fake(key, working_dir, *args):
    return InstallStep(key, f"Step {key}", (sys.executable, str(FAKE), *args), working_dir)


def _collect(installer):
    started, lines = [], []
    installer.step_started.connect(
        lambda index, total, step: started.append((index, total, step.key))
    )
    installer.line_received.connect(lines.append)
    return started, lines


def _install(qtbot, installer, steps):
    with qtbot.waitSignal(installer.finished, timeout=TIMEOUT_MS) as blocker:
        installer.start(steps)
    return tuple(blocker.args)


def _start_sleeping(qtbot, installer, tmp_path, pid_file):
    with qtbot.waitSignal(installer.line_received, timeout=TIMEOUT_MS):
        installer.start((_fake("sleep", tmp_path, "sleep", str(pid_file)),))
    return int(pid_file.read_text())


def test_runs_command_steps_in_order_and_streams_their_lines(qtbot, tmp_path):
    installer = RuntimeInstaller()
    started, lines = _collect(installer)
    steps = (
        _python("one", "print('a'); print('b')", tmp_path),
        _python("two", "print('c')", tmp_path),
    )

    ok, message = _install(qtbot, installer, steps)

    assert ok
    assert message
    assert started == [(1, 2, "one"), (2, 2, "two")]
    assert lines == ["a", "b", "c"]
    assert not installer.is_running()


def test_commands_run_in_the_step_working_dir_with_unbuffered_output(qtbot, tmp_path):
    installer = RuntimeInstaller()
    _, lines = _collect(installer)
    code = "import os; print(os.getcwd()); print(os.environ.get('PYTHONUNBUFFERED'))"

    _install(qtbot, installer, (_python("cwd", code, tmp_path),))

    assert lines == [str(tmp_path), "1"]


def test_action_steps_run_in_process(qtbot, tmp_path):
    installer = RuntimeInstaller()
    started, _ = _collect(installer)
    folder = tmp_path / "data"
    action = InstallStep("data_dir", "Create", action=lambda: folder.mkdir())

    ok, _ = _install(qtbot, installer, (action, _python("after", "pass", tmp_path)))

    assert ok
    assert folder.is_dir()
    assert [key for _, _, key in started] == ["data_dir", "after"]


def test_marker_holds_the_pinned_commit_after_success(qtbot, tmp_path):
    installer = RuntimeInstaller()
    marker = tmp_path / ".install" / "verified.commit"

    _install(qtbot, installer, (_python("verify", "pass", tmp_path, marker),))

    assert marker.read_text(encoding="utf-8").strip() == SD_SCRIPTS_COMMIT


def test_marker_holds_the_step_stamp_after_success(qtbot, tmp_path):
    installer = RuntimeInstaller()
    marker = tmp_path / ".install" / "requirements.commit"
    step = InstallStep(
        "requirements",
        "Step requirements",
        (sys.executable, "-c", "pass"),
        tmp_path,
        marker,
        stamp="abc +torchvision",
    )

    _install(qtbot, installer, (step,))

    assert marker.read_text(encoding="utf-8").strip() == "abc +torchvision"


def test_failed_command_stops_the_install_without_writing_its_marker(qtbot, tmp_path):
    installer = RuntimeInstaller()
    started, lines = _collect(installer)
    marker = tmp_path / ".install" / "requirements.commit"
    steps = (
        _python("requirements", "print('no torch'); raise SystemExit(3)", tmp_path, marker),
        _python("never", "print('should not run')", tmp_path),
    )

    ok, message = _install(qtbot, installer, steps)

    assert not ok
    assert "Step requirements" in message
    assert "3" in message
    assert lines == ["no torch"]
    assert [key for _, _, key in started] == ["requirements"]
    assert not marker.exists()
    assert not installer.is_running()


def test_failing_action_stops_the_install(qtbot, tmp_path):
    installer = RuntimeInstaller()
    started, _ = _collect(installer)

    def broken():
        raise OSError("disk full")

    steps = (
        InstallStep("accelerate_config", "Write", action=broken),
        _python("never", "", tmp_path),
    )

    ok, message = _install(qtbot, installer, steps)

    assert not ok
    assert "disk full" in message
    assert [key for _, _, key in started] == ["accelerate_config"]


def test_missing_program_fails_the_install(qtbot, tmp_path):
    installer = RuntimeInstaller()
    step = InstallStep("venv", "Create the venv", ("/nonexistent/python3",), tmp_path)

    ok, message = _install(qtbot, installer, (step,))

    assert not ok
    assert "/nonexistent/python3" in message
    assert not installer.is_running()


def test_crashed_command_fails_the_install(qtbot, tmp_path):
    installer = RuntimeInstaller()

    ok, message = _install(qtbot, installer, (_fake("crash", tmp_path, "crash"),))

    assert not ok
    assert "Step crash" in message


def test_unwritable_marker_fails_the_install(qtbot, tmp_path):
    installer = RuntimeInstaller()
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    step = _python("verify", "pass", tmp_path, blocker / "verified.commit")

    ok, message = _install(qtbot, installer, (step,))

    assert not ok
    assert "verified.commit" in message


def test_carriage_returns_split_progress_lines(qtbot, tmp_path):
    installer = RuntimeInstaller()
    _, lines = _collect(installer)

    _install(qtbot, installer, (_fake("text", tmp_path, "text", "10%\\r50%\\rdone\\n", "0"),))

    assert lines == ["10%", "50%", "done"]


def test_no_steps_finishes_successfully(qtbot):
    installer = RuntimeInstaller()

    ok, _ = _install(qtbot, installer, ())

    assert ok


def test_start_while_running_is_rejected(qtbot, tmp_path):
    installer = RuntimeInstaller()
    _start_sleeping(qtbot, installer, tmp_path, tmp_path / "child.pid")

    with pytest.raises(RuntimeError):
        installer.start(())

    with qtbot.waitSignal(installer.finished, timeout=TIMEOUT_MS):
        installer.cancel()


def test_cancel_when_idle_does_nothing():
    RuntimeInstaller().cancel()


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX only")
def test_cancel_kills_the_process_group_and_skips_later_steps(qtbot, tmp_path):
    installer = RuntimeInstaller()
    started, _ = _collect(installer)
    pid_file = tmp_path / "child.pid"
    marker = tmp_path / "marker.commit"
    steps = (
        InstallStep(
            "sleep", "Sleep", (sys.executable, str(FAKE), "sleep", str(pid_file)), tmp_path, marker
        ),
        _python("never", "", tmp_path),
    )
    with qtbot.waitSignal(installer.line_received, timeout=TIMEOUT_MS):
        installer.start(steps)
    child_pid = int(pid_file.read_text())

    with qtbot.waitSignal(installer.finished, timeout=TIMEOUT_MS) as blocker:
        installer.cancel()

    assert tuple(blocker.args) == (False, CANCELLED_MESSAGE)
    assert wait_until_dead(child_pid)
    assert not marker.exists()
    assert [key for _, _, key in started] == ["sleep"]
    assert not installer.has_pending_kill()


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX only")
def test_shutdown_stops_the_group_and_blocks_until_exit(qtbot, tmp_path):
    installer = RuntimeInstaller()
    child_pid = _start_sleeping(qtbot, installer, tmp_path, tmp_path / "child.pid")

    installer.shutdown()

    assert not installer.is_running()
    assert wait_until_dead(child_pid)


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX only")
def test_shutdown_escalates_to_sigkill_when_sigterm_is_ignored(qtbot, tmp_path):
    installer = RuntimeInstaller(kill_grace_ms=300)
    results = []
    installer.finished.connect(lambda ok, message: results.append((ok, message)))
    with qtbot.waitSignal(installer.line_received, timeout=TIMEOUT_MS):
        installer.start((_fake("stubborn", tmp_path, "ignore-term"),))

    started = time.monotonic()
    installer.shutdown()

    assert not installer.is_running()
    assert time.monotonic() - started < 5
    assert results == [(False, CANCELLED_MESSAGE)]


def test_shutdown_when_idle_does_nothing():
    RuntimeInstaller().shutdown()


def test_cancel_before_the_first_step_runs_nothing(qtbot, tmp_path):
    installer = RuntimeInstaller()
    started, _ = _collect(installer)

    with qtbot.waitSignal(installer.finished, timeout=TIMEOUT_MS) as blocker:
        installer.start((_python("never", "", tmp_path),))
        installer.cancel()

    assert tuple(blocker.args) == (False, CANCELLED_MESSAGE)
    assert started == []


def test_shutdown_before_the_first_step_finishes_once(qtbot, tmp_path):
    installer = RuntimeInstaller()
    started, _ = _collect(installer)
    results = []
    installer.finished.connect(lambda ok, message: results.append((ok, message)))
    installer.start((_python("never", "", tmp_path),))

    installer.shutdown()
    qtbot.wait(50)  # let the deferred first step run (and do nothing)

    assert results == [(False, CANCELLED_MESSAGE)]
    assert started == []
    assert not installer.is_running()


def test_restart_after_shutdown_ignores_the_previous_run(qtbot, tmp_path):
    installer = RuntimeInstaller()
    started, _ = _collect(installer)
    steps = (
        _python("slow", "import time; time.sleep(0.5)", tmp_path),
        _python("next", "", tmp_path),
    )
    installer.start(steps)
    installer.shutdown()

    installer.start(steps)
    qtbot.wait(100)  # both the stale and the new deferred first step have run

    assert [key for _, _, key in started] == ["slow"]
    installer.shutdown()


def _flag_step(flag, tmp_path):
    code = f"import pathlib; pathlib.Path({str(flag)!r}).write_text('ran')"
    return _python("flag", code, tmp_path)


def test_cancel_from_step_started_does_not_run_the_step(qtbot, tmp_path):
    installer = RuntimeInstaller()
    installer.step_started.connect(lambda *_: installer.cancel())
    flag = tmp_path / "ran"

    with qtbot.waitSignal(installer.finished, timeout=TIMEOUT_MS) as blocker:
        installer.start((_flag_step(flag, tmp_path),))

    assert tuple(blocker.args) == (False, CANCELLED_MESSAGE)
    assert not flag.exists()


def test_shutdown_from_step_started_does_not_run_the_step(qtbot, tmp_path):
    installer = RuntimeInstaller()
    results = []
    installer.finished.connect(lambda ok, message: results.append((ok, message)))
    installer.step_started.connect(lambda *_: installer.shutdown())
    flag = tmp_path / "ran"

    installer.start((_flag_step(flag, tmp_path),))
    qtbot.waitUntil(lambda: bool(results), timeout=TIMEOUT_MS)
    qtbot.wait(300)  # long enough for a stray process to write the flag

    assert results == [(False, CANCELLED_MESSAGE)]
    assert not flag.exists()
    assert not installer.is_running()


def test_deleted_installer_ignores_its_pending_step(qtbot):
    installer = RuntimeInstaller()
    installer.start(())

    shiboken6.delete(installer)
    qtbot.wait(50)  # the deferred first step must not touch the deleted object
