import sys
import time
from pathlib import Path
from types import MappingProxyType

import pytest

from anima_trainer.training.command_builder import LaunchSpec
from anima_trainer.training.log_parser import CheckpointSaved, EpochStarted, StepProgress
from anima_trainer.training.runner import TrainingRunner
from tests.process_helpers import is_alive

pytestmark = pytest.mark.integration

FAKE = Path(__file__).parents[2] / "fixtures" / "fake_accelerate.py"
TIMEOUT_MS = 15000


def _spec(tmp_path, *args, env=None):
    return LaunchSpec(
        program=Path(sys.executable),
        arguments=(str(FAKE), *args),
        env=MappingProxyType(env or {"PYTHONUNBUFFERED": "1"}),
        working_dir=tmp_path,
    )


def _run(qtbot, runner, spec, log_path):
    with qtbot.waitSignal(runner.finished, timeout=TIMEOUT_MS) as blocker:
        runner.start(spec, log_path)
    return blocker.args[0]


def _collect(runner):
    lines, events = [], []
    runner.line_received.connect(lines.append)
    runner.event_received.connect(events.append)
    return lines, events


def test_replays_real_log_into_lines_events_and_log_file(qtbot, tmp_path, citron_fixtures):
    fixture = citron_fixtures / "training_log_excerpt.log"
    runner = TrainingRunner()
    lines, events = _collect(runner)
    log_path = tmp_path / "logs" / "run.log"

    outcome = _run(qtbot, runner, _spec(tmp_path, "replay", str(fixture), "0"), log_path)

    assert (outcome.exit_code, outcome.crashed, outcome.stopped) == (0, False, False)
    assert outcome.log_path == log_path
    assert lines == fixture.read_text(encoding="utf-8").splitlines()
    assert EpochStarted(1, 10) in events
    assert [e.epoch for e in events if isinstance(e, CheckpointSaved)] == [1, None]
    written = log_path.read_text(encoding="utf-8").splitlines()
    assert written[0].startswith("Command: ")
    assert written[1].startswith("Started: ")
    assert written[3:] == lines


def test_carriage_return_updates_become_separate_lines(qtbot, tmp_path):
    runner = TrainingRunner()
    lines, events = _collect(runner)
    text = "steps: | 1/4 [00:01<00:03, 1.0s/it]\\rsteps: | 2/4 [00:02<00:02, 1.0s/it]\\n"

    _run(qtbot, runner, _spec(tmp_path, "text", text, "0"), tmp_path / "r.log")

    assert [e.step for e in events if isinstance(e, StepProgress)] == [1, 2]
    assert len(lines) == 2


def test_unterminated_last_line_is_flushed(qtbot, tmp_path):
    runner = TrainingRunner()
    lines, _ = _collect(runner)

    _run(qtbot, runner, _spec(tmp_path, "text", "no newline", "0"), tmp_path / "r.log")

    assert lines == ["no newline"]


def test_non_zero_exit_code_is_reported(qtbot, tmp_path):
    runner = TrainingRunner()

    outcome = _run(qtbot, runner, _spec(tmp_path, "text", "boom\\n", "3"), tmp_path / "r.log")

    assert outcome.exit_code == 3
    assert not outcome.stopped
    assert runner.log_tail() == ("boom",)


def test_environment_overrides_are_applied(qtbot, tmp_path):
    runner = TrainingRunner()
    lines, _ = _collect(runner)
    spec = _spec(tmp_path, "env", "CUDA_VISIBLE_DEVICES", env={"CUDA_VISIBLE_DEVICES": "7"})

    _run(qtbot, runner, spec, tmp_path / "r.log")

    assert lines == ["7"]


def test_crash_is_reported(qtbot, tmp_path):
    runner = TrainingRunner()

    outcome = _run(qtbot, runner, _spec(tmp_path, "crash"), tmp_path / "r.log")

    assert outcome.crashed
    assert not outcome.stopped
    if sys.platform != "win32":
        # oom_hint relies on this: QProcess reports the signal number on a signal death.
        assert outcome.exit_code == 9


def test_missing_program_emits_failed_to_start(qtbot, tmp_path):
    runner = TrainingRunner()
    spec = LaunchSpec(Path("/nonexistent/accelerate"), (), MappingProxyType({}), tmp_path)

    with qtbot.waitSignal(runner.failed_to_start, timeout=TIMEOUT_MS) as blocker:
        runner.start(spec, tmp_path / "r.log")

    assert "/nonexistent/accelerate" in blocker.args[0]
    assert not runner.is_running()


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX only")
def test_stop_terminates_the_whole_process_group(qtbot, tmp_path):
    runner = TrainingRunner()
    pid_file = tmp_path / "child.pid"
    with qtbot.waitSignal(runner.line_received, timeout=TIMEOUT_MS):
        runner.start(_spec(tmp_path, "sleep", str(pid_file)), tmp_path / "r.log")
    child_pid = int(pid_file.read_text())

    with qtbot.waitSignal(runner.finished, timeout=TIMEOUT_MS) as blocker:
        runner.stop()

    assert blocker.args[0].stopped
    deadline = time.monotonic() + 5
    while is_alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not is_alive(child_pid)


def test_start_while_running_is_rejected(qtbot, tmp_path):
    runner = TrainingRunner()
    pid_file = tmp_path / "child.pid"
    with qtbot.waitSignal(runner.line_received, timeout=TIMEOUT_MS):
        runner.start(_spec(tmp_path, "sleep", str(pid_file)), tmp_path / "r.log")

    with pytest.raises(RuntimeError):
        runner.start(_spec(tmp_path, "env", "X"), tmp_path / "r2.log")

    with qtbot.waitSignal(runner.finished, timeout=TIMEOUT_MS):
        runner.stop()


def test_stop_when_idle_does_nothing():
    TrainingRunner().stop()


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX only")
def test_shutdown_stops_the_group_and_blocks_until_exit(qtbot, tmp_path):
    runner = TrainingRunner()
    pid_file = tmp_path / "child.pid"
    with qtbot.waitSignal(runner.line_received, timeout=TIMEOUT_MS):
        runner.start(_spec(tmp_path, "sleep", str(pid_file)), tmp_path / "r.log")
    child_pid = int(pid_file.read_text())

    runner.shutdown()

    assert not runner.is_running()
    deadline = time.monotonic() + 5
    while is_alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not is_alive(child_pid)


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX only")
def test_shutdown_escalates_to_sigkill_when_sigterm_is_ignored(qtbot, tmp_path):
    runner = TrainingRunner(kill_grace_ms=300)
    outcomes = []
    runner.finished.connect(outcomes.append)
    with qtbot.waitSignal(runner.line_received, timeout=TIMEOUT_MS):
        runner.start(_spec(tmp_path, "ignore-term"), tmp_path / "r.log")

    started = time.monotonic()
    runner.shutdown()

    assert not runner.is_running()
    assert time.monotonic() - started < 5
    assert outcomes[0].stopped
    assert outcomes[0].crashed


def test_kill_timer_is_cancelled_when_the_run_ends(qtbot, tmp_path):
    runner = TrainingRunner()
    pid_file = tmp_path / "child.pid"
    with qtbot.waitSignal(runner.line_received, timeout=TIMEOUT_MS):
        runner.start(_spec(tmp_path, "sleep", str(pid_file)), tmp_path / "r.log")

    with qtbot.waitSignal(runner.finished, timeout=TIMEOUT_MS):
        runner.stop()

    assert not runner.has_pending_kill()


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full")
def test_log_write_failure_warns_once_and_keeps_streaming(qtbot, tmp_path):
    runner = TrainingRunner()
    lines, _ = _collect(runner)
    warnings = []
    runner.warning.connect(warnings.append)
    spec = _spec(tmp_path, "text", "a\\nb\\nc\\n", "0")

    with qtbot.waitSignal(runner.finished, timeout=TIMEOUT_MS) as blocker:
        runner.start(spec, Path("/dev/full"))
    qtbot.waitUntil(lambda: bool(warnings), timeout=TIMEOUT_MS)

    assert lines == ["a", "b", "c"]
    assert len(warnings) == 1
    assert "log file" in warnings[0].lower()
    assert blocker.args[0].log_path is None  # the log was abandoned, not saved


def test_unopenable_log_raises_without_leaking_state(tmp_path):
    runner = TrainingRunner()
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")

    with pytest.raises(OSError):
        runner.start(_spec(tmp_path, "env", "X"), blocker / "r.log")

    assert not runner.is_running()
