from datetime import datetime

import pytest

from anima_trainer.runtime.install_plan import InstallPlan, InstallStep
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.workflow import install_session
from anima_trainer.workflow.install_session import InstallSession
from tests.fakes import FakeInstaller

NOW = datetime(2026, 10, 24, 9, 30, 0)
STEPS = (InstallStep("venv", "Create the training venv"), InstallStep("verify", "Check the GPU"))
LOG_NAME = "install_2026-10-24_093000.log"


@pytest.fixture
def paths(tmp_path):
    return RuntimePaths(tmp_path / "data")


@pytest.fixture
def parts(qtbot, paths):
    installer = FakeInstaller()
    plans = []

    def plan(for_paths):
        plans.append(for_paths)
        return InstallPlan(STEPS, ())

    session = InstallSession(installer, plan=plan, clock=lambda: NOW)
    log, steps, results = [], [], []
    session.log_line.connect(log.append)
    session.step_changed.connect(steps.append)
    session.finished.connect(lambda ok, message: results.append((ok, message)))
    return {
        "session": session,
        "installer": installer,
        "plans": plans,
        "log": log,
        "steps": steps,
        "results": results,
    }


def _log_file(paths):
    return paths.logs_dir / LOG_NAME


def test_start_runs_the_planned_steps_and_opens_the_install_log(parts, paths):
    problems = parts["session"].start(paths)

    assert problems == ()
    assert parts["plans"] == [paths]
    assert parts["installer"].started == [STEPS]
    assert parts["session"].is_running()
    assert parts["log"][0] == f"Install log: {_log_file(paths)}"
    assert _log_file(paths).exists()


def test_a_plan_that_is_not_ready_is_not_started(qtbot, paths):
    installer = FakeInstaller()
    session = InstallSession(installer, plan=lambda _paths: InstallPlan(STEPS, ("git not found.",)))

    problems = session.start(paths)

    assert problems == ("git not found.",)
    assert installer.started == []
    assert not paths.root.exists()  # nothing was written for a plan that cannot run


def test_steps_and_output_reach_the_window_and_the_log_file(parts, paths):
    parts["session"].start(paths)
    installer = parts["installer"]

    installer.step_started.emit(1, 2, STEPS[0])
    installer.line_received.emit("created venv")
    installer.finish(True, "Installation completed.")

    assert parts["steps"] == ["Step 1/2: Create the training venv"]
    assert parts["log"][1:] == [
        "Step 1/2: Create the training venv",
        "created venv",
        "OK: Installation completed.",
    ]
    assert parts["results"] == [(True, "Installation completed.")]
    written = _log_file(paths).read_text(encoding="utf-8").splitlines()
    assert written[0] == f"Started: {NOW.isoformat()}"
    assert written[1:] == parts["log"][1:]


def test_a_failed_install_is_reported_as_an_error(parts, paths):
    parts["session"].start(paths)

    parts["installer"].finish(False, "Check the GPU failed (exit code 1).")

    assert parts["log"][-1] == "ERROR: Check the GPU failed (exit code 1)."
    assert parts["results"] == [(False, "Check the GPU failed (exit code 1).")]
    assert not parts["session"].is_running()


def test_an_unwritable_log_prevents_the_start(parts, paths):
    paths.root.mkdir(parents=True)
    paths.logs_dir.write_text("not a folder")

    problems = parts["session"].start(paths)

    assert len(problems) == 1
    assert "install log" in problems[0]
    assert parts["installer"].started == []


def test_a_log_write_failure_warns_once_and_keeps_streaming(parts, paths, monkeypatch):
    class BrokenFile:
        def write(self, _text):
            raise OSError("disk full")

        def flush(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(install_session, "open", lambda *_a, **_k: BrokenFile(), raising=False)
    parts["session"].start(paths)

    parts["installer"].line_received.emit("one")
    parts["installer"].line_received.emit("two")

    warnings = [line for line in parts["log"] if line.startswith("WARN:")]
    assert len(warnings) == 1
    assert "disk full" in warnings[0]
    assert parts["log"][-2:] == ["one", "two"]


def test_start_while_running_is_rejected(parts, paths):
    parts["session"].start(paths)

    with pytest.raises(RuntimeError):
        parts["session"].start(paths)


def test_cancel_and_shutdown_reach_the_installer(parts, paths):
    parts["session"].start(paths)

    parts["session"].cancel()
    parts["session"].shutdown()

    assert parts["installer"].cancels == 1
    assert parts["installer"].shut_down == 1


def test_the_log_is_closed_when_the_installer_cannot_start(parts, paths, monkeypatch):
    def broken(_steps):
        raise RuntimeError("busy")

    monkeypatch.setattr(parts["installer"], "start", broken)

    with pytest.raises(RuntimeError):
        parts["session"].start(paths)

    parts["installer"].line_received.emit("late line")
    assert "late line" not in _log_file(paths).read_text(encoding="utf-8")
