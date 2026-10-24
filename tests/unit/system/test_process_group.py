import os
import signal

import pytest

from anima_trainer.system import process_group


class FakeProcess:
    def __init__(self, pid):
        self._pid = pid
        self.calls = []

    def processId(self):
        return self._pid

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")


@pytest.fixture
def killpg_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        process_group.os, "killpg", lambda pid, signum: calls.append((pid, signum)), raising=False
    )
    return calls


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX only")
def test_posix_signals_go_to_the_whole_group(killpg_calls):
    process = FakeProcess(1234)

    process_group.terminate_group(process)
    process_group.kill_group(process)

    assert killpg_calls == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]
    assert process.calls == []


def test_without_process_groups_qprocess_terminate_and_kill_are_used(monkeypatch, killpg_calls):
    monkeypatch.setattr(process_group, "_POSIX", False)
    process = FakeProcess(1234)

    process_group.terminate_group(process)
    process_group.kill_group(process)

    assert process.calls == ["terminate", "kill"]
    assert killpg_calls == []


def test_a_process_without_pid_falls_back_to_qprocess(killpg_calls):
    process = FakeProcess(0)

    process_group.terminate_group(process)

    assert process.calls == ["terminate"]
    assert killpg_calls == []


def test_a_group_that_already_exited_is_ignored(monkeypatch):
    def gone(pid, signum):
        raise ProcessLookupError

    monkeypatch.setattr(process_group.os, "killpg", gone, raising=False)
    monkeypatch.setattr(process_group, "_POSIX", True)

    process_group.kill_group(FakeProcess(1234))
