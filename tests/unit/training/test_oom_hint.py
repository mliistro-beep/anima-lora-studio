import subprocess

from anima_trainer.training.oom_hint import (
    GPU_OOM_HINT,
    SYSTEM_OOM_HINT,
    oom_hint,
    read_kernel_log,
)

CUDA_OOM = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB."


def test_success_never_gives_a_hint():
    assert oom_hint(0, (CUDA_OOM,), "Out of memory: Killed process 123") is None


def test_cuda_out_of_memory_in_log():
    assert oom_hint(1, ("Traceback ...", CUDA_OOM), None) == GPU_OOM_HINT


def test_legacy_runtime_error_message():
    assert oom_hint(1, ("RuntimeError: CUDA out of memory.",), None) == GPU_OOM_HINT


def test_kernel_oom_killer_in_recent_kernel_log():
    kernel_log = "\n".join(["noise"] * 10 + ["[Sun Oct 19] Out of memory: Killed process 4242"])

    assert oom_hint(-9, (), kernel_log) == SYSTEM_OOM_HINT


def test_kernel_messages_older_than_tail_are_ignored():
    kernel_log = "\n".join(["oom_reaper: reaped process"] + ["noise"] * 40)

    assert oom_hint(1, (), kernel_log) is None


def test_sigkill_without_evidence_still_suggests_system_oom():
    assert oom_hint(-9, (), None) == SYSTEM_OOM_HINT
    assert oom_hint(137, (), None) == SYSTEM_OOM_HINT


def test_generic_failure_gives_no_hint():
    assert oom_hint(1, ("ValueError: bad config",), "all quiet") is None


def test_hints_keep_citron_advice():
    assert "network_dim=8" in GPU_OOM_HINT
    assert "resolution=512" in SYSTEM_OOM_HINT


def test_read_kernel_log_returns_stdout(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="kernel text", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert read_kernel_log() == "kernel text"


def test_read_kernel_log_returns_none_when_unavailable(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("dmesg")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert read_kernel_log() is None


def test_read_kernel_log_returns_none_when_not_permitted(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="Operation not permitted")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert read_kernel_log() is None


def test_crash_by_sigkill_as_reported_by_qprocess_suggests_system_oom():
    # QProcess reports a signal death as crashed=True with the signal number as exit code.
    assert oom_hint(9, (), None, crashed=True) == SYSTEM_OOM_HINT


def test_normal_exit_code_nine_is_not_a_signal():
    assert oom_hint(9, (), None, crashed=False) is None
