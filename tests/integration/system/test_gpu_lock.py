"""The lock between processes that use the GPU of one data folder (window, terminal)."""

import os
import time

import pytest

from anima_trainer.system.gpu_lock import GpuLock, lock_refusal
from tests.lock_helpers import held_by_another_process, start_holder

pytestmark = pytest.mark.integration


@pytest.fixture
def file(tmp_path):
    return tmp_path / "gpu.lock"


def test_a_free_lock_is_taken_and_given_back(file):
    lock, other = GpuLock(), GpuLock()
    assert lock.acquire("training", file) is None
    assert other.acquire("training", file) is not None
    lock.release("training")
    assert other.acquire("training", file) is None
    other.release("training")
    assert not file.exists()


def test_one_process_can_hold_it_for_several_jobs(file):
    lock, other = GpuLock(), GpuLock()
    assert lock.acquire("training", file) is None
    assert lock.acquire("generator", file) is None
    lock.release("training")
    assert other.acquire("training", file) is not None
    lock.release("generator")
    assert other.acquire("training", file) is None
    other.release("training")


def test_taking_it_twice_for_one_job_needs_one_release(file):
    lock, other = GpuLock(), GpuLock()
    lock.acquire("training", file)
    lock.acquire("training", file)
    lock.release("training")
    assert other.acquire("training", file) is None
    other.release("training")


def test_giving_back_a_lock_never_taken_does_nothing(file):
    lock = GpuLock()
    lock.release("training")
    assert lock.acquire("training", file) is None
    lock.release("generator")
    assert GpuLock().acquire("training", file) is not None
    lock.release("training")


def test_another_process_holding_it_is_named_in_the_refusal(file):
    with held_by_another_process(file) as pid:
        refusal = GpuLock().acquire("training", file)
    assert refusal is not None
    assert str(pid) in refusal


def test_a_long_training_does_not_look_abandoned(file):
    with held_by_another_process(file):
        hours_ago = time.time() - 10 * 3600
        os.utime(file, (hours_ago, hours_ago))
        assert GpuLock().acquire("training", file) is not None


def test_a_lock_left_by_a_dead_process_does_not_block(file):
    holder = start_holder(file)
    holder.kill()
    holder.wait()
    lock = GpuLock()
    assert lock.acquire("training", file) is None
    lock.release("training")


def test_the_data_folder_is_created_when_missing(tmp_path):
    lock = GpuLock()
    assert lock.acquire("install", tmp_path / "new" / "gpu.lock") is None
    lock.release("install")


def test_a_folder_that_cannot_be_created_is_a_refusal(tmp_path):
    (tmp_path / "file").write_text("")
    refusal = GpuLock().acquire("training", tmp_path / "file" / "gpu.lock")
    assert refusal is not None
    assert "gpu.lock" in refusal


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root ignores the permissions of a folder",
)
def test_a_lock_that_cannot_be_written_is_a_refusal_not_a_busy_message(tmp_path):
    folder = tmp_path / "readonly"
    folder.mkdir()
    folder.chmod(0o500)
    try:
        refusal = GpuLock().acquire("training", folder / "gpu.lock")
    finally:
        folder.chmod(0o700)
    assert refusal is not None
    assert "gpu.lock" in refusal
    assert "process" not in refusal


def test_another_data_folder_cannot_be_locked_while_one_is_held(tmp_path):
    lock = GpuLock()
    lock.acquire("training", tmp_path / "a" / "gpu.lock")
    assert lock.acquire("generator", tmp_path / "b" / "gpu.lock") is not None
    lock.release("training")


def test_a_check_on_a_free_lock_does_not_keep_it(file):
    assert lock_refusal(file) is None
    lock = GpuLock()
    assert lock.acquire("training", file) is None
    lock.release("training")


def test_a_check_names_the_process_holding_the_lock(file):
    with held_by_another_process(file) as pid:
        refusal = lock_refusal(file)
    assert refusal is not None
    assert str(pid) in refusal


def test_a_check_inside_a_process_that_holds_the_lock_is_a_refusal(file):
    lock = GpuLock()
    lock.acquire("generator", file)
    assert lock_refusal(file) is not None
    lock.release("generator")
