"""Training, generation and install share one lock; another process holding it stops them."""

from datetime import datetime
from pathlib import Path

import pytest

from anima_trainer.preview.inference_command import GenerationRequest
from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.runtime.data_dir import open_data_dir
from anima_trainer.runtime.install_plan import InstallPlan
from anima_trainer.system.gpu_lock import GpuLock
from anima_trainer.training.runner import RunOutcome
from anima_trainer.training.start_service import StartRequest
from anima_trainer.workflow.generator_session import GeneratorSession
from anima_trainer.workflow.install_session import InstallSession
from anima_trainer.workflow.training_session import SessionState, TrainingSession
from tests.e2e.helpers import INSTALL_STEPS, make_runtime
from tests.fakes import FakeDownloader, FakeInstaller, FakeRunner
from tests.lock_helpers import held_by_another_process

pytestmark = pytest.mark.integration
NOW = datetime(2026, 11, 5, 9, 0, 0)
BASE_MODEL = "anima-base-v1.0"
REQUEST = StartRequest("0", 1, BASE_MODEL)
LOG = Path("x.log")


@pytest.fixture
def paths(qapp, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.png").write_bytes(b"")
    (images / "a.txt").write_text("caption")
    return make_runtime(tmp_path / "data", images)


def _free(paths):
    """True when another process could take the lock now."""
    other = GpuLock()
    free = other.acquire("probe", paths.gpu_lock_file) is None
    other.release("probe")
    return free


def _training(paths, lock):
    runner = FakeRunner()
    session = TrainingSession(
        paths,
        open_data_dir(paths.root).settings,
        runner,
        FakeDownloader(),
        clock=lambda: NOW,
        kernel_log=lambda: None,
        lock=lock,
    )
    assert session.configure(session.settings).success
    return session, runner


def _generation(tmp_path):
    return GenerationRequest(
        base_model=BASE_MODEL, lora=None, lora_multiplier=1.0, prompt="a girl", negative_prompt="",
        size=SampleSize(1024, 1024), steps=25, cfg_scale=4.0, flow_shift=3.0, seed=1,
        sampler="er_sde", output=tmp_path / "out" / "a.png",
    )  # fmt: skip


def _install(lock):
    installer = FakeInstaller()
    plan = lambda _paths: InstallPlan(INSTALL_STEPS, ())  # noqa: E731
    return InstallSession(installer, plan=plan, clock=lambda: NOW, lock=lock), installer


def test_a_training_holds_the_lock_until_it_ends(paths):
    session, runner = _training(paths, GpuLock())
    session.start(REQUEST)
    assert session.state is SessionState.TRAINING
    assert not _free(paths)
    runner.finished.emit(RunOutcome(0, False, False, LOG))
    assert session.state is SessionState.IDLE
    assert _free(paths)


def test_a_training_is_refused_while_another_process_holds_the_lock(paths):
    session, runner = _training(paths, GpuLock())
    lines = []
    session.log_line.connect(lines.append)
    with held_by_another_process(paths.gpu_lock_file) as pid:
        session.start(REQUEST)
    assert session.state is SessionState.IDLE
    assert runner.started == []
    assert any(line.startswith("ERROR: ") and str(pid) in line for line in lines)


def test_a_training_that_cannot_launch_gives_the_lock_back(paths):
    session, runner = _training(paths, GpuLock())
    runner.start_error = OSError("no log file")
    session.start(REQUEST)
    assert session.state is SessionState.IDLE
    assert _free(paths)


def test_a_training_that_fails_to_start_gives_the_lock_back(paths):
    session, runner = _training(paths, GpuLock())
    session.start(REQUEST)
    runner.failed_to_start.emit("no accelerate")
    assert _free(paths)


def test_the_shutdown_of_a_training_gives_the_lock_back(paths):
    session, _runner = _training(paths, GpuLock())
    session.start(REQUEST)
    session.shutdown()
    assert _free(paths)


def test_a_download_before_the_training_counts_as_training(paths):
    paths.dit_model(BASE_MODEL).unlink()
    session, _runner = _training(paths, GpuLock())
    session.start(REQUEST)
    assert session.state is SessionState.DOWNLOADING
    assert not _free(paths)
    session.stop()
    session._downloader.cancelled.emit()
    assert _free(paths)


def test_a_model_download_alone_does_not_take_the_lock(paths):
    paths.dit_model(BASE_MODEL).unlink()
    session, _runner = _training(paths, GpuLock())
    session.download_model(BASE_MODEL)
    assert session.state is SessionState.DOWNLOADING
    assert _free(paths)


def test_a_generation_holds_the_lock_until_it_ends(paths, tmp_path):
    runner = FakeRunner()
    session = GeneratorSession(runner, clock=lambda: NOW, lock=GpuLock())
    assert session.generate(paths, (_generation(tmp_path),), "0", False) is None
    assert not _free(paths)
    runner.line_received.emit(f"IMAGE: {tmp_path / 'out' / 'a.png'}")
    runner.finished.emit(RunOutcome(0, False, False, LOG))
    assert not session.is_running()
    assert _free(paths)


def test_a_generation_is_refused_while_another_process_holds_the_lock(paths, tmp_path):
    runner = FakeRunner()
    session = GeneratorSession(runner, clock=lambda: NOW, lock=GpuLock())
    with held_by_another_process(paths.gpu_lock_file) as pid:
        refusal = session.generate(paths, (_generation(tmp_path),), "0", False)
    assert str(pid) in refusal
    assert runner.started == []
    assert not session.is_running()


def test_the_shutdown_of_a_generation_gives_the_lock_back(paths, tmp_path):
    session = GeneratorSession(FakeRunner(), clock=lambda: NOW, lock=GpuLock())
    session.generate(paths, (_generation(tmp_path),), "0", False)
    session.shutdown()
    assert _free(paths)


def test_a_generation_can_join_a_training_of_the_same_process(paths, tmp_path):
    lock = GpuLock()
    training, training_runner = _training(paths, lock)
    generator_runner = FakeRunner()
    generator = GeneratorSession(generator_runner, clock=lambda: NOW, lock=lock)
    training.start(REQUEST)
    assert generator.generate(paths, (_generation(tmp_path),), "0", True) is None
    training_runner.finished.emit(RunOutcome(0, False, False, LOG))
    assert not _free(paths)  # the image is still being generated
    generator_runner.finished.emit(RunOutcome(1, False, False, LOG))
    assert _free(paths)


def test_an_install_holds_the_lock_until_it_ends(paths):
    session, installer = _install(GpuLock())
    assert session.start(paths) == ()
    assert not _free(paths)
    installer.finished.emit(True, "done")
    assert _free(paths)


def test_an_install_is_refused_while_another_process_holds_the_lock(paths):
    session, installer = _install(GpuLock())
    with held_by_another_process(paths.gpu_lock_file) as pid:
        problems = session.start(paths)
    assert str(pid) in problems[0]
    assert not installer.is_running()


def test_an_install_with_nothing_to_do_gives_the_lock_back(paths):
    lock = GpuLock()
    plan = lambda _paths: InstallPlan((), ("no python",))  # noqa: E731
    session = InstallSession(FakeInstaller(), plan=plan, clock=lambda: NOW, lock=lock)
    assert session.start(paths) == ("no python",)
    assert _free(paths)


def test_the_shutdown_of_an_install_gives_the_lock_back(paths):
    session, _installer = _install(GpuLock())
    session.start(paths)
    session.shutdown()
    assert _free(paths)
