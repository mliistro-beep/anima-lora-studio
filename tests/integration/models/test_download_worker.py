import os

import pytest

from anima_trainer.models import download_worker
from anima_trainer.models.download_worker import ModelDownloader

pytestmark = pytest.mark.integration

TIMEOUT_MS = 10000


@pytest.fixture
def downloader():
    instance = ModelDownloader(chunk_size=1000)
    yield instance
    instance.shutdown()
    assert not instance.is_running()


def test_worker_emits_progress_and_finished(qtbot, downloader, file_server, tmp_path):
    state, url = file_server
    state.payload = os.urandom(5000)
    target = tmp_path / "model.safetensors"
    progress = []
    downloader.progress.connect(lambda done, total: progress.append((done, total)))

    with qtbot.waitSignal(downloader.finished, timeout=TIMEOUT_MS) as blocker:
        downloader.start(url, target)

    assert blocker.args == [str(target)]
    assert target.read_bytes() == state.payload
    assert progress[-1] == (5000, 5000)


def test_worker_is_idle_when_terminal_signal_arrives(qtbot, downloader, file_server, tmp_path):
    state, url = file_server
    state.status_override = 500
    running_at_signal = []
    downloader.failed.connect(lambda _: running_at_signal.append(downloader.is_running()))

    with qtbot.waitSignal(downloader.failed, timeout=TIMEOUT_MS) as blocker:
        downloader.start(url, tmp_path / "model.safetensors")

    assert "500" in blocker.args[0]
    assert running_at_signal == [False]


def test_worker_can_be_cancelled(qtbot, downloader, file_server, tmp_path):
    state, url = file_server
    state.payload = os.urandom(200_000)
    state.block_delay_s = 0.01
    downloader.progress.connect(lambda *_: downloader.cancel())

    with qtbot.waitSignal(downloader.cancelled, timeout=TIMEOUT_MS):
        downloader.start(url, tmp_path / "model.safetensors")

    assert not (tmp_path / "model.safetensors").exists()


def test_unexpected_exception_is_reported_as_failure(qtbot, downloader, monkeypatch, tmp_path):
    def explode(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(download_worker, "download_file", explode)

    with qtbot.waitSignal(downloader.failed, timeout=TIMEOUT_MS) as blocker:
        downloader.start("http://127.0.0.1/x", tmp_path / "m.safetensors")

    assert "unexpected" in blocker.args[0]


def test_start_while_running_is_rejected(qtbot, downloader, file_server, tmp_path):
    state, url = file_server
    state.payload = os.urandom(100_000)
    state.block_delay_s = 0.01

    downloader.start(url, tmp_path / "a.safetensors")

    with pytest.raises(RuntimeError):
        downloader.start(url, tmp_path / "b.safetensors")


def test_shutdown_stops_a_running_download_and_waits(qtbot, file_server, tmp_path):
    state, url = file_server
    state.payload = os.urandom(500_000)
    state.block_delay_s = 0.01
    instance = ModelDownloader(chunk_size=1000)
    instance.start(url, tmp_path / "model.safetensors")

    instance.shutdown()

    assert not instance.is_running()
    assert not (tmp_path / "model.safetensors").exists()


def test_shutdown_when_idle_does_nothing():
    ModelDownloader().shutdown()
