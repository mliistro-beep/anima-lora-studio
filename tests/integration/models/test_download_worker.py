import os

import pytest

from anima_trainer.models.download_worker import ModelDownloader

pytestmark = pytest.mark.integration


def test_worker_emits_progress_and_finished(qtbot, file_server, tmp_path):
    state, url = file_server
    state.payload = os.urandom(5000)
    target = tmp_path / "model.safetensors"
    downloader = ModelDownloader()
    progress = []
    downloader.progress.connect(lambda done, total: progress.append((done, total)))

    with qtbot.waitSignal(downloader.finished, timeout=5000) as blocker:
        downloader.start(url, target)

    assert blocker.args == [str(target)]
    assert target.read_bytes() == state.payload
    assert progress[-1] == (5000, 5000)
    qtbot.waitUntil(lambda: not downloader.is_running())


def test_worker_reports_failure(qtbot, file_server, tmp_path):
    state, url = file_server
    state.status_override = 500
    downloader = ModelDownloader()

    with qtbot.waitSignal(downloader.failed, timeout=5000) as blocker:
        downloader.start(url, tmp_path / "model.safetensors")

    assert "500" in blocker.args[0]


def test_worker_can_be_cancelled(qtbot, file_server, tmp_path):
    state, url = file_server
    state.payload = os.urandom(50_000)
    downloader = ModelDownloader(chunk_size=1000)
    downloader.progress.connect(lambda *_: downloader.cancel())

    with qtbot.waitSignal(downloader.cancelled, timeout=5000):
        downloader.start(url, tmp_path / "model.safetensors")

    assert not (tmp_path / "model.safetensors").exists()


def test_start_while_running_is_rejected(qtbot, file_server, tmp_path):
    state, url = file_server
    state.payload = os.urandom(50_000)
    downloader = ModelDownloader(chunk_size=100)

    with qtbot.waitSignal(downloader.finished, timeout=10000):
        downloader.start(url, tmp_path / "a.safetensors")
        with pytest.raises(RuntimeError):
            downloader.start(url, tmp_path / "b.safetensors")
