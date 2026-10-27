import hashlib

import pytest

from anima_trainer.models.catalog import ExpectedFile
from anima_trainer.models.check_worker import ModelFileChecker
from anima_trainer.models.linked_models import LinkedModel

pytestmark = pytest.mark.integration

TIMEOUT_MS = 10000
CONTENT = bytes(range(256)) * 400


@pytest.fixture
def checker():
    instance = ModelFileChecker(chunk_size=1000)
    yield instance
    instance.shutdown()
    assert not instance.is_running()


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(CONTENT)
    return path


def _expected(content=CONTENT):
    return ExpectedFile(hashlib.sha256(content).hexdigest(), len(content))


def test_matching_file_finishes_with_its_link(qtbot, checker, model_file):
    progress = []
    checker.progress.connect(lambda done, total: progress.append((done, total)))

    with qtbot.waitSignal(checker.finished, timeout=TIMEOUT_MS) as blocker:
        checker.start(model_file, _expected())

    link = blocker.args[0]
    assert isinstance(link, LinkedModel) and link.path == model_file
    qtbot.waitUntil(lambda: bool(progress), timeout=TIMEOUT_MS)
    assert progress[-1][1] == len(CONTENT)


def test_wrong_file_fails_with_the_reason(qtbot, checker, model_file):
    with qtbot.waitSignal(checker.failed, timeout=TIMEOUT_MS) as blocker:
        checker.start(model_file, _expected(bytes(reversed(CONTENT))))

    assert "SHA-256" in blocker.args[0]


def test_check_can_be_cancelled(qtbot, checker, model_file, paused_file_check):
    def cancel(*_):
        checker.cancel()
        paused_file_check.set()

    checker.progress.connect(cancel)

    with qtbot.waitSignal(checker.cancelled, timeout=TIMEOUT_MS):
        checker.start(model_file, _expected())
