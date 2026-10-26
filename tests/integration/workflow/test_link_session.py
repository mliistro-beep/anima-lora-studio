import hashlib

import pytest
from PySide6.QtCore import QSettings

from anima_trainer.models.catalog import VAE_FILENAME, ExpectedFile
from anima_trainer.models.check_worker import ModelFileChecker
from anima_trainer.models.linked_models import link_for
from anima_trainer.models.linked_store import load_links, save_link
from anima_trainer.workflow.link_session import LinkSession

pytestmark = pytest.mark.integration

TIMEOUT_MS = 10000
CONTENT = b"vae weights" * 100


@pytest.fixture
def store(tmp_path):
    return QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def session(store):
    expected = {VAE_FILENAME: ExpectedFile(hashlib.sha256(CONTENT).hexdigest(), len(CONTENT))}
    instance = LinkSession(store, ModelFileChecker(chunk_size=100), expected)
    yield instance
    instance.shutdown()


@pytest.fixture
def vae_file(tmp_path):
    path = tmp_path / "mine" / "vae.safetensors"
    path.parent.mkdir()
    path.write_bytes(CONTENT)
    return path


def test_matching_file_is_saved_as_a_link(qtbot, session, store, vae_file):
    with qtbot.waitSignal(session.finished, timeout=TIMEOUT_MS) as blocker:
        assert session.check(VAE_FILENAME, vae_file) is None
        assert session.is_running()

    ok, message = blocker.args
    assert ok
    assert message == f"OK: VAE file accepted (SHA-256 matches): {vae_file}"
    assert load_links(store)[VAE_FILENAME].path == vae_file
    assert not session.is_running()


def test_wrong_file_is_rejected_and_not_saved(qtbot, session, store, vae_file):
    vae_file.write_bytes(bytes(reversed(CONTENT)))

    with qtbot.waitSignal(session.finished, timeout=TIMEOUT_MS) as blocker:
        session.check(VAE_FILENAME, vae_file)

    ok, message = blocker.args
    assert not ok
    assert message.startswith(f"ERROR: {vae_file} is not a compatible VAE file: SHA-256")
    assert dict(load_links(store)) == {}


def test_cancelled_check_saves_nothing(qtbot, session, store, vae_file):
    session.progress.connect(lambda *_: session.cancel())

    with qtbot.waitSignal(session.finished, timeout=TIMEOUT_MS) as blocker:
        session.check(VAE_FILENAME, vae_file)

    assert blocker.args == [False, "INFO: File check cancelled."]
    assert dict(load_links(store)) == {}


def test_unknown_model_and_second_check_are_refused(session, vae_file):
    assert session.check("other", vae_file) == "Unknown model: other"
    session.check(VAE_FILENAME, vae_file)

    assert session.check(VAE_FILENAME, vae_file) == "A file check is already running"


def test_unlink_forgets_the_file(session, store, vae_file):
    save_link(store, VAE_FILENAME, link_for(vae_file))

    assert session.unlink(VAE_FILENAME) is None

    assert dict(session.links()) == {}


def test_save_failure_is_reported(qtbot, session, store, vae_file, monkeypatch):
    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr("anima_trainer.workflow.link_session.save_link", fail)
    monkeypatch.setattr("anima_trainer.workflow.link_session.remove_link", fail)

    with qtbot.waitSignal(session.finished, timeout=TIMEOUT_MS) as blocker:
        session.check(VAE_FILENAME, vae_file)

    assert blocker.args == [False, "ERROR: disk full"]
    assert session.unlink(VAE_FILENAME) == "disk full"
