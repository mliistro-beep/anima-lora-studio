import hashlib
import os

import pytest

from anima_trainer.models.catalog import EXPECTED_FILES, QWEN3_FILENAME, ExpectedFile
from anima_trainer.models.file_check import (
    CheckCancelled,
    ModelFileRejected,
    check_model_file,
)
from anima_trainer.models.linked_models import is_unchanged

CONTENT = b"anima weights" * 10


def _expected(content=CONTENT):
    return ExpectedFile(hashlib.sha256(content).hexdigest(), len(content))


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(CONTENT)
    return path


def test_matching_file_is_accepted_as_an_unchanged_link(model_file):
    link = check_model_file(model_file, _expected(), chunk_size=16)

    assert link.path == model_file
    assert link.size == len(CONTENT)
    assert is_unchanged(link)


def test_progress_reports_every_chunk_up_to_the_size(model_file):
    seen = []

    check_model_file(model_file, _expected(), on_progress=lambda d, t: seen.append((d, t)))

    assert seen[-1] == (len(CONTENT), len(CONTENT))


def test_wrong_size_is_rejected_without_reading_the_file(model_file):
    def fail(*_args):
        raise AssertionError("hashed a file of the wrong size")

    with pytest.raises(ModelFileRejected, match="size"):
        check_model_file(model_file, _expected(CONTENT + b"x"), on_progress=fail)


def test_same_size_but_different_content_is_rejected(model_file):
    other = bytes(reversed(CONTENT))

    with pytest.raises(ModelFileRejected, match="SHA-256"):
        check_model_file(model_file, _expected(other))


def test_missing_file_and_folder_are_rejected(tmp_path):
    with pytest.raises(ModelFileRejected, match="not a file"):
        check_model_file(tmp_path / "nope", _expected())
    with pytest.raises(ModelFileRejected, match="not a file"):
        check_model_file(tmp_path, _expected())


def test_cancel_stops_the_check(model_file):
    with pytest.raises(CheckCancelled):
        check_model_file(model_file, _expected(), is_cancelled=lambda: True, chunk_size=16)


def test_file_changed_while_checking_is_rejected(model_file):
    def rewrite(_done, _total):
        model_file.write_bytes(CONTENT)  # same bytes, new modification time

    os.utime(model_file, ns=(0, 0))
    with pytest.raises(ModelFileRejected, match="changed"):
        check_model_file(model_file, _expected(), on_progress=rewrite)


def test_unreadable_file_is_rejected(model_file, monkeypatch):
    def broken_open(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", broken_open)

    with pytest.raises(ModelFileRejected, match="denied"):
        check_model_file(model_file, _expected())


def test_catalog_knows_every_downloadable_model():
    from anima_trainer.models.catalog import BASE_MODEL_URLS, SUPPORT_MODEL_URLS

    assert set(EXPECTED_FILES) == {*BASE_MODEL_URLS, *SUPPORT_MODEL_URLS}
    assert all(len(f.sha256) == 64 and f.size > 0 for f in EXPECTED_FILES.values())
    assert EXPECTED_FILES[QWEN3_FILENAME].size == 1192135096
