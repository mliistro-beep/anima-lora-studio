from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from anima_trainer.models.catalog import VAE_FILENAME
from anima_trainer.models.linked_models import LinkedModel
from anima_trainer.models.linked_store import (
    LINKS_GROUP,
    load_links,
    remove_link,
    save_link,
)

LINK = LinkedModel(Path("/models/qwen_image_vae.safetensors"), 253806246, 123456789)


@pytest.fixture
def store(tmp_path):
    return QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat)


def test_saved_link_survives_a_restart(store):
    save_link(store, VAE_FILENAME, LINK)

    reopened = QSettings(store.fileName(), QSettings.Format.IniFormat)
    assert dict(load_links(reopened)) == {VAE_FILENAME: LINK}


def test_removed_link_is_gone(store):
    save_link(store, VAE_FILENAME, LINK)
    save_link(store, "anima-preview", LINK)

    remove_link(store, VAE_FILENAME)

    assert set(load_links(store)) == {"anima-preview"}


def test_no_links_by_default(store):
    assert dict(load_links(store)) == {}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("anima-preview", "not json"),
        ("anima-preview", '{"path": "/m", "size": "big", "mtime_ns": 1}'),
        ("anima-preview", '{"path": "", "size": 1, "mtime_ns": 1}'),
        ("anima-preview", "[1, 2]"),
        ("unknown-model", '{"path": "/m", "size": 1, "mtime_ns": 1}'),
    ],
)
def test_damaged_or_unknown_entries_are_ignored(store, key, value):
    store.setValue(f"{LINKS_GROUP}/{key}", value)

    assert dict(load_links(store)) == {}


def test_save_failure_is_raised(tmp_path):
    folder = tmp_path / "readonly"
    folder.mkdir()
    store = QSettings(str(folder / "app.ini"), QSettings.Format.IniFormat)
    folder.chmod(0o500)
    try:
        with pytest.raises(OSError, match="could not be saved"):
            save_link(store, VAE_FILENAME, LINK)
    finally:
        folder.chmod(0o700)
