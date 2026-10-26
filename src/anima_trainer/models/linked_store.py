"""Saves the model files the user linked, in QSettings: they do not depend on the data folder."""

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from PySide6.QtCore import QSettings

from anima_trainer.models.catalog import EXPECTED_FILES
from anima_trainer.models.linked_models import LinkedModel

LINKS_GROUP = "linked_models"


def load_links(store: QSettings) -> Mapping[str, LinkedModel]:
    """Links by model key; damaged entries and unknown models are skipped."""
    store.beginGroup(LINKS_GROUP)
    try:
        entries = {key: store.value(key) for key in store.childKeys()}
    finally:
        store.endGroup()
    links = {key: _parse(value) for key, value in entries.items() if key in EXPECTED_FILES}
    return MappingProxyType({key: link for key, link in links.items() if link is not None})


def save_link(store: QSettings, key: str, link: LinkedModel) -> None:
    """Raises OSError when the settings file cannot be written."""
    text = json.dumps({"path": str(link.path), "size": link.size, "mtime_ns": link.mtime_ns})
    store.setValue(f"{LINKS_GROUP}/{key}", text)
    _sync(store)


def remove_link(store: QSettings, key: str) -> None:
    """Raises OSError when the settings file cannot be written."""
    store.remove(f"{LINKS_GROUP}/{key}")
    _sync(store)


def _sync(store: QSettings) -> None:
    store.sync()
    if store.status() != QSettings.Status.NoError:
        raise OSError(f"The model file choice could not be saved to {store.fileName()}")


def _parse(value: object) -> LinkedModel | None:
    try:
        data = json.loads(str(value))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    path, size, mtime_ns = data.get("path"), data.get("size"), data.get("mtime_ns")
    if not (isinstance(path, str) and path and isinstance(size, int) and isinstance(mtime_ns, int)):
        return None
    return LinkedModel(Path(path), size, mtime_ns)
