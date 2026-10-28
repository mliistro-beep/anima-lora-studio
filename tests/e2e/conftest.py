import pytest
from PySide6.QtWidgets import QMessageBox


@pytest.fixture(autouse=True)
def confirm_quit_at_teardown(monkeypatch):
    """A real question dialog would block forever when a test ends with work running."""
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes)


@pytest.fixture
def dataset(tmp_path):
    """One captioned image, enough for Configure."""
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"")
    (folder / "a.txt").write_text("caption")
    return folder
