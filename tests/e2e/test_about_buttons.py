import pytest
from PySide6.QtCore import QSettings

from anima_trainer.main import create_window
from anima_trainer.runtime.data_dir import DATA_DIR_KEY

pytestmark = pytest.mark.e2e


@pytest.fixture
def window(qtbot, tmp_path):
    store = QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat)
    store.setValue(DATA_DIR_KEY, str(tmp_path / "data"))
    window = create_window(store, detect=lambda _paths: ("0",))
    qtbot.addWidget(window)
    window.show()
    return window


def test_license_button_shows_the_gpl_notice(window):
    window.license_button.click()

    dialog = window.about_dialog
    assert dialog is not None and dialog.isVisible()
    assert dialog.windowTitle() == "License"
    text = dialog.browser.toPlainText()
    assert "GPL-3.0-or-later" in text
    assert "Mauro Liistro" in text


def test_credits_button_credits_author_citron_and_models(window):
    window.credits_button.click()

    dialog = window.about_dialog
    assert dialog is not None and dialog.isVisible()
    assert dialog.windowTitle() == "Credits"
    text = dialog.browser.toPlainText()
    for name in ("Mauro Liistro", "Citron Legacy", "CircleStone Labs", "Qwen", "kohya-ss"):
        assert name in text


def test_buttons_stay_visible_on_every_tab(window):
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index)

        assert window.license_button.isVisible()
        assert window.credits_button.isVisible()


def test_opening_again_replaces_the_previous_dialog(window, qtbot):
    window.license_button.click()
    first = window.about_dialog
    first.close()

    window.credits_button.click()

    assert window.about_dialog is not first
    assert window.about_dialog.windowTitle() == "Credits"
