import signal

import pytest
from PySide6.QtCore import QSettings

from anima_trainer.main import APP_NAME, create_window, install_quit_signals
from anima_trainer.settings.citron_root import ROOT_KEY


@pytest.mark.e2e
def test_main_window_opens_with_app_title(qtbot, tmp_path):
    store = QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat)
    store.setValue(ROOT_KEY, str(tmp_path / "citron"))
    window = create_window(store, detect=lambda _paths: ("0",))
    qtbot.addWidget(window)

    window.show()

    assert window.isVisible()
    assert window.windowTitle() == APP_NAME
    assert window.session.paths.root == tmp_path / "citron"
    assert window.close()


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="needs POSIX signals")
def test_terminal_signals_quit_the_app_through_the_given_callback():
    calls = []
    before = signal.getsignal(signal.SIGTERM)
    restore = install_quit_signals(lambda: calls.append("quit"))
    try:
        signal.raise_signal(signal.SIGTERM)
        signal.raise_signal(signal.SIGINT)
    finally:
        restore()

    assert calls == ["quit", "quit"]
    assert signal.getsignal(signal.SIGTERM) == before
