import pytest

from anima_trainer.main import APP_NAME, create_window


@pytest.mark.e2e
def test_main_window_opens_with_app_title(qtbot):
    window = create_window()
    qtbot.addWidget(window)

    window.show()

    assert window.isVisible()
    assert window.windowTitle() == APP_NAME
