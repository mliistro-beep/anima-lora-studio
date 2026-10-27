from anima_trainer.ui.dialogs.text_dialog import TextDialog


def test_text_dialog_shows_the_html_and_opens_links_externally(qtbot):
    dialog = TextDialog("Credits", '<p><a href="https://example.org">Example</a></p>')
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "Credits"
    assert "Example" in dialog.browser.toPlainText()
    assert dialog.browser.openExternalLinks()


def test_text_dialog_close_button_closes_it(qtbot):
    dialog = TextDialog("License", "<p>text</p>")
    qtbot.addWidget(dialog)
    dialog.show()

    dialog.close_button.click()

    assert not dialog.isVisible()
