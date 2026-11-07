"""The Generator's list of LoRAs: file, strength and on/off, at most MAX_LORAS."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog

from anima_trainer.preview.generator_settings import MAX_LORAS, LoraEntry
from anima_trainer.ui.widgets.lora_table import TRIGGER_COLUMN, LoraTable


def _table(qtbot):
    table = LoraTable()
    qtbot.addWidget(table)
    return table


def test_entries_round_trip_with_their_strength_and_switch(qtbot):
    table = _table(qtbot)
    entries = (
        LoraEntry(path="/l/a.safetensors", strength=-0.5),
        LoraEntry(path="/l/b.safetensors", strength=1.25, enabled=False),
    )

    table.set_entries(entries)

    assert table.entries() == entries
    assert table.rowCount() == 2


def test_the_file_name_is_shown_and_the_full_path_is_in_the_tooltip(qtbot):
    table = _table(qtbot)

    table.set_entries((LoraEntry(path="/l/style.safetensors"),))

    item = table.item(0, 1)
    assert (item.text(), item.toolTip()) == ("style.safetensors", "/l/style.safetensors")


def test_add_takes_the_chosen_files_with_strength_one(qtbot, monkeypatch):
    table = _table(qtbot)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames", lambda *_a: (["/l/a.safetensors", "/l/b.safetensors"], "")
    )

    table.add_button.click()

    assert table.entries() == (
        LoraEntry(path="/l/a.safetensors"),
        LoraEntry(path="/l/b.safetensors"),
    )


def test_a_file_already_in_the_list_is_not_added_twice(qtbot):
    table = _table(qtbot)
    table.set_entries((LoraEntry(path="/l/a.safetensors", strength=0.3),))

    table.add_files(["/l/a.safetensors"])

    assert table.entries() == (LoraEntry(path="/l/a.safetensors", strength=0.3),)


def test_no_more_than_the_maximum_is_added_and_add_is_then_disabled(qtbot):
    table = _table(qtbot)

    skipped = table.add_files([f"/l/{n}.safetensors" for n in range(MAX_LORAS + 2)])

    assert len(table.entries()) == MAX_LORAS
    assert skipped == 2
    assert not table.add_button.isEnabled()


def test_remove_takes_the_selected_rows_out(qtbot):
    table = _table(qtbot)
    table.add_files(["/l/a.safetensors", "/l/b.safetensors", "/l/c.safetensors"])
    table.selectRow(1)

    table.remove_button.click()

    assert [entry.path for entry in table.entries()] == ["/l/a.safetensors", "/l/c.safetensors"]
    assert table.add_button.isEnabled()


def test_the_strength_goes_from_minus_two_to_two(qtbot):
    table = _table(qtbot)
    table.add_files(["/l/a.safetensors"])
    spin = table.strength_box(0)

    spin.setValue(5.0)

    assert spin.value() == 2.0
    assert table.entries()[0].strength == 2.0


def test_while_generating_the_list_cannot_be_changed(qtbot):
    table = _table(qtbot)
    table.add_files(["/l/a.safetensors"])

    table.set_editable(False)

    assert not table.add_button.isEnabled()
    assert not table.remove_button.isEnabled()
    assert not table.strength_box(0).isEnabled()
    table.set_editable(True)
    assert table.add_button.isEnabled()


def test_the_trigger_is_shown_and_kept_with_its_lora(qtbot):
    table = _table(qtbot)
    entries = (LoraEntry(path="/l/a.safetensors", trigger="@style"),)

    table.set_entries(entries)

    assert table.item(0, TRIGGER_COLUMN).text() == "@style"
    assert table.entries() == entries


def test_a_typed_trigger_is_kept(qtbot):
    table = _table(qtbot)
    table.add_files(["/l/a.safetensors"])

    table.item(0, TRIGGER_COLUMN).setText(" watercolor ")

    assert table.entries()[0].trigger == "watercolor"


def test_an_added_file_gets_the_trigger_found_in_it(qtbot):
    table = LoraTable(trigger_for=lambda path: "@" + path.rsplit("/", 1)[-1].split(".")[0])
    qtbot.addWidget(table)

    table.add_files(["/l/style.safetensors"])

    assert table.entries()[0].trigger == "@style"


def test_only_the_trigger_can_be_typed_in(qtbot):
    table = _table(qtbot)
    table.add_files(["/l/a.safetensors"])

    editable = Qt.ItemFlag.ItemIsEditable
    assert table.item(0, TRIGGER_COLUMN).flags() & editable
    assert not table.item(0, 1).flags() & editable
