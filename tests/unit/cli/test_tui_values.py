"""What the TUI remembers for the next time, in a file of its own."""

from anima_trainer.cli.tui_values import TuiValues, load_tui_values, save_tui_values


def test_nothing_saved_gives_empty_values(tmp_path):
    loaded = load_tui_values(tmp_path / "tui.json")
    assert loaded.values == TuiValues()
    assert loaded.warnings == ()


def test_saved_values_come_back(tmp_path):
    file = tmp_path / "tui.json"
    values = TuiValues(project="p", dataset="/d", output="/o", preset="Person")
    save_tui_values(file, values)
    assert load_tui_values(file).values == values


def test_a_broken_file_gives_empty_values_and_a_warning(tmp_path):
    file = tmp_path / "tui.json"
    file.write_text("{broken")
    loaded = load_tui_values(file)
    assert loaded.values == TuiValues()
    assert len(loaded.warnings) == 1
