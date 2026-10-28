import os
from pathlib import Path

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QFileDialog

from anima_trainer.training.progress import ProgressState
from anima_trainer.ui.widgets.image_grid import ImageGrid
from anima_trainer.ui.widgets.log_view import LogView
from anima_trainer.ui.widgets.path_picker import PathPicker
from anima_trainer.ui.widgets.progress_panel import ProgressPanel


def test_path_picker_round_trips_text(qtbot):
    picker = PathPicker("Images", "folder", placeholder="/path")
    qtbot.addWidget(picker)

    picker.set_path("/data/images")

    assert picker.path() == "/data/images"
    assert picker.edit.placeholderText() == "/path"


def test_path_picker_browse_folder_sets_path_and_emits(qtbot, monkeypatch):
    picker = PathPicker("Images", "folder")
    qtbot.addWidget(picker)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: "/chosen")

    with qtbot.waitSignal(picker.path_changed) as blocker:
        picker.browse_button.click()

    assert picker.path() == "/chosen"
    assert blocker.args == ["/chosen"]


def test_path_picker_cancelled_browse_keeps_path(qtbot, monkeypatch):
    picker = PathPicker("Config", "file", file_filter="TOML (*.toml)")
    qtbot.addWidget(picker)
    picker.set_path("/old.toml")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))

    picker.browse_button.click()

    assert picker.path() == "/old.toml"


def test_path_picker_file_mode_uses_file_dialog(qtbot, monkeypatch):
    picker = PathPicker("Config", "file")
    qtbot.addWidget(picker)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("/a.toml", "TOML"))

    picker.browse_button.click()

    assert picker.path() == "/a.toml"


def test_log_view_appends_and_clears(qtbot):
    view = LogView("Training Log")
    qtbot.addWidget(view)

    view.append_line("one")
    view.append_line("two")

    assert view.text() == "one\ntwo"
    view.clear()
    assert view.text() == ""


def test_log_view_collapses_tqdm_redraws(qtbot):
    view = LogView("Training Log")
    qtbot.addWidget(view)

    view.append_line("epoch 1/10")
    view.append_line("steps:   1%|  | 1/100 [00:01<01:40]")
    view.append_line("steps:   2%|  | 2/100 [00:02<01:38]")

    assert view.text() == "epoch 1/10\nsteps:   2%|  | 2/100 [00:02<01:38]"


def test_log_view_set_lines_replaces_content(qtbot):
    view = LogView("Status")
    qtbot.addWidget(view)
    view.append_line("old")

    view.set_lines(("a", "b"))

    assert view.text() == "a\nb"


def test_log_view_keeps_only_the_newest_lines(qtbot):
    view = LogView("Training Log", max_lines=3)
    qtbot.addWidget(view)

    for i in range(5):
        view.append_line(f"line {i}")

    assert view.text() == "line 2\nline 3\nline 4"


def test_log_view_copy_puts_text_on_clipboard(qtbot):
    view = LogView("Status")
    qtbot.addWidget(view)
    view.set_lines(("copied",))

    view.copy_button.click()

    assert QApplication.clipboard().text() == "copied"


def test_progress_panel_shows_training_state(qtbot):
    panel = ProgressPanel()
    qtbot.addWidget(panel)
    state = ProgressState(
        epoch=3,
        total_epochs=10,
        step=50,
        total_steps=200,
        loss=0.12345,
        elapsed_s=125,
        remaining_s=3725,
        last_checkpoint=Path("/out/proj-000002.safetensors"),
    )

    panel.show_progress(state)

    assert panel.epoch_label.text() == "Epoch 3 / 10"
    assert (panel.bar.value(), panel.bar.maximum()) == (50, 200)
    assert panel.loss_label.text() == "Loss: 0.1235"
    assert panel.time_label.text() == "Elapsed 02:05  |  Remaining 1:02:05"
    assert panel.checkpoint_label.text() == "Last checkpoint: proj-000002.safetensors"


def test_progress_panel_empty_state(qtbot):
    panel = ProgressPanel()
    qtbot.addWidget(panel)

    panel.show_progress(ProgressState())

    assert panel.epoch_label.text() == "Epoch - / -"
    assert panel.loss_label.text() == "Loss: -"
    assert panel.checkpoint_label.text() == "Last checkpoint: -"


def test_progress_panel_shows_download_in_permille(qtbot):
    panel = ProgressPanel()
    qtbot.addWidget(panel)

    panel.show_download(3 * 1024**3, 4 * 1024**3)

    assert (panel.bar.value(), panel.bar.maximum()) == (750, 1000)
    assert panel.epoch_label.text() == "Downloading model: 3.00 GB / 4.00 GB"


def test_progress_panel_download_with_unknown_size(qtbot):
    panel = ProgressPanel()
    qtbot.addWidget(panel)

    panel.show_download(512 * 1024**2, None)

    assert panel.bar.maximum() == 0  # busy indicator
    assert panel.epoch_label.text() == "Downloading model: 0.50 GB"


def test_image_grid_loads_again_a_thumbnail_that_was_half_written(qtbot, tmp_path):
    # A failed load must not be cached: the same path and mtime may hold the full file later.
    grid = ImageGrid()
    qtbot.addWidget(grid)
    path = tmp_path / "sample.png"
    path.write_bytes(b"\x89PNG half")
    stamp = path.stat().st_mtime_ns
    grid.set_grid(("Epoch 1",), ("Prompt 1",), {(0, 0): path})
    assert grid.item(0, 0).icon().isNull()

    image = QImage(8, 8, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    assert image.save(str(path))
    os.utime(path, ns=(stamp, stamp))
    grid.set_grid(("Epoch 1",), ("Prompt 1",), {(0, 0): path})

    assert not grid.item(0, 0).icon().isNull()
