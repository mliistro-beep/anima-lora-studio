"""The TUI generator's LoRAs, on a screen of their own: one two-line card per LoRA, with the
count of those on. Done gives the edited list back to the form; Cancel and Escape leave it."""

from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Header, Label, Static

from anima_trainer.cli.generator_values import add_loras, lora_count_text, loras_from_rows
from anima_trainer.cli.tui.folder_picker import FilePicker, start_folder
from anima_trainer.cli.tui.generator_rows import LoraRow
from anima_trainer.preview.generator_settings import MAX_LORAS, LoraEntry
from anima_trainer.preview.lora_files import lora_trigger

LORA_CSS = """
.lora-row { height: auto; border-bottom: dashed $foreground 25%; padding-bottom: 1; }
.lora-head, .lora-body, .buttons { height: auto; }
.lora-name { width: 1fr; margin-left: 1; text-style: bold; }
.lora-body { margin-left: 4; }
.lora-label { width: auto; margin-right: 1; color: $text-muted; }
.lora-trigger { width: 1fr; margin-right: 2; }
.lora-strength { width: 8; }
.lora-remove { width: auto; margin-left: 2; }
#loras { height: auto; }
#lora-errors { height: auto; color: $error; }
.buttons Button { margin-right: 2; }
"""


class LoraScreen(Screen[tuple[LoraEntry, ...] | None]):
    DEFAULT_CSS = LORA_CSS
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, loras: tuple[LoraEntry, ...], start: Path) -> None:
        super().__init__()
        self._loras = loras
        self._start = start  # where Add LoRA browses when the list is empty

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Label(f"LoRAs (at most {MAX_LORAS}), with their trigger and strength")
            yield Static(id="lora-screen-count", markup=False)
            yield Vertical(*(LoraRow(entry) for entry in self._loras), id="loras")
            yield Button("Add LoRA", id="add-lora", compact=True)
            yield Static(id="lora-errors", markup=False)
            with Horizontal(classes="buttons"):
                yield Button("Done", variant="primary", id="loras-done")
                yield Button("Cancel", id="loras-cancel")
        yield Footer()

    def on_mount(self) -> None:
        self._update_count()

    def on_checkbox_changed(self, _event: Checkbox.Changed) -> None:
        self._update_count()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button
        if button.has_class("lora-remove"):
            await button.query_ancestor(LoraRow).remove()
            self._update_count()
        elif button.id == "add-lora":
            self.app.push_screen(FilePicker(self._browse_from()), self._add)
        elif button.id == "loras-done":
            self._done()
        elif button.id == "loras-cancel":
            self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _rows(self) -> list[LoraRow]:
        return list(self.query(LoraRow))

    def _update_count(self) -> None:
        enabled = tuple(_is_on(row) for row in self._rows())
        self.query_one("#lora-screen-count", Static).update(lora_count_text(enabled))

    def _done(self) -> None:
        loras, errors = loras_from_rows([row.values() for row in self._rows()])
        if errors:
            self.query_one("#lora-errors", Static).update("\n".join(errors))
            return
        self.dismiss(loras)

    def _add(self, file: str | None) -> None:
        if file is None:
            return
        # Doubles are found by path, which a card cannot change: the values first shown are
        # enough, and only the new cards are mounted, so typed triggers and strengths stay.
        current = tuple(row.entry for row in self._rows())
        entries, notes = add_loras(current, (file,), lambda path: lora_trigger(Path(path)))
        self.query_one("#loras", Vertical).mount_all(
            LoraRow(entry) for entry in entries[len(current) :]
        )
        self.query_one("#lora-errors", Static).update("\n".join(notes))
        self.call_after_refresh(self._update_count)

    def _browse_from(self) -> Path:
        rows = self._rows()
        if rows:
            return start_folder(rows[-1].entry.path, Path.home())
        return self._start


def _is_on(row: LoraRow) -> bool:
    """The row's switch; a card just added may not have built it yet: its entry says."""
    boxes = row.query(".lora-on").results(Checkbox)
    box = next(boxes, None)
    return box.value if box is not None else row.entry.enabled
