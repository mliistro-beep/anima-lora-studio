"""The rows of the TUI generator's lists: a LoRA, and an entry of the queue."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Checkbox, Input, Static

from anima_trainer.preview.generator_settings import GenerationEntry, LoraEntry
from anima_trainer.preview.queue_text import entry_summary


class LoraRow(Horizontal):
    """One LoRA: on/off, file name (the full path in its tooltip), trigger, strength."""

    def __init__(self, entry: LoraEntry) -> None:
        super().__init__(classes="lora-row")
        self.entry = entry

    def compose(self) -> ComposeResult:
        on = Checkbox(value=self.entry.enabled, classes="lora-on", compact=True)
        on.tooltip = "Used, and its trigger added to the prompt, while on."
        yield on
        name = Static(Path(self.entry.path).name, classes="lora-name", markup=False)
        name.tooltip = self.entry.path
        yield name
        yield Input(self.entry.trigger, placeholder="trigger", classes="lora-trigger", compact=True)
        yield Input(str(self.entry.strength), classes="lora-strength", compact=True)
        yield Button("Remove", classes="lora-remove", compact=True)

    def values(self) -> dict[str, object]:
        return {
            "path": self.entry.path,
            "enabled": self.query_one(".lora-on", Checkbox).value,
            "trigger": self.query_one(".lora-trigger", Input).value,
            "strength": self.query_one(".lora-strength", Input).value,
        }

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("lora-remove"):
            event.stop()
            self.remove()


class QueueRow(Horizontal):
    """One entry of the queue, numbered from 1; its Remove button carries its index in `name`,
    for the screen, which renumbers the others."""

    def __init__(self, index: int, entry: GenerationEntry) -> None:
        super().__init__(classes="queue-row")
        self._index = index
        self._entry = entry

    def compose(self) -> ComposeResult:
        yield Static(
            entry_summary(self._index + 1, self._entry), classes="queue-text", markup=False
        )
        yield Button("Remove", name=str(self._index), classes="queue-remove", compact=True)
