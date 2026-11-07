"""The TUI's generator: its form and the screen of a generation.

The form shows the window's generator settings (<data>/generator.json): the base model or a
model file, the LoRAs with on/off, trigger and strength, the pre-prompt, prompt and negative
prompt, the image settings and the folder. No project checkpoints here: a trained LoRA is added
like any other.
"""

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
    TextArea,
)

from anima_trainer.cli.generator_values import add_loras, image_defaults, settings_from_fields
from anima_trainer.cli.train_preparation import TerminalContext
from anima_trainer.cli.tui.folder_picker import FilePicker, FolderPicker, start_folder
from anima_trainer.preview.form_fields import GENERATOR_FIELDS
from anima_trainer.preview.generator_settings import (
    MAX_LORAS,
    GeneratorSettings,
    LoraEntry,
    load_generator_settings,
)
from anima_trainer.preview.lora_files import lora_trigger
from anima_trainer.preview.settings import cfg_after_sampler_change
from anima_trainer.settings.choices import BASE_MODELS

if TYPE_CHECKING:
    from anima_trainer.cli.tui.app import TrainerApp

MAX_LOG_LINES = 5000
FIELDS_PER_ROW = 4
# The project checkpoint's strength has no use here; the negative prompt gets a full row.
GRID_FIELDS = tuple(
    spec for spec in GENERATOR_FIELDS if spec.name not in {"lora_multiplier", "negative_prompt"}
)
PATH_FIELDS = ("model_file", "output_folder")
GENERATOR_CSS = """
.field-label { color: $text-muted; }
.path-row, .lora-row, .field-row, .buttons { height: auto; }
.path-row Input { width: 1fr; }
.path-row Button, .lora-row Button { width: auto; margin-left: 2; }
.lora-name { width: 1fr; }
.lora-trigger { width: 1fr; }
.lora-strength { width: 8; }
.field-cell { width: 1fr; height: auto; margin-right: 1; }
#prompt { height: 5; }
#loras { height: auto; }
#errors { height: auto; color: $error; }
.buttons { margin-top: 1; }
.buttons Button { margin-right: 2; }
#log { height: 1fr; border: solid $primary; }
#bar { margin: 1 0; }
"""
NOT_STARTED_STATUS = "Generation not started."


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


class GeneratorScreen(Screen[None]):
    DEFAULT_CSS = GENERATOR_CSS

    def __init__(self, context: TerminalContext) -> None:
        super().__init__()
        loaded = load_generator_settings(context.paths.generator_settings_file)
        self._settings = loaded.settings
        self._warnings = loaded.warnings
        self._sampler: str = loaded.settings.sampler

    def compose(self) -> ComposeResult:
        settings = self._settings
        yield Header()
        with VerticalScroll():
            yield Label("Base model (used when no model file is chosen)", classes="field-label")
            yield Select(
                [(Text(name), name) for name in BASE_MODELS],
                value=settings.base_model,
                allow_blank=False,
                id="base_model",
                compact=True,
            )
            yield from _path_field(
                "Model file (blank: the base model)", settings.model_file, "model_file"
            )
            yield Label(
                f"LoRAs (at most {MAX_LORAS}): on, file, trigger, strength", classes="field-label"
            )
            yield Vertical(*(LoraRow(entry) for entry in settings.loras), id="loras")
            yield Button("Add LoRA", id="add-lora", compact=True)
            yield Label("Pre-prompt (e.g. score_9, masterpiece)", classes="field-label")
            yield Input(settings.pre_prompt, id="pre_prompt", compact=True)
            yield Label("Prompt (after the pre-prompt and the triggers)", classes="field-label")
            yield TextArea(settings.prompt, id="prompt")
            yield Label("Negative Prompt", classes="field-label")
            yield Input(settings.negative_prompt, id="negative_prompt", compact=True)
            for start in range(0, len(GRID_FIELDS), FIELDS_PER_ROW):
                with Horizontal(classes="field-row"):
                    for spec in GRID_FIELDS[start : start + FIELDS_PER_ROW]:
                        with Vertical(classes="field-cell"):
                            yield Label(spec.label, classes="field-label")
                            yield _setting_widget(
                                spec.name, getattr(settings, spec.name), spec.choices
                            )
            yield from _path_field(
                "Output folder (blank: <data folder>/generated)",
                settings.output_folder,
                "output_folder",
            )
            yield Static("\n".join(self._warnings), id="errors", markup=False)
            with Horizontal(classes="buttons"):
                yield Button("Generate", variant="primary", id="generate")
                yield Button("Reset to defaults", id="reset-generator")
                yield Button("Back", id="back-generator")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt", TextArea).focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "sampler" or event.value is Select.NULL:
            return
        new = str(event.value)
        cfg = self.query_one("#cfg_scale", Input)
        try:
            value = float(cfg.value)
        except ValueError:  # a value being typed that is not a number yet stays as it is
            value = None
        if value is not None:
            changed = cfg_after_sampler_change(self._sampler, new, value)
            if changed != value:
                cfg.value = str(changed)
        self._sampler = new

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        if button == "generate":
            self._generate()
        elif button == "reset-generator":
            self._reset()
        elif button == "back-generator":
            self.app.pop_screen()
        elif button == "add-lora":
            self.app.push_screen(FilePicker(self._lora_folder()), self._add_lora)
        elif button == "browse-model_file":
            self._browse(self.query_one("#model_file", Input), folder=False)
        elif button == "browse-output_folder":
            self._browse(self.query_one("#output_folder", Input), folder=True)

    def _generate(self) -> None:
        settings = settings_from_fields(self._values())
        if isinstance(settings, tuple):
            self._show_errors(settings)
            return
        self._show_errors(())
        self._settings = settings
        self.app.push_screen(GenerationScreen(settings))

    def _values(self) -> dict[str, object]:
        values: dict[str, object] = self._settings.model_dump()  # keeps what is not shown here
        for spec in GRID_FIELDS:
            widget = self.query_one(f"#{spec.name}")
            values[spec.name] = widget.value if isinstance(widget, Select) else _text(widget)
        for name in ("pre_prompt", "negative_prompt", *PATH_FIELDS):
            values[name] = self.query_one(f"#{name}", Input).value
        values["base_model"] = self.query_one("#base_model", Select).value
        values["prompt"] = self.query_one("#prompt", TextArea).text
        values["loras"] = [row.values() for row in self.query(LoraRow)]
        return values

    def _reset(self) -> None:
        for name, value in image_defaults().items():
            found = self.query(f"#{name}")
            widget = found.first() if found else None
            if isinstance(widget, Select):
                self._sampler = str(value)
                widget.value = value
            elif isinstance(widget, Input):
                widget.value = str(value)

    def _add_lora(self, file: str | None) -> None:
        if file is None:
            return
        rows = list(self.query(LoraRow))
        current = tuple(row.entry for row in rows)
        entries, warnings = add_loras(current, (file,), lambda path: lora_trigger(Path(path)))
        self.query_one("#loras", Vertical).mount_all(
            LoraRow(entry) for entry in entries[len(current) :]
        )
        self._show_errors(warnings)

    def _lora_folder(self) -> Path:
        rows = list(self.query(LoraRow))
        if rows:
            return start_folder(rows[-1].entry.path, Path.home())
        return start_folder(self.query_one("#model_file", Input).value, Path.home())

    def _browse(self, field: Input, folder: bool) -> None:
        def fill(chosen: str | None) -> None:
            if chosen is not None:
                field.value = chosen
            field.focus()

        start = start_folder(field.value, Path.home())
        self.app.push_screen(FolderPicker(start) if folder else FilePicker(start), fill)

    def _show_errors(self, lines: tuple[str, ...]) -> None:
        self.query_one("#errors", Static).update("\n".join(lines))


class GenerationScreen(Screen[None]):
    DEFAULT_CSS = GENERATOR_CSS
    BINDINGS: ClassVar[list[BindingType]] = [Binding("s", "stop", "Stop the generation")]

    def __init__(self, settings: GeneratorSettings) -> None:
        super().__init__()
        self._settings = settings

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("Starting...", id="status", markup=False)
            yield ProgressBar(id="bar", show_eta=False)
            yield RichLog(id="log", max_lines=MAX_LOG_LINES, wrap=True, markup=False)
            with Horizontal(classes="buttons"):
                yield Button("Stop", variant="error", id="stop")
                yield Button("Back to the generator", id="back", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        cast("TrainerApp", self.app).start_generation(self._settings, self)

    def add_line(self, line: str) -> None:
        self.query_one("#log", RichLog).write(line)

    def show_step(self, done: int, total: int) -> None:
        self.query_one("#status", Static).update(f"Step {done}/{total}")
        self.query_one("#bar", ProgressBar).update(total=total, progress=done)

    def add_image(self, path: Path) -> None:
        self.add_line(f"OK: Image saved: {path}")

    def show_end(self, ok: bool, message: str) -> None:
        self.add_line(message)
        if ok:
            status = "Generation completed."
        elif message.startswith("INFO: "):
            status = "Generation stopped."
        else:
            status = "Generation failed: see the log above."
        self._ended(status)

    def show_not_started(self) -> None:
        self._ended(NOT_STARTED_STATUS)

    def action_stop(self) -> None:
        cast("TrainerApp", self.app).stop_generation()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop":
            self.action_stop()
        elif event.button.id == "back":
            self.app.pop_screen()

    def _ended(self, status: str) -> None:
        self.query_one("#status", Static).update(status)
        self.query_one("#stop", Button).disabled = True
        self.query_one("#back", Button).disabled = False


def _setting_widget(name: str, value: object, choices: tuple[str, ...]) -> Input | Select[str]:
    if choices:
        options = [(Text(choice), choice) for choice in choices]
        return Select(options, value=str(value), allow_blank=False, id=name, compact=True)
    return Input(str(value), id=name, compact=True)


def _path_field(label: str, value: str, name: str) -> ComposeResult:
    yield Label(label, classes="field-label")
    with Horizontal(classes="path-row"):
        yield Input(value, id=name, compact=True)
        yield Button("Browse", id=f"browse-{name}", compact=True)


def _text(widget: object) -> str:
    return widget.value if isinstance(widget, Input) else ""
