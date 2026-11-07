"""The screens of the TUI: the choices, the summary to approve, the training."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
)

from anima_trainer.cli.terminal_output import progress_text
from anima_trainer.cli.train_options import DEFAULT_RESOLUTION, TrainOptions
from anima_trainer.cli.train_plan import EXIT_FAILED, EXIT_OK, EXIT_STOPPED
from anima_trainer.cli.train_preparation import (
    TerminalContext,
    TrainPlan,
    plan_lines,
    prepare_training,
)
from anima_trainer.cli.tui.folder_picker import FolderPicker, start_folder
from anima_trainer.cli.tui_values import load_tui_values
from anima_trainer.presets.catalog import find_preset
from anima_trainer.system.gpu_lock import lock_refusal
from anima_trainer.training.progress import ProgressState

if TYPE_CHECKING:
    from anima_trainer.cli.tui.app import TrainerApp

MAX_LOG_LINES = 5000
SCREEN_CSS = """
.field-label { margin-top: 0; color: $text-muted; }
.buttons { height: auto; margin-top: 1; }
.folder-row { height: auto; }
.folder-row Input { width: 1fr; }
.folder-row Button { width: auto; margin-left: 2; }
.buttons Button { margin-right: 2; }
#errors, #preset-about { height: auto; }
#errors { color: $error; }
#log { height: 1fr; border: solid $primary; }
#bar { margin: 1 0; }
QuitDialog { align: center middle; }
#quit-box { width: 70; height: auto; padding: 1 2; border: thick $warning; background: $surface; }
"""
# An option named at the start of a word; the dashes inside a path are left alone.
_OPTION_DASHES = re.compile(r"(?<!\S)--(?=[a-z])")
BROWSE_BUTTONS = {"browse-dataset": "dataset", "browse-output": "output"}
NOT_STARTED_STATUS = "Training not started."
END_STATUS = {
    EXIT_OK: "Training completed.",
    EXIT_FAILED: "Training failed: see the log above.",
    EXIT_STOPPED: "Training stopped.",
}


class FormScreen(Screen[None]):
    DEFAULT_CSS = SCREEN_CSS

    def __init__(self, context: TerminalContext) -> None:
        super().__init__()
        self._terminal = context
        self._remembered = load_tui_values(context.paths.tui_values_file)

    def compose(self) -> ComposeResult:
        values = self._remembered.values
        presets = self._terminal.presets
        preset = find_preset(presets, values.preset) if values.preset else None
        # Text: a plain string here would be read as markup, and "[wip]" would vanish.
        options = [(Text(item.name), item.name) for item in presets.presets]
        gone = (
            (f"WARN: Preset '{values.preset}' is no longer available.",)
            if values.preset and preset is None
            else ()
        )
        yield Header()
        with VerticalScroll():
            yield from _field("Project name", Input(values.project, id="project", compact=True))
            yield from _folder_field("Dataset folder", values.dataset, "dataset")
            yield from _folder_field("Output folder", values.output, "output")
            yield Label("Preset", classes="field-label")
            yield Select(
                options,
                prompt="Choose a preset",
                value=preset.name if preset else Select.NULL,
                id="preset",
                compact=True,
            )
            yield Static(
                self._about(preset.name if preset else None), id="preset-about", markup=False
            )
            yield from _field(
                "Epochs (proposed by the preset)",
                Input(
                    self._suggested_epochs(preset.name if preset else None),
                    type="integer",
                    id="epochs",
                ),
            )
            yield from _field("Repeats", Input(type="integer", id="repeats", compact=True))
            yield from _field(
                "Resolution (768 is safe on 12 GB of VRAM)",
                Input(str(DEFAULT_RESOLUTION), type="integer", id="resolution", compact=True),
            )
            warnings = (*self._remembered.warnings, *presets.warnings, *gone)
            yield Static("\n".join(warnings), id="errors", markup=False)
            with Horizontal(classes="buttons"):
                yield Button("Continue", variant="primary", id="continue")
                yield Button("Menu", id="menu-form")
                yield Button("Quit", id="quit-form")
        yield Footer()

    def on_mount(self) -> None:
        for name in ("project", "dataset", "output"):
            field = self.query_one(f"#{name}", Input)
            if not field.value:
                field.focus()
                return
        if self.query_one("#preset", Select).is_blank():
            self.query_one("#preset", Select).focus()
            return
        self.query_one("#repeats", Input).focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        chosen = None if event.value is Select.NULL else str(event.value)
        self.query_one("#preset-about", Static).update(self._about(chosen))
        self.query_one("#epochs", Input).value = self._suggested_epochs(chosen)

    def on_input_submitted(self) -> None:
        self._continue()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "continue":
            self._continue()
        elif event.button.id == "quit-form":
            cast("TrainerApp", self.app).exit()
        elif event.button.id == "menu-form":
            self.app.pop_screen()
        elif event.button.id in BROWSE_BUTTONS:
            self._browse(self.query_one(f"#{BROWSE_BUTTONS[event.button.id]}", Input))

    def _browse(self, field: Input) -> None:
        def fill(folder: str | None) -> None:
            if folder is not None:
                field.value = folder
            field.focus()

        self.app.push_screen(FolderPicker(start_folder(field.value, Path.home())), fill)

    def _continue(self) -> None:
        options = self._options()
        if isinstance(options, tuple):
            self._show_errors(options)
            return
        plan = prepare_training(options, self._terminal)
        if isinstance(plan, tuple):
            self._show_errors(plan)
            return
        # Before the summary to approve; Start takes the lock again (see TerminalRun.begin).
        refusal = lock_refusal(self._terminal.paths.gpu_lock_file)
        if refusal is not None:
            self._show_errors((f"ERROR: {refusal}",))
            return
        self._show_errors(())
        self.app.push_screen(SummaryScreen(plan, self._terminal))

    def _options(self) -> TrainOptions | tuple[str, ...]:
        select = self.query_one("#preset", Select)
        if select.is_blank():
            return ("ERROR: preset is required",)
        numbers: dict[str, int | None] = {}
        errors = []
        for name in ("epochs", "repeats", "resolution"):
            text = self.query_one(f"#{name}", Input).value.strip()
            try:
                numbers[name] = int(text) if text else None
            except ValueError:
                errors.append(f"ERROR: {name} must be a whole number")
        if errors:
            return tuple(errors)
        resolution = numbers["resolution"]
        return TrainOptions(
            list_presets=False,
            preset=str(select.value),
            project=self.query_one("#project", Input).value,
            dataset=self.query_one("#dataset", Input).value,
            output=self.query_one("#output", Input).value,
            epochs=numbers["epochs"],
            repeats=numbers["repeats"],
            resolution=DEFAULT_RESOLUTION if resolution is None else resolution,
            yes=True,
        )

    def _show_errors(self, lines: tuple[str, ...]) -> None:
        # The shared checks name command line options; here the same names are field labels.
        shown = "\n".join(_OPTION_DASHES.sub("", line) for line in lines)
        self.query_one("#errors", Static).update(shown)

    def _about(self, name: str | None) -> str:
        preset = find_preset(self._terminal.presets, name) if name else None
        return preset.description if preset else ""

    def _suggested_epochs(self, name: str | None) -> str:
        preset = find_preset(self._terminal.presets, name) if name else None
        suggested = preset.values.get("max_train_epochs") if preset else None
        return str(suggested) if suggested else ""


class SummaryScreen(Screen[None]):
    DEFAULT_CSS = SCREEN_CSS

    def __init__(self, plan: TrainPlan, context: TerminalContext) -> None:
        super().__init__()
        self._plan = plan
        self._lines = plan_lines(plan, context)

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static("\n".join(self._lines), id="summary", markup=False)
            with Horizontal(classes="buttons"):
                yield Button("Start", variant="success", id="start")
                yield Button("Edit", id="edit")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#start", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.app.push_screen(TrainingScreen(self._plan))
        elif event.button.id == "edit":
            self.app.pop_screen()


class TrainingScreen(Screen[None]):
    DEFAULT_CSS = SCREEN_CSS
    BINDINGS: ClassVar[list[BindingType]] = [Binding("s", "stop", "Stop the training")]

    def __init__(self, plan: TrainPlan) -> None:
        super().__init__()
        self._plan = plan

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("Starting...", id="status", markup=False)
            yield ProgressBar(id="bar", show_eta=False)
            yield RichLog(id="log", max_lines=MAX_LOG_LINES, wrap=True, markup=False)
            with Horizontal(classes="buttons"):
                yield Button("Stop", variant="error", id="stop")
                yield Button("Back to the form", id="back", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        cast("TrainerApp", self.app).start_training(self._plan, self)

    def add_line(self, line: str) -> None:
        self.query_one("#log", RichLog).write(line)

    def show_progress(self, state: ProgressState) -> None:
        if state.total_steps is None and state.step == 0:
            return
        self.query_one("#status", Static).update(progress_text(state))
        self.query_one("#bar", ProgressBar).update(total=state.total_steps, progress=state.step)

    def show_end(self, exit_code: int) -> str:
        """Show how the run ended; returns the same text for after the interface has closed."""
        status = END_STATUS.get(exit_code, NOT_STARTED_STATUS)
        self.query_one("#status", Static).update(status)
        self.query_one("#stop", Button).disabled = True
        self.query_one("#back", Button).disabled = False
        return status

    def action_stop(self) -> None:
        cast("TrainerApp", self.app).stop_training()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop":
            self.action_stop()
        elif event.button.id == "back":
            self.app.pop_screen()  # this screen
            self.app.pop_screen()  # the summary


class QuitDialog(ModalScreen[bool]):
    DEFAULT_CSS = SCREEN_CSS

    def compose(self) -> ComposeResult:
        with Vertical(id="quit-box"):
            yield Label(
                "Quitting stops the training or the generation that is running.\n"
                "Inside tmux you can leave it running instead: detach with Ctrl+B, then D."
            )
            with Horizontal(classes="buttons"):
                yield Button("Stop and quit", variant="error", id="quit-yes")
                yield Button("Keep it running", variant="primary", id="quit-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "quit-yes")


def _folder_field(label: str, value: str, name: str) -> ComposeResult:
    yield Label(label, classes="field-label")
    with Horizontal(classes="folder-row"):
        yield Input(value, id=name, compact=True)
        yield Button("Browse", id=f"browse-{name}", compact=True)


def _field(label: str, field: Input) -> ComposeResult:
    yield Label(label, classes="field-label")
    yield field
