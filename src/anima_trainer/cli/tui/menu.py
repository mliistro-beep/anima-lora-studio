"""The first screen of the TUI: train a LoRA or generate images."""

from typing import TYPE_CHECKING, cast

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from anima_trainer.cli.tui.generator_screens import GeneratorScreen
from anima_trainer.cli.tui.screens import FormScreen

if TYPE_CHECKING:
    from anima_trainer.cli.tui.app import TrainerApp

MENU_CSS = """
MenuScreen { align: center middle; }
#menu-box { width: 60; height: auto; padding: 1 2; border: thick $primary; }
#menu-box Button { width: 100%; margin-top: 1; }
"""


class MenuScreen(Screen[None]):
    DEFAULT_CSS = MENU_CSS

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="menu-box"):
            yield Static("What do you want to do?", markup=False)
            yield Button("Train a LoRA", variant="primary", id="menu-training")
            yield Button("Generate images", id="menu-generator")
            yield Button("Quit", id="menu-quit")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#menu-training", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        app = cast("TrainerApp", self.app)
        if event.button.id == "menu-training":
            app.push_screen(FormScreen(app.terminal))
        elif event.button.id == "menu-generator":
            app.push_screen(GeneratorScreen(app.terminal))
        elif event.button.id == "menu-quit":
            app.exit()
