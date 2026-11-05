"""The TUI application: keeps the training and runs Qt's events inside Textual's loop.

Both the training's signals and Textual's screens stay on this one thread, so slots can update
the widgets directly.
"""

import os
import threading
from collections.abc import Callable

from PySide6.QtCore import QCoreApplication
from textual.app import App

from anima_trainer.cli.terminal_output import is_progress_line
from anima_trainer.cli.terminal_run import TerminalRun
from anima_trainer.cli.train_plan import EXIT_FAILED, EXIT_OK
from anima_trainer.cli.train_preparation import TerminalContext, TrainPlan
from anima_trainer.cli.tui.screens import FormScreen, QuitDialog, TrainingScreen
from anima_trainer.cli.tui_values import TuiValues, save_tui_values
from anima_trainer.training.progress import ProgressState

QT_EVENTS_INTERVAL_S = 0.05
# After a signal the terminal may be gone, and Textual can hang while closing on it. Nothing is
# left to protect by then: the training has ended and its lock was given back.
FORCED_EXIT_DELAY_S = 3.0


class TrainerApp(App[None]):
    TITLE = "Anima LoRA Studio"
    ENABLE_COMMAND_PALETTE = False

    def __init__(
        self,
        context: TerminalContext,
        qt: QCoreApplication,
        force_exit: Callable[[int], None] = os._exit,
    ) -> None:
        super().__init__()
        self._force_exit = force_exit
        self._forced_exit_timer: threading.Timer | None = None
        self._signalled = False
        self._end_shown = False
        self.terminal = context
        self.exit_code = EXIT_OK
        self.last_note = ""
        self._qt_app = qt
        self._training: TerminalRun | None = None
        self._training_screen: TrainingScreen | None = None
        self._quit_requested = False
        self._exit_when_ended = False

    @property
    def final_exit_code(self) -> int:
        """Textual reports an exception in a screen by closing with a return code of its own."""
        if self.exit_code == EXIT_OK and self.return_code:
            return EXIT_FAILED
        return self.exit_code

    @property
    def run_active(self) -> bool:
        return self._training is not None and self._training.active

    def on_mount(self) -> None:
        self.set_interval(QT_EVENTS_INTERVAL_S, self._process_qt_events)
        self.push_screen(FormScreen(self.terminal))

    def request_quit(self) -> None:
        """From a signal handler (a lost terminal, SIGTERM): recorded, see _process_qt_events."""
        self._quit_requested = True

    def start_training(self, plan: TrainPlan, screen: TrainingScreen) -> None:
        """Called by the training screen once it can show lines."""
        self._training_screen = screen
        self._end_shown = False
        self.exit_code = EXIT_OK
        self._remember(plan, screen)
        self._training = TerminalRun(
            self.terminal.paths,
            plan.settings,
            self.terminal.previews.settings,
            self._on_line,
            self._on_progress,
            self._on_ended,
        )
        if not self._training.begin():
            self._on_ended()

    def stop_training(self) -> None:
        if self._training is not None:
            self._training.stop()

    def shutdown(self) -> None:
        if self._forced_exit_timer is not None:
            self._forced_exit_timer.cancel()
        if self._training is not None:
            self._training.shutdown()

    async def action_quit(self) -> None:
        if self.run_active:
            self.push_screen(QuitDialog(), self._on_quit_answer)
        else:
            self.exit()

    def _on_quit_answer(self, confirmed: bool | None) -> None:
        if confirmed:
            self._stop_and_exit()

    def _process_qt_events(self) -> None:
        self._qt_app.processEvents()
        if self._quit_requested:
            self._quit_requested = False
            self._signalled = True
            self._stop_and_exit()

    def _stop_and_exit(self) -> None:
        if self.run_active:
            self._exit_when_ended = True
            self.stop_training()
        else:
            self._leave()

    def _remember(self, plan: TrainPlan, screen: TrainingScreen) -> None:
        values = TuiValues(
            project=plan.settings.project_name,
            dataset=plan.settings.image_directory,
            output=plan.settings.output_directory,
            preset=plan.preset.name,
        )
        try:
            save_tui_values(self.terminal.paths.tui_values_file, values)
        except OSError as error:
            screen.add_line(f"WARN: The chosen values could not be saved: {error}")

    def _on_line(self, line: str) -> None:
        if self._training_screen is not None and not is_progress_line(line):
            self._training_screen.add_line(line)

    def _on_progress(self, state: ProgressState) -> None:
        if self._training_screen is not None:
            self._training_screen.show_progress(state)

    def _on_ended(self) -> None:
        if self._training is None or self._training_screen is None or self._end_shown:
            return
        self._end_shown = True
        self.exit_code = self._training.exit_code
        self.last_note = self._training_screen.show_end(self.exit_code)
        if self._exit_when_ended:
            self._leave()

    def _leave(self) -> None:
        if self._signalled and self._forced_exit_timer is None:
            self._forced_exit_timer = threading.Timer(
                FORCED_EXIT_DELAY_S, self._force_exit, (self.exit_code,)
            )
            self._forced_exit_timer.daemon = True
            self._forced_exit_timer.start()
        self.exit()
