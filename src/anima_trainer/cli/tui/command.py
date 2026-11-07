"""`anima-trainer tui`: open the interface, and leave no training behind when it closes."""

import argparse
import sys
from collections.abc import Sequence

from PySide6.QtCore import QCoreApplication

from anima_trainer.cli.terminal_output import TerminalOutput, silence_if_lost
from anima_trainer.cli.train_plan import EXIT_NOT_STARTED
from anima_trainer.cli.train_preparation import load_terminal_context
from anima_trainer.cli.tui.app import TrainerApp
from anima_trainer.runtime.data_dir import SETTINGS_APPLICATION, SETTINGS_ORGANIZATION
from anima_trainer.system.quit_signals import install_quit_signals


def run_tui(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="anima-trainer tui",
        description=(
            "Choose a preset, check the total steps and train, or generate images with LoRAs, "
            "in the terminal. Quitting stops what is running: for remote use run it inside "
            "tmux, which survives a lost connection."
        ),
    )
    parser.parse_args(arguments)
    output = TerminalOutput(sys.stdout, interactive=False)
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        output.line("ERROR: This needs a terminal. Without one, use: anima-trainer train --help")
        return EXIT_NOT_STARTED
    qt = QCoreApplication.instance() or QCoreApplication(["anima-trainer"])
    qt.setOrganizationName(SETTINGS_ORGANIZATION)
    qt.setApplicationName(SETTINGS_APPLICATION)
    app = TrainerApp(load_terminal_context(), qt)
    restore = install_quit_signals(app.request_quit)
    try:
        app.run()
    finally:
        app.shutdown()  # also after a crash of the interface: nothing stays on the GPU
        restore()
    if app.last_note:
        output.line(app.last_note)
    output.detach()
    silence_if_lost(sys.stderr)  # Textual draws there, and may have left something to flush
    return app.final_exit_code
