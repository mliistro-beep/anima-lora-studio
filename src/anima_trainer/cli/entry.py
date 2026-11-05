"""The `anima-trainer` command: the graphical app, or a terminal command.

Decides before importing anything graphical, so a terminal command starts on a machine without
display libraries. Unknown arguments go to the graphical app, which passes them to Qt.
"""

import os
import sys
from collections.abc import Sequence

from anima_trainer.cli.train_plan import EXIT_NOT_STARTED, EXIT_STOPPED

USAGE = """\
usage: anima-trainer [Qt arguments]
       anima-trainer COMMAND [--help]

Without a command, opens the graphical app.

commands:
  train    train a LoRA from a preset, without a window
  tui      the same in a terminal interface, with a summary to approve or change
"""
HELP_FLAGS = {"-h", "--help"}
QT_PLATFORM_OPTION = "-platform"
DISPLAY_VARIABLES = ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM")
NO_DISPLAY_MESSAGE = (
    "ERROR: No display here, so the graphical app cannot open.\n"
    "In a terminal use: anima-trainer tui, or anima-trainer train --help\n"
)


def main(argv: Sequence[str] | None = None) -> int:
    """`argv` picks the command; the graphical app always reads its Qt arguments from sys.argv."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments[0] if arguments else None
    if command in HELP_FLAGS:
        sys.stdout.write(USAGE)
        return 0
    if command == "train":
        from anima_trainer.cli.train_command import run_train  # noqa: PLC0415

        try:
            return run_train(arguments[1:])
        except KeyboardInterrupt:  # Ctrl+C before the training started
            return EXIT_STOPPED
    if command == "tui":
        from anima_trainer.cli.tui.command import run_tui  # noqa: PLC0415

        try:
            return run_tui(arguments[1:])
        except KeyboardInterrupt:  # Ctrl+C before the interface took over the terminal
            return EXIT_STOPPED
    if _no_display(arguments):
        sys.stdout.write(NO_DISPLAY_MESSAGE)
        return EXIT_NOT_STARTED
    from anima_trainer.main import main as open_window  # noqa: PLC0415

    return open_window()


def _no_display(arguments: Sequence[str]) -> bool:
    """On Linux Qt aborts with a long message of its own when it has nowhere to show a window."""
    if not sys.platform.startswith("linux") or QT_PLATFORM_OPTION in arguments:
        return False
    return not any(os.environ.get(name) for name in DISPLAY_VARIABLES)


if __name__ == "__main__":
    sys.exit(main())
