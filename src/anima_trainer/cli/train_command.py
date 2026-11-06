"""`anima-trainer train`: Configure and Start from a preset, without a window.

Only QtCore is used here, so the command also works over SSH on a machine with no display. The
models and the runtime come from the graphical app; nothing is downloaded from here.
"""

import sys
from collections.abc import Callable, Sequence

from PySide6.QtCore import QCoreApplication, QTimer

from anima_trainer.cli.terminal_output import TerminalOutput
from anima_trainer.cli.terminal_run import TerminalRun
from anima_trainer.cli.train_options import TrainOptions, parse_train_options
from anima_trainer.cli.train_plan import EXIT_NOT_STARTED, EXIT_OK
from anima_trainer.cli.train_preparation import (
    TerminalContext,
    TrainPlan,
    load_terminal_context,
    plan_lines,
    prepare_training,
)
from anima_trainer.cli.train_summary import preset_list_lines
from anima_trainer.runtime.data_dir import SETTINGS_APPLICATION, SETTINGS_ORGANIZATION
from anima_trainer.system.gpu_lock import lock_refusal
from anima_trainer.system.process_group import KILL_GRACE_MS
from anima_trainer.system.quit_signals import SIGNAL_POLL_MS, install_quit_signals

CONFIRMATIONS = frozenset({"y", "yes"})
NOT_STARTED_NOTE = "INFO: Training not started."
MS_PER_SECOND = 1000
STOPPING_NOTE = f"INFO: Stopping the training (forced after {KILL_GRACE_MS // MS_PER_SECOND} s)..."


class _StopRequest:
    """Ctrl+C, SIGTERM or a lost terminal: stop the run like the window's Stop button.

    The handler only records the signal, and the event loop acts on it. A signal during the start
    would otherwise be lost (a stop does nothing while the session is idle), and writing from a
    handler can interrupt a write already in progress on stdout. The training has its own session,
    so a signal lost here would leave it running with no parent.
    """

    def __init__(self, stop: Callable[[], None], output: TerminalOutput) -> None:
        self._stop = stop
        self._output = output
        self._requested = False
        self._applied = False

    def request(self) -> None:
        self._requested = True

    def apply(self) -> None:
        if self._requested and not self._applied:
            self._applied = True
            self._output.line(STOPPING_NOTE)
            self._stop()


def run_train(arguments: Sequence[str]) -> int:
    options = parse_train_options(arguments)
    app = QCoreApplication.instance() or QCoreApplication(["anima-trainer"])
    app.setOrganizationName(SETTINGS_ORGANIZATION)
    app.setApplicationName(SETTINGS_APPLICATION)
    output = TerminalOutput(sys.stdout, interactive=sys.stdout.isatty())
    try:
        return _train(app, options, output)
    finally:
        output.detach()


def _train(app: QCoreApplication, options: TrainOptions, output: TerminalOutput) -> int:
    context = load_terminal_context()
    _print(output, *context.presets.warnings)
    if options.list_presets:
        _print(output, *preset_list_lines(context.presets.presets))
        return EXIT_OK
    plan = prepare_training(options, context)
    if not isinstance(plan, TrainPlan):
        _print(output, *plan)
        return EXIT_NOT_STARTED
    _print(output, *plan_lines(plan, context), "")
    # Before the question: a busy GPU would refuse the start only after the user said yes. The
    # start takes the lock again, since another process can take it during the question.
    refusal = lock_refusal(context.paths.gpu_lock_file)
    if refusal is not None:
        _print(output, f"ERROR: {refusal}", NOT_STARTED_NOTE)
        return EXIT_NOT_STARTED
    if not _confirmed(options.yes, output):
        return EXIT_NOT_STARTED
    return _run(app, plan, context, output)


def _confirmed(yes: bool, output: TerminalOutput) -> bool:
    if yes:
        return True
    # Both ends: with the output in a file the question would be invisible, and it would wait.
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        _print(output, "ERROR: No terminal to ask for confirmation: add --yes to start.")
        return False
    _print(output, "Start the training? [y/N]")
    try:
        answer = sys.stdin.readline()
    except KeyboardInterrupt:
        _print(output, "", NOT_STARTED_NOTE)
        raise
    if answer.strip().lower() in CONFIRMATIONS:
        return True
    _print(output, NOT_STARTED_NOTE)
    return False


def _run(
    app: QCoreApplication, plan: TrainPlan, context: TerminalContext, output: TerminalOutput
) -> int:
    run = TerminalRun(
        context.paths,
        plan.settings,
        context.previews.settings,
        output.line,
        output.progress,
        app.quit,
    )
    stop = _StopRequest(run.stop, output)
    restore = install_quit_signals(stop.request)
    wake = QTimer()  # Also lets Python run signal handlers, which wait for a bytecode boundary.
    wake.timeout.connect(stop.apply)
    wake.start(SIGNAL_POLL_MS)
    try:
        if run.begin():
            app.exec()
    finally:
        run.shutdown()
        restore()
    return run.exit_code


def _print(output: TerminalOutput, *lines: str) -> None:
    for line in lines:
        output.line(line)
