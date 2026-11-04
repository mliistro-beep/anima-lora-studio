"""`anima-trainer train`: Configure and Start with the saved settings, without a window.

Only QtCore is used here, so the command also works over SSH on a machine with no display. The
models and the runtime come from the graphical app; nothing is downloaded from here.
"""

import argparse
import sys
from collections.abc import Sequence

from PySide6.QtCore import QCoreApplication, QSettings, QTimer

from anima_trainer.cli.terminal_output import TerminalOutput
from anima_trainer.cli.train_plan import (
    EXIT_NOT_STARTED,
    exit_code_for,
    missing_model_problems,
    runtime_problem_lines,
    settings_problem_lines,
)
from anima_trainer.models.download_worker import ModelDownloader
from anima_trainer.models.linked_store import load_links
from anima_trainer.preview.settings import load_preview_settings
from anima_trainer.runtime.data_dir import (
    SETTINGS_APPLICATION,
    SETTINGS_ORGANIZATION,
    default_data_dir,
    load_data_dir,
    open_data_dir,
)
from anima_trainer.system.gpu_memory import read_all_gpu_memory
from anima_trainer.system.memory_safety import read_memory_safety, warnings_only
from anima_trainer.system.process_group import KILL_GRACE_MS
from anima_trainer.system.quit_signals import SIGNAL_POLL_MS, install_quit_signals
from anima_trainer.system.resource_monitor import ResourceMonitor
from anima_trainer.training.runner import RunOutcome, TrainingRunner
from anima_trainer.training.start_service import StartRequest
from anima_trainer.workflow.training_session import SessionState, TrainingSession

MS_PER_SECOND = 1000
STOPPING_NOTE = f"INFO: Stopping the training (forced after {KILL_GRACE_MS // MS_PER_SECOND} s)..."


class _StopRequest:
    """Ctrl+C, SIGTERM or a lost terminal: stop the run like the window's Stop button.

    The handler only records the signal, and the event loop acts on it. A signal during
    session.start() would otherwise be lost (stop() does nothing while the session is idle), and
    writing from a handler can interrupt a write already in progress on stdout. The training has
    its own session, so a signal lost here would leave it running with no parent.
    """

    def __init__(self, session: TrainingSession, output: TerminalOutput) -> None:
        self._session = session
        self._output = output
        self._requested = False
        self._applied = False

    def request(self) -> None:
        self._requested = True

    def apply(self) -> None:
        if self._requested and not self._applied:
            self._applied = True
            self._output.line(STOPPING_NOTE)
            self._session.stop()


def run_train(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="anima-trainer train",
        description="Train a LoRA with the settings saved by the last Configure.",
    )
    parser.parse_args(arguments)
    app = QCoreApplication.instance() or QCoreApplication(["anima-trainer"])
    app.setOrganizationName(SETTINGS_ORGANIZATION)
    app.setApplicationName(SETTINGS_APPLICATION)
    output = TerminalOutput(sys.stdout, interactive=sys.stdout.isatty())
    try:
        return _train(app, output)
    finally:
        output.detach()


def _train(app: QCoreApplication, output: TerminalOutput) -> int:
    store = QSettings()
    opened = open_data_dir(load_data_dir(store, default_data_dir()))
    links = load_links(store)
    paths = opened.paths.with_links(links)
    previews = load_preview_settings(paths.preview_settings_file)
    blockers = (
        runtime_problem_lines(opened.problems)  # without the runtime the rest says little
        or settings_problem_lines((*opened.warnings, *previews.warnings))
        or missing_model_problems(paths, opened.settings.base_model)
    )
    _print(output, *warnings_only(read_memory_safety()), *blockers)
    if blockers:
        return EXIT_NOT_STARTED

    runner = TrainingRunner(monitor=ResourceMonitor())
    session = TrainingSession(paths, opened.settings, runner, ModelDownloader())
    session.set_gpu_memory(read_all_gpu_memory())
    result = session.configure(opened.settings, previews.settings)
    _print(output, *result.messages, "")
    if not result.success:
        return EXIT_NOT_STARTED
    return _run(app, session, runner, output)


def _run(
    app: QCoreApplication,
    session: TrainingSession,
    runner: TrainingRunner,
    output: TerminalOutput,
) -> int:
    outcome: list[RunOutcome] = []
    session.log_line.connect(output.line)
    session.progress_changed.connect(output.progress)
    session.state_changed.connect(lambda state: app.quit() if state is SessionState.IDLE else None)
    runner.finished.connect(outcome.append)
    stop = _StopRequest(session, output)
    restore = install_quit_signals(stop.request)
    wake = QTimer()  # Also lets Python run signal handlers, which wait for a bytecode boundary.
    wake.timeout.connect(stop.apply)
    wake.start(SIGNAL_POLL_MS)
    try:
        session.start(
            StartRequest(
                session.settings.gpu_index,
                session.settings.num_cpu_threads_per_process,
                session.settings.base_model,
            )
        )
        if session.state is not SessionState.IDLE:
            app.exec()
    finally:
        session.shutdown()
        restore()
    return exit_code_for(outcome[0] if outcome else None)


def _print(output: TerminalOutput, *lines: str) -> None:
    for line in lines:
        output.line(line)
