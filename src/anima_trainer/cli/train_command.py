"""`anima-trainer train`: Configure and Start from a preset, without a window.

Only QtCore is used here, so the command also works over SSH on a machine with no display. The
models and the runtime come from the graphical app; nothing is downloaded from here.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from pydantic_core import ErrorDetails
from PySide6.QtCore import QCoreApplication, QSettings, QTimer

from anima_trainer.cli.terminal_output import TerminalOutput
from anima_trainer.cli.train_options import (
    TrainOptions,
    missing_option_lines,
    parse_train_options,
)
from anima_trainer.cli.train_plan import (
    EXIT_NOT_STARTED,
    EXIT_OK,
    exit_code_for,
    missing_model_problems,
    runtime_problem_lines,
    settings_problem_lines,
)
from anima_trainer.cli.train_summary import (
    base_model_problem,
    preset_list_lines,
    resolution_warning,
    settings_for,
    summary_lines,
)
from anima_trainer.dataset.validator import DatasetError, DatasetReport, validate_dataset
from anima_trainer.models.download_worker import ModelDownloader
from anima_trainer.models.linked_store import load_links
from anima_trainer.presets.catalog import LoadedPresets, find_preset, load_presets
from anima_trainer.presets.preset_file import Preset
from anima_trainer.preview.settings import load_preview_settings
from anima_trainer.runtime.data_dir import (
    SETTINGS_APPLICATION,
    SETTINGS_ORGANIZATION,
    OpenedRuntime,
    default_data_dir,
    load_data_dir,
    open_data_dir,
)
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.system.gpu_lock import TERMINAL_JOB, GpuLock
from anima_trainer.system.gpu_memory import read_all_gpu_memory
from anima_trainer.system.memory_safety import read_memory_safety, warnings_only
from anima_trainer.system.process_group import KILL_GRACE_MS
from anima_trainer.system.quit_signals import SIGNAL_POLL_MS, install_quit_signals
from anima_trainer.system.resource_monitor import ResourceMonitor
from anima_trainer.training.runner import RunOutcome, TrainingRunner
from anima_trainer.training.start_service import StartRequest
from anima_trainer.workflow.training_session import SessionState, TrainingSession

CONFIRMATIONS = frozenset({"y", "yes"})
NOT_STARTED_NOTE = "INFO: Training not started."
MS_PER_SECOND = 1000
STOPPING_NOTE = f"INFO: Stopping the training (forced after {KILL_GRACE_MS // MS_PER_SECOND} s)..."


@dataclass(frozen=True)
class _Plan:
    preset: Preset
    settings: TrainingSettings
    report: DatasetReport


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
    store = QSettings()
    opened = open_data_dir(load_data_dir(store, default_data_dir()))
    paths = opened.paths.with_links(load_links(store))
    presets = load_presets(paths.presets_dir)
    _print(output, *presets.warnings)
    if options.list_presets:
        _print(output, *preset_list_lines(presets.presets))
        return EXIT_OK
    previews = load_preview_settings(paths.preview_settings_file)
    plan = _plan(options, opened, paths, presets, previews.warnings)
    if not isinstance(plan, _Plan):
        _print(output, *plan)
        return EXIT_NOT_STARTED
    gpus = read_all_gpu_memory()
    warning = resolution_warning(plan.settings, gpus)
    _print(
        output,
        *warnings_only(read_memory_safety()),
        *summary_lines(plan.preset, plan.settings, plan.report, previews.settings),
        *((warning,) if warning else ()),
        "",
    )
    if not _confirmed(options.yes, output):
        return EXIT_NOT_STARTED
    lock = GpuLock()
    # Before Configure: a refused command leaves the configs as they were.
    refusal = lock.acquire(TERMINAL_JOB, paths.gpu_lock_file)
    if refusal is not None:
        _print(output, f"ERROR: {refusal}")
        return EXIT_NOT_STARTED
    try:
        runner = TrainingRunner(monitor=ResourceMonitor())
        session = TrainingSession(paths, plan.settings, runner, ModelDownloader(), lock=lock)
        # Read again: time can pass at the question, and previews fit what is free now.
        session.set_gpu_memory(read_all_gpu_memory())
        # Not persisted: the window finds its settings as the user left them.
        result = session.configure(plan.settings, previews.settings, persist=False)
        _print(output, *result.messages, "")
        if not result.success:
            return EXIT_NOT_STARTED
        return _run(app, session, runner, output)
    finally:
        lock.release(TERMINAL_JOB)


def _plan(
    options: TrainOptions,
    opened: OpenedRuntime,
    paths: RuntimePaths,
    presets: LoadedPresets,
    preview_warnings: tuple[str, ...],
) -> "_Plan | tuple[str, ...]":
    """What this training would be, or the lines saying why it cannot start. Writes nothing."""
    if options.preset is None:
        return ("ERROR: --preset is required (see --list-presets)",)
    preset = find_preset(presets, options.preset)
    if preset is None:
        names = ", ".join(item.name for item in presets.presets)
        return (f"ERROR: Unknown preset '{options.preset}'. Available: {names}",)
    missing = missing_option_lines(options, preset)
    if missing:
        return missing
    try:
        settings = settings_for(options, preset, opened.settings)
    except ValidationError as error:
        return tuple(f"ERROR: {_field_error(detail)}" for detail in error.errors())
    problem = base_model_problem(settings)
    if problem is not None:
        return (problem,)
    blockers = (
        runtime_problem_lines(opened.problems)  # without the runtime the rest says little
        or settings_problem_lines((*opened.warnings, *preview_warnings))
        or missing_model_problems(paths, settings.base_model)
    )
    if blockers:
        return blockers
    try:
        report = validate_dataset(Path(settings.image_directory))
    except DatasetError as error:
        return (f"ERROR: {error}",)
    if report.image_count == 0:
        return (f"ERROR: No images found in {settings.image_directory}",)
    return _Plan(preset, settings, report)


def _field_error(detail: ErrorDetails) -> str:
    return f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"


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
