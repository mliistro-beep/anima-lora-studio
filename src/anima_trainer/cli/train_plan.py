"""Pure decisions of a terminal training: what blocks it and how the process exits."""

from pathlib import Path

from anima_trainer.models.training_model import TrainingModelRejected, check_training_model
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.training.outcome_report import completed
from anima_trainer.training.runner import RunOutcome

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOT_STARTED = 2
EXIT_STOPPED = 130  # the shell convention for a command ended by Ctrl+C

GRAPHICAL_APP_NOTE = "run anima-trainer without arguments to open the graphical app."


def missing_model_problems(
    paths: RuntimePaths, base_model: str, model_file: str = ""
) -> tuple[str, ...]:
    """The terminal never downloads: models are installed or linked from the graphical app.

    With a model file of the user's the base model is not needed (model_file_problems checks it).
    """
    missing = list(paths.missing_required_models())
    if not model_file and not paths.dit_model(base_model).exists():
        missing.append(base_model)
    if not missing:
        return ()
    names = ", ".join(missing)
    return (f"ERROR: Missing models: {names}. To download or link them, {GRAPHICAL_APP_NOTE}",)


def model_file_problems(model_file: str) -> tuple[str, ...]:
    """Why the user's model file cannot be trained on; nothing without a model file.

    `model_file` is already absolute here (settings_for), as the command line meant it.
    """
    if not model_file:
        return ()
    try:
        check_training_model(Path(model_file))
    except TrainingModelRejected as error:
        return (f"ERROR: {error}",)
    return ()


def runtime_problem_lines(problems: tuple[str, ...]) -> tuple[str, ...]:
    if not problems:
        return ()
    errors = tuple(f"ERROR: {problem}" for problem in problems)
    return (*errors, f"To install the training runtime, {GRAPHICAL_APP_NOTE}")


def settings_problem_lines(warnings: tuple[str, ...]) -> tuple[str, ...]:
    """Saved settings that had to fall back to defaults: nobody is there to notice before the
    run, and Configure would write the defaults over the user's file."""
    if not warnings:
        return ()
    lines = tuple(f"WARN: {warning}" for warning in warnings)
    return (*lines, f"ERROR: Fix the saved settings first; {GRAPHICAL_APP_NOTE}")


def exit_code_for(outcome: RunOutcome | None) -> int:
    """`outcome` is None when no training process ran."""
    if outcome is None:
        return EXIT_NOT_STARTED
    if completed(outcome):
        return EXIT_OK
    return EXIT_STOPPED if outcome.stopped else EXIT_FAILED
