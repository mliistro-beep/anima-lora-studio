"""Pure decisions of a terminal training: what blocks it and how the process exits."""

from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.training.outcome_report import completed
from anima_trainer.training.runner import RunOutcome

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOT_STARTED = 2
EXIT_STOPPED = 130  # the shell convention for a command ended by Ctrl+C

GRAPHICAL_APP_NOTE = "run anima-trainer without arguments to open the graphical app."


def missing_model_problems(paths: RuntimePaths, base_model: str) -> tuple[str, ...]:
    """The terminal never downloads: models are installed or linked from the graphical app."""
    missing = list(paths.missing_required_models())
    if not paths.dit_model(base_model).exists():
        missing.append(base_model)
    if not missing:
        return ()
    names = ", ".join(missing)
    return (f"ERROR: Missing models: {names}. To download or link them, {GRAPHICAL_APP_NOTE}",)


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
