"""End-of-run report for the training log, as Citron printed it (with text markers)."""

from anima_trainer.training.runner import RunOutcome


def completed(outcome: RunOutcome) -> bool:
    """A clean exit counts as completed even if Stop was clicked just before it."""
    return outcome.exit_code == 0 and not outcome.crashed


def run_failed(outcome: RunOutcome) -> bool:
    """A real failure: neither completed nor stopped by the user."""
    return not outcome.stopped and not completed(outcome)


def outcome_messages(outcome: RunOutcome, output_dir: str, hint: str | None) -> tuple[str, ...]:
    """`hint` (an OOM suggestion) is shown only for real failures, not for a user stop."""
    if completed(outcome):
        lines: tuple[str, ...] = (
            "",
            "OK: Training completed successfully!",
            f"LoRA saved to: {output_dir}",
        )
    elif outcome.stopped:
        lines = ("", "INFO: Training stopped.")
    else:
        reason = (
            f"killed by signal {outcome.exit_code}"
            if outcome.crashed
            else f"exit code: {outcome.exit_code}"
        )
        lines = ("", f"ERROR: Training failed ({reason})")
    log_line = (
        f"Log saved to: {outcome.log_path}"
        if outcome.log_path is not None
        else "WARN: The log file is incomplete."
    )
    lines = (*lines, log_line)
    if hint and run_failed(outcome):
        lines = (*lines, f"HINT: {hint}")
    return lines
