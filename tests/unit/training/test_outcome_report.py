from pathlib import Path

from anima_trainer.training.outcome_report import outcome_messages, run_failed
from anima_trainer.training.runner import RunOutcome

LOG = Path("/citron/logs/proj.log")


def test_success_reports_lora_and_log_location():
    lines = outcome_messages(RunOutcome(0, False, False, LOG), "/out", None)

    assert lines == (
        "",
        "OK: Training completed successfully!",
        "LoRA saved to: /out",
        f"Log saved to: {LOG}",
    )


def test_failure_reports_exit_code_and_hint():
    lines = outcome_messages(RunOutcome(1, False, False, LOG), "/out", "Try smaller dim.")

    assert "ERROR: Training failed (exit code: 1)" in lines
    assert f"Log saved to: {LOG}" in lines
    assert lines[-1] == "HINT: Try smaller dim."


def test_crash_reports_the_signal():
    lines = outcome_messages(RunOutcome(9, True, False, LOG), "/out", None)

    assert "ERROR: Training failed (killed by signal 9)" in lines


def test_stopped_run_is_not_reported_as_a_failure():
    lines = outcome_messages(RunOutcome(15, True, True, LOG), "/out", "ignored")

    assert "INFO: Training stopped." in lines
    assert not any(line.startswith(("ERROR", "HINT")) for line in lines)


def test_incomplete_log_is_reported():
    lines = outcome_messages(RunOutcome(0, False, False, None), "/out", None)

    assert lines[-1] == "WARN: The log file is incomplete."


def test_stop_that_arrives_after_a_clean_exit_counts_as_completed():
    lines = outcome_messages(RunOutcome(0, False, True, LOG), "/out", None)

    assert "OK: Training completed successfully!" in lines


def test_run_failed():
    assert run_failed(RunOutcome(1, False, False, LOG))
    assert run_failed(RunOutcome(9, True, False, LOG))
    assert not run_failed(RunOutcome(0, False, False, LOG))
    assert not run_failed(RunOutcome(15, True, True, LOG))
