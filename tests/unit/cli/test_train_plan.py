"""Checks before a terminal training and its exit code."""

from pathlib import Path

from anima_trainer.cli.train_plan import (
    EXIT_FAILED,
    EXIT_NOT_STARTED,
    EXIT_OK,
    EXIT_STOPPED,
    exit_code_for,
    missing_model_problems,
    runtime_problem_lines,
    settings_problem_lines,
)
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.training.runner import RunOutcome

BASE_MODEL = "anima-base-v1.0"
LOG = Path("run.log")


def _paths_with_models(root, *present):
    paths = RuntimePaths(root)
    files = {"Qwen3": paths.qwen3_model, "VAE": paths.vae_model, "DiT": paths.dit_model(BASE_MODEL)}
    for name in present:
        files[name].parent.mkdir(parents=True, exist_ok=True)
        files[name].write_bytes(b"")
    return paths


def test_no_problem_when_every_model_is_there(tmp_path):
    paths = _paths_with_models(tmp_path, "Qwen3", "VAE", "DiT")
    assert missing_model_problems(paths, BASE_MODEL) == ()


def test_each_missing_model_is_named_and_points_to_the_graphical_app(tmp_path):
    paths = _paths_with_models(tmp_path, "VAE")
    problems = missing_model_problems(paths, BASE_MODEL)
    assert len(problems) == 1
    assert problems[0].startswith("ERROR: Missing models: Qwen3, ")
    assert BASE_MODEL in problems[0]
    assert "graphical app" in problems[0]


def test_an_unstarted_run_has_its_own_exit_code():
    assert exit_code_for(None) == EXIT_NOT_STARTED


def test_a_completed_run_exits_with_zero():
    assert exit_code_for(RunOutcome(0, False, False, LOG)) == EXIT_OK


def test_a_failed_run_exits_with_failure():
    assert exit_code_for(RunOutcome(3, False, False, LOG)) == EXIT_FAILED
    assert exit_code_for(RunOutcome(9, True, False, LOG)) == EXIT_FAILED


def test_a_stopped_run_has_its_own_exit_code():
    assert exit_code_for(RunOutcome(143, False, True, LOG)) == EXIT_STOPPED


def test_a_run_that_finished_while_stopping_counts_as_completed():
    assert exit_code_for(RunOutcome(0, False, True, LOG)) == EXIT_OK


def test_the_exit_codes_are_all_different():
    assert len({EXIT_OK, EXIT_FAILED, EXIT_NOT_STARTED, EXIT_STOPPED}) == 4


def test_no_runtime_problem_gives_no_lines():
    assert runtime_problem_lines(()) == ()


def test_runtime_problems_become_errors_and_point_to_the_graphical_app():
    lines = runtime_problem_lines(("Training script not found: x",))
    assert lines[0] == "ERROR: Training script not found: x"
    assert "graphical app" in lines[-1]


def test_settings_without_warnings_give_no_lines():
    assert settings_problem_lines(()) == ()


def test_settings_warnings_become_a_blocker_that_points_to_the_graphical_app():
    lines = settings_problem_lines(("learning_rate invalid",))
    assert lines[0] == "WARN: learning_rate invalid"
    assert lines[1].startswith("ERROR: ")
    assert "graphical app" in lines[1]
