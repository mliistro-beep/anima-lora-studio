import pytest

from anima_trainer.runtime.accelerate_config import ACCELERATE_GPU_YAML, write_accelerate_config
from anima_trainer.runtime.install_plan import (
    NO_GIT_MESSAGE,
    SD_SCRIPTS_COMMIT,
    SD_SCRIPTS_URL,
    plan_install,
    requirements_marker,
    sd_scripts_marker,
    verified_marker,
)
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.runtime.python_finder import NO_PYTHON_MESSAGE, PythonInterpreter

PYTHON = PythonInterpreter("/usr/bin/python3.12", (3, 12))
GIT = "/usr/bin/git"
ALL_STEPS = [
    "data_dir",
    "venv",
    "pip",
    "sd_scripts_init",
    "sd_scripts_fetch",
    "sd_scripts_checkout",
    "requirements",
    "accelerate_config",
    "verify",
]


def _touch(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _install_everything(paths):
    _touch(paths.venv_python)
    _touch(paths.sd_scripts_dir / ".git" / "HEAD")
    _touch(paths.train_script)
    _touch(sd_scripts_marker(paths), SD_SCRIPTS_COMMIT)
    _touch(requirements_marker(paths), SD_SCRIPTS_COMMIT)
    _touch(verified_marker(paths), SD_SCRIPTS_COMMIT)
    write_accelerate_config(paths)


@pytest.fixture
def paths(tmp_path):
    return RuntimePaths(tmp_path / "data", platform="posix")


def _keys(plan):
    return [step.key for step in plan.steps]


def test_fresh_folder_needs_every_step_in_order(paths):
    plan = plan_install(paths, PYTHON, GIT)

    assert _keys(plan) == ALL_STEPS
    assert plan.problems == ()
    assert plan.ready


def test_complete_install_needs_nothing(paths):
    _install_everything(paths)

    plan = plan_install(paths, None, None)

    assert plan.steps == ()
    assert plan.ready


def test_existing_venv_is_kept(paths):
    _touch(paths.venv_python)

    keys = _keys(plan_install(paths, PYTHON, GIT))

    assert "venv" not in keys and "pip" not in keys and "data_dir" not in keys


def test_interrupted_clone_resumes_without_init(paths):
    _touch(paths.venv_python)
    _touch(paths.sd_scripts_dir / ".git" / "HEAD")

    keys = _keys(plan_install(paths, PYTHON, GIT))

    assert keys[:2] == ["sd_scripts_fetch", "sd_scripts_checkout"]


def test_requirements_rerun_when_marker_is_for_another_commit(paths):
    _install_everything(paths)
    _touch(requirements_marker(paths), "0123abc")

    assert _keys(plan_install(paths, PYTHON, GIT)) == ["requirements", "verify"]


def test_edited_accelerate_config_is_rewritten(paths):
    _install_everything(paths)
    paths.accelerate_config_file.write_text("mixed_precision: fp16\n")

    assert _keys(plan_install(paths, PYTHON, GIT)) == ["accelerate_config", "verify"]


def test_missing_python_blocks_only_when_the_venv_is_needed(paths):
    fresh = plan_install(paths, None, GIT)
    _touch(paths.venv_python)
    with_venv = plan_install(paths, None, GIT)

    assert fresh.problems == (NO_PYTHON_MESSAGE,)
    assert not fresh.ready
    assert with_venv.problems == ()


def test_missing_git_blocks_only_when_sd_scripts_is_needed(paths):
    fresh = plan_install(paths, PYTHON, None)
    _touch(paths.train_script)
    _touch(sd_scripts_marker(paths), SD_SCRIPTS_COMMIT)
    with_scripts = plan_install(paths, PYTHON, None)

    assert fresh.problems == (NO_GIT_MESSAGE,)
    assert with_scripts.problems == ()


def test_commands_use_the_found_python_and_the_venv(paths):
    steps = {step.key: step for step in plan_install(paths, PYTHON, GIT).steps}

    assert steps["venv"].command == (PYTHON.path, "-m", "venv", str(paths.venv_dir))
    assert steps["pip"].command[:3] == (str(paths.venv_python), "-m", "pip")
    assert steps["verify"].command[0] == str(paths.venv_python)
    assert steps["verify"].working_dir == paths.root


def test_sd_scripts_is_fetched_at_the_pinned_commit(paths):
    steps = {step.key: step for step in plan_install(paths, PYTHON, GIT).steps}

    assert steps["sd_scripts_init"].command == (GIT, "init", str(paths.sd_scripts_dir))
    fetch = steps["sd_scripts_fetch"].command
    assert fetch[0] == GIT and SD_SCRIPTS_URL in fetch and SD_SCRIPTS_COMMIT in fetch
    assert steps["sd_scripts_checkout"].command[-1] == "FETCH_HEAD"


def test_requirements_run_inside_sd_scripts_and_leave_a_marker(paths):
    step = next(s for s in plan_install(paths, PYTHON, GIT).steps if s.key == "requirements")

    assert step.command[-2:] == ("-r", "requirements.txt")
    assert step.working_dir == paths.sd_scripts_dir
    assert step.marker == requirements_marker(paths)


def test_checkout_and_verify_leave_markers(paths):
    steps = {step.key: step for step in plan_install(paths, PYTHON, GIT).steps}

    assert steps["sd_scripts_checkout"].marker == sd_scripts_marker(paths)
    assert steps["verify"].marker == verified_marker(paths)
    assert steps["sd_scripts_fetch"].marker is None


def test_in_process_steps_create_the_folder_and_the_config(paths):
    steps = {step.key: step for step in plan_install(paths, PYTHON, GIT).steps}

    steps["data_dir"].run_action()
    steps["accelerate_config"].run_action()

    assert paths.root.is_dir()
    assert paths.accelerate_config_file.read_text(encoding="utf-8") == ACCELERATE_GPU_YAML
    assert steps["data_dir"].command == ()


def test_every_step_has_a_title(paths):
    assert all(step.title for step in plan_install(paths, PYTHON, GIT).steps)


def test_unreadable_marker_and_config_are_redone_instead_of_crashing(paths):
    _install_everything(paths)
    requirements_marker(paths).write_bytes(b"\xff\xfe")
    paths.accelerate_config_file.write_bytes(b"\xff\xfe")

    assert _keys(plan_install(paths, PYTHON, GIT)) == [
        "requirements",
        "accelerate_config",
        "verify",
    ]


def test_command_steps_have_no_in_process_action(paths):
    step = next(s for s in plan_install(paths, PYTHON, GIT).steps if s.key == "venv")

    step.run_action()

    assert step.action is None
    assert not paths.venv_dir.exists()


def test_failed_or_interrupted_gpu_check_is_planned_again(paths):
    _install_everything(paths)
    verified_marker(paths).unlink()

    assert _keys(plan_install(paths, PYTHON, GIT)) == ["verify"]


def test_new_pinned_commit_refetches_sd_scripts_then_reinstalls(paths):
    _install_everything(paths)
    for marker in (sd_scripts_marker(paths), requirements_marker(paths), verified_marker(paths)):
        marker.write_text("0123abc")

    keys = _keys(plan_install(paths, PYTHON, GIT))

    assert keys == ["sd_scripts_fetch", "sd_scripts_checkout", "requirements", "verify"]


def test_interrupted_checkout_is_redone_even_if_the_script_exists(paths):
    _install_everything(paths)
    sd_scripts_marker(paths).unlink()

    keys = _keys(plan_install(paths, PYTHON, GIT))

    assert keys[:3] == ["sd_scripts_fetch", "sd_scripts_checkout", "requirements"]


def test_markers_that_cannot_be_read_count_as_missing(paths):
    _install_everything(paths)
    requirements_marker(paths).unlink()
    requirements_marker(paths).mkdir()
    paths.accelerate_config_file.unlink()
    paths.accelerate_config_file.mkdir()

    assert _keys(plan_install(paths, PYTHON, GIT)) == [
        "requirements",
        "accelerate_config",
        "verify",
    ]


def test_markers_live_in_the_install_state_folder(paths):
    folder = paths.root / ".install"

    assert {
        m.parent
        for m in (sd_scripts_marker(paths), requirements_marker(paths), verified_marker(paths))
    } == {folder}
