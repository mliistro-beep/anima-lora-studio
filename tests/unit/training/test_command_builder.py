from pathlib import Path

import pytest

from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.training.command_builder import build_training_launch

ROOT = Path("/home/user/citron-anima-lora-trainer-ui")
TRAIN_TOML = ROOT / "configs/x_training.toml"
DATASET_TOML = ROOT / "configs/x_dataset.toml"


def _launch(gpu_index="0", cpu_threads=1):
    return build_training_launch(
        RuntimePaths(ROOT, platform="posix"), TRAIN_TOML, DATASET_TOML, gpu_index, cpu_threads
    )


def test_arguments_match_citron_command():
    launch = _launch()

    assert launch.program == ROOT / ".venv/bin/accelerate"
    assert launch.arguments == (
        "launch",
        "--config_file",
        "app_configs/accelerate_gpu.yaml",
        "--num_cpu_threads_per_process",
        "1",
        "--gpu_ids",
        "0",
        str(ROOT / "sd-scripts/anima_train_network.py"),
        "--config_file",
        str(TRAIN_TOML),
        "--dataset_config",
        str(DATASET_TOML),
    )


def test_runs_from_the_data_dir_with_gpu_environment():
    launch = _launch(gpu_index="1")

    assert launch.working_dir == ROOT
    assert dict(launch.env) == {"CUDA_VISIBLE_DEVICES": "1", "PYTHONUNBUFFERED": "1"}


@pytest.mark.parametrize(("threads", "expected"), [(0, "1"), (-2, "1"), (4, "4")])
def test_cpu_threads_are_at_least_one(threads, expected):
    arguments = _launch(cpu_threads=threads).arguments

    assert arguments[arguments.index("--num_cpu_threads_per_process") + 1] == expected


def test_env_is_read_only():
    with pytest.raises(TypeError):
        _launch().env["X"] = "1"


def test_display_is_shell_quoted():
    launch = build_training_launch(
        RuntimePaths(Path("/opt/my citron"), platform="posix"), TRAIN_TOML, DATASET_TOML, "0", 1
    )

    assert launch.display().startswith("'/opt/my citron/.venv/bin/accelerate' launch ")


def test_previews_run_the_preview_launcher_with_the_same_arguments():
    from anima_trainer.runtime_scripts import TRAIN_WITH_PREVIEWS

    plain = _launch()
    previews = build_training_launch(
        RuntimePaths(ROOT, platform="posix"), TRAIN_TOML, DATASET_TOML, "0", 1, with_previews=True
    )

    script_index = plain.arguments.index(str(ROOT / "sd-scripts/anima_train_network.py"))
    assert previews.arguments[script_index] == str(TRAIN_WITH_PREVIEWS)
    assert previews.arguments[:script_index] == plain.arguments[:script_index]
    assert previews.arguments[script_index + 1 :] == plain.arguments[script_index + 1 :]
    assert previews.env["ANIMA_TRAINER_SD_SCRIPTS"] == str(ROOT / "sd-scripts")
    assert "ANIMA_TRAINER_SD_SCRIPTS" not in plain.env


def test_previews_leave_no_bytecode_next_to_the_scripts():
    previews = build_training_launch(
        RuntimePaths(ROOT, platform="posix"), TRAIN_TOML, DATASET_TOML, "0", 1, with_previews=True
    )

    assert previews.env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "PYTHONDONTWRITEBYTECODE" not in _launch().env  # the same environment as Citron
