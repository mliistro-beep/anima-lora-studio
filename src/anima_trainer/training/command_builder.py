"""Builds the `accelerate launch` invocation used by Citron to run sd-scripts training."""

import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.runtime_scripts import TRAIN_WITH_PREVIEWS

# Where train_with_previews.py finds sd-scripts' `library` and the training script.
SD_SCRIPTS_ENV = "ANIMA_TRAINER_SD_SCRIPTS"


@dataclass(frozen=True)
class LaunchSpec:
    program: Path
    arguments: tuple[str, ...]
    env: Mapping[str, str]
    working_dir: Path

    def display(self) -> str:
        return " ".join(shlex.quote(part) for part in (str(self.program), *self.arguments))


def build_training_launch(
    paths: RuntimePaths,
    training_config: Path,
    dataset_config: Path,
    gpu_index: str,
    cpu_threads: int,
    with_previews: bool = False,
) -> LaunchSpec:
    """Same arguments and environment as Citron's start_training(), using the training venv.

    With previews the app's launcher runs in place of the training script (same arguments):
    it adds ER-SDE previews and skips a preview that runs out of GPU memory.
    """
    script = TRAIN_WITH_PREVIEWS if with_previews else paths.train_script
    arguments = (
        "launch",
        "--config_file", str(paths.accelerate_config),
        "--num_cpu_threads_per_process", str(max(cpu_threads, 1)),
        "--gpu_ids", gpu_index,
        str(script),
        "--config_file", str(training_config),
        "--dataset_config", str(dataset_config),
    )  # fmt: skip
    env = {"CUDA_VISIBLE_DEVICES": gpu_index, "PYTHONUNBUFFERED": "1"}
    if with_previews:
        env = {**env, SD_SCRIPTS_ENV: str(paths.sd_scripts_dir)}
    return LaunchSpec(paths.venv_accelerate, arguments, MappingProxyType(env), paths.root)
