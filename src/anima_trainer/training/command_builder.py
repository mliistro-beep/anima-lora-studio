"""Builds the `accelerate launch` invocation used by Citron to run sd-scripts training."""

import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from anima_trainer.runtime.paths import RuntimePaths


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
) -> LaunchSpec:
    """Same arguments and environment as Citron's start_training(), using the training venv."""
    arguments = (
        "launch",
        "--config_file", str(paths.accelerate_config),
        "--num_cpu_threads_per_process", str(max(cpu_threads, 1)),
        "--gpu_ids", gpu_index,
        str(paths.train_script),
        "--config_file", str(training_config),
        "--dataset_config", str(dataset_config),
    )  # fmt: skip
    env = MappingProxyType({"CUDA_VISIBLE_DEVICES": gpu_index, "PYTHONUNBUFFERED": "1"})
    return LaunchSpec(paths.venv_accelerate, arguments, env, paths.root)
