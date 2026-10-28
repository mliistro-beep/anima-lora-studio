"""Shared setup of the E2E tests: an installed data folder and the values the fakes use."""

import json
from datetime import datetime
from pathlib import Path

from anima_trainer.runtime.data_dir import open_data_dir
from anima_trainer.runtime.install_plan import InstallStep
from anima_trainer.runtime.paths import RuntimePaths
from tests.runtime_helpers import mark_installed

NOW = datetime(2026, 10, 20, 9, 0, 0)
TIMEOUT_MS = 5000
GPUS = ("0: NVIDIA A", "1: NVIDIA B")
INSTALL_STEPS = (InstallStep("venv", "Create the training venv"),)
PROJECT = "saved_proj"  # project name make_runtime saves unless the config changes it


def make_runtime(
    root: Path, dataset: Path, config: dict[str, object] | None = None
) -> RuntimePaths:
    """A data folder with the runtime marked installed, empty model files and a saved
    config.json for `dataset` (output in <root>/out); `config` overrides its values."""
    opened = open_data_dir(root)
    paths = opened.paths
    for file in (
        paths.train_script,
        paths.venv_accelerate,
        paths.accelerate_config_file,
        paths.qwen3_model,
        paths.vae_model,
        paths.dit_model("anima-base-v1.0"),
    ):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"")
    mark_installed(paths)
    saved = {
        "project_name": PROJECT,
        "image_directory": str(dataset),
        "output_directory": str(root / "out"),
        "network_dim": 16,
        "gpu_index": "1",
        "cache_latents": False,
        **(config or {}),
    }
    paths.config_file.write_text(json.dumps(saved))
    return paths
