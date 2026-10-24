"""Which model files of the data folder are present, for the Settings tab."""

from dataclasses import dataclass
from pathlib import Path

from anima_trainer.models.catalog import QWEN3_FILENAME, VAE_FILENAME
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.choices import BASE_MODELS


@dataclass(frozen=True)
class ModelStatus:
    label: str
    path: Path
    present: bool
    download_key: str  # for model_download(): the base model name or the file name


def model_statuses(paths: RuntimePaths) -> tuple[ModelStatus, ...]:
    support = (("Qwen3", paths.qwen3_model, QWEN3_FILENAME), ("VAE", paths.vae_model, VAE_FILENAME))
    required = tuple(ModelStatus(label, path, path.exists(), key) for label, path, key in support)
    dits = tuple(
        ModelStatus(f"DiT {name}", paths.dit_model(name), paths.dit_model(name).exists(), name)
        for name in BASE_MODELS
    )
    return (*required, *dits)
