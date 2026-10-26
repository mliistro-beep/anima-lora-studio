"""Which model files are present, in the data folder or linked by the user, for the Settings tab."""

from dataclasses import dataclass
from pathlib import Path

from anima_trainer.models.catalog import QWEN3_FILENAME, VAE_FILENAME, model_label
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.choices import BASE_MODELS


@dataclass(frozen=True)
class ModelStatus:
    label: str
    path: Path
    present: bool
    download_key: str  # for model_download(): the base model name or the file name
    linked: bool = False  # the file is the user's own, checked by SHA-256
    link_broken: bool = False  # a linked file was moved or changed since it was checked


def model_statuses(paths: RuntimePaths) -> tuple[ModelStatus, ...]:
    support = ((paths.qwen3_model, QWEN3_FILENAME), (paths.vae_model, VAE_FILENAME))
    dits = tuple((paths.dit_model(name), name) for name in BASE_MODELS)
    return tuple(
        ModelStatus(
            model_label(key),
            path,
            path.exists(),
            key,
            paths.is_linked(key),
            paths.has_broken_link(key),
        )
        for path, key in (*support, *dits)
    )
