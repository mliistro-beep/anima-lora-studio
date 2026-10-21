"""Which model files of the data folder are present, for the Settings tab."""

from dataclasses import dataclass
from pathlib import Path

from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.choices import BASE_MODELS


@dataclass(frozen=True)
class ModelStatus:
    label: str
    path: Path
    present: bool
    base_model: str | None  # set for DiT models, which can be downloaded


def model_statuses(paths: RuntimePaths) -> tuple[ModelStatus, ...]:
    required = tuple(
        ModelStatus(label, path, path.exists(), None) for label, path in paths.required_models()
    )
    dits = tuple(
        ModelStatus(f"DiT {name}", paths.dit_model(name), paths.dit_model(name).exists(), name)
        for name in BASE_MODELS
    )
    return (*required, *dits)
