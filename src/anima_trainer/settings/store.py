"""Load and save settings in Citron's config.json format.

Unlike Citron, problems are reported as warnings instead of being silently swallowed:
an unreadable file falls back to defaults, an invalid value falls back to that field's default.
"""

from dataclasses import dataclass
from pathlib import Path

from anima_trainer.settings.json_store import load_model, save_model
from anima_trainer.settings.schema import TrainingSettings


@dataclass(frozen=True)
class LoadResult:
    settings: TrainingSettings
    warnings: tuple[str, ...]


def load_settings(path: Path) -> LoadResult:
    loaded = load_model(path, TrainingSettings)
    return LoadResult(loaded.model, loaded.warnings)


def save_settings(path: Path, settings: TrainingSettings) -> None:
    """Write config.json atomically, formatted exactly like Citron's json.dump(indent=2)."""
    save_model(path, settings)
