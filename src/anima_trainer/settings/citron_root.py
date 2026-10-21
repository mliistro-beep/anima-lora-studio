"""Where the Citron installation lives, stored with QSettings.

config.json lives inside the Citron root, so the root itself needs app-level storage.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSettings

from anima_trainer.models.paths import CitronPaths
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.settings.store import load_settings

ROOT_KEY = "citron/root"
# Where Citron's own setup instructions clone it.
DEFAULT_ROOT_CANDIDATES = (Path.home() / "citron-anima-lora-trainer-ui",)


def load_citron_root(
    store: QSettings, candidates: Sequence[Path] = DEFAULT_ROOT_CANDIDATES
) -> Path | None:
    """The saved root, else the first existing candidate, else None."""
    saved = store.value(ROOT_KEY, "")
    if isinstance(saved, str) and saved.strip():
        return Path(saved.strip())
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def save_citron_root(store: QSettings, root: Path) -> None:
    """Raises OSError when the settings file cannot be written (e.g. a read-only folder)."""
    store.setValue(ROOT_KEY, str(root))
    store.sync()
    if store.status() != QSettings.Status.NoError:
        raise OSError(f"The Citron folder could not be saved to {store.fileName()}")


def citron_root_problems(paths: CitronPaths) -> tuple[str, ...]:
    """Why this folder cannot be used to train; empty when it looks like a Citron install."""
    if not paths.root.is_dir():
        return (f"Citron folder does not exist: {paths.root}",)
    checks = (
        (paths.train_script, "Training script not found"),
        (paths.venv_accelerate, "accelerate not found in the Citron venv"),
    )
    return tuple(f"{message}: {path}" for path, message in checks if not path.exists())


@dataclass(frozen=True)
class OpenedCitron:
    paths: CitronPaths
    settings: TrainingSettings
    warnings: tuple[str, ...]  # config.json problems
    problems: tuple[str, ...]  # install problems that prevent training


def open_citron_root(root: Path) -> OpenedCitron:
    """Paths, saved settings (defaults if the folder is missing) and problems of a root."""
    paths = CitronPaths(root.expanduser())
    loaded = load_settings(paths.config_file)
    return OpenedCitron(paths, loaded.settings, loaded.warnings, citron_root_problems(paths))
