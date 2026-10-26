"""Where the app's data folder lives, stored with QSettings.

config.json lives inside the data folder, so the folder itself needs app-level storage.
Citron's own installation is never used: a fresh install must work without it.
"""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths

from anima_trainer.runtime.install_plan import install_pending
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.settings.store import load_settings

DATA_DIR_KEY = "runtime/data_dir"
DATA_DIR_NAME = "anima-trainer"
REPAIR_MESSAGE = "Training runtime incomplete or out of date: press Repair in Settings"


def default_data_dir() -> Path:
    """~/.local/share/anima-trainer on Linux (honours XDG_DATA_HOME)."""
    location = QStandardPaths.StandardLocation.GenericDataLocation
    return Path(QStandardPaths.writableLocation(location)) / DATA_DIR_NAME


def load_data_dir(store: QSettings, default: Path) -> Path:
    """The saved data folder, else `default`."""
    saved = store.value(DATA_DIR_KEY, "")
    if isinstance(saved, str) and saved.strip():
        return Path(saved.strip())
    return default


def save_data_dir(store: QSettings, data_dir: Path) -> None:
    """Raises OSError when the settings file cannot be written (e.g. a read-only folder)."""
    store.setValue(DATA_DIR_KEY, str(data_dir))
    store.sync()
    if store.status() != QSettings.Status.NoError:
        raise OSError(f"The data folder could not be saved to {store.fileName()}")


def resolve_data_dir(data_dir: Path) -> Path:
    """The absolute data folder: install commands run inside it, so a relative one would nest."""
    return data_dir.expanduser().resolve()


def data_dir_problem(data_dir: Path) -> str | None:
    """Why `data_dir` cannot be a data folder: None for a folder or a path that can be created."""
    # The nearest part that exists decides: the rest of the path can be created under it.
    existing = next((part for part in (data_dir, *data_dir.parents) if part.exists()), None)
    if existing is None or existing.is_dir():
        return None
    return f"Not a folder: {existing}"


def runtime_problems(paths: RuntimePaths) -> tuple[str, ...]:
    """Why this folder cannot be used to train yet; empty when the runtime is installed."""
    if not paths.root.is_dir():
        return (f"Training runtime not installed yet in {paths.root}",)
    checks = (
        (paths.train_script, "Training script not found"),
        (paths.venv_accelerate, "accelerate not found in the training venv"),
        (paths.accelerate_config_file, "accelerate config not found"),
    )
    missing = tuple(f"{message}: {path}" for path, message in checks if not path.exists())
    # A failed GPU check or a new pinned commit leaves every file in place.
    if not missing and install_pending(paths):
        return (REPAIR_MESSAGE,)
    return missing


@dataclass(frozen=True)
class OpenedRuntime:
    paths: RuntimePaths
    settings: TrainingSettings
    warnings: tuple[str, ...]  # config.json problems
    problems: tuple[str, ...]  # install problems that prevent training


def open_data_dir(data_dir: Path) -> OpenedRuntime:
    """Paths, saved settings (defaults if the folder is missing) and problems of a data folder."""
    paths = RuntimePaths(resolve_data_dir(data_dir))
    loaded = load_settings(paths.config_file)
    return OpenedRuntime(paths, loaded.settings, loaded.warnings, runtime_problems(paths))
