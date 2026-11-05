"""What the TUI proposes the next time, in a file of its own: config.json belongs to the window.

Epochs and repeats are not here: they depend on the dataset and are chosen for each training.
"""

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from anima_trainer.settings.json_store import load_model, save_model


class TuiValues(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    project: str = ""
    dataset: str = ""
    output: str = ""
    preset: str = ""


@dataclass(frozen=True)
class LoadedTuiValues:
    values: TuiValues
    warnings: tuple[str, ...]


def load_tui_values(path: Path) -> LoadedTuiValues:
    loaded = load_model(path, TuiValues)
    return LoadedTuiValues(loaded.model, loaded.warnings)


def save_tui_values(path: Path, values: TuiValues) -> None:
    """Raises OSError when the file cannot be written."""
    save_model(path, values)
