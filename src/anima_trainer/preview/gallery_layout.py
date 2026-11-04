"""Preview images arranged for the gallery: one row per epoch, one column per prompt."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from anima_trainer.preview.sample_files import SampleImage

MAX_LABEL_CHARS = 40


@dataclass(frozen=True)
class GalleryLayout:
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    cells: Mapping[tuple[int, int], Path]  # (row, column) -> image


def _prompt_label(index: int, prompts: tuple[str, ...]) -> str:
    if index >= len(prompts):
        return f"Prompt {index + 1}"
    text = prompts[index]
    if len(text) > MAX_LABEL_CHARS:
        text = text[:MAX_LABEL_CHARS].rsplit(" ", 1)[0] + "..."
    return f"{index + 1}: {text}"


def _epoch_label(epoch: int) -> str:
    return "Before training" if epoch == 0 else f"Epoch {epoch}"


def preview_layout(samples: tuple[SampleImage, ...], prompts: tuple[str, ...]) -> GalleryLayout:
    epochs = sorted({sample.epoch for sample in samples})
    column_count = max([len(prompts), *(sample.prompt_index + 1 for sample in samples)])
    row_of = {epoch: row for row, epoch in enumerate(epochs)}
    return GalleryLayout(
        rows=tuple(_epoch_label(epoch) for epoch in epochs),
        columns=tuple(_prompt_label(index, prompts) for index in range(column_count)),
        cells={(row_of[sample.epoch], sample.prompt_index): sample.path for sample in samples},
    )
