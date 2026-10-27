"""LoRA files of a project in its output folder: sd-scripts' epoch checkpoints
({output_name}-{epoch:06d}.safetensors) and the final {output_name}.safetensors."""

import re
from dataclasses import dataclass
from pathlib import Path

SUFFIX = ".safetensors"


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    epoch: int | None  # None for the final LoRA

    @property
    def label(self) -> str:
        return "Final" if self.epoch is None else f"Epoch {self.epoch}"


def list_checkpoints(output_dir: Path, project: str) -> tuple[Checkpoint, ...]:
    """Epoch checkpoints in epoch order, then the final LoRA."""
    try:
        entries = tuple(entry for entry in output_dir.iterdir() if entry.is_file())
    except OSError:
        return ()
    epoch_name = re.compile(rf"^{re.escape(project)}-(\d{{6}}){re.escape(SUFFIX)}$")
    epochs = sorted(
        (
            Checkpoint(entry, int(match[1]))
            for entry in entries
            if (match := epoch_name.match(entry.name))
        ),
        key=lambda checkpoint: checkpoint.epoch or 0,
    )
    finals = tuple(Checkpoint(entry, None) for entry in entries if entry.name == project + SUFFIX)
    return (*epochs, *finals)
