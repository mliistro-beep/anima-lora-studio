"""Dataset folder check, ported from Citron's validate_dataset()."""

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})
CAPTION_EXTENSION = ".txt"


class DatasetError(Exception):
    """The dataset path does not exist or is not a directory."""


@dataclass(frozen=True)
class DatasetReport:
    image_count: int
    missing_captions: tuple[str, ...]
    warnings: tuple[str, ...]


def validate_dataset(image_dir: Path) -> DatasetReport:
    """Count images in a flat folder and list those without a matching .txt caption."""
    if not image_dir.exists():
        raise DatasetError(f"Directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise DatasetError(f"Not a directory: {image_dir}")

    files = [entry for entry in image_dir.iterdir() if entry.is_file()]
    images = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
    caption_stems = {f.stem for f in files if f.suffix.lower() == CAPTION_EXTENSION}
    missing = tuple(sorted(f.name for f in images if f.stem not in caption_stems))
    return DatasetReport(len(images), missing, _warnings(len(images), len(missing)))


def _warnings(image_count: int, missing_count: int) -> tuple[str, ...]:
    warnings: tuple[str, ...] = ()
    if image_count == 0:
        warnings = (*warnings, "No image files found in directory.")
    if missing_count:
        warnings = (*warnings, f"{missing_count} image(s) are missing caption (.txt) files.")
    return warnings
