"""The progress line of a generation, the same in the window and in the terminal:
`img 03/10 | batch 2/4 | step 12/25`. A batch is an entry of the queue; the form is one."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageProgress:
    """The image being made: its place among all the images, and its batch."""

    image: int
    images: int
    batch: int
    batches: int


def progress_line(progress: ImageProgress, done: int, total: int) -> str:
    parts = [
        f"img {progress.image:02d}/{progress.images:02d}",
        f"batch {progress.batch}/{progress.batches}",
    ]
    if total:
        parts.append(f"step {done}/{total}")
    return " | ".join(parts)
