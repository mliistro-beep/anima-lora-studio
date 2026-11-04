"""Preview images written by sd-scripts in <output_dir>/sample/.

Names follow anima_train_utils: {output_name}_e{epoch:06d}_{prompt:02d}_{%Y%m%d%H%M%S}[_{seed}].png
(epoch 0 is the sample taken before training). Step-based samples are not used by the app.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"


@dataclass(frozen=True)
class SampleImage:
    path: Path
    epoch: int
    prompt_index: int
    created: datetime
    seed: int | None


def _pattern(project: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(project)}_e(\d{{6}})_(\d{{2,}})_(\d{{14}})(?:_(\d+))?\.png$")


def parse_sample_name(path: Path, project: str) -> SampleImage | None:
    return _parse(path, _pattern(project))


def _parse(path: Path, pattern: re.Pattern[str]) -> SampleImage | None:
    match = pattern.match(path.name)
    if match is None:
        return None
    try:
        created = datetime.strptime(match[3], TIMESTAMP_FORMAT)
    except ValueError:
        return None
    seed = int(match[4]) if match[4] is not None else None
    return SampleImage(path, int(match[1]), int(match[2]), created, seed)


def scan_samples(
    folder: Path, project: str, since: datetime | None = None
) -> tuple[SampleImage, ...]:
    """Images of `project` from its last run, none older than `since`, by epoch and prompt."""
    try:
        entries = tuple(folder.iterdir())
    except OSError:
        return ()
    pattern = _pattern(project)
    parsed = (_parse(entry, pattern) for entry in entries)
    images = [
        image for image in parsed if image is not None and (since is None or image.created >= since)
    ]
    return _last_run(images)


def _last_run(images: list[SampleImage]) -> tuple[SampleImage, ...]:
    """The newest images whose (epoch, prompt) only grow with time: a run makes them in that
    order, so an image at or past the next one's position belongs to an earlier run.

    From the images alone, so a run that failed before its first preview does not hide the last
    good one. A run whose first preview is past where a stopped run ended (epoch 2, then every
    5 epochs) cannot be told apart from it, and the two are shown together.
    """
    ordered = sorted(images, key=lambda item: (item.created, item.epoch, item.prompt_index))
    run: list[SampleImage] = []
    for image in reversed(ordered):
        if run and (image.epoch, image.prompt_index) >= (run[-1].epoch, run[-1].prompt_index):
            break
        run.append(image)
    return tuple(reversed(run))
