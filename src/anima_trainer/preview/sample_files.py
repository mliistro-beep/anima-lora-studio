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
    """Images of `project`, one per (epoch, prompt): the newest, and none older than `since`."""
    try:
        entries = tuple(folder.iterdir())
    except OSError:
        return ()
    pattern = _pattern(project)
    parsed = (_parse(entry, pattern) for entry in entries)
    images = [
        image for image in parsed if image is not None and (since is None or image.created >= since)
    ]
    # Oldest first, so a later run's image of the same slot wins.
    newest = {
        (image.epoch, image.prompt_index): image
        for image in sorted(images, key=lambda item: item.created)
    }
    return tuple(newest[key] for key in sorted(newest))
