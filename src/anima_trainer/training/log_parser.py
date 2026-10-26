"""Turns sd-scripts output lines into typed training events.

The format is free text (tqdm bars, rich logging), so parsing is deliberately tolerant:
unrecognized lines simply produce no event and remain visible in the raw log.
"""

import re
from dataclasses import dataclass
from pathlib import Path

SECONDS_PER_UNIT = (3600, 60, 1)


@dataclass(frozen=True)
class EpochStarted:
    epoch: int
    total: int


@dataclass(frozen=True)
class StepProgress:
    step: int
    total: int
    elapsed_s: int | None
    remaining_s: int | None
    loss: float | None


@dataclass(frozen=True)
class CheckpointSaved:
    path: Path
    epoch: int | None


@dataclass(frozen=True)
class SampleGenerationStarted:
    step: int


@dataclass(frozen=True)
class ModelSaved:
    pass


TrainingEvent = EpochStarted | StepProgress | CheckpointSaved | SampleGenerationStarted | ModelSaved

_EPOCH = re.compile(r"^epoch (\d+)/(\d+)$")
_STEPS = re.compile(
    r"^steps:.*?\|\s*(\d+)/(\d+)\s*\[([^<\]]+)<([^,\]]+)(?:,[^\]]*?avr_loss=([0-9.eE+-]+))?"
)
_CHECKPOINT = re.compile(r"^saving checkpoint: (.+)$")
_CHECKPOINT_EPOCH = re.compile(r"-(\d{6})$")
_SAMPLE = re.compile(r"Generating sample images at step (\d+)")
_MODEL_SAVED = re.compile(r"\bINFO\s+model saved\.")


def parse_duration(text: str) -> int | None:
    """tqdm "MM:SS" or "H:MM:SS" to seconds; None for "?" or anything unexpected."""
    parts = text.strip().split(":")
    if not 2 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        return None
    units = SECONDS_PER_UNIT[-len(parts) :]
    return sum(int(part) * unit for part, unit in zip(parts, units, strict=True))


def parse_line(line: str) -> TrainingEvent | None:
    # tqdm redraws with carriage returns: only the last update in a chunk matters.
    text = line.split("\r")[-1].strip()
    if not text:
        return None
    return (
        _parse_epoch(text)
        or _parse_steps(text)
        or _parse_checkpoint(text)
        or _parse_sample(text)
        or _parse_model_saved(text)
    )


def _parse_epoch(text: str) -> EpochStarted | None:
    match = _EPOCH.match(text)
    return EpochStarted(int(match[1]), int(match[2])) if match else None


def _parse_steps(text: str) -> StepProgress | None:
    match = _STEPS.match(text)
    if not match:
        return None
    loss = float(match[5]) if match[5] else None
    return StepProgress(
        int(match[1]), int(match[2]), parse_duration(match[3]), parse_duration(match[4]), loss
    )


def _parse_checkpoint(text: str) -> CheckpointSaved | None:
    match = _CHECKPOINT.match(text)
    if not match:
        return None
    path = Path(match[1].strip())
    epoch_match = _CHECKPOINT_EPOCH.search(path.stem)
    return CheckpointSaved(path, int(epoch_match[1]) if epoch_match else None)


def _parse_sample(text: str) -> SampleGenerationStarted | None:
    match = _SAMPLE.search(text)
    return SampleGenerationStarted(int(match[1])) if match else None


def _parse_model_saved(text: str) -> ModelSaved | None:
    return ModelSaved() if _MODEL_SAVED.search(text) else None
