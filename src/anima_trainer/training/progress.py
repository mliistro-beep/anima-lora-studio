"""Training progress folded from parser events, for the progress panel."""

from dataclasses import dataclass, replace
from pathlib import Path

from anima_trainer.training.log_parser import (
    CheckpointSaved,
    EpochStarted,
    ModelSaved,
    StepProgress,
    TrainingEvent,
)

UNKNOWN_DURATION = "--:--"
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class ProgressState:
    epoch: int | None = None
    total_epochs: int | None = None
    step: int = 0
    total_steps: int | None = None
    loss: float | None = None
    elapsed_s: int | None = None
    remaining_s: int | None = None
    last_checkpoint: Path | None = None
    completed: bool = False


def apply_event(state: ProgressState, event: TrainingEvent) -> ProgressState:
    """Return the state after `event`; the given state is left unchanged."""
    if isinstance(event, EpochStarted):
        return replace(state, epoch=event.epoch, total_epochs=event.total)
    if isinstance(event, StepProgress):
        return replace(
            state,
            step=event.step,
            total_steps=event.total,
            elapsed_s=event.elapsed_s,
            remaining_s=event.remaining_s,
            # The first tqdm updates carry no loss yet.
            loss=event.loss if event.loss is not None else state.loss,
        )
    if isinstance(event, CheckpointSaved):
        return replace(state, last_checkpoint=event.path)
    if isinstance(event, ModelSaved):
        return replace(state, completed=True)
    return state


def format_duration(seconds: int | None) -> str:
    """tqdm style: MM:SS, or H:MM:SS from one hour on."""
    if seconds is None:
        return UNKNOWN_DURATION
    hours, rest = divmod(seconds, SECONDS_PER_HOUR)
    minutes, secs = divmod(rest, SECONDS_PER_MINUTE)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
