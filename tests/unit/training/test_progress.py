from pathlib import Path

import pytest

from anima_trainer.training.log_parser import (
    CheckpointSaved,
    EpochStarted,
    ModelSaved,
    SampleGenerationStarted,
    StepProgress,
)
from anima_trainer.training.progress import ProgressState, apply_event, format_duration


def test_initial_state_is_empty():
    state = ProgressState()

    assert (state.epoch, state.total_epochs, state.step, state.total_steps) == (None, None, 0, None)
    assert state.loss is None and not state.completed


def test_epoch_started_sets_epoch_and_total():
    state = apply_event(ProgressState(), EpochStarted(3, 10))

    assert (state.epoch, state.total_epochs) == (3, 10)


def test_step_progress_updates_steps_times_and_loss():
    state = apply_event(ProgressState(), StepProgress(120, 6000, 65, 3600, 0.25))

    assert (state.step, state.total_steps) == (120, 6000)
    assert (state.elapsed_s, state.remaining_s, state.loss) == (65, 3600, 0.25)


def test_step_without_loss_keeps_previous_loss():
    first = apply_event(ProgressState(), StepProgress(1, 10, 1, 9, 0.5))

    second = apply_event(first, StepProgress(2, 10, 2, 8, None))

    assert second.loss == 0.5
    assert second.step == 2


def test_apply_event_does_not_mutate_the_previous_state():
    before = ProgressState()

    apply_event(before, EpochStarted(1, 2))

    assert before == ProgressState()


def test_checkpoint_saved_records_path():
    state = apply_event(ProgressState(), CheckpointSaved(Path("/out/x-000001.safetensors"), 1))

    assert state.last_checkpoint == Path("/out/x-000001.safetensors")


def test_model_saved_marks_completed():
    assert apply_event(ProgressState(), ModelSaved()).completed


def test_sample_generation_leaves_state_unchanged():
    state = ProgressState(step=5)

    assert apply_event(state, SampleGenerationStarted(5)) == state


@pytest.mark.parametrize(
    ("step", "total", "expected"), [(0, None, 0.0), (50, 200, 0.25), (300, 200, 1.0), (1, 0, 0.0)]
)
def test_fraction_done(step, total, expected):
    assert ProgressState(step=step, total_steps=total).fraction_done() == expected


@pytest.mark.parametrize(
    ("seconds", "expected"), [(None, "--:--"), (0, "00:00"), (125, "02:05"), (3725, "1:02:05")]
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected
