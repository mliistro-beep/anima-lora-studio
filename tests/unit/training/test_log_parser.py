from pathlib import Path

import pytest

from anima_trainer.training.log_parser import (
    CheckpointSaved,
    EpochStarted,
    ModelSaved,
    SampleGenerationStarted,
    StepProgress,
    parse_duration,
    parse_line,
    parse_output,
)


def test_epoch_line():
    assert parse_line("epoch 3/10") == EpochStarted(epoch=3, total=10)


def test_step_line_with_loss():
    line = "steps:  10%|#         | 600/6000 [22:53<3:26:03,  2.29s/it, avr_loss=0.115]"

    assert parse_line(line) == StepProgress(
        step=600, total=6000, elapsed_s=1373, remaining_s=12363, loss=0.115
    )


def test_step_line_without_loss():
    event = parse_line("steps:   0%|          | 1/6000 [00:19<32:34:49, 19.55s/it]")

    assert event == StepProgress(step=1, total=6000, elapsed_s=19, remaining_s=117289, loss=None)


def test_step_line_with_unknown_remaining_time():
    event = parse_line("steps:   0%|          | 0/6000 [00:00<?, ?it/s]")

    assert event == StepProgress(step=0, total=6000, elapsed_s=0, remaining_s=None, loss=None)


def test_epoch_checkpoint_has_epoch_number():
    path = "/out/lora/my_lora-000002.safetensors"

    assert parse_line(f"saving checkpoint: {path}") == CheckpointSaved(path=Path(path), epoch=2)


def test_final_checkpoint_has_no_epoch_number():
    path = "/out/lora/my_lora.safetensors"

    assert parse_line(f"saving checkpoint: {path}") == CheckpointSaved(path=Path(path), epoch=None)


def test_model_saved_line_with_rich_logging_prefix():
    line = "2026-10-19 01:34:58 INFO     model saved.          train_network.py:1863"

    assert parse_line(line) == ModelSaved()


def test_sample_generation_line():
    line = "2026-10-19 01:00:00 INFO     Generating sample images at step 600   utils.py:545"

    assert parse_line(line) == SampleGenerationStarted(step=600)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        "2026-10-18 21:58:17 INFO     epoch is incremented. current_epoch: 0, epoch: 1",
        "random",
    ],
)
def test_unrelated_lines_give_no_event(line):
    assert parse_line(line) is None


def test_carriage_return_progress_uses_last_update():
    chunk = (
        "steps:   0%| | 1/10 [00:01<00:09, 1.00s/it]\rsteps:  20%| | 2/10 [00:02<00:08, 1.00s/it]"
    )

    assert parse_line(chunk).step == 2


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("00:19", 19),
        ("22:53", 1373),
        ("3:26:03", 12363),
        ("1:00:00:00", None),
        ("?", None),
        ("ab:cd", None),
    ],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


def test_real_log_excerpt_produces_expected_event_sequence(citron_fixtures):
    text = (citron_fixtures / "training_log_excerpt.log").read_text(encoding="utf-8")

    events = parse_output(text)

    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "StepProgress"
    assert kinds[1] == "EpochStarted"
    assert events[1] == EpochStarted(epoch=1, total=10)
    checkpoints = [e for e in events if isinstance(e, CheckpointSaved)]
    assert [c.epoch for c in checkpoints] == [1, None]
    assert EpochStarted(epoch=2, total=10) in events
    assert isinstance(events[-2], ModelSaved)
    assert events[-1] == StepProgress(
        step=6000, total=6000, elapsed_s=13014, remaining_s=0, loss=0.108
    )
