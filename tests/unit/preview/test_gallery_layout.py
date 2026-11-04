from datetime import datetime
from pathlib import Path

from anima_trainer.preview.gallery_layout import GalleryLayout, preview_layout
from anima_trainer.preview.sample_files import SampleImage

TIME = datetime(2026, 10, 27, 10, 0, 0)


def _sample(epoch, index):
    return SampleImage(Path(f"/s/e{epoch}_{index}.png"), epoch, index, TIME, 42)


def test_rows_are_epochs_and_columns_are_prompts():
    samples = (_sample(0, 0), _sample(1, 0), _sample(1, 1), _sample(3, 1))

    layout = preview_layout(samples, ("a girl", "a cat"))

    assert layout == GalleryLayout(
        rows=("Before training", "Epoch 1", "Epoch 3"),
        columns=("1: a girl", "2: a cat"),
        cells={(0, 0): samples[0].path, (1, 0): samples[1].path, (1, 1): samples[2].path,
               (2, 1): samples[3].path},
    )  # fmt: skip


def test_long_prompts_are_shortened_and_unknown_prompts_get_a_number():
    long_prompt = "masterpiece, best quality, a girl with long silver hair in a garden"

    layout = preview_layout((_sample(1, 0), _sample(1, 2)), (long_prompt,))

    assert layout.columns[0] == "1: masterpiece, best quality, a girl with..."
    assert layout.columns[1:] == ("Prompt 2", "Prompt 3")
    assert set(layout.cells) == {(0, 0), (0, 2)}


def test_no_samples_give_an_empty_layout():
    assert preview_layout((), ("a",)) == GalleryLayout((), ("1: a",), {})
