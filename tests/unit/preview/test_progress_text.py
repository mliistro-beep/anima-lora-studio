"""The progress line of a generation, the same in the window and in the terminal."""

from anima_trainer.preview.progress_text import ImageProgress, progress_line


def test_the_line_has_the_image_the_batch_and_the_steps():
    progress = ImageProgress(image=3, images=10, batch=2, batches=4)
    assert progress_line(progress, 12, 25) == "img 03/10 | batch 2/4 | step 12/25"


def test_before_the_first_step_the_steps_are_left_out():
    progress = ImageProgress(image=1, images=1, batch=1, batches=1)
    assert progress_line(progress, 0, 0) == "img 01/01 | batch 1/1"
