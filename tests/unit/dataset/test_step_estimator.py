import pytest

from anima_trainer.dataset.step_estimator import estimate_steps


def test_matches_real_citron_run():
    # 60 images x 10 repeats, batch 1, 10 epochs -> 6000 steps (see fixture log).
    estimate = estimate_steps(image_count=60, repeats=10, batch_size=1, grad_accum=1, epochs=10)

    assert estimate.steps_per_epoch == 600
    assert estimate.total_steps == 6000


def test_rounds_steps_per_epoch_up():
    estimate = estimate_steps(image_count=7, repeats=3, batch_size=2, grad_accum=2, epochs=4)

    assert estimate.steps_per_epoch == 6
    assert estimate.total_steps == 24


@pytest.mark.parametrize(("batch_size", "grad_accum"), [(0, 1), (1, 0), (-3, -1)])
def test_non_positive_batch_and_accum_are_treated_as_one(batch_size, grad_accum):
    estimate = estimate_steps(
        image_count=5, repeats=2, batch_size=batch_size, grad_accum=grad_accum, epochs=1
    )

    assert estimate.steps_per_epoch == 10


def test_zero_images_gives_zero_steps():
    estimate = estimate_steps(image_count=0, repeats=10, batch_size=1, grad_accum=1, epochs=10)

    assert estimate.total_steps == 0
