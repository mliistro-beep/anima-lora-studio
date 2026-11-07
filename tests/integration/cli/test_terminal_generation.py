"""One generation started from the terminal: settings saved, the lock, the queue, its end."""

from datetime import datetime

import pytest

from anima_trainer.cli.terminal_generation import TerminalGeneration
from anima_trainer.preview.generator_settings import (
    GenerationEntry,
    GeneratorSettings,
    load_generator_settings,
)
from anima_trainer.preview.inference_command import default_generated_folder
from anima_trainer.training.runner import RunOutcome
from tests.e2e.helpers import make_runtime
from tests.fakes import FakeRunner
from tests.lock_helpers import held_by_another_process

pytestmark = pytest.mark.integration
NOW = datetime(2026, 11, 7, 12, 0, 0)


@pytest.fixture
def paths(qapp, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    return make_runtime(tmp_path / "data", images, {"gpu_index": "1"})


@pytest.fixture
def parts(paths):
    runner = FakeRunner()
    events = {"lines": [], "steps": [], "images": [], "ended": []}
    run = TerminalGeneration(
        paths,
        on_line=events["lines"].append,
        on_step=lambda done, total: events["steps"].append((done, total)),
        on_image=events["images"].append,
        on_ended=lambda ok, message: events["ended"].append((ok, message)),
        runner=runner,
        clock=lambda: NOW,
    )
    yield run, runner, events
    run.shutdown()


def _finish(runner, exit_code=0, stopped=False):
    runner.finished.emit(RunOutcome(exit_code, crashed=False, stopped=stopped, log_path=None))


def test_begin_saves_the_settings_and_starts_on_the_window_gpu(paths, parts):
    run, runner, events = parts
    settings = GeneratorSettings(prompt="a cat", pre_prompt="score_9")

    assert run.begin(settings)

    assert run.active
    assert load_generator_settings(paths.generator_settings_file).settings == settings
    assert "INFO: Prompt: score_9, a cat" in events["lines"]
    spec, _log = runner.started[0]
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "1"
    assert str(default_generated_folder(paths)) in " ".join(spec.arguments)


def test_wrong_settings_start_nothing_and_say_why(parts):
    run, runner, events = parts

    assert not run.begin(GeneratorSettings())

    assert not run.active
    assert runner.started == []
    assert "ERROR: Write a prompt first." in events["lines"]


def test_a_gpu_held_by_another_process_refuses_the_start(paths, parts):
    run, runner, events = parts
    with held_by_another_process(paths.gpu_lock_file):
        assert not run.begin(GeneratorSettings(prompt="a cat"))
    assert runner.started == []
    assert any(line.startswith("ERROR: ") for line in events["lines"])


def test_steps_images_and_the_end_are_reported(tmp_path, parts):
    run, runner, events = parts
    run.begin(GeneratorSettings(prompt="a cat"))
    image = tmp_path / "image.png"

    runner.line_received.emit("STEP: 5/25")
    runner.line_received.emit(f"IMAGE: {image}")
    _finish(runner)

    assert events["steps"] == [(5, 25)]
    assert events["images"] == [image]
    assert events["ended"] == [(True, "OK: 1 image generated.")]
    assert not run.active


def test_stop_ends_the_generation_as_stopped(parts):
    run, runner, events = parts
    run.begin(GeneratorSettings(prompt="a cat"))

    run.stop()
    _finish(runner, exit_code=-15, stopped=True)

    assert runner.stopped == 1
    assert events["ended"] == [(False, "INFO: Generation stopped.")]


def test_after_the_end_the_lock_is_free_for_another_generation(parts):
    run, runner, _events = parts
    run.begin(GeneratorSettings(prompt="a cat"))
    _finish(runner)

    assert run.begin(GeneratorSettings(prompt="a dog"))
    assert len(runner.started) == 2


def test_shutdown_stops_the_process(parts):
    run, runner, _events = parts
    run.begin(GeneratorSettings(prompt="a cat"))

    run.shutdown()

    assert runner.shut_down == 1  # the real runner waits there for the process to exit


def test_the_queue_is_generated_with_every_entry_in_order(parts):
    run, runner, events = parts
    queue = (GenerationEntry(prompt="a cat"), GenerationEntry(prompt="a dog"))

    assert run.begin(GeneratorSettings(prompt="not queued", queue=queue), queue=True)

    assert "INFO: Queue: 2 entries, 2 images." in events["lines"]
    assert "a cat" in " ".join(runner.started[0][0].arguments)


def test_an_empty_queue_starts_nothing(parts):
    run, runner, events = parts

    assert not run.begin(GeneratorSettings(prompt="a cat"), queue=True)

    assert runner.started == []
    assert any("The queue is empty" in line for line in events["lines"])
