"""The TUI's menu and its generator, driven with Textual's test pilot and a fake runner."""

import json

import pytest
from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from anima_trainer.cli.tui.folder_picker import FilePicker
from anima_trainer.cli.tui.generator_screens import (
    GenerationScreen,
    GeneratorScreen,
    LoraRow,
    QueueRow,
)
from anima_trainer.cli.tui.menu import MenuScreen
from anima_trainer.cli.tui.screens import FormScreen, QuitDialog
from anima_trainer.preview.generator_settings import load_generator_settings
from anima_trainer.training.runner import RunOutcome
from tests.e2e.helpers import make_runtime
from tests.fakes import FakeRunner
from tests.integration.cli.test_tui import _app, _input, _log, _play, _text, _until
from tests.safetensors_helpers import checkpoint_keys, kohya_blocks, write_safetensors

pytestmark = pytest.mark.integration
GENERATOR = "#menu-generator"
COLUMNS = (".lora-on", ".lora-name", ".lora-trigger")


@pytest.fixture
def paths(qapp, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    return make_runtime(tmp_path / "data", images, {"gpu_index": "0"})


@pytest.fixture
def runner():
    return FakeRunner()


def _lora(folder, name, trigger=None):
    metadata = {"modelspec.trigger_phrase": trigger} if trigger else None
    return write_safetensors(folder / name, kohya_blocks(28), metadata)


def _save(paths, **values):
    paths.generator_settings_file.write_text(json.dumps(values))


def _rows(app):
    return list(app.screen.query(LoraRow))


def _prompt(app):
    return app.screen.query_one("#prompt", TextArea)


async def _add(pilot, path):
    await pilot.click("#add-lora")
    await _until(pilot, lambda: isinstance(pilot.app.screen, FilePicker))
    pilot.app.screen.dismiss(str(path))
    await _until(pilot, lambda: isinstance(pilot.app.screen, GeneratorScreen))
    await pilot.pause()


def test_the_menu_comes_first_and_leads_to_both_parts(paths):
    app = _app(paths)

    async def scenario(pilot):
        assert isinstance(app.screen, MenuScreen)
        await pilot.click("#menu-training")
        await _until(pilot, lambda: isinstance(app.screen, FormScreen))
        await pilot.click("#menu-form")
        await _until(pilot, lambda: isinstance(app.screen, MenuScreen))
        await pilot.click(GENERATOR)
        await _until(pilot, lambda: isinstance(app.screen, GeneratorScreen))
        await pilot.click("#back-generator")
        await _until(pilot, lambda: isinstance(app.screen, MenuScreen))

    _play(app, scenario, choice=None)


def test_the_generator_shows_the_settings_saved_by_the_window(paths, tmp_path):
    lora = _lora(tmp_path, "style.safetensors")
    _save(
        paths,
        prompt="a cat",
        pre_prompt="score_9",
        steps=30,
        loras=[{"path": str(lora), "strength": 0.8, "enabled": False, "trigger": "sks"}],
    )
    app = _app(paths)

    async def scenario(pilot):
        assert _prompt(app).text == "a cat"
        assert _input(app, "pre_prompt").value == "score_9"
        assert _input(app, "steps").value == "30"
        (row,) = _rows(app)
        assert "style.safetensors" in str(row.query_one(".lora-name", Static).content)
        assert row.query_one(".lora-trigger", Input).value == "sks"
        assert row.query_one(".lora-strength", Input).value == "0.8"
        assert not row.query_one(".lora-on", Checkbox).value

    _play(app, scenario, GENERATOR)


def test_an_added_lora_gets_its_trigger_and_remove_takes_it_away(paths, tmp_path):
    first = _lora(tmp_path, "a.safetensors", trigger="aaa")
    second = _lora(tmp_path, "b.safetensors")
    app = _app(paths)

    async def scenario(pilot):
        await _add(pilot, first)
        await _add(pilot, second)
        rows = _rows(app)
        assert [row.query_one(".lora-trigger", Input).value for row in rows] == ["aaa", ""]
        rows[0].query_one(".lora-remove", Button).press()
        await pilot.pause()
        (left,) = _rows(app)
        assert "b.safetensors" in str(left.query_one(".lora-name", Static).content)

    _play(app, scenario, GENERATOR)


def test_the_cfg_follows_the_sampler_only_while_at_its_default(paths):
    app = _app(paths)

    async def scenario(pilot):
        sampler = app.screen.query_one("#sampler", Select)
        assert _input(app, "cfg_scale").value == "6.5"
        sampler.value = "euler"
        await pilot.pause()
        assert _input(app, "cfg_scale").value == "7.0"
        _input(app, "cfg_scale").value = "5"
        sampler.value = "er_sde"
        await pilot.pause()
        assert _input(app, "cfg_scale").value == "5"

    _play(app, scenario, GENERATOR)


def test_reset_puts_back_the_image_settings_and_keeps_prompts_and_loras(paths, tmp_path):
    lora = _lora(tmp_path, "a.safetensors")
    _save(paths, prompt="a cat", steps=40, seed=7, loras=[{"path": str(lora)}])
    app = _app(paths)

    async def scenario(pilot):
        await pilot.click("#reset-generator")
        await pilot.pause()
        assert (_input(app, "steps").value, _input(app, "seed").value) == ("25", "42")
        assert _prompt(app).text == "a cat"
        assert len(_rows(app)) == 1

    _play(app, scenario, GENERATOR)


def test_wrong_values_are_shown_and_nothing_starts(paths, runner):
    app = _app(paths, generation_runner=runner)

    async def scenario(pilot):
        _prompt(app).text = "a cat"
        _input(app, "steps").value = "many"
        await pilot.click("#generate")
        await pilot.pause()
        assert isinstance(app.screen, GeneratorScreen)
        assert "Steps" in _text(app, "errors")

    _play(app, scenario, GENERATOR)
    assert runner.started == []


def test_generate_runs_with_the_loras_and_shows_the_image(paths, runner, tmp_path):
    write_safetensors(paths.dit_model("anima-base-v1.0"), checkpoint_keys(28))
    lora = _lora(tmp_path, "a.safetensors", trigger="sks")
    image = tmp_path / "image.png"
    app = _app(paths, generation_runner=runner)

    async def scenario(pilot):
        await _add(pilot, lora)
        _input(app, "pre_prompt").value = "score_9"
        _prompt(app).text = "a cat"
        await pilot.click("#generate")
        await _until(pilot, lambda: isinstance(app.screen, GenerationScreen))
        await _until(pilot, lambda: runner.started)
        assert "INFO: Prompt: score_9, sks, a cat" in _log(app)
        runner.line_received.emit("STEP: 25/25")
        runner.line_received.emit(f"IMAGE: {image}")
        runner.finished.emit(RunOutcome(0, crashed=False, stopped=False, log_path=None))
        await _until(pilot, lambda: "completed" in _text(app, "status"))
        assert str(image) in _log(app)
        assert app.screen.query_one("#stop", Button).disabled
        await pilot.click("#back")
        await _until(pilot, lambda: isinstance(app.screen, GeneratorScreen))

    _play(app, scenario, GENERATOR)
    saved = load_generator_settings(paths.generator_settings_file).settings
    assert (saved.prompt, saved.loras[0].trigger) == ("a cat", "sks")
    spec, _log_path = runner.started[0]
    assert str(lora) in spec.arguments


def test_quitting_during_a_generation_asks_and_yes_stops_it(paths, runner):
    app = _app(paths, generation_runner=runner)

    async def scenario(pilot):
        _prompt(app).text = "a cat"
        await pilot.click("#generate")
        await _until(pilot, lambda: runner.started)
        await pilot.press("ctrl+q")
        await _until(pilot, lambda: isinstance(app.screen, QuitDialog))
        await pilot.click("#quit-yes")
        await _until(pilot, lambda: runner.stopped)
        runner.finished.emit(RunOutcome(-15, crashed=False, stopped=True, log_path=None))
        await pilot.pause()

    _play(app, scenario, GENERATOR)
    assert not app.run_active


def _queue_texts(app):
    return [str(row.query_one(".queue-text", Static).content) for row in app.screen.query(QueueRow)]


def test_a_random_seed_makes_the_chosen_number_of_images(paths, runner):
    app = _app(paths, generation_runner=runner)

    async def scenario(pilot):
        _prompt(app).text = "a cat"
        app.screen.query_one("#random_seed", Checkbox).value = True
        _input(app, "count").value = "2"
        await pilot.click("#generate")
        await _until(pilot, lambda: runner.started)
        runner.line_received.emit("IMAGE: /x.png")
        runner.finished.emit(RunOutcome(0, crashed=False, stopped=False, log_path=None))
        await _until(pilot, lambda: len(runner.started) == 2)

    _play(app, scenario, GENERATOR)
    seeds = [spec.arguments[spec.arguments.index("--seed") + 1] for spec, _ in runner.started]
    assert len(set(seeds)) == 2


def test_the_queue_takes_the_form_shows_it_numbered_and_removes_an_entry(paths):
    app = _app(paths)

    async def scenario(pilot):
        for prompt in ("a cat", "a dog", "a bird"):
            _prompt(app).text = prompt
            app.screen.query_one("#queue-add", Button).press()  # a click would miss: rows move it
            await pilot.pause()
        assert [text[:8] for text in _queue_texts(app)] == ["1. a cat", "2. a dog", "3. a bir"]
        list(app.screen.query(QueueRow))[0].query_one(".queue-remove", Button).press()
        await pilot.pause()
        assert [text[:8] for text in _queue_texts(app)] == ["1. a dog", "2. a bir"]
        app.screen.query_one("#queue-clear", Button).press()
        await pilot.pause()
        assert _queue_texts(app) == []

    _play(app, scenario, GENERATOR)


def test_generate_queue_runs_the_entries_and_saves_the_queue(paths, runner):
    app = _app(paths, generation_runner=runner)

    async def scenario(pilot):
        _prompt(app).text = "a cat"
        await pilot.click("#queue-add")
        _prompt(app).text = "not queued"
        await pilot.click("#generate-queue")
        await _until(pilot, lambda: runner.started)
        assert "INFO: Queue: 1 entries, 1 images." in _log(app)

    _play(app, scenario, GENERATOR)
    assert "a cat" in " ".join(runner.started[0][0].arguments)
    saved = load_generator_settings(paths.generator_settings_file).settings
    assert [entry.prompt for entry in saved.queue] == ["a cat"]


def test_the_saved_queue_is_shown(paths):
    _save(paths, queue=[{"prompt": "a cat", "random_seed": True, "count": 3}])
    app = _app(paths)

    async def scenario(pilot):
        assert _queue_texts(app) == ["1. a cat | no LoRA | random seed x3"]

    _play(app, scenario, GENERATOR)


def test_reset_turns_the_random_seed_off(paths):
    app = _app(paths)

    async def scenario(pilot):
        app.screen.query_one("#random_seed", Checkbox).value = True
        _input(app, "count").value = "5"
        await pilot.click("#reset-generator")
        await pilot.pause()
        assert not app.screen.query_one("#random_seed", Checkbox).value
        assert _input(app, "count").value == "1"

    _play(app, scenario, GENERATOR)


def test_the_progress_line_shows_image_batch_and_steps_next_to_the_bar(paths, runner):
    app = _app(paths, generation_runner=runner)

    async def scenario(pilot):
        _prompt(app).text = "a cat"
        app.screen.query_one("#random_seed", Checkbox).value = True
        _input(app, "count").value = "2"
        await pilot.click("#generate")
        await _until(pilot, lambda: runner.started)
        runner.line_received.emit("STEP: 12/25")
        await _until(pilot, lambda: _text(app, "status") == "img 01/02 | batch 1/1 | step 12/25")
        bar = app.screen.query_one("#bar")
        assert bar.parent is app.screen.query_one("#status").parent

    _play(app, scenario, GENERATOR)


def test_the_lora_columns_have_a_separator(paths, tmp_path):
    _save(paths, prompt="a cat", loras=[{"path": str(_lora(tmp_path, "a.safetensors"))}])
    app = _app(paths)

    async def scenario(pilot):
        (row,) = _rows(app)
        sides = {name: row.query_one(name).styles.border_right[0] for name in COLUMNS}
        assert all(sides.values()), sides

    _play(app, scenario, GENERATOR)
