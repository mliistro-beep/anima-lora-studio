"""The Generator's random seed, images in a row and queue, in the real window."""

import json

from anima_trainer.preview.generator_settings import load_generator_settings
from tests.e2e.test_previews_generator import OK, _option, app  # noqa: F401


def _seeds(runner):
    return [_option(spec.arguments, "--seed") for spec, _log in runner.started]


def test_a_random_seed_makes_the_chosen_number_of_images_with_new_seeds(app):  # noqa: F811
    window, _, generator_runner, _paths = app
    tab = window.generator_tab
    tab.prompt.setPlainText("a cat")
    tab.form.field_widget("random_seed").setChecked(True)
    tab.form.set_value("count", 3)

    tab.generate_button.click()
    for _ in range(2):
        generator_runner.line_received.emit("IMAGE: /x.png")
        generator_runner.finished.emit(OK)

    assert len(generator_runner.started) == 3
    assert len(set(_seeds(generator_runner))) == 3
    assert "Generating image 3/3" in tab.log.text()


def test_reset_turns_the_random_seed_off_and_back_to_one_image(app):  # noqa: F811
    window, _, _, _ = app
    tab = window.generator_tab
    tab.form.field_widget("random_seed").setChecked(True)
    tab.form.set_value("count", 4)

    tab.reset_button.click()

    assert (tab.form.values()["random_seed"], tab.form.values()["count"]) == (False, 1)


def test_the_queue_generates_every_entry_and_is_saved(app):  # noqa: F811
    window, _, generator_runner, paths = app
    tab = window.generator_tab
    tab.prompt.setPlainText("a cat")
    tab.queue.add_button.click()
    tab.prompt.setPlainText("a dog")
    tab.form.field_widget("random_seed").setChecked(True)
    tab.form.set_value("count", 2)
    tab.queue.add_button.click()
    tab.prompt.setPlainText("not queued")

    tab.generate_queue_button.click()
    for _ in range(2):
        generator_runner.line_received.emit("IMAGE: /x.png")
        generator_runner.finished.emit(OK)

    prompts = [
        next(part for part in spec.arguments if part.startswith("--prompt="))
        for spec, _log in generator_runner.started
    ]
    assert len(prompts) == 3
    assert "a cat" in prompts[0] and "a dog" in prompts[1] and "a dog" in prompts[2]
    saved = load_generator_settings(paths.generator_settings_file).settings
    assert [entry.prompt for entry in saved.queue] == ["a cat", "a dog"]
    assert saved.prompt == "not queued"


def test_an_empty_queue_says_what_to_do(app):  # noqa: F811
    window, _, generator_runner, _ = app
    tab = window.generator_tab

    tab.generate_queue_button.click()

    assert generator_runner.started == []
    assert "The queue is empty" in tab.log.text()


def test_add_current_with_a_wrong_form_shows_why_and_adds_nothing(app):  # noqa: F811
    window, _, _, _ = app
    tab = window.generator_tab
    tab.form.field_widget("cfg_scale").setText("lots")

    tab.queue.add_button.click()

    assert tab.queue.entries() == ()
    assert "ERROR: CFG Scale" in tab.log.text()


def test_the_saved_queue_is_shown_again(app):  # noqa: F811
    window, _, _, paths = app
    paths.generator_settings_file.write_text(
        json.dumps({"queue": [{"prompt": "a cat"}, {"prompt": "a dog", "random_seed": True}]})
    )

    window.generator_tab.set_settings(
        load_generator_settings(paths.generator_settings_file).settings
    )

    assert [entry.prompt for entry in window.generator_tab.queue.entries()] == ["a cat", "a dog"]


def test_the_window_shows_the_same_progress_line(app):  # noqa: F811
    window, _, generator_runner, _ = app
    tab = window.generator_tab
    tab.prompt.setPlainText("a cat")
    tab.queue.add_button.click()
    tab.queue.add_button.click()

    tab.generate_queue_button.click()
    generator_runner.line_received.emit("STEP: 5/25")

    assert tab.progress_text.text() == "img 01/02 | batch 1/2 | step 5/25"
