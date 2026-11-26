"""The preview fields of the TUI's training form, over the shared preview.json settings."""

from anima_trainer.cli.tui_previews import (
    PreviewFields,
    fields_of,
    kept_prompts_note,
    previews_from_fields,
)
from anima_trainer.preview.settings import PreviewSettings

SAVED = PreviewSettings(
    enabled=True,
    prompts=("first", "second", "third"),
    negative_prompt="lowres",
    pre_prompt="score_9",
    trigger="ohwx",
    steps=30,
    sampler="euler",
)


def _fields(**changes):
    values = {
        "enabled": True,
        "pre_prompt": "best",
        "trigger": "abc",
        "prompt": "new",
        "negative_prompt": "blurry",
    }
    return PreviewFields(**{**values, **changes})


def test_the_fields_show_the_first_prompt_of_the_saved_settings():
    assert fields_of(SAVED) == PreviewFields(True, "score_9", "ohwx", "first", "lowres")
    assert fields_of(PreviewSettings()) == PreviewFields(False, "", "", "", "")


def test_the_other_prompts_are_named_as_kept():
    assert kept_prompts_note(SAVED) == "+2 more prompts from the window, kept."
    assert kept_prompts_note(SAVED.with_changes(prompts=("a", "b"))) == (
        "+1 more prompt from the window, kept."
    )
    assert kept_prompts_note(SAVED.with_changes(prompts=("a",))) == ""


def test_the_fields_replace_only_their_values_and_keep_the_rest():
    result = previews_from_fields(SAVED, _fields())

    assert result == SAVED.with_changes(
        pre_prompt="best",
        trigger="abc",
        prompts=("new", "second", "third"),
        negative_prompt="blurry",
    )
    assert (result.steps, result.sampler) == (30, "euler")


def test_previews_can_be_switched_off_and_the_other_prompts_stay():
    result = previews_from_fields(SAVED, _fields(enabled=False))

    assert result.enabled is False
    assert result.prompts == ("new", "second", "third")


def test_a_cleared_prompt_field_removes_the_prompt_it_showed():
    assert previews_from_fields(SAVED, _fields(enabled=False, prompt=" ")).prompts == (
        "second",
        "third",
    )
    only_one = SAVED.with_changes(prompts=("first",))
    assert previews_from_fields(only_one, _fields(enabled=False, prompt="")).prompts == ()


def test_a_first_prompt_is_required_when_previews_are_on():
    assert previews_from_fields(SAVED, _fields(prompt="  ")) == (
        "ERROR: Previews are on: write a preview prompt.",
    )


def test_a_first_prompt_is_added_to_empty_settings():
    result = previews_from_fields(PreviewSettings(), _fields(prompt="1girl"))

    assert result.prompts == ("1girl",)
    assert result.enabled is True
