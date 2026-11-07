import pytest
from pydantic import ValidationError

from anima_trainer.preview.generator_settings import (
    MAX_COUNT,
    MAX_LORAS,
    MAX_QUEUE,
    GenerationEntry,
    GeneratorSettings,
    LoraEntry,
    load_generator_settings,
    save_generator_settings,
)


def test_defaults_match_the_preview_defaults():
    settings = GeneratorSettings()

    assert (settings.sampler, settings.steps, settings.cfg_scale, settings.seed) == (
        "er_sde", 25, 6.5, 42,
    )  # fmt: skip
    assert (settings.width, settings.height, settings.lora_multiplier) == (1024, 1024, 1.0)


@pytest.mark.parametrize(
    "field, value",
    [("width", 1000), ("height", 4096), ("lora_multiplier", -0.1), ("lora_multiplier", 2.5),
     ("steps", 0), ("cfg_scale", True), ("sampler", "x")],
)  # fmt: skip
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        GeneratorSettings(**{field: value})


def test_prompt_is_required_to_generate():
    assert GeneratorSettings().problems() == ("Write a prompt first.",)
    assert GeneratorSettings(prompt="  a girl ").problems() == ()
    assert GeneratorSettings(prompt="  a girl ").prompt == "a girl"


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "generator.json"
    settings = GeneratorSettings(prompt="a", width=832, height=1216, lora_multiplier=0.7)

    save_generator_settings(path, settings)

    assert load_generator_settings(path).settings == settings


def test_the_new_choices_default_to_the_catalog_model_no_loras_and_the_default_folder():
    settings = GeneratorSettings()

    assert settings.base_model == "anima-base-v1.0"
    assert (settings.model_file, settings.loras, settings.output_folder) == ("", (), "")


def test_a_file_saved_before_these_choices_still_loads(tmp_path):
    path = tmp_path / "generator.json"
    path.write_text('{"prompt": "a cat", "seed": 3, "lora_multiplier": 0.5}')

    result = load_generator_settings(path)

    assert result.warnings == ()
    assert (result.settings.prompt, result.settings.seed, result.settings.loras) == ("a cat", 3, ())


def test_up_to_eight_loras_are_kept_in_order_with_their_strength(tmp_path):
    loras = tuple(LoraEntry(path=f" /l/{n}.safetensors ", strength=n / 4 - 1) for n in range(8))

    settings = GeneratorSettings(loras=loras)

    assert [entry.path for entry in settings.loras] == [f"/l/{n}.safetensors" for n in range(8)]
    assert settings.loras[0].strength == -1.0
    assert all(entry.enabled for entry in settings.loras)
    assert MAX_LORAS == 8


def test_a_ninth_lora_is_refused():
    loras = tuple(LoraEntry(path=f"/l/{n}.safetensors") for n in range(MAX_LORAS + 1))

    with pytest.raises(ValidationError):
        GeneratorSettings(loras=loras)


@pytest.mark.parametrize(
    "values",
    [{"path": "  "}, {"path": "/l/a.safetensors", "strength": 2.5},
     {"path": "/l/a.safetensors", "strength": -2.1},
     {"path": "/l/a.safetensors", "strength": True}],
)  # fmt: skip
def test_a_lora_without_a_file_or_with_a_strength_out_of_range_is_refused(values):
    with pytest.raises(ValidationError):
        LoraEntry(**values)


def test_the_enabled_loras_are_the_ones_used():
    settings = GeneratorSettings(
        loras=(
            LoraEntry(path="/l/a.safetensors"),
            LoraEntry(path="/l/b.safetensors", enabled=False),
        )
    )

    assert [entry.path for entry in settings.enabled_loras()] == ["/l/a.safetensors"]


def test_loras_model_file_and_folder_survive_a_save(tmp_path):
    path = tmp_path / "generator.json"
    settings = GeneratorSettings(
        model_file="/m/merge.safetensors",
        loras=(LoraEntry(path="/l/a.safetensors", strength=0.6, enabled=False),),
        output_folder="/pictures/anima",
    )

    save_generator_settings(path, settings)

    assert load_generator_settings(path).settings == settings


def test_a_change_keeps_the_loras_and_checks_them_again():
    settings = GeneratorSettings(loras=(LoraEntry(path="/l/a.safetensors", strength=0.5),))

    changed = settings.with_changes(seed=9)

    assert (changed.seed, changed.loras) == (9, settings.loras)
    with pytest.raises(ValidationError):
        settings.with_changes(loras=(*settings.loras,) * (MAX_LORAS + 1))


def test_the_final_prompt_is_pre_prompt_then_triggers_of_enabled_loras_then_prompt():
    settings = GeneratorSettings(
        pre_prompt=" score_9, masterpiece ",
        prompt="1girl, castle",
        loras=(
            LoraEntry(path="/l/a.safetensors", trigger=" @bisley "),
            LoraEntry(path="/l/b.safetensors", trigger="off_trigger", enabled=False),
            LoraEntry(path="/l/c.safetensors"),
            LoraEntry(path="/l/d.safetensors", trigger="watercolor"),
        ),
    )

    assert settings.final_prompt() == "score_9, masterpiece, @bisley, watercolor, 1girl, castle"


def test_without_pre_prompt_or_triggers_the_final_prompt_is_the_prompt():
    assert GeneratorSettings(prompt="a cat").final_prompt() == "a cat"


def test_trigger_and_pre_prompt_survive_a_save(tmp_path):
    path = tmp_path / "generator.json"
    settings = GeneratorSettings(
        pre_prompt="score_9", loras=(LoraEntry(path="/l/a.safetensors", trigger="@x"),)
    )

    save_generator_settings(path, settings)

    assert load_generator_settings(path).settings == settings


def test_a_new_entry_has_a_fixed_seed_one_image_and_the_queue_is_empty():
    settings = GeneratorSettings()
    assert (settings.random_seed, settings.count, settings.queue) == (False, 1, ())


@pytest.mark.parametrize("count", [0, MAX_COUNT + 1, True])
def test_the_count_goes_from_one_to_ten(count):
    with pytest.raises(ValidationError):
        GeneratorSettings(count=count)


def test_the_form_becomes_a_queue_entry_without_the_folder_or_the_checkpoint_strength():
    lora = LoraEntry(path="/l/a.safetensors", trigger="sks")
    settings = GeneratorSettings(
        prompt="a cat", random_seed=True, count=5, loras=(lora,), output_folder="/out"
    )
    entry = settings.entry()
    assert isinstance(entry, GenerationEntry)
    assert (entry.prompt, entry.random_seed, entry.count, entry.loras) == (
        "a cat",
        True,
        5,
        (lora,),
    )
    assert not hasattr(entry, "output_folder")
    assert not hasattr(entry, "lora_multiplier")


def test_the_queue_keeps_at_most_ten_entries_in_order():
    entries = tuple(GenerationEntry(prompt=f"p{n}") for n in range(MAX_QUEUE))
    assert GeneratorSettings(queue=entries).queue == entries
    with pytest.raises(ValidationError):
        GeneratorSettings(queue=(*entries, GenerationEntry(prompt="one more")))


def test_the_queue_survives_a_save(tmp_path):
    file = tmp_path / "generator.json"
    entry = GenerationEntry(prompt="a dog", random_seed=True, count=3, sampler="euler")
    save_generator_settings(file, GeneratorSettings(prompt="a cat", queue=(entry,)))
    assert load_generator_settings(file).settings.queue == (entry,)


def test_a_queue_entry_has_its_own_final_prompt_and_problems():
    entry = GenerationEntry(pre_prompt="score_9", prompt="a dog")
    assert entry.final_prompt() == "score_9, a dog"
    assert GenerationEntry().problems() == ("Write a prompt first.",)
