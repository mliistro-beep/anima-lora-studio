"""How a queue entry is shown in the window and in the terminal."""

from anima_trainer.preview.generator_settings import GenerationEntry, LoraEntry
from anima_trainer.preview.queue_text import entry_summary

LORA = LoraEntry(path="/l/a.safetensors")


def test_a_fixed_seed_entry_shows_its_prompt_loras_and_seed():
    entry = GenerationEntry(prompt="a cat", seed=7, loras=(LORA,))
    assert entry_summary(3, entry) == "3. a cat | 1 LoRA | seed 7"


def test_a_random_seed_entry_shows_how_many_images():
    entry = GenerationEntry(prompt="a cat", random_seed=True, count=5, loras=(LORA, LORA))
    assert entry_summary(1, entry) == "1. a cat | 2 LoRAs | random seed x5"


def test_only_the_loras_that_are_on_count_and_a_long_prompt_is_cut():
    entry = GenerationEntry(
        pre_prompt="score_9", prompt="x" * 80, loras=(LoraEntry(path="/l/b", enabled=False),)
    )
    summary = entry_summary(2, entry)
    assert summary.startswith("2. score_9, " + "x" * 31 + "...")
    assert " | no LoRA | " in summary
