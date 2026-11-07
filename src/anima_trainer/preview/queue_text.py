"""How a queue entry is shown, the same in the window and in the terminal."""

from anima_trainer.preview.generator_settings import GenerationEntry

PROMPT_CHARACTERS = 40


def entry_summary(number: int, entry: GenerationEntry) -> str:
    """`3. score_9, a cat... | 2 LoRAs | random seed x5`, or `| seed 42` with a fixed seed."""
    prompt = entry.final_prompt()
    if len(prompt) > PROMPT_CHARACTERS:
        prompt = f"{prompt[:PROMPT_CHARACTERS]}..."
    count = len(entry.enabled_loras())
    loras = "no LoRA" if count == 0 else f"{count} LoRA{'' if count == 1 else 's'}"
    seed = f"random seed x{entry.count}" if entry.random_seed else f"seed {entry.seed}"
    return f"{number}. {prompt} | {loras} | {seed}"
