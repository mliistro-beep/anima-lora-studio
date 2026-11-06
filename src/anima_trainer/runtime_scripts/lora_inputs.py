"""The LoRA options of generate.py: `--lora FILE --lora_multiplier STRENGTH`, repeated.

Runs in the training venv next to generate.py and imports nothing of the app, which tests it
through anima_trainer.runtime_scripts.lora_inputs.
"""

from collections.abc import Sequence


def pair_loras(
    loras: Sequence[str] | None, strengths: Sequence[float] | None
) -> tuple[tuple[str, float], ...]:
    """(file, strength) in the given order; raises ValueError when the counts differ."""
    loras, strengths = list(loras or ()), list(strengths or ())
    if len(loras) != len(strengths):
        raise ValueError(
            f"{len(loras)} --lora but {len(strengths)} --lora_multiplier: give one of each"
        )
    return tuple(zip(loras, strengths, strict=True))
