"""Preview prompts in the JSON form sd-scripts reads (`sample_prompts` ending in .json).

JSON instead of the .txt prompt lines: there " --" starts an option, so a prompt containing it
would be cut, while JSON passes any text and every key (flow_shift included) as is.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anima_trainer.preview.settings import PreviewSettings


@dataclass(frozen=True)
class SampleSize:
    width: int
    height: int


@dataclass(frozen=True)
class SampleRequest:
    """Previews to add to a training run, with the size already fitted to the GPU."""

    settings: PreviewSettings
    size: SampleSize


def build_sample_prompts(settings: PreviewSettings, size: SampleSize) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "prompt": prompt,
            "negative_prompt": settings.negative_prompt,
            "width": size.width,
            "height": size.height,
            "sample_steps": settings.steps,
            "scale": settings.cfg_scale,
            "flow_shift": settings.flow_shift,
            "seed": settings.seed,
            # sd-scripts' name for the sampler option; its Anima sampler ignores it, the
            # train_with_previews.py launcher reads it to pick ER-SDE or Euler.
            "sample_sampler": settings.sampler,
        }
        for prompt in settings.prompts
    )


def sample_config_keys(settings: PreviewSettings, prompts_file: Path) -> dict[str, Any]:
    """Keys added to the training TOML; only when previews are on."""
    return {
        "sample_prompts": str(prompts_file),
        "sample_every_n_epochs": settings.every_n_epochs,
        "sample_at_first": settings.sample_at_first,
    }
