"""Settings of the Generator tab, kept in <data>/generator.json.

The generator does not depend on the training form: its own base model (one of the catalog, or
a DiT file of the user's), up to MAX_LORAS LoRA files with their strength, and its own output
folder (blank: the default one, see inference_command.default_generated_folder). A file saved
before these choices existed still loads: the missing fields take their defaults.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from anima_trainer.preview.settings import DEFAULT_CFG, MAX_SIDE, MIN_SIDE, Sampler
from anima_trainer.preview.vram_profile import GENERATOR_SIDE_STEP
from anima_trainer.settings.choices import BaseModelName
from anima_trainer.settings.json_store import load_model, save_model
from anima_trainer.settings.schema import DecimalNumber, WholeNumber

MAX_LORAS = 8  # the user runs 6 together in Krita
MIN_STRENGTH = -2.0
MAX_STRENGTH = 2.0
PROMPT_SEPARATOR = ", "


def _check_side(value: int) -> int:
    if value % GENERATOR_SIDE_STEP:
        raise ValueError(f"must be a multiple of {GENERATOR_SIDE_STEP}")
    return value


Side = Annotated[WholeNumber, Field(ge=MIN_SIDE, le=MAX_SIDE), AfterValidator(_check_side)]


def _required_path(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("a LoRA needs a file")
    return value


class LoraEntry(BaseModel):
    """One LoRA of the list: its file, its strength and whether it is used now."""

    model_config = ConfigDict(frozen=True, extra="ignore", allow_inf_nan=False)

    path: Annotated[str, AfterValidator(_required_path)]
    strength: DecimalNumber = Field(1.0, ge=MIN_STRENGTH, le=MAX_STRENGTH)
    enabled: bool = True
    trigger: Annotated[str, AfterValidator(str.strip)] = ""  # added to the prompt while enabled


class GeneratorSettings(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="ignore", validate_default=True, allow_inf_nan=False
    )

    prompt: Annotated[str, AfterValidator(str.strip)] = ""
    pre_prompt: Annotated[str, AfterValidator(str.strip)] = ""  # e.g. score_9, masterpiece
    negative_prompt: str = ""
    width: Side = 1024
    height: Side = 1024
    sampler: Sampler = "er_sde"
    steps: WholeNumber = Field(25, ge=1, le=200)
    cfg_scale: DecimalNumber = Field(DEFAULT_CFG, ge=1.0, le=20.0)
    flow_shift: DecimalNumber = Field(3.0, gt=0, le=20.0)
    seed: WholeNumber = Field(42, ge=0)
    lora_multiplier: DecimalNumber = Field(1.0, ge=0.0, le=2.0)  # the project's checkpoint
    base_model: BaseModelName = "anima-base-v1.0"
    model_file: Annotated[str, AfterValidator(str.strip)] = ""  # wins over base_model when set
    loras: tuple[LoraEntry, ...] = Field((), max_length=MAX_LORAS)
    output_folder: Annotated[str, AfterValidator(str.strip)] = ""

    def with_changes(self, **changes: Any) -> Self:
        """Return a new, re-validated copy with the given fields replaced."""
        return type(self).model_validate({**self.model_dump(), **changes})

    def enabled_loras(self) -> tuple[LoraEntry, ...]:
        return tuple(entry for entry in self.loras if entry.enabled)

    def final_prompt(self) -> str:
        """Pre-prompt, then the triggers of the enabled LoRAs in their order, then the prompt."""
        triggers = (entry.trigger for entry in self.enabled_loras())
        parts = (self.pre_prompt, *triggers, self.prompt)
        return PROMPT_SEPARATOR.join(part for part in parts if part)

    def problems(self) -> tuple[str, ...]:
        return () if self.prompt else ("Write a prompt first.",)


@dataclass(frozen=True)
class GeneratorLoadResult:
    settings: GeneratorSettings
    warnings: tuple[str, ...]


def load_generator_settings(path: Path) -> GeneratorLoadResult:
    loaded = load_model(path, GeneratorSettings)
    return GeneratorLoadResult(loaded.model, loaded.warnings)


def save_generator_settings(path: Path, settings: GeneratorSettings) -> None:
    """Raises OSError when the file cannot be written."""
    save_model(path, settings)
