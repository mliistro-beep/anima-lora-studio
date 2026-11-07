"""Settings of the epoch previews, kept in <data>/preview.json so config.json stays Citron's.

Defaults: previews off (TOMLs identical to Citron), ER-SDE at 25 steps, size chosen from the
GPU memory unless the user fixes it.
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from anima_trainer.settings.json_store import load_model, save_model
from anima_trainer.settings.schema import DecimalNumber, WholeNumber

MAX_PROMPTS = 8
SIDE_STEP = 16  # sd-scripts rounds sample sizes down to a multiple of 16
MIN_SIDE = 256
MAX_SIDE = 2048
Sampler = Literal["er_sde", "euler"]
# The CFG each sampler starts from (user's choice after real runs, 2026-11-07: 4 was too low).
SAMPLER_CFG = MappingProxyType({"er_sde": 6.5, "euler": 7.0})
DEFAULT_SAMPLER = "er_sde"
DEFAULT_CFG = SAMPLER_CFG[DEFAULT_SAMPLER]


def cfg_after_sampler_change(previous: str, new: str, cfg: float) -> float:
    """A CFG still at the previous sampler's default moves to the new one's; one the user chose
    stays."""
    if cfg == SAMPLER_CFG.get(previous) and new in SAMPLER_CFG:
        return SAMPLER_CFG[new]
    return cfg


def _check_side(value: int) -> int:
    if value % SIDE_STEP:
        raise ValueError(f"must be a multiple of {SIDE_STEP}")
    return value


def _clean_prompts(prompts: tuple[str, ...]) -> tuple[str, ...]:
    cleaned = tuple(prompt.strip() for prompt in prompts if prompt.strip())
    if len(cleaned) > MAX_PROMPTS:
        raise ValueError(f"at most {MAX_PROMPTS} preview prompts")
    return cleaned


Side = Annotated[WholeNumber, Field(ge=MIN_SIDE, le=MAX_SIDE), AfterValidator(_check_side)]


class PreviewSettings(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="ignore", validate_default=True, allow_inf_nan=False
    )

    enabled: bool = False
    prompts: Annotated[tuple[str, ...], AfterValidator(_clean_prompts)] = ()
    negative_prompt: str = ""
    auto_size: bool = True
    width: Side = 1024
    height: Side = 1024
    sampler: Sampler = "er_sde"
    steps: WholeNumber = Field(25, ge=1, le=200)
    cfg_scale: DecimalNumber = Field(DEFAULT_CFG, ge=1.0, le=20.0)
    flow_shift: DecimalNumber = Field(3.0, gt=0, le=20.0)
    seed: WholeNumber = Field(42, ge=0)
    every_n_epochs: WholeNumber = Field(1, ge=1)
    sample_at_first: bool = False

    def with_changes(self, **changes: Any) -> Self:
        """Return a new, re-validated copy with the given fields replaced."""
        return type(self).model_validate({**self.model_dump(), **changes})

    def problems(self) -> tuple[str, ...]:
        if self.enabled and not self.prompts:
            return ("Previews are on but there is no preview prompt.",)
        return ()


@dataclass(frozen=True)
class PreviewLoadResult:
    settings: PreviewSettings
    warnings: tuple[str, ...]


def load_preview_settings(path: Path) -> PreviewLoadResult:
    loaded = load_model(path, PreviewSettings)
    return PreviewLoadResult(loaded.model, loaded.warnings)


def save_preview_settings(path: Path, settings: PreviewSettings) -> None:
    """Raises OSError when the file cannot be written."""
    save_model(path, settings)
