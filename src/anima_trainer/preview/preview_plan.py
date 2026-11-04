"""What Configure adds to a training for the previews: the prompts at a size that fits the
GPU, the report lines, or the reason it cannot configure."""

from dataclasses import dataclass

from anima_trainer.preview.sample_prompts import SampleRequest, SampleSize
from anima_trainer.preview.settings import PreviewSettings
from anima_trainer.preview.vram_profile import fit_preview_size

SAMPLER_NAMES = {"er_sde": "ER-SDE", "euler": "Euler"}


@dataclass(frozen=True)
class PreviewPlan:
    request: SampleRequest | None = None  # None: no previews in this training
    messages: tuple[str, ...] = ()
    error: str | None = None


def plan_previews(settings: PreviewSettings, gpu_total_mib: int | None) -> PreviewPlan:
    if not settings.enabled:
        return PreviewPlan()
    problems = settings.problems()
    if problems:
        return PreviewPlan(error=problems[0])
    requested = SampleSize(settings.width, settings.height)
    choice = fit_preview_size(gpu_total_mib, requested, settings.auto_size)
    header = ("", "Previews:", f"  {choice.note}")
    if choice.size is None:
        return PreviewPlan(messages=header)
    return PreviewPlan(
        SampleRequest(settings, choice.size), (*header, f"  INFO: {_summary(settings)}")
    )


def _summary(settings: PreviewSettings) -> str:
    count = len(settings.prompts)
    prompts = f"{count} prompt" if count == 1 else f"{count} prompts"
    epochs = "epoch" if settings.every_n_epochs == 1 else "epochs"
    first = ", also before training" if settings.sample_at_first else ""
    return (
        f"{prompts}, every {settings.every_n_epochs} {epochs}{first}, "
        f"{SAMPLER_NAMES[settings.sampler]}, {settings.steps} steps, CFG {settings.cfg_scale}."
    )
