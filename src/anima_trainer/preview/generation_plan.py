"""From the Generator settings to the images to generate, or the ERROR: lines that stop it.

The generator needs no training project: its own model (a catalog model or a DiT file of the
user's), its LoRAs, and its own folder (blank: <data>/generated). A project only adds its
checkpoints, one image each (Compare Epochs) or the chosen one, applied before the user's LoRAs.

Files are read only when something of the user's is involved: a model file is checked, and with
LoRAs the model's block count is read to check each enabled LoRA. Only errors stop the plan; the
warnings about a LoRA (text encoder part, unknown keys) come from generate.py in the log.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from anima_trainer.preview.checkpoints import Checkpoint
from anima_trainer.preview.generator_settings import GeneratorSettings
from anima_trainer.preview.inference_command import (
    GenerationRequest,
    LoraUse,
    default_generated_folder,
    output_image_path,
)
from anima_trainer.preview.lora_files import (
    CheckpointFileError,
    LoraFileError,
    check_lora,
    read_checkpoint_blocks,
    read_lora,
)
from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.runtime.paths import RuntimePaths

NAME_WITHOUT_PROJECT = "image"
NO_LORA_LABEL = "No LoRA"
LORAS_LABEL = "LoRAs"


@dataclass(frozen=True)
class GenerationTarget:
    """What the training side adds: the project name for the file names, and the
    checkpoints to apply (None: no checkpoint of the project)."""

    project: str
    checkpoints: tuple[Checkpoint | None, ...]


def plan_generation(
    settings: GeneratorSettings, paths: RuntimePaths, target: GenerationTarget, now: datetime
) -> tuple[GenerationRequest, ...] | tuple[str, ...]:
    problems = settings.problems()
    if problems:
        return tuple(f"ERROR: {problem}" for problem in problems)
    own_model = bool(settings.model_file)
    dit = Path(settings.model_file) if own_model else paths.dit_model(settings.base_model)
    loras = tuple(LoraUse(Path(entry.path), entry.strength) for entry in settings.enabled_loras())
    errors = _file_errors(dit, own_model, loras)
    if errors:
        return errors
    folder = (
        Path(settings.output_folder) if settings.output_folder else default_generated_folder(paths)
    )
    prefix = target.project or NAME_WITHOUT_PROJECT
    return tuple(
        _request(
            settings,
            dit,
            checkpoint,
            loras,
            output_image_path(folder, prefix, _label(checkpoint, loras), settings.seed, now),
        )
        for checkpoint in target.checkpoints
    )


def _file_errors(dit: Path, own_model: bool, loras: tuple[LoraUse, ...]) -> tuple[str, ...]:
    if not own_model and not loras:
        return ()
    try:
        blocks = read_checkpoint_blocks(dit)
    except CheckpointFileError as error:
        return (f"ERROR: {error}",)
    errors: list[str] = []
    for lora in loras:
        try:
            lines = check_lora(read_lora(lora.path), blocks)
        except LoraFileError as error:
            lines = (f"ERROR: {error}",)
        errors += [line for line in lines if line.startswith("ERROR: ")]
    return tuple(errors)


def _label(checkpoint: Checkpoint | None, loras: tuple[LoraUse, ...]) -> str:
    if checkpoint is not None:
        return checkpoint.label
    return LORAS_LABEL if loras else NO_LORA_LABEL


def _request(
    settings: GeneratorSettings,
    dit: Path,
    checkpoint: Checkpoint | None,
    loras: tuple[LoraUse, ...],
    output: Path,
) -> GenerationRequest:
    project_lora = (LoraUse(checkpoint.path, settings.lora_multiplier),) if checkpoint else ()
    return GenerationRequest(
        dit=dit,
        loras=(*project_lora, *loras),
        prompt=settings.final_prompt(),
        negative_prompt=settings.negative_prompt,
        size=SampleSize(settings.width, settings.height),
        steps=settings.steps,
        cfg_scale=settings.cfg_scale,
        flow_shift=settings.flow_shift,
        seed=settings.seed,
        sampler=settings.sampler,
        output=output,
    )
