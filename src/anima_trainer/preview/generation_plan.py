"""From the Generator settings to the images to generate, or the ERROR: lines that stop it.

The generator needs no training project: its own model (a catalog model or a DiT file of the
user's), its LoRAs, and its own folder (blank: <data>/generated). A project only adds its
checkpoints, one image each (Compare Epochs) or the chosen one, applied before the user's LoRAs.

Each entry gives one image, or `count` images with a new seed each when its seed is random. The
queue gives the images of all its entries in order, without the project checkpoint.

Files are read only when something of the user's is involved: a model file is checked, and with
LoRAs the model's block count is read to check each enabled LoRA. Only errors stop the plan; the
warnings about a LoRA (text encoder part, unknown keys) come from generate.py in the log.
"""

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from anima_trainer import __version__
from anima_trainer.preview.checkpoints import Checkpoint
from anima_trainer.preview.generator_settings import GenerationEntry, GeneratorSettings
from anima_trainer.preview.image_metadata import ImageMetadata, image_metadata
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
from anima_trainer.settings.field_specs import INT_MAX

NAME_WITHOUT_PROJECT = "image"
NO_LORA_LABEL = "No LoRA"
LORAS_LABEL = "LoRAs"
ERROR_PREFIX = "ERROR: "
EMPTY_QUEUE = f"{ERROR_PREFIX}The queue is empty: add the form to it first."


@dataclass(frozen=True)
class GenerationTarget:
    """What the training side adds: the project name for the file names, and the
    checkpoints to apply (None: no checkpoint of the project)."""

    project: str
    checkpoints: tuple[Checkpoint | None, ...]


def random_seed() -> int:
    """A new seed, within what the window's seed field holds."""
    return secrets.randbelow(INT_MAX + 1)


def plan_generation(
    settings: GeneratorSettings,
    paths: RuntimePaths,
    target: GenerationTarget,
    now: datetime,
    new_seed: Callable[[], int] = random_seed,
) -> tuple[GenerationRequest, ...] | tuple[str, ...]:
    """The form's images: one per checkpoint for each seed. Compare Epochs (more than one
    checkpoint) uses a single seed, so the checkpoints are compared on the same image."""
    entry = settings.entry()
    seeds = _seeds(entry, new_seed, single=len(target.checkpoints) > 1)
    images = _plan_entry(entry, paths, target.checkpoints, settings.lora_multiplier, seeds, batch=1)
    if not images or isinstance(images[0], str):
        return tuple(str(line) for line in images)
    destination = _Destination(_folder(settings, paths), target.project or NAME_WITHOUT_PROJECT)
    return _named(_images(images), destination, now, batches=1)


def plan_queue(
    settings: GeneratorSettings,
    paths: RuntimePaths,
    now: datetime,
    new_seed: Callable[[], int] = random_seed,
) -> tuple[GenerationRequest, ...] | tuple[str, ...]:
    """The images of every queue entry in order, without the project checkpoint; any error
    stops the whole queue and names its entry."""
    if not settings.queue:
        return (EMPTY_QUEUE,)
    images: list[_Image] = []
    errors: list[str] = []
    for batch, entry in enumerate(settings.queue, start=1):
        seeds = _seeds(entry, new_seed, single=False)
        for item in _plan_entry(entry, paths, (None,), settings.lora_multiplier, seeds, batch):
            if isinstance(item, _Image):
                images.append(item)
            else:
                errors.append(item.replace(ERROR_PREFIX, f"{ERROR_PREFIX}Queue {batch}: ", 1))
    if errors:
        return tuple(errors)
    destination = _Destination(_folder(settings, paths), NAME_WITHOUT_PROJECT)
    return _named(tuple(images), destination, now, batches=len(settings.queue))


@dataclass(frozen=True)
class _Destination:
    folder: Path
    prefix: str


@dataclass(frozen=True)
class _Image:
    """One planned image, before its file name is chosen."""

    entry: GenerationEntry
    dit: Path
    loras: tuple[LoraUse, ...]  # the project checkpoint first, if any
    seed: int
    label: str
    metadata: ImageMetadata
    batch: int


def _folder(settings: GeneratorSettings, paths: RuntimePaths) -> Path:
    if settings.output_folder:
        return Path(settings.output_folder)
    return default_generated_folder(paths)


def _seeds(entry: GenerationEntry, new_seed: Callable[[], int], single: bool) -> tuple[int, ...]:
    if not entry.random_seed:
        return (entry.seed,)
    return tuple(new_seed() for _ in range(1 if single else entry.count))


def _images(items: tuple[_Image, ...] | tuple[str, ...]) -> tuple[_Image, ...]:
    return tuple(item for item in items if isinstance(item, _Image))


def _plan_entry(
    entry: GenerationEntry,
    paths: RuntimePaths,
    checkpoints: tuple[Checkpoint | None, ...],
    checkpoint_strength: float,
    seeds: tuple[int, ...],
    batch: int,
) -> tuple[_Image, ...] | tuple[str, ...]:
    problems = entry.problems()
    if problems:
        return tuple(f"{ERROR_PREFIX}{problem}" for problem in problems)
    own_model = bool(entry.model_file)
    dit = Path(entry.model_file) if own_model else paths.dit_model(entry.base_model)
    loras = tuple(LoraUse(Path(lora.path), lora.strength) for lora in entry.enabled_loras())
    errors = _file_errors(dit, own_model, loras)
    if errors:
        return errors
    images: list[_Image] = []
    for seed in seeds:
        for checkpoint in checkpoints:
            project = LoraUse(checkpoint.path, checkpoint_strength) if checkpoint else None
            images.append(
                _Image(
                    entry=entry,
                    dit=dit,
                    loras=(*((project,) if project else ()), *loras),
                    seed=seed,
                    label=_label(checkpoint, loras),
                    metadata=image_metadata(entry, seed, project, __version__),
                    batch=batch,
                )
            )
    return tuple(images)


def _named(
    images: tuple[_Image, ...], destination: _Destination, now: datetime, batches: int
) -> tuple[GenerationRequest, ...]:
    """The requests with their files; the images of a group are numbered in order."""
    numbered = len(images) > 1
    used: set[Path] = set()
    requests: list[GenerationRequest] = []
    for position, image in enumerate(images, start=1):
        output = output_image_path(
            destination.folder,
            destination.prefix,
            image.label,
            image.seed,
            now,
            used,
            number=position if numbered else None,
        )
        used.add(output)
        requests.append(_request(image, output, batches))
    return tuple(requests)


def _file_errors(dit: Path, own_model: bool, loras: tuple[LoraUse, ...]) -> tuple[str, ...]:
    if not own_model and not loras:
        return ()
    try:
        blocks = read_checkpoint_blocks(dit)
    except CheckpointFileError as error:
        return (f"{ERROR_PREFIX}{error}",)
    errors: list[str] = []
    for lora in loras:
        try:
            lines = check_lora(read_lora(lora.path), blocks)
        except LoraFileError as error:
            lines = (f"{ERROR_PREFIX}{error}",)
        errors += [line for line in lines if line.startswith(ERROR_PREFIX)]
    return tuple(errors)


def _label(checkpoint: Checkpoint | None, loras: tuple[LoraUse, ...]) -> str:
    if checkpoint is not None:
        return checkpoint.label
    return LORAS_LABEL if loras else NO_LORA_LABEL


def _request(image: _Image, output: Path, batches: int) -> GenerationRequest:
    entry = image.entry
    return GenerationRequest(
        dit=image.dit,
        loras=image.loras,
        prompt=entry.final_prompt(),
        negative_prompt=entry.negative_prompt,
        size=SampleSize(entry.width, entry.height),
        steps=entry.steps,
        cfg_scale=entry.cfg_scale,
        flow_shift=entry.flow_shift,
        seed=image.seed,
        sampler=entry.sampler,
        output=output,
        metadata=image.metadata,
        batch=image.batch,
        batches=batches,
    )
