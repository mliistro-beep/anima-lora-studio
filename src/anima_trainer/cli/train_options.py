"""The options of `anima-trainer train`.

Required values are checked after parsing, not by argparse: the message about the epochs names
what the chosen preset suggests, and --list-presets needs none of them.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from anima_trainer.presets.preset_file import Preset

DEFAULT_RESOLUTION = 768


@dataclass(frozen=True)
class TrainOptions:
    list_presets: bool
    preset: str | None
    project: str | None
    dataset: str | None
    output: str | None
    epochs: int | None
    repeats: int | None
    resolution: int
    yes: bool


def parse_train_options(arguments: Sequence[str]) -> TrainOptions:
    """Exits with code 2 and a usage message, like any command, on a malformed option."""
    parser = argparse.ArgumentParser(
        prog="anima-trainer train", description="Train a LoRA from a preset, without a window."
    )
    parser.add_argument("--list-presets", action="store_true", help="show the presets and exit")
    parser.add_argument("--preset", help="preset name")
    parser.add_argument("--project", help="project name, also the name of the LoRA file")
    parser.add_argument("--dataset", help="folder with the images and captions")
    parser.add_argument("--output", help="folder for the LoRA files")
    parser.add_argument("--epochs", type=int, help="number of epochs")
    parser.add_argument("--repeats", type=int, help="repeats of each image in an epoch")
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"training resolution (default {DEFAULT_RESOLUTION})",
    )
    parser.add_argument("--yes", action="store_true", help="start without asking for confirmation")
    return TrainOptions(**vars(parser.parse_args(arguments)))


def missing_option_lines(options: TrainOptions, preset: Preset) -> tuple[str, ...]:
    """An ERROR line for each value of the training that was not given."""
    texts = (
        ("--project", options.project),
        ("--dataset", options.dataset),
        ("--output", options.output),
    )
    lines = [f"ERROR: {name} is required" for name, value in texts if not (value or "").strip()]
    if options.epochs is None:
        suggested = preset.values.get("max_train_epochs")
        hint = f" (preset {preset.name} suggests {suggested})" if suggested else ""
        lines.append(f"ERROR: --epochs is required{hint}")
    if options.repeats is None:
        lines.append("ERROR: --repeats is required")
    return tuple(lines)
