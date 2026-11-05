"""The presets available: the built-in ones, then the user's own from a folder.

A file that cannot be used becomes a warning: it must not hide the presets that work.
"""

from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from anima_trainer.presets.preset_file import Preset, PresetError, parse_preset

BUILTIN_FILES = ("style.toml", "character.toml", "concept.toml", "person.toml")


@dataclass(frozen=True)
class LoadedPresets:
    presets: tuple[Preset, ...]
    warnings: tuple[str, ...]


def load_presets(user_folder: Path) -> LoadedPresets:
    builtin_folder = resources.files("anima_trainer.presets") / "builtin"
    sources: list[tuple[Traversable, bool]] = [
        (builtin_folder / name, True) for name in BUILTIN_FILES
    ]
    presets: dict[str, Preset] = {}
    warnings: list[str] = []
    try:
        user_files = sorted(user_folder.glob("*.toml"), key=lambda item: item.name)
    except OSError as error:
        warnings.append(f"WARN: Presets folder {user_folder} could not be read: {error}")
    else:
        sources += [(path, False) for path in user_files]
    for source, builtin in sources:
        try:
            preset = parse_preset(source.read_text(encoding="utf-8"), builtin)
        except (OSError, UnicodeDecodeError, PresetError) as error:
            warnings.append(_ignored(source.name, str(error)))
            continue
        key = preset.name.casefold()
        if key in presets:
            warnings.append(_ignored(source.name, f"the name '{preset.name}' is already used"))
            continue
        presets[key] = preset
    return LoadedPresets(tuple(presets.values()), tuple(warnings))


def find_preset(loaded: LoadedPresets, name: str) -> Preset | None:
    wanted = name.casefold()
    return next((preset for preset in loaded.presets if preset.name.casefold() == wanted), None)


def _ignored(file_name: str, reason: str) -> str:
    return f"WARN: Preset {file_name} ignored: {reason}"
