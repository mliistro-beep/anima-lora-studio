"""Load and save settings in Citron's config.json format.

Unlike Citron, problems are reported as warnings instead of being silently swallowed:
an unreadable file falls back to defaults, an invalid value falls back to that field's default.
"""

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from anima_trainer.settings.schema import TrainingSettings

JSON_INDENT = 2
DEFAULT_FILE_MODE = 0o666


@dataclass(frozen=True)
class LoadResult:
    settings: TrainingSettings
    warnings: tuple[str, ...]


def load_settings(path: Path) -> LoadResult:
    if not path.exists():
        return LoadResult(TrainingSettings(), ())
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return LoadResult(TrainingSettings(), (f"{path} could not be read: {error}",))
    if not isinstance(raw, dict):
        return LoadResult(TrainingSettings(), (f"{path} could not be read: not a JSON object",))
    return _validate_per_field(raw)


def _validate_per_field(raw: dict[str, Any]) -> LoadResult:
    known = {key: value for key, value in raw.items() if key in TrainingSettings.model_fields}
    try:
        return LoadResult(TrainingSettings.model_validate(known), ())
    except ValidationError as error:
        invalid = {str(detail["loc"][0]) for detail in error.errors() if detail["loc"]}
    valid = {key: value for key, value in known.items() if key not in invalid}
    warnings = tuple(
        f"Invalid value for '{key}' ({known[key]!r}), using default." for key in sorted(invalid)
    )
    return LoadResult(TrainingSettings.model_validate(valid), warnings)


def save_settings(path: Path, settings: TrainingSettings) -> None:
    """Write config.json atomically, formatted exactly like Citron's json.dump(indent=2).

    The temporary file is fsynced before the rename and given the permissions the old
    file had (or the umask default), since mkstemp always creates it as 0600.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(settings.model_dump(), indent=JSON_INDENT)
    mode = _target_mode(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _target_mode(path: Path) -> int:
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        umask = os.umask(0)
        os.umask(umask)
        return DEFAULT_FILE_MODE & ~umask
