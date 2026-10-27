"""Load a pydantic settings model from a JSON object file and save it atomically.

Problems are reported as warnings instead of being silently swallowed: an unreadable file
falls back to defaults, an invalid value falls back to that field's default.
"""

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

JSON_INDENT = 2
DEFAULT_FILE_MODE = 0o666


def _read_process_umask() -> int:
    # os.umask can only be read by setting it, and the umask is process-wide: do it once
    # at import, before worker threads exist, instead of on every save.
    umask = os.umask(0)
    os.umask(umask)
    return umask


_PROCESS_UMASK = _read_process_umask()


@dataclass(frozen=True)
class LoadedModel[M: BaseModel]:
    model: M
    warnings: tuple[str, ...]


def load_model[M: BaseModel](path: Path, model_type: type[M]) -> LoadedModel[M]:
    if not path.exists():
        return LoadedModel(model_type(), ())
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return LoadedModel(model_type(), (f"{path} could not be read: {error}",))
    if not isinstance(raw, dict):
        return LoadedModel(model_type(), (f"{path} could not be read: not a JSON object",))
    return _validate_per_field(raw, model_type)


def _validate_per_field[M: BaseModel](raw: dict[str, Any], model_type: type[M]) -> LoadedModel[M]:
    known = {key: value for key, value in raw.items() if key in model_type.model_fields}
    try:
        return LoadedModel(model_type.model_validate(known), ())
    except ValidationError as error:
        invalid = {str(detail["loc"][0]) for detail in error.errors() if detail["loc"]}
    valid = {key: value for key, value in known.items() if key not in invalid}
    warnings = tuple(
        f"Invalid value for '{key}' ({known[key]!r}), using default." for key in sorted(invalid)
    )
    return LoadedModel(model_type.model_validate(valid), warnings)


def save_model(path: Path, model: BaseModel) -> None:
    """Write the model as JSON (indent 2) atomically.

    The temporary file is fsynced before the rename and given the permissions the old
    file had (or the umask default), since mkstemp always creates it as 0600.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(model.model_dump(mode="json"), indent=JSON_INDENT)
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
        return DEFAULT_FILE_MODE & ~_PROCESS_UMASK
