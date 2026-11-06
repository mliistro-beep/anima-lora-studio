"""The tensor names of a .safetensors file, read from its header only (no weights, no torch).

The format: 8 bytes with the header length (little-endian), then the header as a JSON object
of tensor entries, plus an optional `__metadata__` entry. Nothing else of this package is
imported, so the module can be copied as it is (Anima Merger Studio does).
"""

import json
import struct
from pathlib import Path

LENGTH_BYTES = 8
# The limit of the safetensors format itself; a larger value means the file is not one.
MAX_HEADER_BYTES = 100 * 1024 * 1024
METADATA_KEY = "__metadata__"


class SafetensorsHeaderError(Exception):
    """The file cannot be read, or is not a .safetensors file."""


def read_tensor_names(path: Path) -> tuple[str, ...]:
    """The tensor names in the order of the header; raises SafetensorsHeaderError."""
    header = _read_header(path)
    names = tuple(name for name in header if name != METADATA_KEY)
    if not all(isinstance(header[name], dict) for name in names):
        raise _not_safetensors(path, "a header entry is not a tensor")
    return names


def _read_header(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as file:
            length_bytes = file.read(LENGTH_BYTES)
            if len(length_bytes) < LENGTH_BYTES:
                raise _not_safetensors(path, "the file is too short")
            (length,) = struct.unpack("<Q", length_bytes)
            if length > MAX_HEADER_BYTES:
                raise _not_safetensors(path, f"its header would be {length} bytes")
            text = file.read(length)
    except OSError as error:
        raise SafetensorsHeaderError(f"{path} cannot be read: {error.strerror or error}") from error
    if len(text) < length:
        raise _not_safetensors(path, "the file ends inside its header")
    try:
        header = json.loads(text.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _not_safetensors(path, "its header is not valid JSON") from error
    if not isinstance(header, dict):
        raise _not_safetensors(path, "its header is not a JSON object")
    return header


def _not_safetensors(path: Path, reason: str) -> SafetensorsHeaderError:
    return SafetensorsHeaderError(f"{path.name} is not a .safetensors file ({reason}).")
