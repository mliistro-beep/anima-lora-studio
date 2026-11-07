"""Tensor names read from the header of a .safetensors file, without the weights."""

import struct

import pytest

from anima_trainer.preview.safetensors_header import (
    MAX_HEADER_BYTES,
    SafetensorsHeaderError,
    read_metadata,
    read_tensor_names,
)
from tests.safetensors_helpers import write_safetensors


def test_the_tensor_names_are_read_and_the_metadata_is_left_out(tmp_path):
    path = write_safetensors(tmp_path / "a.safetensors", ["b", "a"], {"format": "pt"})

    assert read_tensor_names(path) == ("b", "a")


def test_a_missing_file_is_a_header_error_naming_it(tmp_path):
    with pytest.raises(SafetensorsHeaderError, match="missing.safetensors"):
        read_tensor_names(tmp_path / "missing.safetensors")


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"1234",
        struct.pack("<Q", 50) + b"{}",
        struct.pack("<Q", 2) + b"[]",
        struct.pack("<Q", 3) + b"{x}",
        struct.pack("<Q", 7) + b'{"a":1}',
        struct.pack("<Q", 4) + b"\xff\xfe{}",
    ],
    ids=["empty", "short", "cut", "not-object", "bad-json", "entry-not-object", "not-utf8"],
)
def test_a_file_that_is_not_safetensors_is_a_header_error(tmp_path, content):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(content)

    with pytest.raises(SafetensorsHeaderError, match="bad.safetensors"):
        read_tensor_names(path)


def test_a_header_larger_than_the_limit_is_refused_without_reading_it(tmp_path):
    path = tmp_path / "huge.safetensors"
    path.write_bytes(struct.pack("<Q", MAX_HEADER_BYTES + 1) + b"{}")

    with pytest.raises(SafetensorsHeaderError, match="header"):
        read_tensor_names(path)


def test_the_metadata_strings_are_read(tmp_path):
    path = write_safetensors(tmp_path / "a.safetensors", ["w"], {"modelspec.trigger_phrase": "@x"})

    assert read_metadata(path) == {"modelspec.trigger_phrase": "@x"}


def test_a_file_without_metadata_has_none(tmp_path):
    assert read_metadata(write_safetensors(tmp_path / "a.safetensors", ["w"])) == {}


def test_metadata_values_that_are_not_strings_are_left_out(tmp_path):
    path = tmp_path / "odd.safetensors"
    text = b'{"__metadata__": {"a": "1", "b": 2}}'
    path.write_bytes(struct.pack("<Q", len(text)) + text)

    assert read_metadata(path) == {"a": "1"}
