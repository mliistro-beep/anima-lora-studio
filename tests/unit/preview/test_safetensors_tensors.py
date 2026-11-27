import pytest

from anima_trainer.preview.safetensors_header import (
    SafetensorsHeaderError,
    TensorInfo,
    read_tensors,
)
from tests.safetensors_helpers import write_safetensors


def test_the_dtype_and_shape_of_each_tensor_are_read_and_metadata_is_skipped(tmp_path):
    path = write_safetensors(
        tmp_path / "m.safetensors", ["a", "b"], {"k": "v"}, dtype="F16", shapes={"b": (2, 3)}
    )

    assert read_tensors(path) == {"a": TensorInfo("F16", (1,)), "b": TensorInfo("F16", (2, 3))}


@pytest.mark.parametrize(
    "entry",
    [b'{"shape": [1]}', b'{"dtype": "BF16"}', b'{"dtype": "BF16", "shape": [true]}', b"[1]"],
)
def test_an_entry_without_a_dtype_or_a_shape_is_not_a_tensor(tmp_path, entry):
    path = tmp_path / "bad.safetensors"
    text = b'{"a": ' + entry + b"}"
    path.write_bytes(len(text).to_bytes(8, "little") + text)

    with pytest.raises(SafetensorsHeaderError, match="not a tensor"):
        read_tensors(path)
