"""Tiny .safetensors files with chosen tensor names, for tests that read only the header."""

import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from anima_trainer.models.anima_dit_layout import BLOCK_TENSORS, OUTSIDE_TENSORS

TENSOR_BYTES = 2  # one bfloat16 value per tensor


def write_safetensors(
    path: Path,
    names: Iterable[str],
    metadata: dict[str, str] | None = None,
    dtype: str = "BF16",
    shapes: Mapping[str, Sequence[int]] | None = None,
) -> Path:
    header: dict[str, object] = {}
    offset = 0
    for name in names:
        header[name] = {
            "dtype": dtype,
            "shape": list((shapes or {}).get(name, [1])),
            "data_offsets": [offset, offset + TENSOR_BYTES],
        }
        offset += TENSOR_BYTES
    if metadata is not None:
        header["__metadata__"] = metadata
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + bytes(offset))
    return path


def kohya_module(name: str) -> tuple[str, ...]:
    return (f"{name}.lora_down.weight", f"{name}.lora_up.weight", f"{name}.alpha")


def peft_module(path: str) -> tuple[str, ...]:
    return (f"diffusion_model.{path}.lora_A.weight", f"diffusion_model.{path}.lora_B.weight")


def kohya_blocks(count: int) -> list[str]:
    return [
        key
        for block in range(count)
        for module in ("self_attn_q_proj", "mlp_layer1")
        for key in kohya_module(f"lora_unet_blocks_{block}_{module}")
    ]


def peft_blocks(count: int) -> list[str]:
    return [
        key
        for block in range(count)
        for module in ("self_attn.q_proj", "mlp.layer1")
        for key in peft_module(f"blocks.{block}.{module}")
    ]


def checkpoint_keys(blocks: int, prefix: str = "net.") -> list[str]:
    keys = [f"{prefix}x_embedder.proj.1.weight", f"{prefix}final_layer.linear.weight"]
    keys += [f"{prefix}blocks.{block}.self_attn.q_proj.weight" for block in range(blocks)]
    keys += [f"{prefix}llm_adapter.blocks.0.self_attn.q_proj.weight"]
    return keys


def anima_dit_shapes(blocks: int = 28, prefix: str = "net.") -> dict[str, tuple[int, ...]]:
    """The tensor names and shapes of a real Anima DiT; with other blocks, a larger Anima."""
    in_blocks = {
        f"{prefix}blocks.{block}.{name}": shape
        for block in range(blocks)
        for name, shape in BLOCK_TENSORS.items()
    }
    outside = {prefix + name: shape for name, shape in OUTSIDE_TENSORS.items()}
    return {**in_blocks, **outside}


def write_anima_dit(
    path: Path,
    blocks: int = 28,
    prefix: str = "net.",
    dtype: str = "BF16",
    changes: Mapping[str, Sequence[int] | None] | None = None,
) -> Path:
    """A header-only Anima DiT; `changes` gives a name another shape, adds it, or removes it
    (None)."""
    shapes: dict[str, Sequence[int]] = dict(anima_dit_shapes(blocks, prefix))
    for name, shape in (changes or {}).items():
        if shape is None:
            shapes.pop(name)
        else:
            shapes[name] = shape
    return write_safetensors(path, shapes, dtype=dtype, shapes=shapes)
