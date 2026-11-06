"""Tiny .safetensors files with chosen tensor names, for tests that read only the header."""

import json
import struct
from collections.abc import Iterable
from pathlib import Path

TENSOR_BYTES = 2  # one bfloat16 value per tensor


def write_safetensors(
    path: Path, names: Iterable[str], metadata: dict[str, str] | None = None
) -> Path:
    header: dict[str, object] = {}
    offset = 0
    for name in names:
        header[name] = {
            "dtype": "BF16",
            "shape": [1],
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
