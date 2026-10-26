"""Anima models known to Citron: local file names, download URLs and expected files."""

from dataclasses import dataclass
from types import MappingProxyType

DEFAULT_DIT_FILENAME = "anima-base-v1.0.safetensors"
_HF_FILES = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files"
_HF_BASE = f"{_HF_FILES}/diffusion_models"

BASE_MODEL_URLS = MappingProxyType(
    {
        "anima-base-v1.0": f"{_HF_BASE}/anima-base-v1.0.safetensors",
        "anima-preview3-base": f"{_HF_BASE}/anima-preview3-base.safetensors",
        "anima-preview": f"{_HF_BASE}/anima-preview.safetensors",
    }
)

QWEN3_FILENAME = "qwen_3_06b_base.safetensors"
VAE_FILENAME = "qwen_image_vae.safetensors"

# Keyed by local file name (RuntimePaths uses the same constants); URLs as in Citron's setup.
SUPPORT_MODEL_URLS = MappingProxyType(
    {
        QWEN3_FILENAME: f"{_HF_FILES}/text_encoders/{QWEN3_FILENAME}",
        VAE_FILENAME: f"{_HF_FILES}/vae/{VAE_FILENAME}",
    }
)


@dataclass(frozen=True)
class ExpectedFile:
    sha256: str
    size: int  # checked first, so a wrong file is rejected without hashing it


# Keyed like the downloads (base model name, or file name for Qwen3 and the VAE). Values from
# the Hugging Face LFS metadata, matching the files downloaded in the M4 exit run.
EXPECTED_FILES = MappingProxyType(
    {
        "anima-base-v1.0": ExpectedFile(
            "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e", 4182218328
        ),
        "anima-preview3-base": ExpectedFile(
            "14fffe8ad5116cd73b9a4696f6a89d7e5f6efdd24b2e4785603aa891a9b2295b", 4182218360
        ),
        "anima-preview": ExpectedFile(
            "41fa7b78613dfe0d888b3647f70c7fb8cdfda2ff177e78d2e16e06dc810d9dcc", 4182218360
        ),
        QWEN3_FILENAME: ExpectedFile(
            "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba", 1192135096
        ),
        VAE_FILENAME: ExpectedFile(
            "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f", 253806246
        ),
    }
)


def dit_filename(base_model: str) -> str:
    """File name of the DiT weights; unknown models fall back to the default, as in Citron."""
    if base_model in BASE_MODEL_URLS:
        return f"{base_model}.safetensors"
    return DEFAULT_DIT_FILENAME


def model_label(key: str) -> str:
    """Name shown to the user for a model key, as in the Settings tab."""
    return {QWEN3_FILENAME: "Qwen3", VAE_FILENAME: "VAE"}.get(key, f"DiT {key}")
