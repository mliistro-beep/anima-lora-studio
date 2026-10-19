"""Anima base models known to Citron: local file names and download URLs."""

from types import MappingProxyType

DEFAULT_DIT_FILENAME = "anima-base-v1.0.safetensors"
_HF_BASE = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models"

BASE_MODEL_URLS = MappingProxyType(
    {
        "anima-base-v1.0": f"{_HF_BASE}/anima-base-v1.0.safetensors",
        "anima-preview3-base": f"{_HF_BASE}/anima-preview3-base.safetensors",
        "anima-preview": f"{_HF_BASE}/anima-preview.safetensors",
    }
)


def dit_filename(base_model: str) -> str:
    """File name of the DiT weights; unknown models fall back to the default, as in Citron."""
    if base_model in BASE_MODEL_URLS:
        return f"{base_model}.safetensors"
    return DEFAULT_DIT_FILENAME
