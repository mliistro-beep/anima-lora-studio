"""Anima models known to Citron: local file names and download URLs."""

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


def dit_filename(base_model: str) -> str:
    """File name of the DiT weights; unknown models fall back to the default, as in Citron."""
    if base_model in BASE_MODEL_URLS:
        return f"{base_model}.safetensors"
    return DEFAULT_DIT_FILENAME
