"""What the Download buttons of the Settings tab fetch, by download key.

The key is the base model name for a DiT and the local file name for Qwen3 and the VAE.
Each download carries the expected size and SHA-256, checked before the file is kept.
"""

from dataclasses import dataclass
from pathlib import Path

from anima_trainer.models.catalog import (
    BASE_MODEL_URLS,
    EXPECTED_FILES,
    QWEN3_FILENAME,
    SUPPORT_MODEL_URLS,
    VAE_FILENAME,
    ExpectedFile,
    model_label,
)
from anima_trainer.runtime.paths import RuntimePaths


@dataclass(frozen=True)
class ModelDownload:
    description: str  # completes "Downloading ...", as in Citron's DiT message
    url: str
    destination: Path
    expected: ExpectedFile | None = None


def model_download(paths: RuntimePaths, key: str) -> ModelDownload | None:
    """The download for `key`, or None when the file is already there or the key is unknown."""
    download = _support_download(paths, key) or _dit_download(paths, key)
    return _if_missing(download)


def dit_model_download(paths: RuntimePaths, base_model: str) -> ModelDownload | None:
    """The DiT to fetch before training, or None when it is there or the name is unknown."""
    return _if_missing(_dit_download(paths, base_model))


def _if_missing(download: ModelDownload | None) -> ModelDownload | None:
    if download is None or download.destination.exists():
        return None
    return download


def _support_download(paths: RuntimePaths, key: str) -> ModelDownload | None:
    destinations = {QWEN3_FILENAME: paths.qwen3_model, VAE_FILENAME: paths.vae_model}
    if key not in destinations:
        return None
    return ModelDownload(
        model_label(key), SUPPORT_MODEL_URLS[key], destinations[key], EXPECTED_FILES.get(key)
    )


def _dit_download(paths: RuntimePaths, key: str) -> ModelDownload | None:
    url = BASE_MODEL_URLS.get(key)
    if url is None:
        return None
    return ModelDownload(f"base model '{key}'", url, paths.dit_model(key), EXPECTED_FILES.get(key))
