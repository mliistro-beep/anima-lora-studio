import pytest

from anima_trainer.models.catalog import (
    BASE_MODEL_URLS,
    QWEN3_FILENAME,
    SUPPORT_MODEL_URLS,
    VAE_FILENAME,
)
from anima_trainer.models.model_downloads import ModelDownload, model_download
from anima_trainer.runtime.paths import RuntimePaths


def test_qwen3_and_vae_come_from_the_support_urls(tmp_path):
    paths = RuntimePaths(tmp_path)

    assert model_download(paths, QWEN3_FILENAME) == ModelDownload(
        "Qwen3", SUPPORT_MODEL_URLS[QWEN3_FILENAME], paths.qwen3_model
    )
    assert model_download(paths, VAE_FILENAME) == ModelDownload(
        "VAE", SUPPORT_MODEL_URLS[VAE_FILENAME], paths.vae_model
    )


def test_a_base_model_downloads_its_dit_with_the_citron_wording(tmp_path):
    paths = RuntimePaths(tmp_path)

    assert model_download(paths, "anima-preview") == ModelDownload(
        "base model 'anima-preview'",
        BASE_MODEL_URLS["anima-preview"],
        paths.dit_model("anima-preview"),
    )


@pytest.mark.parametrize("key", [QWEN3_FILENAME, "anima-base-v1.0"])
def test_present_files_need_no_download(tmp_path, key):
    paths = RuntimePaths(tmp_path)
    destination = model_download(paths, key).destination
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"")

    assert model_download(paths, key) is None


def test_unknown_keys_have_no_download(tmp_path):
    assert model_download(RuntimePaths(tmp_path), "sdxl") is None
