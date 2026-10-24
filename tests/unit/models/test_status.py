from anima_trainer.models.catalog import QWEN3_FILENAME, VAE_FILENAME
from anima_trainer.models.status import model_statuses
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.choices import BASE_MODELS


def test_lists_required_models_then_every_base_model(tmp_path):
    statuses = model_statuses(RuntimePaths(tmp_path))

    assert [s.label for s in statuses] == ["Qwen3", "VAE", *(f"DiT {m}" for m in BASE_MODELS)]
    assert not any(s.present for s in statuses)


def test_every_model_has_a_download_key(tmp_path):
    statuses = model_statuses(RuntimePaths(tmp_path))

    assert [s.download_key for s in statuses] == [QWEN3_FILENAME, VAE_FILENAME, *BASE_MODELS]


def test_existing_files_are_present(tmp_path):
    paths = RuntimePaths(tmp_path)
    paths.vae_model.parent.mkdir(parents=True)
    paths.vae_model.write_bytes(b"")

    statuses = {s.label: s for s in model_statuses(paths)}

    assert statuses["VAE"].present
    assert statuses["VAE"].path == paths.vae_model
    assert not statuses["Qwen3"].present
