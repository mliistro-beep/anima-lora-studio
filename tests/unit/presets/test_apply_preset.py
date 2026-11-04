"""A preset gives the same training every time, whatever was used last."""

from anima_trainer.presets.apply import apply_preset
from anima_trainer.presets.preset_file import parse_preset
from anima_trainer.settings.schema import TrainingSettings

PRESET = parse_preset('name = "x"\n[settings]\nnetwork_dim = 24\nlearning_rate = 0.0002\n')


def _saved(**changes):
    return TrainingSettings().with_changes(
        project_name="proj",
        image_directory="/images",
        output_directory="/out",
        repeats=7,
        resolution=1024,
        gpu_index="1",
        num_cpu_threads_per_process=4,
        cache_latents=False,
        last_train_config="a.toml",
        last_dataset_config="b.toml",
        **changes,
    )


def test_the_values_of_the_preset_are_applied():
    settings = apply_preset(PRESET, _saved())
    assert settings.network_dim == 24
    assert settings.learning_rate == 0.0002


def test_what_the_preset_does_not_set_comes_from_the_defaults_not_the_last_training():
    settings = apply_preset(PRESET, _saved(seed=1234, max_train_epochs=99, network_alpha=3))
    defaults = TrainingSettings()
    assert settings.seed == defaults.seed
    assert settings.max_train_epochs == defaults.max_train_epochs
    assert settings.network_alpha == defaults.network_alpha


def test_the_values_of_each_training_and_of_this_machine_are_kept():
    saved = _saved()
    settings = apply_preset(PRESET, saved)
    for field in (
        "project_name",
        "image_directory",
        "output_directory",
        "repeats",
        "resolution",
        "gpu_index",
        "num_cpu_threads_per_process",
        "cache_latents",
        "last_train_config",
        "last_dataset_config",
    ):
        assert getattr(settings, field) == getattr(saved, field), field


def test_the_same_preset_gives_the_same_result_after_any_other_training():
    first = apply_preset(PRESET, _saved(seed=1, network_dim=8))
    second = apply_preset(PRESET, _saved(seed=2, network_dim=64))
    assert first == second


def test_the_saved_settings_are_left_unchanged():
    saved = _saved()
    copy = saved.model_copy()
    apply_preset(PRESET, saved)
    assert saved == copy
