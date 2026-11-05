"""From the options to the settings of the training, and what the user approves."""

from pathlib import Path

import pytest

from anima_trainer.cli.train_options import parse_train_options
from anima_trainer.cli.train_summary import (
    base_model_problem,
    preset_list_lines,
    resolution_warning,
    settings_for,
    summary_lines,
)
from anima_trainer.dataset.validator import DatasetReport
from anima_trainer.presets.preset_file import parse_preset
from anima_trainer.preview.settings import PreviewSettings
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.system.gpu_memory import GpuMemory

PERSON = parse_preset(
    'name = "Person"\ndescription = "About 60 images"\n[settings]\n'
    "network_dim = 32\nnetwork_alpha = 16\nmax_train_epochs = 10\ncaption_dropout = 0.05\n"
)
OPTIONS = (
    "--preset", "Person", "--project", "proj", "--dataset", "/images", "--output", "/out",
    "--epochs", "3", "--repeats", "5",
)  # fmt: skip


def _options(*extra):
    return parse_train_options((*OPTIONS, *extra))


def _settings(*extra):
    return settings_for(_options(*extra), PERSON, TrainingSettings())


def test_the_settings_join_the_preset_and_the_options():
    settings = _settings()
    assert settings.network_alpha == 16
    assert settings.caption_dropout == 0.05
    assert settings.project_name == "proj"
    assert settings.image_directory == "/images"
    assert settings.output_directory == "/out"
    assert settings.max_train_epochs == 3
    assert settings.repeats == 5
    assert settings.resolution == 768


def test_the_settings_of_this_machine_come_from_the_saved_ones():
    saved = TrainingSettings().with_changes(gpu_index="1", num_cpu_threads_per_process=4)
    settings = settings_for(_options(), PERSON, saved)
    assert settings.gpu_index == "1"
    assert settings.num_cpu_threads_per_process == 4


def test_a_relative_folder_means_from_where_the_command_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    arguments = list(OPTIONS)
    arguments[arguments.index("--dataset") + 1] = "images"
    arguments[arguments.index("--output") + 1] = "../out"
    settings = settings_for(parse_train_options(arguments), PERSON, TrainingSettings())
    assert settings.image_directory == str(tmp_path / "images")
    assert settings.output_directory == str(tmp_path.parent / "out")


def test_the_home_folder_is_expanded_when_the_shell_did_not(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    arguments = list(OPTIONS)
    arguments[arguments.index("--dataset") + 1] = "~/images"
    settings = settings_for(parse_train_options(arguments), PERSON, TrainingSettings())
    assert settings.image_directory == str(Path(tmp_path) / "images")


@pytest.mark.parametrize(
    "extra", [("--project", "a/b"), ("--epochs", "0"), ("--repeats", "0"), ("--resolution", "8")]
)
def test_a_value_the_settings_refuse_stops_here(extra):
    with pytest.raises(ValueError, match=r"project_name|max_train_epochs|repeats|resolution"):
        _settings(*extra)


def test_only_the_base_model_can_be_trained_from_the_terminal():
    assert base_model_problem(TrainingSettings()) is None
    other = TrainingSettings().with_changes(base_model=_another_model())
    problem = base_model_problem(other)
    assert problem.startswith("ERROR: ")
    assert "anima-base-v1.0" in problem
    assert "graphical app" in problem


def _another_model():
    from typing import get_args  # noqa: PLC0415

    from anima_trainer.settings.choices import BaseModelName  # noqa: PLC0415

    return next(name for name in get_args(BaseModelName) if name != "anima-base-v1.0")


def test_the_summary_shows_samples_and_total_steps():
    report = DatasetReport(60, (), ())
    text = "\n".join(summary_lines(PERSON, _settings(), report, PreviewSettings()))
    assert "Person" in text
    assert "proj" in text
    assert "/images" in text
    assert "/out" in text
    assert "60 images x 5 repeats x 3 epochs = 900 samples" in text
    assert "900 steps" in text
    assert "768" in text
    assert "Previews:  off" in text


def test_the_summary_counts_steps_with_batch_and_accumulation():
    preset = parse_preset('name = "x"\n[settings]\ngradient_accumulation_steps = 2\n')
    settings = settings_for(_options(), preset, TrainingSettings())
    text = "\n".join(summary_lines(preset, settings, DatasetReport(60, (), ()), PreviewSettings()))
    assert "900 samples" in text
    assert "450 steps" in text


def test_the_summary_says_when_previews_are_on():
    previews = PreviewSettings(enabled=True, prompts=("a", "b"))
    text = "\n".join(summary_lines(PERSON, _settings(), DatasetReport(1, (), ()), previews))
    assert "Previews:  on (2 prompts)" in text


def test_the_summary_passes_on_the_warnings_of_the_dataset():
    report = DatasetReport(2, ("a.png",), ("1 image(s) are missing caption (.txt) files.",))
    lines = summary_lines(PERSON, _settings(), report, PreviewSettings())
    assert "WARN: 1 image(s) are missing caption (.txt) files." in lines


@pytest.mark.parametrize(
    ("resolution", "total_mib", "warned"),
    [("768", 12288, False), ("1024", 12288, True), ("1024", 8192, True), ("1024", 16384, False)],
)
def test_a_high_resolution_warns_on_a_small_gpu(resolution, total_mib, warned):
    settings = _settings("--resolution", resolution).with_changes(gpu_index="0")
    warning = resolution_warning(settings, (GpuMemory(0, total_mib, total_mib),))
    assert (warning is not None) == warned


def test_no_warning_when_the_gpu_memory_is_unknown():
    assert resolution_warning(_settings("--resolution", "1024"), ()) is None


def test_the_list_shows_name_description_and_suggested_epochs():
    lines = preset_list_lines((PERSON, parse_preset('name = "Bare"')))
    assert any("Person" in line and "About 60 images" in line and "10" in line for line in lines)
    assert any("Bare" in line for line in lines)


def test_the_summary_counts_one_image_repeat_and_epoch_in_the_singular():
    arguments = list(OPTIONS)
    arguments[arguments.index("--repeats") + 1] = "1"
    arguments[arguments.index("--epochs") + 1] = "1"
    settings = settings_for(parse_train_options(arguments), PERSON, TrainingSettings())
    text = "\n".join(summary_lines(PERSON, settings, DatasetReport(1, (), ()), PreviewSettings()))
    assert "1 image x 1 repeat x 1 epoch = 1 sample" in text
    assert "1 step" in text
    assert "1 steps" not in text


def test_the_summary_says_one_prompt_in_the_singular():
    previews = PreviewSettings(enabled=True, prompts=("a",))
    text = "\n".join(summary_lines(PERSON, _settings(), DatasetReport(1, (), ()), previews))
    assert "Previews:  on (1 prompt)" in text


def test_a_link_to_a_folder_stays_as_the_user_wrote_it(tmp_path, monkeypatch):
    target = tmp_path / "other_disk"
    target.mkdir()
    (tmp_path / "images").symlink_to(target)
    monkeypatch.chdir(tmp_path)
    arguments = list(OPTIONS)
    arguments[arguments.index("--dataset") + 1] = "images"
    settings = settings_for(parse_train_options(arguments), PERSON, TrainingSettings())
    assert settings.image_directory == str(tmp_path / "images")
