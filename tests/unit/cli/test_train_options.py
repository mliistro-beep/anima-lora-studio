"""The options of `anima-trainer train`."""

import pytest

from anima_trainer.cli.train_options import missing_option_lines, parse_train_options
from anima_trainer.presets.preset_file import parse_preset

PRESET = parse_preset('name = "Person"\n[settings]\nmax_train_epochs = 10\n')
ALL = (
    "--preset", "Person", "--project", "p", "--dataset", "/d", "--output", "/o",
    "--epochs", "3", "--repeats", "5",
)  # fmt: skip


def _without(option):
    position = ALL.index(option)
    return (*ALL[:position], *ALL[position + 2 :])


def test_all_the_options_are_read():
    options = parse_train_options((*ALL, "--resolution", "1024", "--yes"))
    assert options.preset == "Person"
    assert options.project == "p"
    assert options.dataset == "/d"
    assert options.output == "/o"
    assert options.epochs == 3
    assert options.repeats == 5
    assert options.resolution == 1024
    assert options.yes


def test_the_resolution_starts_at_768_and_confirmation_is_on():
    options = parse_train_options(ALL)
    assert options.resolution == 768
    assert not options.yes


def test_complete_options_have_nothing_missing():
    assert missing_option_lines(parse_train_options(ALL), PRESET) == ()


def test_the_epochs_are_required_and_the_preset_suggests_them():
    lines = missing_option_lines(parse_train_options(_without("--epochs")), PRESET)
    assert lines == ("ERROR: --epochs is required (preset Person suggests 10)",)


def test_the_suggestion_is_left_out_when_the_preset_has_none():
    preset = parse_preset('name = "Bare"')
    lines = missing_option_lines(parse_train_options(_without("--epochs")), preset)
    assert lines == ("ERROR: --epochs is required",)


@pytest.mark.parametrize("option", ["--project", "--dataset", "--output", "--repeats"])
def test_each_value_of_a_training_is_required(option):
    lines = missing_option_lines(parse_train_options(_without(option)), PRESET)
    assert lines == (f"ERROR: {option} is required",)


@pytest.mark.parametrize("option", ["--project", "--dataset", "--output"])
def test_a_blank_value_counts_as_missing(option):
    arguments = list(ALL)
    arguments[arguments.index(option) + 1] = "  "
    lines = missing_option_lines(parse_train_options(arguments), PRESET)
    assert lines == (f"ERROR: {option} is required",)


def test_all_the_missing_options_are_named_at_once():
    lines = missing_option_lines(parse_train_options(("--preset", "Person")), PRESET)
    assert len(lines) == 5


def test_a_value_that_is_not_a_number_stops_with_a_usage_error():
    with pytest.raises(SystemExit) as stopped:
        parse_train_options(("--epochs", "many"))
    assert stopped.value.code == 2


def test_the_list_of_presets_needs_no_other_option():
    assert parse_train_options(("--list-presets",)).list_presets
