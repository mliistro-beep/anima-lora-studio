from datetime import datetime

import pytest

from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.settings.schema import TrainingSettings
from anima_trainer.training.start_service import StartRequest, plan_start

NOW = datetime(2026, 10, 20, 9, 15, 0)


@pytest.fixture
def citron(tmp_path):
    paths = RuntimePaths(tmp_path / "citron")
    paths.train_script.parent.mkdir(parents=True)
    paths.train_script.write_text("")
    return paths


@pytest.fixture
def configured(citron):
    train = citron.configs_dir / "proj_training.toml"
    dataset = citron.configs_dir / "proj_dataset.toml"
    train.parent.mkdir(parents=True)
    train.write_text("")
    dataset.write_text("")
    return TrainingSettings(
        project_name="proj", last_train_config=str(train), last_dataset_config=str(dataset)
    )


def _request(**changes):
    values = {"gpu_index": "0", "cpu_threads": 2, "base_model": "anima-base-v1.0"}
    return StartRequest(**{**values, **changes})


def test_plan_uses_last_generated_configs(citron, configured):
    plan = plan_start(configured, citron, _request(), NOW)

    assert plan.success
    assert plan.spec is not None
    arguments = plan.spec.arguments
    assert arguments[arguments.index("--dataset_config") + 1] == configured.last_dataset_config
    assert configured.last_train_config in arguments
    assert arguments[3:5] == ("--num_cpu_threads_per_process", "2")
    assert plan.log_path == citron.logs_dir / "proj_2026-10-20_091500.log"


def test_plan_reports_gpu_and_command_like_citron(citron, configured):
    plan = plan_start(configured, citron, _request(gpu_index="1"), NOW)

    assert plan.messages[0] == "Using GPU index: 1"
    assert plan.messages[1].startswith("Command: ")
    assert plan.spec is not None and plan.spec.env["CUDA_VISIBLE_DEVICES"] == "1"


def test_custom_config_overrides_the_last_training_config(citron, configured, tmp_path):
    custom = tmp_path / "custom.toml"
    custom.write_text("")

    plan = plan_start(configured, citron, _request(custom_config=f"  {custom} "), NOW)

    assert plan.spec is not None
    assert str(custom) in plan.spec.arguments
    assert configured.last_train_config not in plan.spec.arguments


def test_relative_custom_config_is_checked_against_the_data_dir(citron, configured):
    (citron.root / "mine.toml").write_text("")

    plan = plan_start(configured, citron, _request(custom_config="mine.toml"), NOW)

    assert plan.success
    assert plan.spec is not None and "mine.toml" in plan.spec.arguments


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"last_train_config": ""}, "No training config found"),
        ({"last_train_config": "/nope/train.toml"}, "Training config not found"),
        ({"last_dataset_config": ""}, "No dataset config found"),
        ({"last_dataset_config": "/nope/data.toml"}, "Dataset config not found"),
    ],
)
def test_missing_configs_fail_with_citron_messages(citron, configured, changes, expected):
    plan = plan_start(configured.with_changes(**changes), citron, _request(), NOW)

    assert not plan.success
    assert plan.spec is None
    assert plan.messages[-1].startswith(f"ERROR: {expected}")


def test_missing_training_script_fails(citron, configured):
    citron.train_script.unlink()

    plan = plan_start(configured, citron, _request(), NOW)

    assert not plan.success
    assert "Training script not found" in plan.messages[-1]
