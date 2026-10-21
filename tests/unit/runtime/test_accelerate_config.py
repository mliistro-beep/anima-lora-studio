from anima_trainer.runtime.accelerate_config import ACCELERATE_GPU_YAML, write_accelerate_config
from anima_trainer.runtime.data_dir import runtime_problems
from anima_trainer.runtime.paths import RuntimePaths


def test_yaml_is_identical_to_citron_file(citron_fixtures):
    golden = (citron_fixtures / "accelerate_gpu.yaml").read_text(encoding="utf-8")

    assert golden == ACCELERATE_GPU_YAML


def test_write_creates_the_config_folder_and_file(tmp_path):
    paths = RuntimePaths(tmp_path / "data")

    written = write_accelerate_config(paths)

    assert written == paths.accelerate_config_file
    assert written.read_text(encoding="utf-8") == ACCELERATE_GPU_YAML


def test_write_replaces_an_edited_or_broken_file(tmp_path):
    paths = RuntimePaths(tmp_path / "data")
    paths.accelerate_config_file.parent.mkdir(parents=True)
    paths.accelerate_config_file.write_text("broken: [")

    write_accelerate_config(paths)

    assert paths.accelerate_config_file.read_text(encoding="utf-8") == ACCELERATE_GPU_YAML


def test_written_config_no_longer_counts_as_missing(tmp_path):
    paths = RuntimePaths(tmp_path / "data")

    write_accelerate_config(paths)

    assert not any("accelerate_gpu.yaml" in problem for problem in runtime_problems(paths))
