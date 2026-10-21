import subprocess
from pathlib import Path

from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.system.gpu_info import (
    FALLBACK_CHOICES,
    NO_CUDA_CHOICE,
    detect_gpus,
    parse_nvidia_smi,
    parse_torch_output,
)

PATHS = RuntimePaths(Path("/opt/citron"), platform="posix")


def _runner(outputs):
    """Fake subprocess.run: maps the program name to (returncode, stdout) or an exception."""
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        result = outputs[Path(cmd[0]).name]
        if isinstance(result, Exception):
            raise result
        code, stdout = result
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr="")

    return run, calls


def test_parse_torch_output_lists_devices():
    assert parse_torch_output('["NVIDIA GeForce RTX 3060", "RTX 4090"]\n') == (
        "0: NVIDIA GeForce RTX 3060",
        "1: RTX 4090",
    )


def test_parse_torch_output_without_cuda():
    assert parse_torch_output("null") == (NO_CUDA_CHOICE,)


def test_parse_torch_output_uses_last_line_only():
    assert parse_torch_output('warning: something\n["GPU A"]') == ("0: GPU A",)


def test_parse_torch_output_rejects_garbage():
    assert parse_torch_output("not json") is None
    assert parse_torch_output("[]") is None
    assert parse_torch_output('{"a": 1}') is None


def test_parse_nvidia_smi():
    assert parse_nvidia_smi("0, NVIDIA GeForce RTX 3060\n1, Other\n") == (
        "0: NVIDIA GeForce RTX 3060",
        "1: Other",
    )
    assert parse_nvidia_smi("") is None


def test_detect_uses_citron_venv_torch_first():
    run, calls = _runner({"python": (0, '["NVIDIA GeForce RTX 3060"]')})

    assert detect_gpus(PATHS, run=run) == ("0: NVIDIA GeForce RTX 3060",)
    assert calls[0][0] == str(PATHS.venv_python)


def test_detect_falls_back_to_nvidia_smi():
    run, _ = _runner({"python": FileNotFoundError(), "nvidia-smi": (0, "0, RTX 3060")})

    assert detect_gpus(PATHS, run=run) == ("0: RTX 3060",)


def test_detect_falls_back_when_torch_fails():
    run, _ = _runner({"python": (1, "Traceback"), "nvidia-smi": (0, "0, RTX 3060")})

    assert detect_gpus(PATHS, run=run) == ("0: RTX 3060",)


def test_detect_returns_citron_fallback_when_nothing_works():
    run, _ = _runner(
        {"python": subprocess.TimeoutExpired("python", 1), "nvidia-smi": FileNotFoundError()}
    )

    assert detect_gpus(PATHS, run=run) == FALLBACK_CHOICES
    assert FALLBACK_CHOICES == ("0", "1")


def test_missing_data_dir_does_not_break_nvidia_smi(tmp_path):
    paths = RuntimePaths(tmp_path / "does-not-exist", platform="posix")
    seen_cwds = []

    def run(cmd, **kwargs):
        seen_cwds.append(kwargs.get("cwd"))
        if Path(cmd[0]).name == "python":
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, 0, stdout="0, RTX 3060", stderr="")

    assert detect_gpus(paths, run=run) == ("0: RTX 3060",)
    assert seen_cwds == [None, None]
