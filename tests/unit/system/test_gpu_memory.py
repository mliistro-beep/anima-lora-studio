import subprocess

from anima_trainer.system.gpu_memory import (
    NVIDIA_SMI_MEMORY_COMMAND,
    GpuMemory,
    parse_memory_rows,
    read_gpu_memory,
)


def _run(returncode=0, stdout="", error=None):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if error is not None:
            raise error
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    return run, calls


def test_parse_memory_rows_reads_index_total_and_free_mib():
    rows = parse_memory_rows("0, 12288, 11386\n1, 8192, 100\n")

    assert rows == (GpuMemory(0, 12288, 11386), GpuMemory(1, 8192, 100))


def test_parse_memory_rows_skips_garbage_lines():
    assert parse_memory_rows("0, 12288, [N/A]\nhello\n\n2, 6144, 5000") == (
        GpuMemory(2, 6144, 5000),
    )


def test_read_gpu_memory_picks_the_selected_gpu():
    run, calls = _run(stdout="0, 12288, 11386\n1, 8192, 100\n")

    assert read_gpu_memory("1", run) == GpuMemory(1, 8192, 100)
    assert calls[0][0] == list(NVIDIA_SMI_MEMORY_COMMAND)
    assert calls[0][1]["timeout"] > 0


def test_read_gpu_memory_uses_the_first_of_several_selected_gpus():
    run, _ = _run(stdout="0, 12288, 11386\n1, 8192, 100\n")

    assert read_gpu_memory("0,1", run) == GpuMemory(0, 12288, 11386)


def test_read_gpu_memory_is_none_when_unknown():
    assert read_gpu_memory("3", _run(stdout="0, 12288, 11386")[0]) is None
    assert read_gpu_memory("0", _run(returncode=9, stdout="0, 1, 1")[0]) is None
    assert read_gpu_memory("0", _run(error=FileNotFoundError("nvidia-smi"))[0]) is None
    assert read_gpu_memory("0", _run(error=subprocess.TimeoutExpired("x", 1))[0]) is None
    assert read_gpu_memory("cpu", _run(stdout="0, 1, 1")[0]) is None


def test_non_decimal_digits_are_rejected_instead_of_crashing():
    run, _ = _run(stdout="0, 12288, 11386\n², 1, 1\n")

    assert read_gpu_memory("²", run) is None
    assert parse_memory_rows("², 1, 1") == ()
