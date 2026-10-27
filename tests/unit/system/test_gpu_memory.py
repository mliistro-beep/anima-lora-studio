import subprocess

from anima_trainer.system.gpu_memory import (
    NVIDIA_SMI_MEMORY_COMMAND,
    GpuMemory,
    memory_of,
    parse_memory_rows,
    read_all_gpu_memory,
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


GPUS = (GpuMemory(0, 12288, 11386), GpuMemory(1, 8192, 100))


def test_memory_of_picks_the_selected_gpu():
    assert memory_of(GPUS, "1") == GpuMemory(1, 8192, 100)


def test_memory_of_uses_the_first_of_several_selected_gpus():
    assert memory_of(GPUS, "0,1") == GpuMemory(0, 12288, 11386)


def test_memory_of_is_none_when_unknown():
    assert memory_of(GPUS, "3") is None
    assert memory_of(GPUS, "cpu") is None
    assert memory_of((), "0") is None


def test_non_decimal_digits_are_rejected_instead_of_crashing():
    assert memory_of(GPUS, "\u00b2") is None
    assert parse_memory_rows("\u00b2, 1, 1") == ()


def test_read_all_gpu_memory_lists_every_gpu_or_nothing():
    run, calls = _run(stdout="0, 12288, 11386\n1, 8192, 100\n")

    assert read_all_gpu_memory(run) == GPUS
    assert calls[0][0] == list(NVIDIA_SMI_MEMORY_COMMAND)
    assert calls[0][1]["timeout"] > 0
    assert read_all_gpu_memory(_run(error=FileNotFoundError("nvidia-smi"))[0]) == ()
    assert read_all_gpu_memory(_run(returncode=9, stdout="0, 1, 1")[0]) == ()
