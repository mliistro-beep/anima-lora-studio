from datetime import datetime

from anima_trainer.system.resource_usage import (
    GpuUsage,
    MemoryUsage,
    parse_gpu_usage,
    parse_meminfo,
    resource_line,
)

NOW = datetime(2026, 10, 28, 19, 40, 0, 123456)
MEMINFO = (
    "MemTotal:       14675968 kB\n"
    "MemFree:         1887956 kB\n"
    "MemAvailable:   10485760 kB\n"
    "SwapTotal:      16683008 kB\n"
    "SwapFree:        8493056 kB\n"
)
FULL_MEMORY = MemoryUsage(
    available_mib=10240, total_mib=14332, swap_used_mib=7998, swap_total_mib=16292
)
FULL_GPU = GpuUsage(
    index=0,
    used_mib=9318,
    total_mib=12288,
    temperature_c=72,
    power_w=165.4,
    power_limit_w=170.0,
    utilization_pct=99,
)


def test_meminfo_gives_ram_and_swap_in_mib():
    assert parse_meminfo(MEMINFO) == FULL_MEMORY


def test_meminfo_without_swap_lines_has_no_swap():
    memory = parse_meminfo("MemTotal: 14675968 kB\nMemAvailable: 10485760 kB\n")

    assert memory == MemoryUsage(10240, 14332, None, None)


def test_meminfo_without_ram_lines_is_none():
    assert parse_meminfo("SwapTotal: 16683008 kB\nSwapFree: 8493056 kB\n") is None
    assert parse_meminfo("") is None


def test_nvidia_smi_rows_become_gpu_usage():
    stdout = "0, 9318, 12288, 72, 165.40, 170.00, 99\n1, 0, 8192, 30, 9.5, 120.00, 0\n"

    assert parse_gpu_usage(stdout) == (FULL_GPU, GpuUsage(1, 0, 8192, 30, 9.5, 120.0, 0))


def test_not_available_values_are_none_and_bad_rows_are_skipped():
    stdout = "0, 9318, 12288, [N/A], [N/A], [Not Supported], 99\nbroken row\nx, 1, 2, 3, 4, 5, 6\n"

    assert parse_gpu_usage(stdout) == (GpuUsage(0, 9318, 12288, None, None, None, 99),)


def test_rows_without_memory_are_skipped():
    assert parse_gpu_usage("0, [N/A], 12288, 72, 165.40, 170.00, 99\n") == ()


def test_line_with_everything():
    line = resource_line(NOW, FULL_MEMORY, (FULL_GPU,))

    assert line == (
        "RESOURCES 2026-10-28T19:40:00 | RAM available 10.0 / 14.0 GB | "
        "swap used 7.8 / 15.9 GB | GPU 0: VRAM 9.1 / 12.0 GB, 72 C, 165 / 170 W, load 99%"
    )


def test_line_leaves_out_what_could_not_be_read():
    memory = MemoryUsage(10240, 14332, None, None)
    gpu = GpuUsage(0, 9318, 12288, None, None, None, None)

    line = resource_line(NOW, memory, (gpu,))

    assert line == (
        "RESOURCES 2026-10-28T19:40:00 | RAM available 10.0 / 14.0 GB | GPU 0: VRAM 9.1 / 12.0 GB"
    )


def test_power_without_limit_is_shown_alone():
    gpu = GpuUsage(0, 9318, 12288, 72, 165.4, None, 99)

    assert resource_line(NOW, None, (gpu,)).endswith(
        "GPU 0: VRAM 9.1 / 12.0 GB, 72 C, 165 W, load 99%"
    )


def test_line_says_when_nothing_could_be_read():
    assert resource_line(NOW, None, ()) == "RESOURCES 2026-10-28T19:40:00 | not available"
