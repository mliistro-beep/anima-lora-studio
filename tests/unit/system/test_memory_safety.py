import pytest

from anima_trainer.system.memory_safety import (
    MIN_SWAP_MIB,
    memory_safety_lines,
    read_memory_safety,
    swap_total_mib,
    warnings_only,
)

KIB_PER_MIB = 1024


def _meminfo(swap_mib):
    return (
        "MemTotal:       16276480 kB\n"
        "MemAvailable:   12000000 kB\n"
        f"SwapTotal:      {swap_mib * KIB_PER_MIB} kB\n"
        "SwapFree:        1000000 kB\n"
    )


def test_swap_total_is_read_in_mib():
    assert swap_total_mib(_meminfo(16292)) == 16292


def test_swap_total_is_none_without_the_line():
    assert swap_total_mib("MemTotal: 16276480 kB\n") is None


def test_enough_swap_and_a_daemon_give_only_ok_lines():
    lines = memory_safety_lines(_meminfo(16292), frozenset({"bash", "earlyoom"}), "linux")

    assert lines == ("OK: Swap: 15.9 GB.", "OK: Out-of-memory daemon running: earlyoom.")
    assert warnings_only(lines) == ()


def test_little_swap_warns_and_points_to_the_setup_guide():
    lines = memory_safety_lines(_meminfo(512), frozenset({"systemd-oomd"}), "linux")

    warning = warnings_only(lines)
    assert len(warning) == 1
    assert warning[0].startswith("WARN: Swap is 0.5 GB")
    assert "8-16 GB" in warning[0]
    assert "DOCS/SYSTEM_SETUP.md" in warning[0]
    assert "OK: Out-of-memory daemon running: systemd-oomd." in lines


def test_swap_just_at_the_minimum_does_not_warn():
    lines = memory_safety_lines(_meminfo(MIN_SWAP_MIB), frozenset({"earlyoom"}), "linux")

    assert warnings_only(lines) == ()


def test_no_oom_daemon_warns():
    lines = memory_safety_lines(_meminfo(16292), frozenset({"bash", "plasmashell"}), "linux")

    (warning,) = warnings_only(lines)
    assert warning.startswith("WARN: No out-of-memory daemon (earlyoom or systemd-oomd)")
    assert "DOCS/SYSTEM_SETUP.md" in warning


def test_both_daemons_are_listed_in_order():
    lines = memory_safety_lines(_meminfo(16292), frozenset({"systemd-oomd", "earlyoom"}), "linux")

    assert "OK: Out-of-memory daemon running: earlyoom, systemd-oomd." in lines


def test_unknown_values_are_left_out_instead_of_warning():
    assert memory_safety_lines(None, None, "linux") == ()


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_other_systems_get_no_lines(platform):
    assert memory_safety_lines(_meminfo(0), frozenset(), platform) == ()


def _fake_proc(root, swap_mib, processes):
    root.mkdir()
    (root / "meminfo").write_text(_meminfo(swap_mib))
    (root / "self").mkdir()  # not a process folder: skipped
    (root / "999").mkdir()  # a process that ended before its comm was read
    for pid, name in processes.items():
        (root / str(pid)).mkdir()
        (root / str(pid) / "comm").write_text(f"{name}\n")
    return root


def test_read_memory_safety_reads_meminfo_and_the_process_names(tmp_path):
    proc = _fake_proc(tmp_path / "proc", 512, {1: "systemd", 1186: "earlyoom"})

    lines = read_memory_safety(proc, platform="linux")

    assert lines[0].startswith("WARN: Swap is 0.5 GB")
    assert lines[1] == "OK: Out-of-memory daemon running: earlyoom."


def test_read_memory_safety_without_a_proc_folder_gives_no_lines(tmp_path):
    assert read_memory_safety(tmp_path / "missing", platform="linux") == ()
