from datetime import datetime
from pathlib import Path

from anima_trainer.training.runner import log_file_path, split_lines


def test_split_lines_on_all_line_endings():
    assert split_lines("", "a\nb\r\nc\rd") == (("a", "b", "c"), "d")


def test_split_lines_keeps_incomplete_tail():
    lines, rest = split_lines("par", "tial")

    assert lines == ()
    assert rest == "partial"


def test_split_lines_holds_trailing_carriage_return_for_crlf_across_chunks():
    lines, rest = split_lines("", "one\r")
    more, final = split_lines(rest, "\ntwo\n")

    assert lines == ()
    assert more == ("one", "two")
    assert final == ""


def test_split_lines_preserves_empty_lines():
    assert split_lines("", "a\n\nb\n") == (("a", "", "b"), "")


def test_log_file_path_matches_citron_naming():
    path = log_file_path(Path("/c/logs"), "proj", datetime(2026, 10, 18, 21, 56, 29))

    assert path == Path("/c/logs/proj_2026-10-18_215629.log")
