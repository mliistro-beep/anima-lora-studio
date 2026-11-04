"""What the terminal shows: log lines, progress on one line or sparse, and a lost terminal."""

import io

from anima_trainer.cli.terminal_output import TerminalOutput, progress_text
from anima_trainer.training.progress import ProgressState

TQDM_LINE = "steps:  10%|#  | 2/20 [00:04<00:36,  2.10s/it, avr_loss=0.105]"


class ClosedTerminal(io.StringIO):
    """Writes fail like a terminal whose SSH connection has gone."""

    def write(self, _text):
        raise OSError(5, "Input/output error")


def _state(step, total=100):
    return ProgressState(
        epoch=2,
        total_epochs=10,
        step=step,
        total_steps=total,
        loss=0.105,
        elapsed_s=60,
        remaining_s=600,
    )


def test_log_lines_are_written_one_per_line():
    stream = io.StringIO()
    TerminalOutput(stream, interactive=False).line("OK: done")
    assert stream.getvalue() == "OK: done\n"


def test_tqdm_lines_are_left_to_the_progress_display():
    stream = io.StringIO()
    TerminalOutput(stream, interactive=False).line(TQDM_LINE)
    assert stream.getvalue() == ""


def test_progress_text_shows_steps_epoch_loss_and_times():
    text = progress_text(_state(25))
    assert "25/100" in text
    assert "25%" in text
    assert "epoch 2/10" in text
    assert "loss 0.1050" in text
    assert "01:00" in text
    assert "10:00" in text


def test_progress_text_works_before_the_first_values_arrive():
    assert "0/?" in progress_text(ProgressState())


def test_an_interactive_terminal_redraws_progress_on_one_line():
    stream = io.StringIO()
    output = TerminalOutput(stream, interactive=True)
    output.progress(_state(1))
    output.progress(_state(2))
    assert stream.getvalue().count("\n") == 0
    assert stream.getvalue().count("\r") == 2


def test_a_log_line_after_progress_starts_on_a_new_line():
    stream = io.StringIO()
    output = TerminalOutput(stream, interactive=True)
    output.progress(_state(1))
    output.line("WARN: careful")
    assert stream.getvalue().endswith("\nWARN: careful\n")


def test_without_a_terminal_progress_is_written_every_five_percent():
    stream = io.StringIO()
    output = TerminalOutput(stream, interactive=False)
    for step in range(1, 21):
        output.progress(_state(step))
    written = stream.getvalue().splitlines()
    assert [line.split("/")[0].split()[-1] for line in written] == ["1", "5", "10", "15", "20"]


def test_without_a_terminal_a_new_run_of_steps_is_reported_again():
    stream = io.StringIO()
    output = TerminalOutput(stream, interactive=False)
    output.progress(_state(50))
    output.progress(_state(1))
    assert len(stream.getvalue().splitlines()) == 2


def test_a_state_without_steps_is_not_written():
    stream = io.StringIO()
    TerminalOutput(stream, interactive=True).progress(ProgressState())
    assert stream.getvalue() == ""


def test_a_lost_terminal_does_not_raise_and_stops_writing():
    stream = ClosedTerminal()
    output = TerminalOutput(stream, interactive=True)
    output.line("first")
    output.progress(_state(1))
    output.line("second")
    assert output.terminal_lost


def test_detach_of_a_working_terminal_changes_nothing():
    stream = io.StringIO()
    output = TerminalOutput(stream, interactive=False)
    output.detach()
    output.line("still here")
    assert stream.getvalue() == "still here\n"


def test_detach_after_a_lost_terminal_points_the_stream_at_dev_null(tmp_path):
    file = (tmp_path / "out.txt").open("w", encoding="utf-8")
    original_write = file.write

    class Failing:
        """Delegates to the real file, whose descriptor detach() replaces, but fails writes."""

        fileno = file.fileno

        def write(self, _text):
            raise OSError(5, "Input/output error")

        def flush(self):
            file.flush()

    output = TerminalOutput(Failing(), interactive=False)
    output.line("lost")
    output.detach()
    original_write("goes to /dev/null")
    file.close()
    assert (tmp_path / "out.txt").read_text() == ""


def test_detach_tolerates_a_stream_without_a_descriptor():
    output = TerminalOutput(ClosedTerminal(), interactive=False)
    output.line("lost")
    output.detach()
    assert output.terminal_lost


def test_a_closed_stream_counts_as_a_lost_terminal():
    stream = io.StringIO()
    stream.close()
    output = TerminalOutput(stream, interactive=False)
    output.line("lost")
    output.detach()
    assert output.terminal_lost
