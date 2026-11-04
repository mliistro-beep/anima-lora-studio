"""What a terminal training prints: log lines, plus progress in place of tqdm's own lines.

The terminal can vanish at any time (SSH dropped). Losing it must not stop the shutdown of the
training, so write errors are recorded instead of raised: the log file keeps the full output.
"""

import os
from typing import TextIO

from anima_trainer.training.log_parser import StepProgress, parse_line
from anima_trainer.training.progress import ProgressState, format_duration

# Without a terminal (a file, a pipe) one progress line every 5% of the run.
REPORTS_PER_RUN = 20
CLEAR_TO_END_OF_LINE = "\x1b[K"


def progress_text(state: ProgressState) -> str:
    total = state.total_steps
    parts = [f"steps {state.step}/{total if total else '?'}"]
    if total:
        parts.append(f"{state.step * 100 // total}%")
    if state.epoch is not None:
        parts.append(f"epoch {state.epoch}/{state.total_epochs}")
    if state.loss is not None:
        parts.append(f"loss {state.loss:.4f}")
    elapsed, remaining = format_duration(state.elapsed_s), format_duration(state.remaining_s)
    parts.append(f"{elapsed} elapsed, {remaining} left")
    return " | ".join(parts)


class TerminalOutput:
    def __init__(self, stream: TextIO, interactive: bool) -> None:
        self._stream = stream
        self._interactive = interactive
        self._terminal_lost = False
        self._on_progress_line = False
        self._last_step: int | None = None
        self._last_report = 0

    @property
    def terminal_lost(self) -> bool:
        return self._terminal_lost

    def line(self, text: str) -> None:
        if isinstance(parse_line(text), StepProgress):
            return
        prefix = "\n" if self._on_progress_line else ""
        self._on_progress_line = False
        self._write(f"{prefix}{text}\n")

    def progress(self, state: ProgressState) -> None:
        if state.total_steps is None and state.step == 0:
            return
        if self._interactive:
            self._on_progress_line = True
            self._write(f"\r{progress_text(state)}{CLEAR_TO_END_OF_LINE}")
        elif self._is_report_due(state):
            self._write(f"{progress_text(state)}\n")
        self._last_step = state.step

    def detach(self) -> None:
        """After a lost terminal, point stdout at /dev/null so the exit flush cannot fail."""
        if not self._terminal_lost:
            return
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, self._stream.fileno())
            os.close(devnull)
        except (OSError, ValueError):  # ValueError: closed stream
            return  # no file descriptor to replace: nothing will flush to the lost terminal

    def _is_report_due(self, state: ProgressState) -> bool:
        report = state.step * REPORTS_PER_RUN // state.total_steps if state.total_steps else 0
        due = self._last_step is None or state.step < self._last_step or report != self._last_report
        self._last_report = report
        return due

    def _write(self, text: str) -> None:
        if self._terminal_lost:
            return
        try:
            self._stream.write(text)
            self._stream.flush()
        except (OSError, ValueError):  # ValueError: closed stream
            self._terminal_lost = True
