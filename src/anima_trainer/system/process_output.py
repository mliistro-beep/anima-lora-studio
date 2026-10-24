"""Turns chunks of process output into lines, for the training runner and the installer."""

import re

_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def split_lines(pending: str, data: str) -> tuple[tuple[str, ...], str]:
    """Split on \\n, \\r\\n and \\r (tqdm redraws); a trailing \\r waits for a possible \\n."""
    text = pending + data
    held = "\r" if text.endswith("\r") else ""
    parts = _LINE_BREAK.split(text[: len(text) - len(held)])
    return tuple(parts[:-1]), parts[-1] + held
