"""Signals that end the app: Ctrl+C, SIGTERM and a closed terminal (SIGHUP).

Training processes run in their own session, so the terminal's hangup does not reach them: the
app must catch it and stop them, or they stay on the GPU with no parent.
"""

import signal
from collections.abc import Callable
from types import FrameType

QUIT_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, name)
)
# Python runs signal handlers only between bytecodes; a timer wakes it inside Qt's event loop.
SIGNAL_POLL_MS = 200


def install_quit_signals(quit_app: Callable[[], None]) -> Callable[[], None]:
    """Make the quit signals call `quit_app`; returns a function restoring the old handlers.

    A signal already ignored stays ignored: that is how `nohup` asks to survive a closed terminal.
    """
    previous = {signum: signal.getsignal(signum) for signum in QUIT_SIGNALS}

    def handle(_signum: int, _frame: FrameType | None) -> None:
        quit_app()

    for signum, handler in previous.items():
        if handler != signal.SIG_IGN:
            signal.signal(signum, handle)

    def restore() -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    return restore
