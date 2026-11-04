"""Quit signals call the given function and can be restored."""

import os
import signal
import sys

import pytest

from anima_trainer.system.quit_signals import QUIT_SIGNALS, install_quit_signals

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")


@posix_only
def test_a_closed_terminal_is_a_quit_signal():
    assert signal.SIGHUP in QUIT_SIGNALS


@posix_only
def test_sighup_calls_the_function_and_restore_puts_the_old_handler_back():
    calls = []
    outer = signal.signal(signal.SIGHUP, signal.SIG_DFL)
    try:
        restore = install_quit_signals(lambda: calls.append("quit"))
        try:
            os.kill(os.getpid(), signal.SIGHUP)
        finally:
            restore()
        assert calls == ["quit"]
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_DFL
    finally:
        signal.signal(signal.SIGHUP, outer)


@posix_only
def test_a_signal_already_ignored_stays_ignored_so_nohup_keeps_working():
    outer = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        restore = install_quit_signals(lambda: None)
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        restore()
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, outer)
