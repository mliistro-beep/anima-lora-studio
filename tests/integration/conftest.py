import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@dataclass
class FileServerState:
    payload: bytes = b""
    honor_range: bool = True
    status_override: int | None = None
    broken_status_line: bool = False
    # Throttling: seconds to sleep after each block of `block_size` bytes.
    block_delay_s: float = 0.0
    block_size: int = 1024
    requests: list[dict[str, str]] = field(default_factory=list)


def _handler_for(state: FileServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state.requests.append(dict(self.headers))
            if state.broken_status_line:
                self.wfile.write(b"NOT-HTTP garbage\r\n\r\n")
                return
            if state.status_override:
                self.send_error(state.status_override)
                return
            start = self._range_start()
            if start is not None and start >= len(state.payload):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(state.payload)}")
                self.end_headers()
                return
            body = state.payload[start or 0 :]
            self.send_response(206 if start is not None else 200)
            if start is not None:
                end = len(state.payload) - 1
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(state.payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write(body)

        def _write(self, body):
            if not state.block_delay_s:
                self.wfile.write(body)
                return
            for start in range(0, len(body), state.block_size):
                try:
                    self.wfile.write(body[start : start + state.block_size])
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                time.sleep(state.block_delay_s)

        def _range_start(self):
            header = self.headers.get("Range")
            if not header or not state.honor_range:
                return None
            return int(header.removeprefix("bytes=").split("-")[0])

    return Handler


@pytest.fixture
def file_server():
    state = FileServerState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    yield state, f"{base_url}/model.safetensors"
    server.shutdown()
    server.server_close()


@pytest.fixture
def closed_port_url():
    """A URL on a local port that was just released, so nothing is listening there."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/none"


PAUSE_LIMIT_S = 10


@pytest.fixture
def paused_file_check(monkeypatch):
    """The real model file check, paused after each chunk until the returned event is set.

    A test that cancels from a progress slot sets it there, so the cancel always lands before
    the check reaches the end of the file, however slow the machine (coverage, CI).
    """
    from anima_trainer.models import check_worker

    gate = threading.Event()
    real_check = check_worker.check_model_file

    def paused(path, expected, report, is_cancelled, chunk_size):
        def report_then_wait(done, total):
            report(done, total)
            gate.wait(PAUSE_LIMIT_S)

        return real_check(path, expected, report_then_wait, is_cancelled, chunk_size)

    monkeypatch.setattr(check_worker, "check_model_file", paused)
    return gate
