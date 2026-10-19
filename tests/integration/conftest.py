import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@dataclass
class FileServerState:
    payload: bytes = b""
    honor_range: bool = True
    status_override: int | None = None
    requests: list[dict[str, str]] = field(default_factory=list)


def _handler_for(state: FileServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state.requests.append(dict(self.headers))
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
            self.wfile.write(body)

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
