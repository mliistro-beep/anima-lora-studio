import os

import pytest

from anima_trainer.models.downloader import (
    DownloadCancelled,
    DownloadError,
    download_file,
    partial_path,
)

pytestmark = pytest.mark.integration

PAYLOAD = os.urandom(10_000)


def test_downloads_whole_file_and_reports_progress(file_server, tmp_path):
    state, url = file_server
    state.payload = PAYLOAD
    progress = []
    target = tmp_path / "dit" / "model.safetensors"

    result = download_file(url, target, on_progress=progress.append, chunk_size=4096)

    assert result == target
    assert target.read_bytes() == PAYLOAD
    assert not partial_path(target).exists()
    assert progress[-1].downloaded == len(PAYLOAD)
    assert all(p.total == len(PAYLOAD) for p in progress)


def test_cancel_keeps_partial_file(file_server, tmp_path):
    state, url = file_server
    state.payload = PAYLOAD
    target = tmp_path / "model.safetensors"
    chunks = []

    with pytest.raises(DownloadCancelled):
        download_file(
            url,
            target,
            on_progress=chunks.append,
            is_cancelled=lambda: len(chunks) >= 1,
            chunk_size=4096,
        )

    assert not target.exists()
    assert partial_path(target).stat().st_size == 4096


def test_resumes_from_partial_file_with_range_request(file_server, tmp_path):
    state, url = file_server
    state.payload = PAYLOAD
    target = tmp_path / "model.safetensors"
    partial_path(target).write_bytes(PAYLOAD[:3000])
    progress = []

    download_file(url, target, on_progress=progress.append)

    assert state.requests[-1]["Range"] == "bytes=3000-"
    assert target.read_bytes() == PAYLOAD
    assert progress[0].downloaded > 3000
    assert progress[-1].total == len(PAYLOAD)


def test_restarts_when_server_ignores_range(file_server, tmp_path):
    state, url = file_server
    state.payload = PAYLOAD
    state.honor_range = False
    target = tmp_path / "model.safetensors"
    partial_path(target).write_bytes(b"stale bytes")

    download_file(url, target)

    assert target.read_bytes() == PAYLOAD


def test_already_complete_partial_file_is_finalized(file_server, tmp_path):
    state, url = file_server
    state.payload = PAYLOAD
    target = tmp_path / "model.safetensors"
    partial_path(target).write_bytes(PAYLOAD)

    download_file(url, target)

    assert [r["Range"] for r in state.requests] == [f"bytes={len(PAYLOAD)}-"]
    assert target.read_bytes() == PAYLOAD


def test_oversized_stale_partial_file_is_discarded_and_redownloaded(file_server, tmp_path):
    state, url = file_server
    state.payload = PAYLOAD
    target = tmp_path / "model.safetensors"
    partial_path(target).write_bytes(PAYLOAD + b"stale tail from an older, larger file")

    download_file(url, target)

    assert target.read_bytes() == PAYLOAD
    assert "Range" not in state.requests[-1]


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x"])
def test_only_http_and_https_are_allowed(url, tmp_path):
    with pytest.raises(DownloadError, match="scheme"):
        download_file(url, tmp_path / "m.safetensors")


def test_filesystem_errors_become_download_errors(file_server, tmp_path):
    state, url = file_server
    state.payload = PAYLOAD
    target = tmp_path / "model.safetensors"
    target.mkdir()

    with pytest.raises(DownloadError):
        download_file(url, target)


def test_protocol_errors_become_download_errors(file_server, tmp_path):
    state, url = file_server
    state.broken_status_line = True

    with pytest.raises(DownloadError):
        download_file(url, tmp_path / "model.safetensors")


def test_http_error_raises_download_error(file_server, tmp_path):
    state, url = file_server
    state.status_override = 404

    with pytest.raises(DownloadError, match="404"):
        download_file(url, tmp_path / "model.safetensors")


def test_unreachable_server_raises_download_error(closed_port_url, tmp_path):
    with pytest.raises(DownloadError):
        download_file(closed_port_url, tmp_path / "model.safetensors", timeout_s=2)


def test_sends_user_agent(file_server, tmp_path):
    state, url = file_server
    state.payload = b"x"

    download_file(url, tmp_path / "m.safetensors")

    assert state.requests[0]["User-Agent"].startswith("anima-trainer/")


def _expected(payload=PAYLOAD):
    import hashlib

    from anima_trainer.models.catalog import ExpectedFile

    return ExpectedFile(hashlib.sha256(payload).hexdigest(), len(payload))


def test_download_matching_the_expected_sha256_is_kept(file_server, tmp_path):
    state, url = file_server
    state.payload = PAYLOAD
    target = tmp_path / "model.safetensors"

    assert download_file(url, target, expected=_expected()) == target
    assert target.read_bytes() == PAYLOAD


def test_download_with_a_different_sha256_is_deleted(file_server, tmp_path):
    state, url = file_server
    state.payload = PAYLOAD
    target = tmp_path / "model.safetensors"
    damaged = bytes([PAYLOAD[0] ^ 1]) + PAYLOAD[1:]

    with pytest.raises(DownloadError, match="SHA-256 does not match"):
        download_file(url, target, expected=_expected(damaged))

    assert not target.exists()
    assert not partial_path(target).exists()


def test_download_with_a_different_size_is_deleted(file_server, tmp_path):
    state, url = file_server
    state.payload = PAYLOAD
    target = tmp_path / "model.safetensors"

    with pytest.raises(DownloadError, match="wrong size"):
        download_file(url, target, expected=_expected(PAYLOAD + b"x"))

    assert not target.exists() and not partial_path(target).exists()


def test_complete_partial_file_is_checked_before_it_is_finalized(file_server, tmp_path):
    state, url = file_server
    state.payload = PAYLOAD
    target = tmp_path / "model.safetensors"
    partial_path(target).write_bytes(bytes(len(PAYLOAD)))  # right size, wrong content

    with pytest.raises(DownloadError, match="SHA-256 does not match"):
        download_file(url, target, expected=_expected())

    assert not target.exists() and not partial_path(target).exists()


def test_cancel_during_the_check_keeps_the_partial_file(file_server, tmp_path):
    state, url = file_server
    state.payload = PAYLOAD
    target = tmp_path / "model.safetensors"
    partial_path(target).write_bytes(PAYLOAD)  # complete: only the check is left

    with pytest.raises(DownloadCancelled):
        download_file(url, target, is_cancelled=lambda: True, expected=_expected())

    assert partial_path(target).read_bytes() == PAYLOAD and not target.exists()
