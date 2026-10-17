"""Checksums, model downloading and the model readme.

The download tests talk to a small HTTP server on 127.0.0.1 that serves a fake model plus its
`.sha256` / `.readme` files, so they run offline and exercise the real HTTP client. Digests are
checked against hashlib. The tests marked `models` talk to the real `models` release (run with
`pytest -m models`).
"""

from __future__ import annotations

import csv
import hashlib
import http.server
import os
import threading
import urllib.request
from pathlib import Path

import pytest

import libspeech

MODEL = b"pretend these are the weights of an onnx model"
README = "# tiny.onnx\nWhat it is, who made it, license.\n"
ROOT = Path(__file__).resolve().parents[2]


def _sum_line(data: bytes, name: str = "tiny.onnx") -> bytes:
    return f"{hashlib.sha256(data).hexdigest()}  {name}\n".encode()


# ---- sha256 ------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 3, 55, 56, 63, 64, 65, 119, 120, 1000, 65537])
def test_sha256_matches_hashlib(n):
    data = bytes((i * 7 + 3) & 255 for i in range(n))
    assert libspeech.sha256(data) == hashlib.sha256(data).hexdigest()


def test_sha256_file_matches_hashlib_across_chunk_boundaries(tmp_path):
    # 2 MiB + 777: not a multiple of the 1 MiB read chunk or of the 64-byte block.
    data = os.urandom(2 * 1024 * 1024 + 777)
    p = tmp_path / "blob.bin"
    p.write_bytes(data)
    assert libspeech.sha256_file(p) == hashlib.sha256(data).hexdigest()
    assert libspeech.sha256_file(str(p)) == hashlib.sha256(data).hexdigest()


def test_sha256_errors(tmp_path):
    with pytest.raises(RuntimeError):
        libspeech.sha256_file(tmp_path / "does-not-exist.bin")
    with pytest.raises(TypeError):
        libspeech.sha256("not bytes")  # type: ignore[arg-type]


# ---- a local server ----------------------------------------------------------------------------


class _Server:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.requests: list[str] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests.append(self.path)
                body = outer.files.get(self.path)
                if body is None:  # like a real 404 page
                    page = b"<!DOCTYPE html><html><body>404 Not Found</body></html>"
                    self.send_response(404)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # silence
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/tiny.onnx"

    def publish(self, data: bytes = MODEL):
        self.files = {
            "/tiny.onnx": data,
            "/tiny.onnx.sha256": _sum_line(data),
            "/tiny.onnx.readme": README.encode(),
        }

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = _Server()
    s.publish()
    yield s
    s.close()


def _listing(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir())


def test_fetch_model_downloads_checksum_then_model_then_readme(server, tmp_path):
    path = libspeech.fetch_model(server.url, cache_dir=tmp_path)
    assert Path(path) == tmp_path / "tiny.onnx"
    assert Path(path).read_bytes() == MODEL
    assert (tmp_path / "tiny.onnx.readme").read_text() == README
    assert (tmp_path / "tiny.onnx.sha256").read_bytes() == _sum_line(MODEL)
    assert server.requests == ["/tiny.onnx.sha256", "/tiny.onnx", "/tiny.onnx.readme"]
    assert _listing(tmp_path) == ["tiny.onnx", "tiny.onnx.readme", "tiny.onnx.sha256"]


def test_fetch_model_uses_the_cache(server, tmp_path):
    libspeech.fetch_model(server.url, cache_dir=tmp_path)
    server.requests.clear()
    libspeech.fetch_model(server.url, cache_dir=tmp_path)
    assert server.requests == []


def test_ignore_cache_downloads_again_and_replaces(server, tmp_path):
    libspeech.fetch_model(server.url, cache_dir=tmp_path)
    newer = b"a newer version of the weights"
    server.publish(newer)
    server.requests.clear()
    path = libspeech.fetch_model(server.url, ignore_cache=True, cache_dir=tmp_path)
    assert sorted(server.requests) == ["/tiny.onnx", "/tiny.onnx.readme", "/tiny.onnx.sha256"]
    assert Path(path).read_bytes() == newer
    assert (tmp_path / "tiny.onnx.sha256").read_bytes() == _sum_line(newer)


def test_html_instead_of_a_checksum_is_an_error_and_leaves_nothing(server, tmp_path):
    server.files["/tiny.onnx.sha256"] = b"<!DOCTYPE html><html><body>Not Found</body></html>"
    with pytest.raises(RuntimeError, match="checksum"):
        libspeech.fetch_model(server.url, cache_dir=tmp_path)
    assert server.requests == ["/tiny.onnx.sha256"]  # the model itself was never requested
    assert _listing(tmp_path) == []


def test_wrong_hash_deletes_the_download(server, tmp_path):
    server.files["/tiny.onnx"] = MODEL + b" tampered"
    with pytest.raises(RuntimeError, match="SHA-256"):
        libspeech.fetch_model(server.url, cache_dir=tmp_path)
    assert _listing(tmp_path) == []


def test_missing_model_is_an_error_and_no_error_page_is_cached(server, tmp_path):
    del server.files["/tiny.onnx"]  # the server answers 404 with an HTML page
    with pytest.raises(RuntimeError):
        libspeech.fetch_model(server.url, cache_dir=tmp_path)
    assert _listing(tmp_path) == []  # in particular no tiny.onnx holding the 404 page


def test_corrupted_cached_model_is_repaired(server, tmp_path):
    libspeech.fetch_model(server.url, cache_dir=tmp_path)
    (tmp_path / "tiny.onnx").write_bytes(b"<html>half a download</html>")
    path = libspeech.fetch_model(server.url, cache_dir=tmp_path)
    assert Path(path).read_bytes() == MODEL


def test_url_without_sidecar_files_is_used_unverified(server, tmp_path):
    del server.files["/tiny.onnx.sha256"]
    del server.files["/tiny.onnx.readme"]
    path = libspeech.fetch_model(server.url, cache_dir=tmp_path)
    assert Path(path).read_bytes() == MODEL
    assert _listing(tmp_path) == ["tiny.onnx"]
    with pytest.raises(RuntimeError, match="readme"):
        libspeech.model_readme(server.url, cache_dir=tmp_path)


def test_model_readme_returns_the_text(server, tmp_path):
    assert libspeech.model_readme(server.url, cache_dir=tmp_path) == README


def test_missing_release_model_raises_runtime_error():
    # A bare name must come with a checksum and a readme from the release; one that is not there fails
    # cleanly (the 404 page is not mistaken for a checksum, nothing is left in the cache).
    with pytest.raises(RuntimeError):
        libspeech.SileroVad("definitely-not-a-model-libspeech-has.onnx")


# ---- data/models.csv ---------------------------------------------------------------------------


def _rows():
    with (ROOT / "data" / "models.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def test_models_csv_is_well_formed():
    rows = _rows()
    assert rows, "data/models.csv is empty"
    names = [r["name"] for r in rows]
    assert len(set(names)) == len(names)
    for r in rows:
        assert r["file"] == r["name"] + ".onnx"
        assert len(r["sha256"]) == 64, r["name"]
        assert all(c in "0123456789abcdef" for c in r["sha256"]), r["name"]
        assert int(r["size_bytes"]) > 0
        assert r["license"]
        assert r["source"].startswith("https://")
        assert r["commercial_use"] in ("yes", "no")
        assert r["backend"] in ("gtcrn", "dpdfnet", "facebook", "speechbrain", "silero_vad")


def test_models_csv_covers_every_default_model_and_flags_non_commercial():
    by_name = {r["name"]: r for r in _rows()}
    for default in (
        "facebook-denoiser-dns64",
        "speechbrain-sepformer-wham16k-enhancement",
        "gtcrn_simple",
        "dpdfnet2",
        "silero-vad",
    ):
        assert default in by_name
    for name, r in by_name.items():
        if name.startswith("facebook-"):
            assert r["commercial_use"] == "no"
            assert r["license"] == "CC-BY-NC-4.0"


@pytest.mark.models
def test_models_csv_matches_the_release_checksums():
    # Every row's digest must equal the .sha256 file in the release (a few bytes each).
    for r in _rows():
        url = f"https://github.com/MohammadRaziei/libspeech/releases/download/models/{r['file']}.sha256"
        with urllib.request.urlopen(url, timeout=60) as resp:
            text = resp.read().decode()
        assert text.split()[0] == r["sha256"], r["name"]
        assert text.split()[1] == r["file"]


@pytest.mark.models
def test_real_release_model_download_verify_readme_and_ignore_cache(tmp_path):
    row = next(r for r in _rows() if r["name"] == "gtcrn_simple")
    path = Path(libspeech.fetch_model(row["file"], cache_dir=tmp_path))
    assert libspeech.sha256_file(path) == row["sha256"]
    assert path.stat().st_size == int(row["size_bytes"])
    assert "GTCRN" in libspeech.model_readme(row["file"], cache_dir=tmp_path)
    assert _listing(tmp_path) == [row["file"], row["file"] + ".readme", row["file"] + ".sha256"]
    # ignore_cache replaces the files (same content, new mtime)
    before = path.stat().st_mtime_ns
    libspeech.fetch_model(row["file"], ignore_cache=True, cache_dir=tmp_path)
    assert path.stat().st_mtime_ns != before
    assert libspeech.sha256_file(path) == row["sha256"]


@pytest.mark.models
def test_denoiser_exposes_model_path_sha256_readme_and_honours_ignore_cache():
    row = next(r for r in _rows() if r["name"] == "gtcrn_simple")
    d = libspeech.Denoiser("gtcrn")
    path = Path(d.model_path)
    assert path.name == row["file"]
    assert d.sha256 == row["sha256"]
    assert "GTCRN" in d.readme()
    d.close()
    assert d.readme()  # still available after close()

    before = path.stat().st_mtime_ns
    libspeech.Denoiser("gtcrn").close()  # cached: nothing is downloaded
    assert path.stat().st_mtime_ns == before
    libspeech.Denoiser("gtcrn", ignore_cache=True).close()  # downloaded again
    assert path.stat().st_mtime_ns != before
    assert libspeech.sha256_file(path) == row["sha256"]


@pytest.mark.models
def test_silero_vad_exposes_readme_and_honours_ignore_cache():
    row = next(r for r in _rows() if r["name"] == "silero-vad")
    v = libspeech.SileroVad("silero-vad.onnx")
    path = Path(v.model_path)
    assert v.sha256 == row["sha256"]
    assert "Silero" in v.readme()
    before = path.stat().st_mtime_ns
    libspeech.SileroVad("silero-vad.onnx", ignore_cache=True)
    assert path.stat().st_mtime_ns != before
