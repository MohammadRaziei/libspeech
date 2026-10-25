"""Helpers for the notebook: checking downloads, running models through libspeech (optional), listening."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
from audio_metrics import snr_db

LIBSPEECH_CACHE = Path.home() / ".libspeech"


def verify_sha256(model_path) -> str:
    """Check a downloaded model against the one-line `<model>.sha256` file next to it; returns the digest.

    Raises RuntimeError on a mismatch (a truncated or corrupt download), FileNotFoundError if a file is missing.
    """
    model = Path(model_path)
    expected = Path(f"{model}.sha256").read_text().split()[0].lower()
    actual = hashlib.sha256(model.read_bytes()).hexdigest()
    if actual != expected:
        message = f"{model.name}: sha256 is {actual}, expected {expected}"
        raise RuntimeError(message)
    return actual


def random_teacher_onnx(workdir="work", model="random:8") -> str:
    """For dry runs: export a randomly initialized small Demucs as the 'teacher' (same weights as `--model random:8`)."""
    from denoiser_model import export_onnx, load_model

    Path(workdir).mkdir(parents=True, exist_ok=True)
    path = Path(workdir) / "teacher_random.onnx"
    export_onnx(load_model(model), path)
    return str(path)


def register_in_libspeech(onnx_path, name: str, cache_dir=None) -> str:
    """Put a local .onnx into libspeech's model cache under `name`, with the .sha256 and .readme libspeech requires.

    `Denoiser("facebook", name)` then loads it without any download. (Passing a local path as `model` would be simpler, but
    in the libspeech source this was written against it is taken as a file name of the models release and gives a 404.)
    Returns `name`.
    """
    cache = Path(cache_dir) if cache_dir else LIBSPEECH_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / name
    shutil.copyfile(onnx_path, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (cache / f"{name}.sha256").write_text(digest + "\n")
    (cache / f"{name}.readme").write_text(f"Local model registered by the quantize_denoiser lab.\nSource file: {onnx_path}\n")
    return name


def run_in_libspeech(name: str, audio: np.ndarray, backend="facebook", threads=1) -> tuple[np.ndarray, float]:
    """(output, real-time factor) of `libspeech.Denoiser(backend, name)` on a 16 kHz clip; one warm-up call is not timed."""
    import libspeech

    with libspeech.Denoiser(backend, name, num_threads=threads) as denoiser:
        denoiser.process(audio[:16000])
        start = time.perf_counter()
        out = np.asarray(denoiser.process(audio))
        elapsed = time.perf_counter() - start
    return out, elapsed / (len(audio) / 16000)


def agreement(candidate: np.ndarray, reference: np.ndarray) -> float:
    n = min(len(candidate), len(reference))
    return snr_db(candidate[:n], reference[:n])


def table(rows: dict[str, dict]) -> str:
    """Plain-text table: one row per label, one column per key of the first row's dict."""
    columns = list(next(iter(rows.values())))
    width = max(len(label) for label in rows) + 2
    lines = ["".ljust(width) + "".join(c.rjust(26) for c in columns)]
    for label, values in rows.items():
        cells = "".join((f"{v:.3f}" if isinstance(v, float) else str(v)).rjust(26) for v in values.values())
        lines.append(label.ljust(width) + cells)
    return "\n".join(lines)


def listen(label: str, audio: np.ndarray, rate: int = 16000):
    """An audio player in the notebook (IPython) with a label above it."""
    from IPython.display import Audio, display

    display(label)
    display(Audio(np.asarray(audio, dtype=np.float32), rate=rate))
