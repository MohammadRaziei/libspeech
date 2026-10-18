"""The Python examples in README.md must run: they are the first code a new user copies.

Each ```python block is executed in order in one shared namespace (later blocks use `x`, `mono`...
from earlier ones), inside a temp directory that holds a stereo 44.1 kHz `sample.wav`. The block that
downloads models is marked `models` like the other tests that need the network (see conftest.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import libspeech

README = Path(__file__).resolve().parents[2] / "README.md"

pytestmark = pytest.mark.skipif(not README.is_file(), reason="README.md is not next to the tests")


def _blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.DOTALL)


def _run(count: int, workdir: Path) -> dict:
    blocks = _blocks()
    assert len(blocks) >= count, "README.md has fewer python examples than this test expects"
    rng = np.random.default_rng(0)
    libspeech.Audio(rng.uniform(-0.5, 0.5, (2, 44100)).astype(np.float32), 44100).save(str(workdir / "sample.wav"))
    namespace: dict = {}
    for number, code in enumerate(blocks[:count], start=1):
        exec(compile(code, f"README.md python block {number}", "exec"), namespace)
    return namespace


def test_audio_and_feature_examples_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = _run(2, tmp_path)
    assert (tmp_path / "sample_16k_mono.wav").is_file()
    assert ns["x"].dtype == np.float32
    assert ns["x"].shape == (16000,)
    assert ns["view"].shape == (2, 44100)
    assert ns["same"].shape == (2, 44100)
    assert ns["z"].shape[0] == 257
    assert ns["y"].shape == ns["x"].shape
    assert len(ns["coefficients"]) > 0


@pytest.mark.models
def test_model_examples_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = _run(3, tmp_path)
    assert ns["clean"].shape == ns["x"].shape
