"""labs/quantize_denoiser: the helpers and the notebooks. NumPy only, so these run in every environment (wheel CI too).

The lab folder is on `pythonpath` (pyproject.toml), so its modules import by name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from audio_metrics import snr_db
from lab_helpers import agreement, register_in_libspeech, table, verify_sha256


def test_register_in_libspeech_writes_the_files_libspeech_requires(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"not really onnx" * 100)
    cache = tmp_path / "cache"
    assert register_in_libspeech(model, "lab-m.onnx", cache_dir=cache) == "lab-m.onnx"
    assert (cache / "lab-m.onnx").read_bytes() == model.read_bytes()
    assert (cache / "lab-m.onnx.sha256").read_text() == hashlib.sha256(model.read_bytes()).hexdigest() + "\n"
    assert "quantize_denoiser" in (cache / "lab-m.onnx.readme").read_text()


def test_verify_sha256_accepts_a_good_download_and_rejects_a_bad_one(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"model bytes" * 50)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    (tmp_path / "m.onnx.sha256").write_text(digest + "\n")
    assert verify_sha256(model) == digest
    (tmp_path / "m.onnx.sha256").write_text(f"{digest.upper()}  m.onnx\n")  # `sha256sum` layout, upper case
    assert verify_sha256(model) == digest
    model.write_bytes(model.read_bytes()[:-1])  # truncated
    with pytest.raises(RuntimeError, match="expected"):
        verify_sha256(model)
    (tmp_path / "gone.onnx").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        verify_sha256(tmp_path / "gone.onnx")  # no .sha256 file next to it


def test_agreement_compares_only_the_common_length():
    ref = np.sin(np.linspace(0, 100, 1000))
    longer = np.concatenate([ref * 1.1, np.ones(7)])
    assert agreement(longer, ref) == pytest.approx(snr_db(ref * 1.1, ref))


def test_table_has_one_row_per_label_and_one_column_per_key():
    lines = table({"a": {"x": 1.0, "y": 2}, "bb": {"x": 3.5, "y": 4}}).splitlines()
    assert len(lines) == 3
    assert "x" in lines[0]
    assert lines[1].startswith("a ")
    assert lines[2].startswith("bb")


NOTEBOOKS = sorted((Path(__file__).resolve().parents[2]).rglob("*.ipynb"))


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebooks_are_clean_and_their_code_compiles(notebook):
    nb = json.loads(notebook.read_text())
    assert nb["nbformat"] == 4
    for number, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [], f"cell {number} carries output; clear it before committing"
        assert cell["execution_count"] is None
        source = "\n".join(line for line in "".join(cell["source"]).splitlines() if not line.lstrip().startswith(("!", "%")))
        compile(source, f"{notebook.name} cell {number}", "exec")


def test_there_is_at_least_one_notebook():
    assert NOTEBOOKS
