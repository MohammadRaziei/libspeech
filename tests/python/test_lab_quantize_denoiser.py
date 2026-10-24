"""labs/quantize_denoiser: the NumPy parts (mixtures, metrics). The PyTorch parts are in test_lab_quantize_denoiser_torch.py.

The lab folder is on `pythonpath` (pyproject.toml), so its modules import by name."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from audio_metrics import si_sdr, snr_db
from lab_helpers import agreement, register_in_libspeech, table
from mixtures import SR, MixtureSource, NoisyRecordings, mix_at_snr


def test_mix_at_snr_hits_the_requested_snr():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(SR).astype(np.float32)
    noise = rng.standard_normal(SR).astype(np.float32) * 5.0
    for snr in (-5.0, 0.0, 12.5, 20.0):
        mix = mix_at_snr(clean, noise, snr)
        assert snr_db(mix, clean) == pytest.approx(snr, abs=1e-2)


def test_mixture_source_is_reproducible_and_within_its_ranges():
    a_mix, a_clean = MixtureSource(seconds=1.0, seed=3).batch(4)
    b_mix, b_clean = MixtureSource(seconds=1.0, seed=3).batch(4)
    np.testing.assert_array_equal(a_mix, b_mix)
    np.testing.assert_array_equal(a_clean, b_clean)
    assert a_mix.shape == (4, SR)
    for mix, clean in zip(a_mix, a_clean):
        assert np.abs(mix).max() <= 10 ** (-3 / 20) + 1e-6  # peak between -30 and -3 dBFS
        assert -5.1 <= snr_db(mix, clean) <= 20.1  # the clean reference keeps the mixture's SNR


def test_mixture_snr_is_spread_over_the_requested_range():
    # The clean reference is scaled together with the mixture, so its SNR against the mixture is the drawn one:
    # 64 draws from U(-5, 20) dB cover the range and average near its middle.
    mixtures, cleans = MixtureSource(seconds=0.5, seed=11).batch(64)
    snrs = [snr_db(m, c) for m, c in zip(mixtures, cleans)]
    assert min(snrs) < 0.0
    assert max(snrs) > 15.0
    assert 5.0 < float(np.mean(snrs)) < 10.0


def test_snr_and_si_sdr_known_values():
    ref = np.sin(np.linspace(0, 200, SR))
    assert snr_db(ref * 1.1, ref) == pytest.approx(20.0, abs=1e-6)  # error is 10 % of the amplitude
    assert si_sdr(ref * 0.3, ref) > 100.0  # a pure gain change costs nothing
    assert si_sdr(ref + 0.1 * np.cos(np.linspace(0, 333, SR)), ref) < 30.0


def test_noisy_recordings_are_contiguous_crops_of_the_files(tmp_path):
    ramp = (np.arange(3 * SR) / (3 * SR)).astype(np.float32)  # 0 ... 1 in equal steps, so a crop is easy to recognise
    sf.write(tmp_path / "ramp.wav", ramp, SR, subtype="FLOAT")
    a, clean = NoisyRecordings(tmp_path, seconds=0.5, seed=1).batch(5)
    b, _ = NoisyRecordings(tmp_path, seconds=0.5, seed=2).batch(5)
    assert clean is None
    assert a.shape == (5, SR // 2)
    np.testing.assert_allclose(np.diff(a, axis=1), 1 / (3 * SR), atol=1e-6)  # every crop is a consecutive stretch
    assert not np.array_equal(a, b)  # other seed, other crops
    np.testing.assert_array_equal(a, NoisyRecordings(tmp_path, seconds=0.5, seed=1).batch(5)[0])


def test_register_in_libspeech_writes_the_files_libspeech_requires(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"not really onnx" * 100)
    cache = tmp_path / "cache"
    assert register_in_libspeech(model, "lab-m.onnx", cache_dir=cache) == "lab-m.onnx"
    assert (cache / "lab-m.onnx").read_bytes() == model.read_bytes()
    assert (cache / "lab-m.onnx.sha256").read_text() == hashlib.sha256(model.read_bytes()).hexdigest() + "\n"
    assert "quantize_denoiser" in (cache / "lab-m.onnx.readme").read_text()


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


NOTEBOOKS = sorted((Path(__file__).resolve().parents[2] / "labs").rglob("*.ipynb"))


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
