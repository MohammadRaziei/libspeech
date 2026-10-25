"""labs/quantize_denoiser: mixtures and metrics. Needs `soundfile` and `scipy`; the module is skipped without them.

The lab's helpers and notebooks are in test_helpers.py, the PyTorch parts in test_torch.py.
The lab folder is on `pythonpath` (pyproject.toml), so its modules import by name.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest


def _optional(name: str):
    """Import an optional test dependency, or skip this whole module (also when the file is run on its own)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return importlib.import_module(name)
    except Exception as exc:
        pytest.skip(f"optional test dependency {name!r} is unavailable: {exc!r}", allow_module_level=True)


sf = _optional("soundfile")
_optional("scipy.signal")
audio_metrics = _optional("audio_metrics")
mixtures = _optional("mixtures")
si_sdr, snr_db = audio_metrics.si_sdr, audio_metrics.snr_db
SR, MixtureSource, NoisyRecordings, mix_at_snr = (
    mixtures.SR,
    mixtures.MixtureSource,
    mixtures.NoisyRecordings,
    mixtures.mix_at_snr,
)


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
