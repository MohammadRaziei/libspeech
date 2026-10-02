"""SIMD backend selection: reported, switchable at run time, overridable by environment,
and numerically consistent with the portable kernels."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import libspeech

np = pytest.importorskip("numpy")

KNOWN_BACKENDS = {"generic", "avx2+fma", "neon"}


def test_backend_name_is_one_of_the_known_backends():
    assert libspeech.simd_backend() in KNOWN_BACKENDS


def test_runtime_switch_selects_generic_and_restores():
    before = libspeech.simd_backend()
    try:
        libspeech.set_simd_enabled(False)
        assert libspeech.simd_enabled() is False
        assert libspeech.simd_backend() == "generic"
    finally:
        libspeech.set_simd_enabled(True)
    assert libspeech.simd_enabled() is True
    assert libspeech.simd_backend() == before


def _compute_all(x16: np.ndarray, x44: np.ndarray):
    stft_re, stft_im = libspeech.STFT(9, libspeech.WindowType.hann, 128).stft(x16, onesided=True)
    p = libspeech.MFCCParams()
    p.sample_rate = 16000
    p.num_mel_filters = 26
    p.num_coefficients = 13
    p.radix2_exp = 9
    p.slide_length = 128
    mfcc = libspeech.MFCC(p).compute(x16)
    rs = libspeech.Resample(44100, 16000).resample(x44)
    return stft_re, stft_im, mfcc, rs


def test_simd_and_generic_kernels_agree():
    rng = np.random.default_rng(0)
    t16 = np.arange(24000) / 16000
    t44 = np.arange(60000) / 44100
    # A noise floor keeps MFCC's log() away from the float rounding floor.
    x16 = (0.4 * np.sin(2 * np.pi * 220 * t16) + 0.01 * rng.standard_normal(t16.size)).astype(np.float32)
    x44 = (0.4 * np.sin(2 * np.pi * 220 * t44)).astype(np.float32)

    fast = _compute_all(x16, x44)
    try:
        libspeech.set_simd_enabled(False)
        portable = _compute_all(x16, x44)
    finally:
        libspeech.set_simd_enabled(True)

    for a, b, tol in zip(fast, portable, (1e-3, 1e-3, 1e-3, 1e-5)):
        assert a.shape == b.shape
        assert np.abs(a - b).max() <= tol * max(1.0, float(np.abs(b).max()))


@pytest.mark.parametrize("value", ["off", "0", "generic", "none"])
def test_environment_override_forces_generic(value):
    code = "import libspeech; print(libspeech.simd_backend())"
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "LIBSPEECH_SIMD": value},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "generic"


def test_legacy_environment_override_still_works():
    code = "import libspeech; print(libspeech.simd_backend())"
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "LIBSPEECH_DISABLE_AVX2": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "generic"
