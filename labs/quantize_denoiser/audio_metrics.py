"""Audio metrics used to compare a quantized denoiser with its fp32 original (NumPy only)."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def snr_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    """10 log10(sum(reference^2) / sum((estimate - reference)^2)): how closely `estimate` follows `reference`."""
    ref = np.asarray(reference, dtype=np.float64)
    err = np.asarray(estimate, dtype=np.float64) - ref
    return float(10.0 * np.log10((np.sum(ref**2) + EPS) / (np.sum(err**2) + EPS)))


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (zero-mean signals)."""
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    est = est - est.mean()
    ref = ref - ref.mean()
    target = (np.dot(est, ref) / (np.dot(ref, ref) + EPS)) * ref
    noise = est - target
    return float(10.0 * np.log10((np.sum(target**2) + EPS) / (np.sum(noise**2) + EPS)))
