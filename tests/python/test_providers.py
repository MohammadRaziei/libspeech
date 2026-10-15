"""Tests for libspeech.available_providers() / libspeech.is_cuda_available()."""

from __future__ import annotations

import libspeech


def test_cpu_provider_always_listed():
    providers = libspeech.available_providers()
    assert isinstance(providers, list)
    assert all(isinstance(p, str) for p in providers)
    assert "CPUExecutionProvider" in providers


def test_is_cuda_available_returns_bool_consistent_with_providers():
    available = libspeech.is_cuda_available()
    assert isinstance(available, bool)
    if "CUDAExecutionProvider" not in libspeech.available_providers():
        assert available is False  # a CPU-only ONNX Runtime can never report CUDA
