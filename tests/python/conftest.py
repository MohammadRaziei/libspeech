"""pytest configuration shared by the Python tests."""

from __future__ import annotations

import importlib.util

import pytest

# labs/quantize_denoiser needs PyTorch, the `denoiser` package and `onnx`; without them its tests are not collected.
collect_ignore = []
if not all(importlib.util.find_spec(m) for m in ("torch", "denoiser", "onnx")):
    collect_ignore.append("test_lab_quantize_denoiser_torch.py")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Tests marked `models` need the real ONNX models (a download plus ~1 GB of RAM), so a plain
    # `pytest` run -- the wheel tests in wheels.yml (cibuildwheel), speech_coverage_python, a
    # developer's local run -- skips them. They run only when selected explicitly:
    #     pytest -m models
    # which is what cmake.yml does (target speech_test_python_models_e2e).
    if "models" in (config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="needs the ONNX models; run with `pytest -m models` (cmake.yml does)")
    for item in items:
        if item.get_closest_marker("models"):
            item.add_marker(skip)
