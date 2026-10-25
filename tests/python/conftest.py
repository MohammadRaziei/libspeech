"""pytest configuration shared by the Python tests."""

from __future__ import annotations

import pytest


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
