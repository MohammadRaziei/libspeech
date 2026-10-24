"""labs/quantize_denoiser, PyTorch parts. Skipped (see conftest.py) without `torch`, `denoiser` and `onnx`.

They use a tiny randomly initialized Demucs, so they check mechanics (grid, export, training signal, the exported model
going through onnxruntime's quantizer unchanged), never the quality of a real model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import distill_qat
import lab_helpers
import numpy as np
import onnx
import onnx_ptq
import pytest
import torch
from denoiser_model import DynamicLengthDemucs, export_onnx, load_model, verify_export
from distill_qat import distill
from mixtures import MixtureSource
from onnx import numpy_helper
from onnx_ptq import evaluate, quantize, sensitivity
from weight_qat import quantize_tensor, quantize_weights, snap_weights

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def tiny():
    return load_model("random:8")


def test_quantize_tensor_lands_on_the_symmetric_per_channel_grid():
    w = torch.randn(6, 4, 8) * torch.rand(6, 1, 1) * 3
    wq = quantize_tensor(w, 8, axis=0)
    scale = w.abs().amax(dim=(1, 2), keepdim=True) / 127
    q = wq / scale
    assert torch.allclose(q, q.round(), atol=1e-4)
    assert q.abs().max() <= 127 + 1e-4
    assert torch.allclose(q.abs().amax(dim=(1, 2)), torch.full((6,), 127.0), atol=1e-4)  # the max element is on the rail
    assert torch.equal(quantize_tensor(wq, 8, axis=0), wq)  # quantizing again changes nothing


def test_quantize_tensor_per_tensor_and_fewer_bits():
    w = torch.randn(5, 7)
    assert len(torch.unique(quantize_tensor(w, 4, axis=None))) <= 15  # int4 symmetric: -7..7
    assert len(torch.unique(quantize_tensor(w, 8, axis=None))) > 15


def test_export_matches_eager_for_several_lengths(tiny, tmp_path):
    export_onnx(tiny, tmp_path / "m.onnx")
    worst = verify_export(tiny, tmp_path / "m.onnx", lengths=(1000, 16000, 16001, 24013))
    assert max(worst.values()) < 1e-3, worst


def test_dynamic_length_wrapper_equals_the_original_forward(tiny):
    x = torch.randn(2, 1, 12345) * 0.1
    with torch.no_grad():
        torch.testing.assert_close(DynamicLengthDemucs(tiny)(x), tiny(x))


def test_distillation_improves_agreement_with_the_teacher():
    teacher = load_model("random:8")
    source = MixtureSource(seconds=1.0, seed=1)
    eval_mixtures, _ = MixtureSource(seconds=1.0, seed=2).batch(4)
    student, report = distill(
        teacher, source, bits=4, steps=80, batch=4, lr=2e-3, warmup=5, eval_mixtures=eval_mixtures, log_every=1000
    )
    assert report["agreement_db_after"] > report["agreement_db_before"] + 1.0, report
    assert report["loss_last"] < report["loss_first"], report  # the first and the last steps are different windows
    assert report["agreement_db_after_snap"] == pytest.approx(report["agreement_db_after"], abs=0.5)
    assert not any(type(m).__name__ == "QuantizedWeights" for m in student.modules())  # snapped back to plain modules
    # The weights themselves must have learned (not only the biases): a visible share of them moved off the grid
    # point plain round-to-nearest would have chosen.
    rtn = quantize_tensor(teacher.encoder[1][0].weight.detach(), 4, axis=0)
    assert (student.encoder[1][0].weight.detach() != rtn).float().mean() > 0.01


def test_snapped_weights_survive_onnxruntime_quantization_unchanged(tmp_path):
    model = load_model("random:8")
    assert quantize_weights(model, 8) > 0
    snap_weights(model)
    export_onnx(model, tmp_path / "grid.onnx")
    calibration, _ = MixtureSource(seconds=1.0, seed=5).batch(4)
    quantize(tmp_path / "grid.onnx", tmp_path / "int8.onnx", calibration, method="minmax", lstm="fp32")

    original = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(tmp_path / "grid.onnx")).graph.initializer}
    quantized = onnx.load(str(tmp_path / "int8.onnx"))
    inits = {i.name: numpy_helper.to_array(i) for i in quantized.graph.initializer}
    checked = 0
    for node in quantized.graph.node:
        if node.op_type != "DequantizeLinear" or not node.input[0].endswith("weight_quantized"):  # biases become int32 on their own scale
            continue
        q, scale = inits[node.input[0]].astype(np.float64), inits[node.input[1]].astype(np.float64)
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
        shape = [1] * q.ndim
        shape[axis] = -1
        restored = q * scale.reshape(shape if scale.ndim else [])
        np.testing.assert_allclose(restored, original[node.input[0][: -len("_quantized")]], rtol=0, atol=1e-6)
        checked += 1
    assert checked == 20  # 5 encoder levels x 2 convs, 5 decoder levels x (1x1 conv + transposed conv)


def test_quantize_and_evaluate_run_end_to_end(tiny, tmp_path):
    export_onnx(tiny, tmp_path / "fp32.onnx")
    calibration, _ = MixtureSource(seconds=1.0, seed=7).batch(4)
    quantize(tmp_path / "fp32.onnx", tmp_path / "int8.onnx", calibration, method="percentile")
    mixtures, cleans = MixtureSource(seconds=1.0, seed=8).batch(3)
    report = evaluate(tmp_path / "fp32.onnx", tmp_path / "int8.onnx", mixtures, cleans)
    assert np.isfinite(report["mean_snr_vs_fp32_db"])
    assert report["size_candidate_mb"] < report["size_fp32_mb"]
    assert report["rtf_fp32"] > 0


def test_sensitivity_ranks_nodes_and_includes_the_lstm(tiny, tmp_path):
    export_onnx(tiny, tmp_path / "fp32.onnx")
    calibration, _ = MixtureSource(seconds=1.0, seed=9).batch(3)
    evals, _ = MixtureSource(seconds=1.0, seed=10).batch(2)
    rows = sensitivity(
        tmp_path / "fp32.onnx", calibration, evals, method="minmax", per_channel=True, workdir=tmp_path / "work", limit=2
    )
    assert len(rows) == 3  # two convolution nodes and the LSTM
    assert "LSTM" in [r["node"] for r in rows]
    scores = [r["mean_snr_vs_fp32_db"] for r in rows]
    assert scores == sorted(scores)  # most sensitive (lowest agreement) first
    assert all(np.isfinite(scores))
    assert not list((tmp_path / "work").glob("trial_*.onnx"))  # trial models are removed after use


def test_every_function_the_notebooks_call_exists():
    modules = {"lab": lab_helpers, "onnx_ptq": onnx_ptq, "distill_qat": distill_qat}
    notebooks = sorted((Path(__file__).resolve().parents[2] / "labs").rglob("*.ipynb"))
    assert notebooks
    for notebook in notebooks:
        code = "\n".join("".join(c["source"]) for c in json.loads(notebook.read_text())["cells"] if c["cell_type"] == "code")
        for module, name in re.findall(r"\b(lab|onnx_ptq|distill_qat)\.(\w+)\(", code):
            assert hasattr(modules[module], name), f"{notebook.name} calls {module}.{name}(), which does not exist"
