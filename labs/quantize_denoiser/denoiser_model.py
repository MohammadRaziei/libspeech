"""Loading the Facebook denoiser (Demucs) and exporting it to a dynamic-length ONNX graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from denoiser import pretrained
from denoiser.demucs import Demucs
from denoiser.resample import downsample2, upsample2
from denoiser.utils import deserialize_model
from torch.nn import functional as F

PRETRAINED = ("dns48", "dns64", "master64")


def load_model(spec: str, device: str = "cpu") -> Demucs:
    """`dns48`, `dns64`, `master64` (downloaded by the denoiser package), a `.th` file of a serialized model,
    or `random:<hidden>` for a randomly initialized small model (tests, dry runs)."""
    if spec in PRETRAINED:
        model = getattr(pretrained, spec)()
    elif spec.startswith("random:"):
        torch.manual_seed(0)
        model = Demucs(hidden=int(spec.split(":", 1)[1]), sample_rate=16000)
    else:
        model = deserialize_model(torch.load(spec, map_location="cpu", weights_only=False))
    return model.to(device).eval()


def _at_least_one(value):
    return torch.clamp(value, min=1) if isinstance(value, torch.Tensor) else max(value, 1)


class DynamicLengthDemucs(torch.nn.Module):
    """`Demucs.forward` with the length arithmetic written in tensor operations.

    Tracing then keeps the input length dynamic instead of baking in the length of the example input. Eager results are
    identical to the original forward.
    """

    def __init__(self, demucs: Demucs):
        super().__init__()
        self.demucs = demucs

    def valid_length(self, length):
        d = self.demucs
        length = length * d.resample
        for _ in range(d.depth):
            length = _at_least_one((length - d.kernel_size + d.stride - 1) // d.stride + 1)
        for _ in range(d.depth):
            length = (length - 1) * d.stride + d.kernel_size
        return (length + d.resample - 1) // d.resample

    def forward(self, mix):
        d = self.demucs
        std = mix.mean(dim=1, keepdim=True).std(dim=-1, keepdim=True)
        x = mix / (d.floor + std)
        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))
        for _ in range({1: 0, 2: 1, 4: 2}[d.resample]):
            x = upsample2(x)
        skips = []
        for encode in d.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = d.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in d.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., : x.shape[-1]]
            x = decode(x)
        for _ in range({1: 0, 2: 1, 4: 2}[d.resample]):
            x = downsample2(x)
        return std * x[..., :length]


def export_onnx(model: Demucs, path, opset: int = 13) -> None:
    """Write a fp32 graph with input `input` (batch, 1, length) and output `output`, like the models libspeech ships."""
    wrapper = DynamicLengthDemucs(model).eval()
    dummy = torch.randn(1, 1, 16037) * 0.1  # a length that is not already valid, so the padding is traced
    kwargs = {
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {"input": {0: "batch_size", 2: "length"}, "output": {0: "batch_size", 2: "length"}},
        "opset_version": opset,
    }
    with torch.no_grad():
        try:
            torch.onnx.export(wrapper, dummy, str(path), dynamo=False, **kwargs)
        except TypeError:  # PyTorch older than 2.5 has no `dynamo` argument and always uses the legacy exporter
            torch.onnx.export(wrapper, dummy, str(path), **kwargs)


def verify_export(model: Demucs, onnx_path, lengths=(1000, 16000, 16001, 24013)) -> dict[int, float]:
    """Largest |eager - onnxruntime| per tested input length (relative to the eager output's peak)."""
    session = ort.InferenceSession(str(Path(onnx_path)), providers=["CPUExecutionProvider"])
    gen = torch.Generator().manual_seed(1)
    worst = {}
    for n in lengths:
        x = torch.randn(1, 1, n, generator=gen) * 0.1
        with torch.no_grad():
            eager = model(x).numpy()
        got = session.run(None, {"input": x.numpy()})[0]
        worst[n] = float(np.abs(eager - got).max() / (np.abs(eager).max() + 1e-12))
    return worst
