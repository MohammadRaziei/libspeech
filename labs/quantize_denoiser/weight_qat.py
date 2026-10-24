"""Weight fake-quantization for quantization-aware distillation.

The weights are quantized the way onnxruntime quantizes them afterwards: symmetric int8 range [-qmax, qmax], scale =
max|w| / qmax per output channel (conv) or per tensor (LSTM), round to nearest even. A weight tensor that went through
`snap_weights` therefore sits exactly on the grid onnxruntime will use, and re-quantizing the exported model loses nothing.
Gradients pass through the rounding unchanged (straight-through estimator).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call


def quantize_tensor(w: torch.Tensor, bits: int = 8, axis: int | None = None) -> torch.Tensor:
    """Round `w` to the symmetric integer grid and back to float. `axis` is the channel axis, None for per tensor."""
    qmax = 2 ** (bits - 1) - 1
    if axis is None:
        amax = w.detach().abs().max()
    else:
        dims = [d for d in range(w.dim()) if d != axis]
        amax = w.detach().abs().amax(dim=dims, keepdim=True)
    scale = (amax / qmax).clamp_min(torch.finfo(w.dtype).tiny)
    return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale


def fake_quant(w: torch.Tensor, bits: int = 8, axis: int | None = None) -> torch.Tensor:
    """`quantize_tensor` in the forward pass, identity in the backward pass."""
    return w + (quantize_tensor(w, bits, axis) - w).detach()


class QuantizedWeights(nn.Module):
    """Runs `inner` with the named weight parameters replaced by their fake-quantized values."""

    def __init__(self, inner: nn.Module, names: list[str], bits: int, axis: int | None):
        super().__init__()
        self.inner = inner
        self.names = names
        self.bits = bits
        self.axis = axis

    def forward(self, *args, **kwargs):
        params = dict(self.inner.named_parameters())
        replaced = {n: fake_quant(params[n], self.bits, self.axis) for n in self.names}
        return functional_call(self.inner, replaced, args, kwargs)


def _wrap(module: nn.Module, bits: int, lstm_axis: int | None):
    if isinstance(module, nn.Conv1d):
        return QuantizedWeights(module, ["weight"], bits, 0)
    if isinstance(module, nn.ConvTranspose1d):
        return QuantizedWeights(module, ["weight"], bits, 1)  # weight is (in, out, k): output channels on axis 1
    if isinstance(module, nn.LSTM):
        names = [n for n, _ in module.named_parameters() if n.startswith("weight")]
        return QuantizedWeights(module, names, bits, lstm_axis)
    return None


def quantize_weights(model: nn.Module, bits: int = 8, lstm_axis: int | None = None, keep_first_last: bool = False):
    """Wrap every Conv1d / ConvTranspose1d / LSTM of `model` in place. Returns how many modules were wrapped."""
    skip = set()
    if keep_first_last:
        skip = {id(model.encoder[0][0]), id(model.decoder[-1][2])}
    count = 0

    def visit(parent: nn.Module) -> None:
        nonlocal count
        for name, child in list(parent.named_children()):
            wrapped = None if id(child) in skip else _wrap(child, bits, lstm_axis)
            if wrapped is not None:
                setattr(parent, name, wrapped)
                count += 1
            else:
                visit(child)

    visit(model)
    return count


@torch.no_grad()
def snap_weights(model: nn.Module) -> None:
    """Replace every wrapped module by its inner module with the weights moved onto the integer grid."""

    def visit(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, QuantizedWeights):
                params = dict(child.inner.named_parameters())
                for n in child.names:
                    params[n].copy_(quantize_tensor(params[n], child.bits, child.axis))
                setattr(parent, name, child.inner)
            else:
                visit(child)

    visit(model)
