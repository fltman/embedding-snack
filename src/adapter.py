"""Trainable bridge between Model A's and Model B's hidden spaces."""
from __future__ import annotations

import torch
from torch import nn


class Adapter(nn.Module):
    """Linear projection d_in -> d_out, followed by LayerNorm in d_out space."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=True)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


@torch.no_grad()
def init_orthogonal(adapter: Adapter) -> None:
    """Initialize the linear weight as a random (semi-)orthogonal matrix.

    Used for the Phase 1 sanity test: a fixed random projection from A to B.
    `decode_from_hidden` output is expected to be garbage. We only verify that
    the pipeline runs end-to-end without errors.

    MPS does not implement `aten::linalg_qr`, which `nn.init.orthogonal_` uses.
    Initialize on CPU/fp32, then copy back to the adapter's device/dtype.
    """
    w = adapter.linear.weight
    cpu_w = torch.empty_like(w, device="cpu", dtype=torch.float32)
    nn.init.orthogonal_(cpu_w)
    w.copy_(cpu_w.to(device=w.device, dtype=w.dtype))
    nn.init.zeros_(adapter.linear.bias)
