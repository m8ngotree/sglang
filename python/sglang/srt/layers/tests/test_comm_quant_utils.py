"""
Tests for comm_quant_utils.py

Covers quantize_fp8 and dequantize_fp8. quantized_all_gather requires
a multi-GPU environment and belongs in integration tests.

Run with: pytest python/sglang/srt/layers/tests/test_comm_quant_utils.py -v
"""

import itertools

import pytest
import torch

from sglang.srt.layers.comm_quant_utils import (
    QuantizedTensor,
    dequantize_fp8,
    quantize_fp8,
)


# ── Round-trip accuracy ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape,per_token",
    list(
        itertools.product(
            [(512, 4096), (8, 128, 4096)],  # 2D and 3D
            [False, True],                  # per-tensor and per-token
        )
    ),
)
def test_roundtrip_accuracy(shape, per_token):
    """quantize → dequantize gives approximately the original values."""
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    restored = dequantize_fp8(quantize_fp8(x, per_token=per_token))

    rel_error = (x - restored).abs() / x.abs().clamp(min=1e-5)
    assert rel_error.mean().item() < 0.02, (
        f"Mean relative error {rel_error.mean().item():.4f} exceeds 2% "
        f"for shape={shape}, per_token={per_token}"
    )


# ── FP8 representation ─────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", [(512, 4096), (8, 128, 4096)])
def test_fp8_output_is_half_the_bytes(shape):
    """FP8 tensor occupies exactly half the bytes of the BF16 input."""
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    quantized = quantize_fp8(x)
    assert quantized.data.nbytes == x.nbytes // 2


@pytest.mark.parametrize("shape", [(512, 4096), (8, 128, 4096)])
def test_shape_preserved_after_roundtrip(shape):
    """Output shape and dtype match the input after a full round-trip."""
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    restored = dequantize_fp8(quantize_fp8(x))
    assert restored.shape == x.shape
    assert restored.dtype == x.dtype


# ── Scale shapes ───────────────────────────────────────────────────────────


def test_per_tensor_produces_scalar_scale():
    """Per-tensor mode computes one scale for the whole tensor."""
    x = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
    quantized = quantize_fp8(x, per_token=False)
    assert quantized.scales.numel() == 1


def test_per_token_produces_one_scale_per_row():
    """Per-token mode computes one scale per row (token)."""
    x = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
    quantized = quantize_fp8(x, per_token=True)
    assert quantized.scales.shape[0] == 512


# ── Robustness ─────────────────────────────────────────────────────────────


def test_non_contiguous_input():
    """Non-contiguous tensors (e.g. from .T) are made contiguous before quantizing."""
    x = torch.randn(4096, 512, dtype=torch.bfloat16, device="cuda").T
    assert not x.is_contiguous()
    restored = dequantize_fp8(quantize_fp8(x))
    assert restored.shape == x.shape


# ── Per-token vs per-tensor trade-off ──────────────────────────────────────


def test_per_token_better_accuracy_when_row_magnitudes_differ():
    """
    Per-token quantization outperforms per-tensor when rows have very
    different magnitudes, because each row gets its own optimal scale.
    """
    x = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
    x[1::2] *= 100.0  # every other row is 100x larger

    err_per_tensor = (x - dequantize_fp8(quantize_fp8(x, per_token=False))).abs().mean()
    err_per_token  = (x - dequantize_fp8(quantize_fp8(x, per_token=True))).abs().mean()

    assert err_per_token < err_per_tensor, (
        "Per-token should give lower mean error than per-tensor when "
        "row magnitudes differ significantly"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
