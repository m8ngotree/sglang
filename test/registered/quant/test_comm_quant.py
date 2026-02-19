import unittest

import torch

from sglang.srt.layers.comm_quant_utils import (
    QuantizedTensor,
    dequantize_fp8,
    quantize_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-small-1-gpu")


class TestQuantizeFp8(CustomTestCase):
    """Tests for quantize_fp8 and dequantize_fp8 round-trip correctness.

    quantized_all_gather requires torch.distributed and belongs in
    multi-GPU integration tests.
    """

    def _roundtrip(self, shape, per_token=False):
        x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        restored = dequantize_fp8(quantize_fp8(x, per_token=per_token))
        return x, restored

    # ── round-trip accuracy ───────────────────────────────────────────────

    def test_roundtrip_2d_per_tensor(self):
        x, restored = self._roundtrip((512, 4096), per_token=False)
        rel_error = (x - restored).abs() / x.abs().clamp(min=1e-5)
        self.assertLess(rel_error.mean().item(), 0.02)

    def test_roundtrip_2d_per_token(self):
        x, restored = self._roundtrip((512, 4096), per_token=True)
        rel_error = (x - restored).abs() / x.abs().clamp(min=1e-5)
        self.assertLess(rel_error.mean().item(), 0.02)

    def test_roundtrip_3d(self):
        """3D tensors [batch, seq, hidden] are reshaped internally and restored."""
        x, restored = self._roundtrip((8, 128, 4096))
        self.assertEqual(restored.shape, (8, 128, 4096))
        self.assertEqual(restored.dtype, torch.bfloat16)

    # ── FP8 representation ────────────────────────────────────────────────

    def test_fp8_output_is_half_the_bytes(self):
        """The bandwidth-saving promise: FP8 uses half the bytes of BF16."""
        x = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
        quantized = quantize_fp8(x)
        self.assertEqual(quantized.data.nbytes, x.nbytes // 2)

    def test_per_tensor_produces_scalar_scale(self):
        x = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
        quantized = quantize_fp8(x, per_token=False)
        self.assertEqual(quantized.scales.numel(), 1)

    def test_per_token_produces_one_scale_per_row(self):
        x = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
        quantized = quantize_fp8(x, per_token=True)
        self.assertEqual(quantized.scales.shape[0], 512)

    # ── robustness ────────────────────────────────────────────────────────

    def test_non_contiguous_input(self):
        """Transposed (non-contiguous) tensors are made contiguous before quantizing."""
        x = torch.randn(4096, 512, dtype=torch.bfloat16, device="cuda").T
        self.assertFalse(x.is_contiguous())
        restored = dequantize_fp8(quantize_fp8(x))
        self.assertEqual(restored.shape, x.shape)

    def test_per_token_better_when_row_magnitudes_differ(self):
        """Per-token accuracy beats per-tensor when rows have different scales."""
        x = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        x[1::2] *= 100.0

        err_pt = (x - dequantize_fp8(quantize_fp8(x, per_token=False))).abs().mean()
        err_tk = (x - dequantize_fp8(quantize_fp8(x, per_token=True))).abs().mean()
        self.assertLess(err_tk.item(), err_pt.item())


if __name__ == "__main__":
    unittest.main()
