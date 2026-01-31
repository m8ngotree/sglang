"""
Unit tests for communication quantization utilities.

These tests run single-process (no multi-GPU required). They validate:
- Quantize/dequantize round-trip accuracy
- Shape preservation across different input dimensions
- Per-tensor vs per-token quantization strategies
- The should_use_comm_quant size heuristic
- The get_quantization_error_bound diagnostic utility
"""

import unittest

import torch


class TestQuantizeFp8ForComm(unittest.TestCase):
    """Tests for quantize_fp8_for_comm and dequantize_fp8_from_comm."""

    def _import(self):
        from sglang.srt.layers.comm_quant_utils import (
            QuantizedTensor,
            dequantize_fp8_from_comm,
            quantize_fp8_for_comm,
        )

        return quantize_fp8_for_comm, dequantize_fp8_from_comm, QuantizedTensor

    # --- Round-trip accuracy ---

    def test_roundtrip_per_tensor_2d(self):
        """Per-tensor quantization on a standard 2D tensor preserves values approximately."""
        quantize, dequantize, _ = self._import()
        tensor = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
        quantized = quantize(tensor, use_per_token=False)
        restored = dequantize(quantized, output_dtype=torch.bfloat16)

        self.assertEqual(restored.shape, tensor.shape)
        self.assertEqual(restored.dtype, torch.bfloat16)
        # FP8 per-tensor error is typically < 1% relative for random normal data
        rel_error = (tensor - restored).abs() / tensor.abs().clamp(min=1e-5)
        self.assertLess(rel_error.mean().item(), 0.02)

    def test_roundtrip_per_token_2d(self):
        """Per-token quantization gives equal or better accuracy than per-tensor."""
        quantize, dequantize, _ = self._import()
        tensor = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")

        q_pt = quantize(tensor, use_per_token=False)
        r_pt = dequantize(q_pt, output_dtype=torch.bfloat16)
        err_per_tensor = (tensor - r_pt).abs().mean().item()

        q_tk = quantize(tensor, use_per_token=True)
        r_tk = dequantize(q_tk, output_dtype=torch.bfloat16)
        err_per_token = (tensor - r_tk).abs().mean().item()

        # Per-token should be at least as good (usually better)
        self.assertLessEqual(err_per_token, err_per_tensor * 1.01)

    def test_roundtrip_3d_tensor(self):
        """3D tensors [batch, seq, hidden] are handled transparently."""
        quantize, dequantize, _ = self._import()
        tensor = torch.randn(8, 128, 4096, dtype=torch.bfloat16, device="cuda")
        quantized = quantize(tensor, use_per_token=False)
        restored = dequantize(quantized, output_dtype=torch.bfloat16)

        self.assertEqual(restored.shape, (8, 128, 4096))
        self.assertEqual(restored.dtype, torch.bfloat16)

    # --- Output types and shapes ---

    def test_returns_quantized_tensor_namedtuple(self):
        """quantize_fp8_for_comm returns a QuantizedTensor with .data and .scales."""
        quantize, _, QuantizedTensor = self._import()
        tensor = torch.randn(256, 2048, dtype=torch.bfloat16, device="cuda")
        result = quantize(tensor)

        self.assertIsInstance(result, QuantizedTensor)
        self.assertTrue(hasattr(result, "data"))
        self.assertTrue(hasattr(result, "scales"))
        # FP8 data is half the bytes of BF16
        self.assertEqual(result.data.nbytes, tensor.nbytes // 2)

    def test_per_tensor_scale_shape(self):
        """Per-tensor mode produces a single scalar scale."""
        quantize, _, _ = self._import()
        tensor = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
        result = quantize(tensor, use_per_token=False)
        self.assertEqual(result.scales.numel(), 1)

    def test_per_token_scale_shape(self):
        """Per-token mode produces one scale per row."""
        quantize, _, _ = self._import()
        tensor = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
        result = quantize(tensor, use_per_token=True)
        # One scale per token (row), shape [512, 1]
        self.assertEqual(result.scales.shape[0], 512)

    # --- Edge cases ---

    def test_non_contiguous_input(self):
        """Non-contiguous tensors are handled (made contiguous internally)."""
        quantize, dequantize, _ = self._import()
        # Transpose makes tensor non-contiguous
        tensor = torch.randn(4096, 512, dtype=torch.bfloat16, device="cuda").T
        self.assertFalse(tensor.is_contiguous())

        quantized = quantize(tensor)
        restored = dequantize(quantized, output_dtype=torch.bfloat16)
        self.assertEqual(restored.shape, tensor.shape)

    def test_dequantize_accepts_raw_tensor(self):
        """dequantize_fp8_from_comm works with (raw_fp8_tensor, scales) too."""
        quantize, dequantize, _ = self._import()
        tensor = torch.randn(256, 2048, dtype=torch.bfloat16, device="cuda")
        quantized = quantize(tensor)

        # Pass raw tensor + scales instead of QuantizedTensor
        restored = dequantize(quantized.data, scales=quantized.scales)
        self.assertEqual(restored.shape, tensor.shape)

    def test_output_dtype_fp32(self):
        """Can dequantize to FP32 instead of BF16."""
        quantize, dequantize, _ = self._import()
        tensor = torch.randn(256, 2048, dtype=torch.bfloat16, device="cuda")
        quantized = quantize(tensor)
        restored = dequantize(quantized, output_dtype=torch.float32)
        self.assertEqual(restored.dtype, torch.float32)


class TestShouldUseCommQuant(unittest.TestCase):
    """Tests for the size-based heuristic."""

    def _import(self):
        from sglang.srt.layers.comm_quant_utils import should_use_comm_quant

        return should_use_comm_quant

    def test_large_tensor_returns_true(self):
        should_use = self._import()
        # 2 MB tensor — well above 1 MB threshold
        tensor = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(should_use(tensor))

    def test_small_tensor_returns_false(self):
        should_use = self._import()
        # 512 bytes — well below 1 MB threshold
        tensor = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(should_use(tensor))

    def test_accepts_raw_int(self):
        should_use = self._import()
        self.assertTrue(should_use(2 * 1024 * 1024))   # 2 MB
        self.assertFalse(should_use(512))               # 512 bytes

    def test_custom_threshold(self):
        should_use = self._import()
        # 100-byte tensor, threshold lowered to 50 bytes
        tensor = torch.randn(25, 2, dtype=torch.bfloat16, device="cuda")  # 100 bytes
        self.assertTrue(should_use(tensor, min_size_threshold=50))
        self.assertFalse(should_use(tensor, min_size_threshold=200))


class TestGetQuantizationErrorBound(unittest.TestCase):
    """Tests for the error diagnostic utility."""

    def _import(self):
        from sglang.srt.layers.comm_quant_utils import get_quantization_error_bound

        return get_quantization_error_bound

    def test_error_bound_is_small(self):
        get_error = self._import()
        tensor = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
        error = get_error(tensor, use_per_token=False)
        # Max relative error for FP8 on random normal data should be < 10%
        self.assertLess(error, 0.10)
        # And it should be positive (there IS some error)
        self.assertGreater(error, 0.0)

    def test_per_token_error_leq_per_tensor(self):
        get_error = self._import()
        tensor = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
        err_pt = get_error(tensor, use_per_token=False)
        err_tk = get_error(tensor, use_per_token=True)
        # Per-token should have equal or lower max error
        self.assertLessEqual(err_tk, err_pt * 1.01)


if __name__ == "__main__":
    unittest.main()
