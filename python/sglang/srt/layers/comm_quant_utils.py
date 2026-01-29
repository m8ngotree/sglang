# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Communication Quantization Utilities

This module provides utilities for quantizing tensors before GPU-to-GPU communication
to reduce bandwidth usage. It supports FP8 quantization for operations like AllReduce,
AllGather, and other collective communications.

Key concepts:
- Quantize tensors from BF16/FP32 to FP8 before communication (50% bandwidth reduction)
- Dequantize after communication to restore original precision
- Use per-tensor or per-token quantization strategies
"""

from typing import Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    scaled_fp8_quant,
)


def quantize_fp8_for_comm(
    tensor: torch.Tensor,
    use_per_token: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 format for communication.

    This function converts a BF16/FP32 tensor to FP8, reducing its size by 50%.
    It returns both the quantized tensor and scale factors needed for dequantization.

    Args:
        tensor: Input tensor to quantize. Expected shape: [batch * seq_len, hidden_dim]
                or [M, K] for general matrix dimensions.
        use_per_token: If True, use per-token (per-row) quantization for better accuracy.
                      If False, use per-tensor (single scale) quantization for simplicity.

    Returns:
        Tuple of (quantized_tensor, scales):
            - quantized_tensor: FP8 tensor (half the size of input)
            - scales: Scale factors for dequantization
                     Shape: [1] for per-tensor, [M, 1] for per-token

    Example:
        >>> tensor = torch.randn(1024, 4096, dtype=torch.bfloat16, device='cuda')
        >>> tensor_fp8, scales = quantize_fp8_for_comm(tensor, use_per_token=False)
        >>> print(f"Original: {tensor.nbytes / 1e6:.2f} MB")
        >>> print(f"Quantized: {tensor_fp8.nbytes / 1e6:.2f} MB")
        Original: 8.39 MB
        Quantized: 4.19 MB
    """
    # Ensure tensor is 2D for quantization
    original_shape = tensor.shape
    if tensor.ndim != 2:
        # Reshape to 2D: [total_tokens, hidden_dim]
        tensor = tensor.reshape(-1, tensor.shape[-1])

    # Quantize using existing FP8 kernel
    tensor_fp8, scales = scaled_fp8_quant(
        tensor,
        scale=None,  # Dynamic scaling (compute scale from tensor)
        use_per_token_if_dynamic=use_per_token,
    )

    # Reshape back if needed
    if len(original_shape) != 2:
        tensor_fp8 = tensor_fp8.reshape(*original_shape)

    return tensor_fp8, scales


def dequantize_fp8_from_comm(
    tensor_fp8: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize an FP8 tensor back to higher precision after communication.

    This function converts an FP8 tensor back to BF16/FP32 using the scale factors
    computed during quantization.

    Args:
        tensor_fp8: Quantized FP8 tensor
        scales: Scale factors from quantization
                Shape: [1] for per-tensor, [M, 1] for per-token
        output_dtype: Desired output dtype (default: bfloat16)

    Returns:
        Dequantized tensor in the specified dtype

    Example:
        >>> # After communication
        >>> tensor_restored = dequantize_fp8_from_comm(tensor_fp8, scales)
        >>> error = (tensor - tensor_restored).abs().max()
        >>> print(f"Max reconstruction error: {error:.6f}")
        Max reconstruction error: 0.000234
    """
    # Convert FP8 to the target dtype
    tensor_dequant = tensor_fp8.to(output_dtype)

    # Apply scaling
    if scales.numel() == 1:
        # Per-tensor: single scale for entire tensor
        tensor_dequant = tensor_dequant * scales.item()
    else:
        # Per-token: scale each row/token
        # Reshape tensor for broadcasting if needed
        original_shape = tensor_dequant.shape
        if tensor_dequant.ndim != 2:
            tensor_dequant = tensor_dequant.reshape(-1, tensor_dequant.shape[-1])

        # scales shape: [M, 1], tensor shape: [M, K]
        # Broadcasting will scale each row
        tensor_dequant = tensor_dequant * scales

        # Reshape back
        if len(original_shape) != 2:
            tensor_dequant = tensor_dequant.reshape(*original_shape)

    return tensor_dequant


def should_use_comm_quant(
    tensor_size_bytes: int,
    min_size_threshold: int = 1024 * 1024,  # 1 MB default
) -> bool:
    """
    Determine if communication quantization should be used based on tensor size.

    For very small tensors, the overhead of quantization/dequantization may
    outweigh the bandwidth savings. This function provides a simple heuristic.

    Args:
        tensor_size_bytes: Size of the tensor in bytes
        min_size_threshold: Minimum size in bytes to enable quantization

    Returns:
        True if quantization should be used, False otherwise

    Example:
        >>> tensor = torch.randn(32, 2048, 8192, dtype=torch.bfloat16)
        >>> if should_use_comm_quant(tensor.nbytes):
        >>>     tensor_fp8, scales = quantize_fp8_for_comm(tensor)
        >>>     # ... communicate tensor_fp8 ...
        >>>     tensor = dequantize_fp8_from_comm(tensor_fp8, scales)
    """
    return tensor_size_bytes >= min_size_threshold


# Expose key functions
__all__ = [
    "quantize_fp8_for_comm",
    "dequantize_fp8_from_comm",
    "should_use_comm_quant",
]
