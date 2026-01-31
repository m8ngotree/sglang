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

Utilities for quantizing tensors to FP8 before GPU-to-GPU communication
to reduce bandwidth by 50%. Supports AllGather, AllReduce, and ReduceScatter.

Key points:
- FP8 is a storage/transfer format, not a compute format
- AllGather is ideal (no arithmetic on FP8)
- AllReduce uses AllGather + local sum to avoid FP8 overflow
"""

from typing import NamedTuple, Optional, Union

import torch
import torch.distributed as dist

from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant


class QuantizedTensor(NamedTuple):
    """Container for quantized tensor and its scale factors."""

    data: torch.Tensor  # FP8 quantized tensor
    scales: torch.Tensor  # Scale factors for dequantization


def quantize_fp8_for_comm(
    tensor: torch.Tensor,
    use_per_token: bool = False,
) -> QuantizedTensor:
    """
    Quantize a tensor to FP8 format for communication.

    Args:
        tensor: Input tensor (BF16/FP32, CUDA). Shape: [M, K] or higher dims.
        use_per_token: If True, use per-row quantization for better accuracy.

    Returns:
        QuantizedTensor with data (FP8) and scales ([1] or [M, 1]).
    """
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Reshape to 2D for quantization kernel
    original_shape = tensor.shape
    if tensor.ndim != 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])

    tensor_fp8, scales = scaled_fp8_quant(
        tensor,
        scale=None,
        use_per_token_if_dynamic=use_per_token,
    )

    # Reshape back if needed
    if tensor.ndim != len(original_shape):
        tensor_fp8 = tensor_fp8.reshape(*original_shape)

    return QuantizedTensor(data=tensor_fp8, scales=scales)


def dequantize_fp8_from_comm(
    quantized: Union[QuantizedTensor, torch.Tensor],
    scales: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize an FP8 tensor back to higher precision.

    Args:
        quantized: QuantizedTensor or raw FP8 tensor.
        scales: Required if quantized is a raw tensor.
        output_dtype: Output dtype (default: bfloat16).

    Returns:
        Dequantized tensor.
    """
    if isinstance(quantized, QuantizedTensor):
        tensor_fp8 = quantized.data
        scales = quantized.scales
    else:
        tensor_fp8 = quantized

    tensor_dequant = tensor_fp8.to(output_dtype)

    # Reshape to 2D for scaling
    original_shape = tensor_dequant.shape
    if tensor_dequant.ndim != 2:
        tensor_dequant = tensor_dequant.reshape(-1, tensor_dequant.shape[-1])

    tensor_dequant = tensor_dequant * scales

    # Reshape back
    if tensor_dequant.ndim != len(original_shape):
        tensor_dequant = tensor_dequant.reshape(*original_shape)

    return tensor_dequant


def quantized_all_gather(
    tensor: torch.Tensor,
    world_size: int,
    group: Optional[dist.ProcessGroup] = None,
    use_per_token: bool = False,
) -> torch.Tensor:
    """
    AllGather with FP8 quantization (50% bandwidth savings).

    This is the ideal use case for FP8 communication - no arithmetic needed.

    Args:
        tensor: Input tensor to gather.
        world_size: Number of ranks.
        group: Process group (default: world group).
        use_per_token: Use per-row quantization.

    Returns:
        Gathered tensor. Shape: [tensor.shape[0] * world_size, ...].
    """
    original_dtype = tensor.dtype

    quantized = quantize_fp8_for_comm(tensor, use_per_token=use_per_token)

    gathered_fp8_list = [torch.empty_like(quantized.data) for _ in range(world_size)]
    gathered_scales_list = [torch.empty_like(quantized.scales) for _ in range(world_size)]

    dist.all_gather(gathered_fp8_list, quantized.data, group=group)
    dist.all_gather(gathered_scales_list, quantized.scales, group=group)

    dequantized_list = [
        dequantize_fp8_from_comm(fp8, sc, output_dtype=original_dtype)
        for fp8, sc in zip(gathered_fp8_list, gathered_scales_list)
    ]

    return torch.cat(dequantized_list, dim=0)


def quantized_all_reduce(
    tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    use_per_token: bool = False,
) -> torch.Tensor:
    """
    AllReduce with FP8 quantization (bandwidth savings via AllGather + local sum).

    Direct FP8 SUM would overflow and have incorrect math (different scales per rank).
    Instead: AllGather FP8 tensors, dequantize locally, sum locally.

    Args:
        tensor: Input tensor to reduce.
        group: Process group (default: world group).
        use_per_token: Use per-row quantization.

    Returns:
        Reduced tensor (sum across all ranks).

    Note:
        Uses world_size copies of tensor temporarily (memory overhead).
    """
    original_dtype = tensor.dtype
    world_size = dist.get_world_size(group) if group else dist.get_world_size()

    quantized = quantize_fp8_for_comm(tensor, use_per_token=use_per_token)

    gathered_fp8_list = [torch.empty_like(quantized.data) for _ in range(world_size)]
    gathered_scales_list = [torch.empty_like(quantized.scales) for _ in range(world_size)]

    dist.all_gather(gathered_fp8_list, quantized.data, group=group)
    dist.all_gather(gathered_scales_list, quantized.scales, group=group)

    # Dequantize and sum locally
    result = dequantize_fp8_from_comm(
        gathered_fp8_list[0], gathered_scales_list[0], output_dtype=original_dtype
    )
    for fp8, sc in zip(gathered_fp8_list[1:], gathered_scales_list[1:]):
        result = result + dequantize_fp8_from_comm(fp8, sc, output_dtype=original_dtype)

    return result


def quantized_reduce_scatter(
    tensor: torch.Tensor,
    world_size: int,
    group: Optional[dist.ProcessGroup] = None,
    use_per_token: bool = False,
) -> torch.Tensor:
    """
    ReduceScatter with FP8 quantization.

    Implementation: AllGather + local sum + slice (not a true reduce_scatter,
    but ensures correct math with FP8).

    Args:
        tensor: Input tensor. First dim must be divisible by world_size.
        world_size: Number of ranks.
        group: Process group (default: world group).
        use_per_token: Use per-row quantization.

    Returns:
        Reduced and scattered tensor. Shape: [tensor.shape[0] // world_size, ...].
    """
    rank = dist.get_rank(group) if group else dist.get_rank()

    reduced = quantized_all_reduce(tensor, group=group, use_per_token=use_per_token)

    chunk_size = tensor.shape[0] // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size

    return reduced[start_idx:end_idx].contiguous()


def should_use_comm_quant(
    tensor: Union[torch.Tensor, int],
    min_size_threshold: int = 1024 * 1024,  # 1 MB
) -> bool:
    """
    Heuristic: use quantized communication for tensors >= threshold.

    For small tensors, quantization overhead may exceed bandwidth savings.
    """
    size_bytes = tensor.nbytes if isinstance(tensor, torch.Tensor) else tensor
    return size_bytes >= min_size_threshold


def get_quantization_error_bound(
    tensor: torch.Tensor,
    use_per_token: bool = False,
) -> float:
    """
    Measure max relative error from quantize-dequantize round trip.

    Useful for debugging/testing. Has computational cost - use sparingly.
    """
    quantized = quantize_fp8_for_comm(tensor, use_per_token=use_per_token)
    reconstructed = dequantize_fp8_from_comm(quantized, output_dtype=tensor.dtype)

    abs_error = (tensor - reconstructed).abs()
    abs_values = tensor.abs().clamp(min=1e-10)
    relative_error = abs_error / abs_values

    return relative_error.max().item()


__all__ = [
    "QuantizedTensor",
    "quantize_fp8_for_comm",
    "dequantize_fp8_from_comm",
    "quantized_all_gather",
    "quantized_all_reduce",
    "quantized_reduce_scatter",
    "should_use_comm_quant",
    "get_quantization_error_bound",
]
