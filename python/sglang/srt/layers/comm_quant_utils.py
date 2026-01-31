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
FP8 Communication Quantization for AllGather

Quantize tensors to FP8 before AllGather for 50% bandwidth reduction.

Why only AllGather?
- AllGather is pure data movement (no arithmetic on FP8) - ideal for quantization
- AllReduce/ReduceScatter use ring-reduce with O(1) bandwidth scaling
  Naive quantized implementations would use O(world_size) more bandwidth

Usage:
    from sglang.srt.layers.comm_quant_utils import quantized_all_gather

    result = quantized_all_gather(tensor, world_size, group)
"""

from typing import NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant


class QuantizedTensor(NamedTuple):
    """FP8 quantized tensor with scale factors."""

    data: torch.Tensor  # FP8 tensor
    scales: torch.Tensor  # [1] for per-tensor, [M, 1] for per-token


def quantize_fp8(tensor: torch.Tensor, per_token: bool = False) -> QuantizedTensor:
    """Quantize tensor to FP8."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    original_shape = tensor.shape
    if tensor.ndim != 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])

    tensor_fp8, scales = scaled_fp8_quant(
        tensor, scale=None, use_per_token_if_dynamic=per_token
    )

    if tensor.ndim != len(original_shape):
        tensor_fp8 = tensor_fp8.reshape(*original_shape)

    return QuantizedTensor(data=tensor_fp8, scales=scales)


def dequantize_fp8(quantized: QuantizedTensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize FP8 tensor back to higher precision."""
    tensor = quantized.data.to(dtype)

    original_shape = tensor.shape
    if tensor.ndim != 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])

    tensor = tensor * quantized.scales

    if tensor.ndim != len(original_shape):
        tensor = tensor.reshape(*original_shape)

    return tensor


def quantized_all_gather(
    tensor: torch.Tensor,
    world_size: int,
    group: Optional[dist.ProcessGroup] = None,
    per_token: bool = False,
) -> torch.Tensor:
    """
    AllGather with FP8 quantization (50% bandwidth reduction).

    Args:
        tensor: Local tensor to gather.
        world_size: Number of ranks.
        group: Process group.
        per_token: Use per-row quantization for better accuracy.

    Returns:
        Gathered tensor [world_size * local_size, ...].
    """
    dtype = tensor.dtype
    quantized = quantize_fp8(tensor, per_token=per_token)

    fp8_list = [torch.empty_like(quantized.data) for _ in range(world_size)]
    scales_list = [torch.empty_like(quantized.scales) for _ in range(world_size)]

    dist.all_gather(fp8_list, quantized.data, group=group)
    dist.all_gather(scales_list, quantized.scales, group=group)

    return torch.cat(
        [dequantize_fp8(QuantizedTensor(fp8, sc), dtype) for fp8, sc in zip(fp8_list, scales_list)],
        dim=0,
    )


def should_quantize_comm(tensor: torch.Tensor, threshold_bytes: int = 1024 * 1024) -> bool:
    """Heuristic: quantize if tensor >= threshold (default 1MB)."""
    return tensor.nbytes >= threshold_bytes


__all__ = [
    "QuantizedTensor",
    "quantize_fp8",
    "dequantize_fp8",
    "quantized_all_gather",
    "should_quantize_comm",
]
