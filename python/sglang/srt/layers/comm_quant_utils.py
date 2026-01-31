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
FP8 Communication Quantization Primitives

Quantize tensors to FP8 before collective communication for 50% bandwidth reduction.

IMPORTANT: Only use for data-movement collectives (AllGather, AllToAll) where no
arithmetic is performed on quantized values. Do NOT use for reduction operations
(AllReduce, ReduceScatter) - the naive AllGather-based implementation would use
O(world_size) more memory and bandwidth than optimized ring-reduce algorithms.

Usage:
    from sglang.srt.layers.comm_quant_utils import quantized_all_gather

    # 50% bandwidth savings for AllGather
    result = quantized_all_gather(tensor, world_size, group)
"""

from typing import List, NamedTuple, Optional

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

    This is the ideal use case for FP8 communication - AllGather already requires
    O(world_size × tensor_size) memory, so quantization provides pure bandwidth
    savings with no algorithmic penalty.

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


def quantized_all_gather_into_tensor(
    output: torch.Tensor,
    tensor: torch.Tensor,
    world_size: int,
    group: Optional[dist.ProcessGroup] = None,
    per_token: bool = False,
) -> None:
    """AllGather with FP8, writing into pre-allocated output tensor."""
    result = quantized_all_gather(tensor, world_size, group, per_token)
    output.copy_(result)


def quantized_all_to_all(
    output_tensor_list: List[torch.Tensor],
    input_tensor_list: List[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
    per_token: bool = False,
) -> None:
    """
    AllToAll with FP8 quantization (50% bandwidth reduction).

    Like AllGather, AllToAll is pure data movement with no arithmetic,
    making it ideal for FP8 quantization.

    Args:
        output_tensor_list: List of output tensors (one per rank).
        input_tensor_list: List of input tensors (one per rank).
        group: Process group.
        per_token: Use per-row quantization.
    """
    dtype = input_tensor_list[0].dtype
    world_size = len(input_tensor_list)

    quantized_inputs = [quantize_fp8(t, per_token=per_token) for t in input_tensor_list]

    fp8_outputs = [torch.empty_like(q.data) for q in quantized_inputs]
    scales_outputs = [torch.empty_like(q.scales) for q in quantized_inputs]

    dist.all_to_all(fp8_outputs, [q.data for q in quantized_inputs], group=group)
    dist.all_to_all(scales_outputs, [q.scales for q in quantized_inputs], group=group)

    for i, (fp8, sc) in enumerate(zip(fp8_outputs, scales_outputs)):
        output_tensor_list[i].copy_(dequantize_fp8(QuantizedTensor(fp8, sc), dtype))


def quantized_all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    world_size: int,
    group: Optional[dist.ProcessGroup] = None,
    per_token: bool = False,
) -> None:
    """
    AllToAll on single tensors (split along dim 0, exchange, concatenate).

    Args:
        output: Output tensor, same shape as input.
        input: Input tensor, first dim divisible by world_size.
        world_size: Number of ranks.
        group: Process group.
        per_token: Use per-row quantization.
    """
    dtype = input.dtype

    input_chunks = list(input.chunk(world_size, dim=0))
    quantized_chunks = [quantize_fp8(c, per_token=per_token) for c in input_chunks]

    fp8_outputs = [torch.empty_like(q.data) for q in quantized_chunks]
    scales_outputs = [torch.empty_like(q.scales) for q in quantized_chunks]

    dist.all_to_all(fp8_outputs, [q.data for q in quantized_chunks], group=group)
    dist.all_to_all(scales_outputs, [q.scales for q in quantized_chunks], group=group)

    dequantized = [
        dequantize_fp8(QuantizedTensor(fp8, sc), dtype)
        for fp8, sc in zip(fp8_outputs, scales_outputs)
    ]
    output.copy_(torch.cat(dequantized, dim=0))


def should_quantize_comm(tensor: torch.Tensor, threshold_bytes: int = 1024 * 1024) -> bool:
    """Heuristic: quantize if tensor >= threshold (default 1MB)."""
    return tensor.nbytes >= threshold_bytes


__all__ = [
    "QuantizedTensor",
    "quantize_fp8",
    "dequantize_fp8",
    "quantized_all_gather",
    "quantized_all_gather_into_tensor",
    "quantized_all_to_all",
    "quantized_all_to_all_single",
    "should_quantize_comm",
]
