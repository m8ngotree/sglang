from typing import NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant


class QuantizedTensor(NamedTuple):
    data: torch.Tensor
    scales: torch.Tensor


def quantize_fp8(tensor: torch.Tensor, per_token: bool = False) -> QuantizedTensor:
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


__all__ = [
    "QuantizedTensor",
    "quantize_fp8",
    "dequantize_fp8",
    "quantized_all_gather",
]
