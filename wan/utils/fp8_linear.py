# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
HPU FP8 Linear GEMM utilities for Wan inference models.

Provides WanFP8Linear, a drop-in wrapper for torch.nn.Linear that:
  - Precomputes per-output-channel FP8 weights and scales at initialization time.
  - Frees the original BF16 weights immediately to save memory.
  - Uses HPU fp8_gemm_v2 for fast FP8 GEMM forward on Gaudi hardware.
"""

import logging

import torch
import torch.nn as nn

# Maximum representable value for torch.float8_e4m3fn.
FP8_MAX = 448.0
# Small epsilon used when computing per-channel / per-row FP8 scales to avoid
# division-by-zero for all-zero rows.
SCALE_EPSILON = 1e-8


def dynamic_quant(data: torch.Tensor, single_scale: bool = False):
    """Dynamically quantize *data* to FP8 and return (data_fp8, scale).

    Args:
        data: Input tensor (BF16 / FP32).
        single_scale: When True use a single global scale; otherwise use a
            per-row scale (last dimension treated as the feature axis).

    Returns:
        (data_fp8, scale): data_fp8 is float8_e4m3fn; scale is float32 and
            has the same shape as data except the last dim is 1 (or scalar
            when single_scale=True).
    """
    if single_scale:
        scale = (torch.abs(data).max() + SCALE_EPSILON) / FP8_MAX
    else:
        scale = (torch.abs(data).max(dim=-1).values + SCALE_EPSILON) / FP8_MAX
        scale = scale.unsqueeze(-1)

    if data.device.type == 'hpu':
        data_fp8 = torch.ops.hpu.cast_to_fp8_v2(
            data, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
    else:
        data_fp8 = (data / scale).to(torch.float8_e4m3fn)

    return data_fp8, scale.float()


def apply_fp8_linear_hpu(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor = None,
    bias: torch.Tensor = None,
    trans_B: bool = True,
) -> torch.Tensor:
    """Compute an FP8 GEMM on HPU.

    Args:
        input: Activation tensor (BF16).  May have any leading batch dims.
        weight: Pre-quantized FP8 weight tensor with shape
            ``[out_features, in_features]`` (standard Linear layout).
        weight_scale: Per-output-channel dequantisation scale with shape
            ``[out_features, 1]``.
        input_scale: Optional pre-computed activation scale.  When *None*,
            activations are quantised dynamically (per-row).
        bias: Optional bias tensor (will be cast to the input dtype).
        trans_B: Whether to transpose B inside the GEMM.  Defaults to
            ``True`` so that the standard ``[out, in]`` weight layout works
            with ``A @ B.T``.

    Returns:
        Output tensor with the same dtype as *input* and shape
        ``(*leading_dims, out_features)``.
    """
    x_shape = input.shape
    if len(x_shape) > 2:
        input = input.reshape(-1, x_shape[-1])

    if input_scale is None:
        x_fp8, x_scale = dynamic_quant(input)
    else:
        x_fp8 = torch.ops.hpu.cast_to_fp8_v2(
            input, 1.0 / input_scale, False, False, torch.float8_e4m3fn)[0]
        x_scale = input_scale

    output = torch.ops.hpu.fp8_gemm_v2(
        A=x_fp8,
        trans_A=False,
        B=weight,
        trans_B=trans_B,
        D=None,
        out_dtype=input.dtype,
        A_scale_inv=x_scale,
        B_scale_inv=weight_scale,
        bias=bias,
        accumulate=False,
    )

    if len(x_shape) > 2:
        output = output.reshape(*x_shape[:-1], -1)
    return output


def _quantize_weight_to_fp8_per_channel(weight: torch.Tensor):
    """Quantize *weight* to FP8 with a per-output-channel scale.

    Args:
        weight: Float weight tensor with shape ``[out_features, in_features]``.
            May be BF16, FP16, or FP32; scale computation is done in FP32 to
            preserve numerical accuracy.

    Returns:
        (weight_fp8, scale): weight_fp8 is float8_e4m3fn with the same shape
            as *weight*; scale is float32 with shape ``[out_features, 1]``.
    """
    # Compute scale in FP32 for numerical stability regardless of input dtype.
    w_abs = weight.float().abs()
    w_max = w_abs.max(dim=-1, keepdim=True).values
    scale = (w_max + SCALE_EPSILON) / FP8_MAX  # [out, 1], float32

    if weight.device.type == 'hpu':
        weight_fp8 = torch.ops.hpu.cast_to_fp8_v2(
            weight, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
    else:
        weight_fp8 = (weight.float() / scale).to(torch.float8_e4m3fn)

    return weight_fp8, scale


class WanFP8Linear(nn.Module):
    """FP8-wrapped replacement for ``torch.nn.Linear`` targeting HPU.

    At construction time the BF16 weight is compressed into a per-output-
    channel FP8 representation and the original weight tensor is released to
    free memory.  During forward, activations are quantised on-the-fly
    (per-row, dynamic) and the GEMM is executed via
    ``torch.ops.hpu.fp8_gemm_v2``.

    This class is **not** meant to be constructed directly.  Use
    :func:`wrap_blocks_linear_fp8` to replace all eligible
    ``torch.nn.Linear`` modules inside ``model.blocks``.
    """

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Precompute per-output-channel FP8 weight and scale at init time.
        # linear.weight shape: [out_features, in_features]
        weight_fp8, weight_scale = _quantize_weight_to_fp8_per_channel(
            linear.weight.data)

        # Register as buffers so that .to(device) moves them together with
        # the rest of the model (e.g. when calling model.to(hpu_device)).
        self.register_buffer('weight_fp8', weight_fp8)
        self.register_buffer('weight_scale', weight_scale)

        if linear.bias is not None:
            self.register_buffer('bias', linear.bias.data.clone())
        else:
            self.register_buffer('bias', None)

        # Free the original BF16 weight (and bias) to reclaim memory.
        # NOTE: after this point, the original `linear` object is left in an
        # invalid state and must not be used.
        del linear.weight
        if linear.bias is not None:
            del linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        if bias is not None:
            bias = bias.to(x.dtype)
        return apply_fp8_linear_hpu(
            x, self.weight_fp8, self.weight_scale, bias=bias, trans_B=True)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}')


def wrap_blocks_linear_fp8(model: nn.Module) -> None:
    """Replace every ``nn.Linear`` inside ``model.blocks`` with
    :class:`WanFP8Linear` in-place.

    Only the ``self.blocks`` attribute of *model* is affected; all other
    Linear layers (embeddings, heads, etc.) remain in BF16.

    Args:
        model: A ``WanModel`` or ``WanModel_S2V`` instance whose ``blocks``
            attribute contains the transformer blocks to wrap.
    """
    blocks = getattr(model, 'blocks', None)
    if blocks is None:
        logging.warning(
            'wrap_blocks_linear_fp8: model has no `blocks` attribute, '
            'skipping FP8 wrapping.')
        return

    wrapped_count = 0
    for block_idx, block in enumerate(blocks):
        # Collect (parent_module, child_name) pairs for all Linear layers
        # inside this block.  We iterate over a snapshot of named_modules so
        # that in-place replacement does not interfere with iteration.
        replacements = []
        for module_path, module in block.named_modules():
            if isinstance(module, nn.Linear):
                replacements.append(module_path)

        for module_path in replacements:
            # Walk down to the parent of the target Linear.
            parent = block
            parts = module_path.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]
            linear = getattr(parent, child_name)
            if not isinstance(linear, nn.Linear):
                # Guard against the unlikely case where a module has already
                # been replaced by a previous iteration (e.g. if two paths
                # in named_modules resolve to the same object via aliasing).
                continue
            setattr(parent, child_name, WanFP8Linear(linear))
            wrapped_count += 1

    logging.info(
        f'FP8 wrapping: replaced {wrapped_count} Linear modules across '
        f'{len(blocks)} transformer blocks.')
