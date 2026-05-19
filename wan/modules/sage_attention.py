import torch

import triton
import triton.language as tl

from typing import Any, Optional


# Keep quantization and attention block shapes aligned for numerical correctness.
DEFAULT_ATTN_BLOCK_M = 128
DEFAULT_ATTN_BLOCK_N = 64

_PRINTED_TUNED_CONFIG = False

try:
    _compile_disable = torch.compiler.disable
except AttributeError:
    _compile_disable = torch._dynamo.disable


@triton.jit
def quant_per_block_int8_kernel(
    Input,
    Output,
    Scale,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_oz,
    stride_oh,
    stride_on,
    stride_sz,
    stride_sh,
    sm_scale,
    C: tl.constexpr,
    BLK: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    base_i = Input + off_b * stride_iz + off_h * stride_ih
    base_o = Output + off_b * stride_oz + off_h * stride_oh
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_c = tl.arange(0, C)
    block_mask = offs_n[:, None] < L

    input_ptrs = base_i + offs_n[:, None] * stride_in + offs_c[None, :]
    output_ptrs = base_o + offs_n[:, None] * stride_on + offs_c[None, :]

    x = tl.load(input_ptrs, mask=block_mask, other=0.0).to(tl.float32)
    x *= sm_scale
    scale = tl.maximum(tl.max(tl.abs(x)) / 127.0, 1e-8)
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)

    tl.store(output_ptrs, x_int8, mask=block_mask)
    tl.store(scale_ptrs, scale)


def per_block_int8_triton(
    q,
    k,
    km=None,
    BLKQ=DEFAULT_ATTN_BLOCK_M,
    BLKK=DEFAULT_ATTN_BLOCK_N,
    sm_scale=None,
    tensor_layout="HND",
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if km is not None:
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    grid = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    quant_per_block_int8_kernel[grid](
        q,
        q_int8,
        q_scale,
        qo_len,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_qo,
        stride_h_qo,
        stride_seq_qo,
        q_scale.stride(0),
        q_scale.stride(1),
        sm_scale=(sm_scale * 1.44269504),
        C=head_dim,
        BLK=BLKQ,
    )

    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_per_block_int8_kernel[grid](
        k,
        k_int8,
        k_scale,
        kv_len,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_ko,
        stride_h_ko,
        stride_seq_ko,
        k_scale.stride(0),
        k_scale.stride(1),
        sm_scale=1.0,
        C=head_dim,
        BLK=BLKK,
    )

    return q_int8, q_scale, k_int8, k_scale

@triton.jit
def _attn_fwd_inner(
    acc0,
    acc1,
    l_i,
    m_i,
    q,
    q_scale,
    kv_len,
    K_desc,
    K_scale_ptr,
    V_desc,
    start_m,
    mask_desc,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    V_TILE_SIZE: tl.constexpr,
    V_TILES: tl.constexpr,
    STAGE: tl.constexpr,
    offs_n: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    for start_n in tl.range(0, kv_len, BLOCK_N, num_stages=STAGE):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = K_desc.load([start_n, 0]).T
        k_scale = tl.load(K_scale_ptr + (start_n // BLOCK_N))

        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)

        if HAS_MASK:
            mask_block = mask_desc.load([start_m * BLOCK_M, start_n])
            if mask_block.dtype == tl.int1:
                qk = qk + tl.where(mask_block, 0, -1.0e6)
            else:
                qk = qk + mask_block

        qk += tl.where(k_mask, 0, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc0 = acc0 * alpha[:, None]
        if V_TILES == 2:
            acc1 = acc1 * alpha[:, None]

        v0 = V_desc.load([start_n, 0])
        p = p.to(tl.float16)

        acc0 = tl.dot(p, v0, acc0)

        if V_TILES == 2:
            v1 = V_desc.load([start_n, V_TILE_SIZE])
            acc1 = tl.dot(p, v1, acc1)

        m_i = m_ij
    return acc0, acc1, l_i, m_i
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    Out,
    mask,
    Lse,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_on,
    stride_maskz,
    stride_maskh,
    stride_maskm,
    stride_maskn,
    qo_len,
    kv_len,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    V_TILE_SIZE: tl.constexpr,
    V_TILES: tl.constexpr,
    STAGE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    base_q = Q + (off_z * stride_qz + off_h * stride_qh)
    base_k = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh)
    base_v = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh)
    base_o = Out + (off_z * stride_oz + off_h * stride_oh)

    Q_desc = tl.make_tensor_descriptor(
        base=base_q,
        shape=(qo_len, HEAD_DIM),
        strides=(stride_qn, 1),
        block_shape=(BLOCK_M, HEAD_DIM),
    )
    K_desc = tl.make_tensor_descriptor(
        base=base_k,
        shape=(kv_len, HEAD_DIM),
        strides=(stride_kn, 1),
        block_shape=(BLOCK_N, HEAD_DIM),
    )
    V_desc = tl.make_tensor_descriptor(
        base=base_v,
        shape=(kv_len, HEAD_DIM),
        strides=(stride_vn, 1),
        block_shape=(BLOCK_N, V_TILE_SIZE),
    )
    O_desc = tl.make_tensor_descriptor(
        base=base_o,
        shape=(qo_len, HEAD_DIM),
        strides=(stride_on, 1),
        block_shape=(BLOCK_M, V_TILE_SIZE),
    )
    Q_scale_ptr = Q_scale + q_scale_offset
    K_scale_ptr = K_scale + k_scale_offset

    if HAS_MASK:
        mask_desc = tl.make_tensor_descriptor(
            base=mask + (off_z * stride_maskz + off_h * stride_maskh),
            shape=(qo_len, kv_len),
            strides=(stride_maskm, stride_maskn),
            block_shape=(BLOCK_M, BLOCK_N),
        )
    else:
        mask_desc = None

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc0 = tl.zeros([BLOCK_M, V_TILE_SIZE], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M, V_TILE_SIZE], dtype=tl.float32)

    q = Q_desc.load([start_m * BLOCK_M, 0])
    q_scale = tl.load(Q_scale_ptr + start_m)
    acc0, acc1, l_i, m_i = _attn_fwd_inner(
        acc0,
        acc1,
        l_i,
        m_i,
        q,
        q_scale,
        kv_len,
        K_desc,
        K_scale_ptr,
        V_desc,
        start_m,
        mask_desc,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        V_TILE_SIZE,
        V_TILES,
        STAGE,
        offs_n,
        HAS_MASK,
    )
    acc0 = acc0 / l_i[:, None]
    O_desc.store([start_m * BLOCK_M, 0], acc0.to(Out.type.element_ty))
    if V_TILES == 2:
        acc1 = acc1 / l_i[:, None]
        O_desc.store([start_m * BLOCK_M, V_TILE_SIZE], acc1.to(Out.type.element_ty))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))


def _build_attn_autotune_configs():
    configs = []
    for num_warps in (4, 8, 16):
        for num_stages in (1, 2, 3, 4):
            configs.append(
                triton.Config(
                    {"BLOCK_M": DEFAULT_ATTN_BLOCK_M, "BLOCK_N": DEFAULT_ATTN_BLOCK_N},
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return configs


_attn_fwd_tuned = triton.autotune(
    configs=_build_attn_autotune_configs(),
    key=["qo_len", "kv_len", "HEAD_DIM", "RETURN_LSE"],
)(_attn_fwd)


def _print_selected_attn_config(debug_label="sage_attention"):
    global _PRINTED_TUNED_CONFIG
    if _PRINTED_TUNED_CONFIG:
        return

    cfg = getattr(_attn_fwd_tuned, "best_config", None)
    if cfg is None:
        cache = getattr(_attn_fwd_tuned, "cache", None)
        if isinstance(cache, dict) and cache:
            try:
                cfg = next(reversed(cache.values()))
            except TypeError:
                cfg = list(cache.values())[-1]

    if cfg is None:
        print(f"[{debug_label}] autotune config unavailable")
        _PRINTED_TUNED_CONFIG = True
        return

    print(
        f"[{debug_label}] selected config: {getattr(cfg, 'kwargs', {})}, "
        f"num_warps={getattr(cfg, 'num_warps', None)}, "
        f"num_stages={getattr(cfg, 'num_stages', None)}"
    )
    _PRINTED_TUNED_CONFIG = True


def attn(
    q,
    k,
    v,
    q_scale,
    k_scale,
    tensor_layout="HND",
    attn_mask=None,
    output_dtype=torch.float16,
    return_lse=False,
    tune_kernel=False,
    print_tuned_config=False,
):
    block_m = DEFAULT_ATTN_BLOCK_M
    block_n = DEFAULT_ATTN_BLOCK_N
    stage = 3

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    if attn_mask is not None:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = attn_mask.stride(0), attn_mask.stride(1), attn_mask.stride(2), attn_mask.stride(3)
    else:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = 0, 0, 0, 0

    num_kv_groups = h_qo // h_kv

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device="cpu")

    grid = (triton.cdiv(qo_len, block_m), h_qo, b)

    v_tile_size = 64
    v_tiles = 2 if head_dim == 128 else 1

    launch_kwargs = dict(
        HEAD_DIM=head_dim,
        V_TILE_SIZE=v_tile_size,
        V_TILES=v_tiles,
        STAGE=stage,
        RETURN_LSE=return_lse,
        HAS_MASK=(attn_mask is not None),
    )
    if tune_kernel:
        _attn_fwd_tuned[grid](
            q, k, v, q_scale, k_scale, o, attn_mask, lse,
            stride_bz_q, stride_h_q, stride_seq_q,
            stride_bz_k, stride_h_k, stride_seq_k,
            stride_bz_v, stride_h_v, stride_seq_v,
            stride_bz_o, stride_h_o, stride_seq_o,
            stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask,
            qo_len, kv_len,
            h_qo, num_kv_groups,
            **launch_kwargs,
        )
        if print_tuned_config:
            _print_selected_attn_config()
    else:
        _attn_fwd[grid](
            q, k, v, q_scale, k_scale, o, attn_mask, lse,
            stride_bz_q, stride_h_q, stride_seq_q,
            stride_bz_k, stride_h_k, stride_seq_k,
            stride_bz_v, stride_h_v, stride_seq_v,
            stride_bz_o, stride_h_o, stride_seq_o,
            stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask,
            qo_len, kv_len,
            h_qo, num_kv_groups,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=8 if head_dim == 64 else 16,
            num_stages=3 if head_dim == 64 else 4,
            **launch_kwargs)

    return o, lse


@_compile_disable
def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    quantization_backend: str = "triton",
    is_causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    smooth_k: bool = False,
    return_lse: bool = False,
    tune_kernel: bool = False,
    print_tuned_config: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    dtype = q.dtype
    assert q.is_xpu, "Input tensors must be on xpu."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    if is_causal:
        raise NotImplementedError("Causal mode is not implemented for sageattn_qk_int8_pv_fp16_triton")

    if attn_mask is not None:
        assert attn_mask.dtype == torch.bool or attn_mask.dtype == q.dtype, "attn_mask must be of dtype bool or the same dtype as q."
        assert attn_mask.device == q.device, "All tensors must be on the same device."

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2
    nh_dim = 2 if tensor_layout == "NHD" else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    if quantization_backend == "triton":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(
            q,
            k,
            km=km,
            BLKQ=DEFAULT_ATTN_BLOCK_M,
            BLKK=DEFAULT_ATTN_BLOCK_N,
            sm_scale=sm_scale,
            tensor_layout=tensor_layout,
        )
    elif quantization_backend == "sycl-tla":
        raise NotImplementedError("sycl-tla backend is not implemented in this Triton module")
    else:
        raise ValueError(f"Unsupported quantization backend: {quantization_backend}")

    if attn_mask is not None:
        if tensor_layout == "HND":
            target_shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        elif tensor_layout == "NHD":
            target_shape = (q.shape[0], q.shape[2], q.shape[1], k.shape[1])
        else:
            raise ValueError(f"tensor_layout {tensor_layout} not supported")
        try:
            attn_mask = attn_mask.expand(target_shape)
        except Exception:
            raise AssertionError(f"attn_mask shape {attn_mask.shape} cannot be broadcast to {target_shape}")
        
    o, lse = attn(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        output_dtype=dtype,
        attn_mask=attn_mask,
        return_lse=return_lse,
        tune_kernel=tune_kernel,
        print_tuned_config=print_tuned_config,
    )

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    return o
