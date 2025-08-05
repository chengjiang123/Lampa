# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import repeat, reduce, rearrange, einsum
from .cumsum import chunk_local_cumsum
from .fla_op import exp, safe_exp
from .fla_future.solve_tril import (
    solve_tril)
from .fla_future.wy_fast import (
    recompute_w_u_fwd,
    prepare_wy_repr_bwd)
from .fla_future.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd)
from .fla_future.chunk_delta_h import (
    preprocess_qkw,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64)
from .chunk_o import chunk_bwd_kernel_dv_local
from .base import (
    HType,
    ceil_log,
    get_num_levels,
    make_levels_matrix)
from .chunkwise import (
    assert_contiguous,
    NUM_WARPS_OPTIONS,
    BLOCK_SIZE_OPTIONS,
    NUM_STAGES_OPTIONS,
    chunk_fwd_kernel_o,
    chunk_bwd_kernel_dv)
from .autotune import autotune as custom_autotune
from .hssd_minimal import segsum

BLOCK_SIZE_OPTIONS = [32]
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BLOCK_SIZE_OPTIONS
        for BV in BLOCK_SIZE_OPTIONS
        for num_warps in NUM_WARPS_OPTIONS
        for num_stages in NUM_STAGES_OPTIONS
    ],
    key=["H", "K", "V", "L", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o_intra(
    q,
    k,
    v,
    b,
    g,
    l,
    o,
    Aw,
    llut,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    # offset calculation
    q  += (i_b * T * H + i_h) * K
    k  += (i_b * T * H + i_h) * K
    v  += (i_b * T * H + i_h) * V
    b  += (i_b * T * H + i_h)
    g  += (i_b * T * H + i_h)
    l  += (i_b * T * H + i_h) * L
    o  += (i_b * T * H + i_h) * V
    Aw += (i_b * T * H + i_h) * BT

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_A = tl.where(m_A, b_A, 0)

    p_b    = tl.make_block_ptr(b   , (T,   ), (H     ,  ), (i_t * BT,         ), (BT,   ), (0,  ))
    p_g    = tl.make_block_ptr(g   , (T,   ), (H     ,  ), (i_t * BT,         ), (BT,   ), (0,  ))
    p_v    = tl.make_block_ptr(v   , (T, V ), (H * V , 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o    = tl.make_block_ptr(o   , (T, V ), (H * V , 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_Aw   = tl.make_block_ptr(Aw  , (T, BT), (H * BT, 1), (i_t * BT, 0       ), (BT, BT), (1, 0))
    p_llut = tl.make_block_ptr(llut, (T, T ), (T     , 1), (i_t * BT, i_t * BT), (BT, BT), (1, 0))

    b_b  = tl.load(p_b , boundary_check=(0,  ))
    b_g  = tl.load(p_g , boundary_check=(0,  ))
    b_v  = tl.load(p_v , boundary_check=(0, 1))
    b_Aw = tl.load(p_Aw, boundary_check=(0, 1))

    # lambdas
    vals_level_lut = tl.load(p_llut, boundary_check=(0, 1))
    indices_target = i_t * BT + tl.arange(0, BT)[:, None]
    indices_target = tl.broadcast_to(indices_target, (BT, BT))
    p_la = l + indices_target * H * L + vals_level_lut
    b_l = tl.load(p_la)

    # [BT, BS0] @ [BS0, BS1] -> [BT, BS1]
    b_A2 = tl.dot(b_A, b_Aw.to(b_A.dtype))
    b_A2 = b_A2 * safe_exp(b_g[:, None] - b_g[None, :]) * b_b[None, :] * b_l

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = tl.dot(b_A2.to(b_v.dtype), b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_OPTIONS
        for num_stages in NUM_STAGES_OPTIONS
    ],
    key=["H", "K", "V", "L", "BT", "BK", "BV"],
    restore_value=["dq", "dk", "dg", "dl", "dw"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dqkwgl(
    q,
    k,
    v,
    h,
    g,
    l,
    w,
    u,
    do,
    dh,
    dq,
    dk,
    dg,
    dl,
    dv,
    dw,
    ell_h,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    NT = tl.cdiv(T, BT)
    i_tg = i_b * NT + i_t
    bos  = i_b * T

    # offset calculation
    q  += (bos  * H + i_h) * K
    k  += (bos  * H + i_h) * K
    v  += (bos  * H + i_h) * V
    g  += (bos  * H + i_h)
    l  += (bos  * H + i_h) * L
    w  += (bos  * H + i_h) * K
    u  += (bos  * H + i_h) * V

    do += (bos  * H + i_h) * V
    dq += (bos  * H + i_h) * K
    dk += (bos  * H + i_h) * K
    dv += (bos  * H + i_h) * V
    dw += (bos  * H + i_h) * K

    dg += i_k * B * H * T
    dl += i_k * B * H * T * L
    dg += (bos  * H + i_h)
    dl += (bos  * H + i_h) * L
    h  += (i_tg * H + i_h).to(tl.int64) * K * V
    dh += (i_tg * H + i_h).to(tl.int64) * K * V

    b_dq  = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk  = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds  = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw  = tl.zeros([BT, BK], dtype=tl.float32)
    b_dg  = tl.zeros([BT,   ], dtype=tl.float32)
    b_dlh = tl.zeros([BT,   ], dtype=tl.float32)
    b_dg_last = tl.zeros([1,], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v  = tl.make_block_ptr(v , (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h  = tl.make_block_ptr(h , (V, K), (1    , V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh, (V, K), (1    , V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u  = tl.make_block_ptr(u , (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        # [BT, BV]
        b_v  = tl.load(p_v , boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.load(p_dv, boundary_check=(0, 1))
        b_u  = tl.load(p_u , boundary_check=(0, 1))

        # [BV, BK]
        b_h  = tl.load(p_h , boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        b_dg_last += (tl.sum(b_h * b_dh))

        # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        b_dk += tl.dot(b_u, b_dh.to(b_v.dtype))

        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    tl.debug_barrier()
    o_i   = tl.arange(0, BT)
    p_q   = tl.make_block_ptr(q         , (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k   = tl.make_block_ptr(k         , (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_g   = tl.make_block_ptr(g         , (T,  ), (H    ,  ), (i_t * BT,         ), (BT,   ), (0,  ))
    p_lh  = tl.make_block_ptr(l + ell_h , (T,  ), (H * L,  ), (i_t * BT,         ), (BT,   ), (0,  ))
    p_w   = tl.make_block_ptr(w         , (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dq  = tl.make_block_ptr(dq        , (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk  = tl.make_block_ptr(dk        , (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg  = tl.make_block_ptr(dg        , (T,  ), (H    ,  ), (i_t * BT,         ), (BT,   ), (0,  ))
    p_dlh = tl.make_block_ptr(dl + ell_h, (T,  ), (H * L,  ), (i_t * BT,         ), (BT,   ), (0,  ))
    p_dw  = tl.make_block_ptr(dw        , (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    b_q = tl.load(p_q , boundary_check=(0, 1))
    b_k = tl.load(p_k , boundary_check=(0, 1))
    b_g = tl.load(p_g , boundary_check=(0,  ))
    b_l = tl.load(p_lh, boundary_check=(0,  ))
    b_w = tl.load(p_w , boundary_check=(0, 1))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)

    b_dg_last *= exp(b_g_last)

    b_dw = b_dw * exp(b_g)[:, None]
    b_dg -= tl.sum(b_w * b_dw, axis=1)

    b_dq = b_dq * exp(b_g)[:, None]
    b_dlh += tl.sum(b_dq * b_q, axis=1)
    b_dq = b_dq * b_l[:, None]
    b_dg += tl.sum(b_dq * b_q, axis=1)

    b_dk = b_dk * safe_exp(-b_g + b_g_last)[:, None]
    b_dg -= tl.sum(b_k * b_dk, axis=1)
    b_dg_last += tl.sum(b_dk * b_k)

    b_ds  = tl.where(o_i[:, None] >= o_i[None, :], b_ds * safe_exp(b_g[:, None] - b_g[None, :]), 0)
    b_dla = b_ds * tl.dot(b_q, tl.trans(b_k))
    b_dlh += tl.sum(b_dla, axis=1)
    b_dsg = b_dla * b_l[:, None]
    b_dg += tl.sum(b_dsg, axis=1)
    b_dg -= tl.sum(b_dsg, axis=0)

    b_dsl = (b_ds * b_l[:, None]).to(b_k.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_dsl, b_k)
    b_dk += tl.dot(tl.trans(b_dsl), b_q)

    # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
    # b_dg = tl.dot(tl.where(o_i[:, None] <= o_i[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
    b_dg = tl.where(o_i < min(BT, T - i_t * BT) - 1, b_dg, b_dg + b_dg_last)

    b_dq_old  = tl.load(p_dq , boundary_check=(0, 1))
    b_dk_old  = tl.load(p_dk , boundary_check=(0, 1))
    b_dg_old  = tl.load(p_dg , boundary_check=(0,  ))
    b_dlh_old = tl.load(p_dlh, boundary_check=(0,  ))
    b_dw_old  = tl.load(p_dw , boundary_check=(0, 1))
    b_dq  += b_dq_old
    b_dk  += b_dk_old
    b_dg  += b_dg_old
    b_dlh += b_dlh_old
    b_dw2  = b_dw_old - b_dw
    tl.store(p_dq , b_dq .to(p_dq .dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk , b_dk .to(p_dk .dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg , b_dg .to(p_dg .dtype.element_ty), boundary_check=(0,  ))
    tl.store(p_dlh, b_dlh.to(p_dlh.dtype.element_ty), boundary_check=(0,  ))
    tl.store(p_dw , b_dw2.to(p_dw .dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_OPTIONS
        for num_stages in NUM_STAGES_OPTIONS
    ],
    key=["H", "K", "V", "L", "BT", "BK", "BV"],
    restore_value=["dl"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dqkgl_intra(
    q,
    k,
    v,
    b,
    g,
    l,
    Aw,
    do,
    dq,
    dk,
    dg,
    dl,
    llut,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    # offset calculation
    q  += (i_b * T * H + i_h) * K
    k  += (i_b * T * H + i_h) * K
    v  += (i_b * T * H + i_h) * V
    b  += (i_b * T * H + i_h)
    g  += (i_b * T * H + i_h)
    l  += (i_b * T * H + i_h) * L
    Aw += (i_b * T * H + i_h) * BT
    do += (i_b * T * H + i_h) * V
    dq += (i_b * T * H + i_h) * K
    dk += (i_b * T * H + i_h) * K

    dg += i_k * B * H * T
    dl += i_k * B * H * T * L
    dg += (i_b * T * H + i_h)
    dl += (i_b * T * H + i_h) * L

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg = tl.zeros([BT,   ], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v  = tl.make_block_ptr(v , (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        # [BT, BV]
        b_v  = tl.load(p_v , boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v))

    tl.debug_barrier()
    o_i    = tl.arange(0, BT)
    p_q    = tl.make_block_ptr(q   , (T, K ), (H * K , 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k    = tl.make_block_ptr(k   , (T, K ), (H * K , 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_b    = tl.make_block_ptr(b   , (T,   ), (H     ,  ), (i_t * BT,         ), (BT,   ), (0,  ))
    p_g    = tl.make_block_ptr(g   , (T,   ), (H     ,  ), (i_t * BT,         ), (BT,   ), (0,  ))
    p_Aw   = tl.make_block_ptr(Aw  , (T, BT), (H * BT, 1), (i_t * BT, 0       ), (BT, BT), (1, 0))
    p_dq   = tl.make_block_ptr(dq  , (T, K ), (H * K , 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk   = tl.make_block_ptr(dk  , (T, K ), (H * K , 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg   = tl.make_block_ptr(dg  , (T,   ), (H     ,  ), (i_t * BT,         ), (BT,   ), (0,  ))
    p_llut = tl.make_block_ptr(llut, (T, T ), (T     , 1), (i_t * BT, i_t * BT), (BT, BT), (1, 0))

    b_q  = tl.load(p_q , boundary_check=(0, 1))
    b_k  = tl.load(p_k , boundary_check=(0, 1))
    b_b  = tl.load(p_b , boundary_check=(0,  ))
    b_g  = tl.load(p_g , boundary_check=(0,  ))
    b_Aw = tl.load(p_Aw, boundary_check=(0, 1))

    # lambdas
    vals_level_lut = tl.load(p_llut, boundary_check=(0, 1))
    indices_target = i_t * BT + tl.arange(0, BT)[:, None]
    indices_target = tl.broadcast_to(indices_target, (BT, BT))
    p_la  = l  + indices_target * H * L + vals_level_lut
    p_dla = dl + indices_target * H * L + vals_level_lut
    b_l = tl.load(p_la)

    # [BT, BK] @ [BK, BS0] -> [BT, BS0]
    b_A = tl.dot(b_q, tl.trans(b_k))
    b_A = tl.where(o_i[:, None] >= o_i[None, :], b_A, 0)
    # [BT, BS0] @ [BS0, BS1] -> [BT, BS1]
    b_A2 = tl.dot(b_A, b_Aw.to(b_A.dtype))

    b_ds  = b_ds * safe_exp(b_g[:, None] - b_g[None, :]) * b_b[None, :]
    b_dla = b_ds * b_A2
    b_dsg = b_dla * b_l
    b_dg += tl.sum(b_dsg, axis=1)
    b_dg -= tl.sum(b_dsg, axis=0)

    b_dsl = b_ds * b_l
    # [BT, BS1] @ [BS1, BS0] -> [BT, BS0]
    b_ds2 = tl.dot(b_dsl, tl.trans(b_Aw).to(b_dsl.dtype))
    b_ds2 = tl.where(o_i[:, None] >= o_i[None, :], b_ds2, 0)
    b_ds2 = b_ds2.to(b_k.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_ds2, b_k)
    b_dk += tl.dot(tl.trans(b_ds2), b_q)

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,  ))
    tl.atomic_add(p_dla, b_dla, scope="cta")


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_OPTIONS
        for num_stages in NUM_STAGES_OPTIONS
    ],
    key=["H", "K", "V", "L", "BT", "BK", "BV"],
    restore_value=["dk"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dv_local_intra(
    q,
    k,
    v,
    b,
    g,
    l,
    Aw,
    do,
    dk,
    dv,
    db,
    llut,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    # offset calculation
    q  += (i_b * T * H + i_h) * K
    k  += (i_b * T * H + i_h) * K
    v  += (i_b * T * H + i_h) * V
    b  += (i_b * T * H + i_h)
    g  += (i_b * T * H + i_h)
    l  += (i_b * T * H + i_h) * L
    Aw += (i_b * T * H + i_h) * BT
    do += (i_b * T * H + i_h) * V
    dk += (i_b * T * H + i_h) * K
    dv += (i_b * T * H + i_h) * V
    db += (i_b * T * H + i_h)

    b_A  = tl.zeros([BT, BT], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT    ], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q)

    mask = (tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :])
    b_A = tl.where(mask, b_A, 0)

    p_b    = tl.make_block_ptr(b   , (T ,  ), (H,       ), (i_t * BT,         ), (BT,   ), (0,  ))
    p_g    = tl.make_block_ptr(g   , (T ,  ), (H,       ), (i_t * BT,         ), (BT,   ), (0,  ))
    p_Aw   = tl.make_block_ptr(Aw  , (BT, T), (1, H * BT), (0       , i_t * BT), (BT, BT), (0, 1))
    p_db   = tl.make_block_ptr(db  , (T ,  ), (H,       ), (i_t * BT,         ), (BT,   ), (0,  ))
    p_llut = tl.make_block_ptr(llut, (T , T), (T,      1), (i_t * BT, i_t * BT), (BT, BT), (1, 0))

    b_b  = tl.load(p_b , boundary_check=(0,  ))
    b_g  = tl.load(p_g , boundary_check=(0,  ))
    b_Aw = tl.load(p_Aw, boundary_check=(0, 1))

    # lambdas
    vals_level_lut = tl.load(p_llut, boundary_check=(0, 1))
    indices_target = i_t * BT + tl.arange(0, BT)[:, None]
    indices_target = tl.broadcast_to(indices_target, (BT, BT))
    p_la = l + indices_target * H * L + vals_level_lut
    b_l = tl.load(p_la)

    # [BS1, BS0] @ [BS0, BT] -> [BS1, BT]
    b_A2 = tl.dot(b_Aw.to(b_A.dtype), b_A)
    b_A3 = b_A2 * safe_exp(b_g[None, :] - b_g[:, None]) * b_b[:, None] * tl.trans(b_l)

    for i_v in range(tl.cdiv(V, BV)):
        p_v  = tl.make_block_ptr(v , (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_v  = tl.load(p_v , boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        b_dv = tl.dot(b_A3.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        # [BT, BV] @ [BV, BS1] -> [BT, BS1]
        b_ds += tl.dot(b_do, tl.trans(b_v))

    # ------- UT Transform gradients -------
    b_ds = b_ds * safe_exp(b_g[:, None] - b_g[None, :]) * b_l

    # reduce([BS1, BT] x [BS1, BT]) -> [BS1]
    b_db += tl.sum(b_A2 * tl.trans(b_ds), axis=1)

    # [BT, BS1] x [None, BS1] -> [BT, BS1]
    b_dsb = b_ds * b_b[None, :]
    # [BS0, BT] @ [BT, BS1] -> [BS0, BS1]
    b_dAw = tl.dot(b_A, b_dsb.to(b_A.dtype), allow_tf32=False)

    # b_dAw = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dAw, 0)
    # [BS0, BS1] @ [BS1, BS0] -> [BS0, BS0]
    b_dbkk = tl.dot(b_dAw.to(b_Aw.dtype), b_Aw)
    # [BS1, BS0] @ [BS0, BS0] -> [BS1, BS0]
    b_dbkk = tl.dot(b_Aw, b_dbkk.to(b_Aw.dtype))
    b_dbkk = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dbkk, 0).to(k.dtype.element_ty)
    # from now on, `[BS1, BS0]` in `b_dbkk` will be remapped into `[BS0, BS1]` instead

    for i_k in range(tl.cdiv(K, BK)):
        p_k  = tl.make_block_ptr(k , (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        b_k  = tl.load(p_k , boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))

        # [BS0, BS1] @ [BS1, BK] -> [BS0, BK]
        b_dkb = tl.dot(b_dbkk, b_k, allow_tf32=False)
        # [BS0, BK] x [BS0, None] -> [BS0, BK]
        b_dk += b_dkb * b_b[:, None]

        # [BS0, BK] x [BS0, None] -> [BS0, BK]
        b_kb  = (b_k * b_b[:, None]).to(b_k.dtype)
        # [BS1, BS0] @ [BS0, BK] -> [BS1, BK]
        b_dk += tl.dot(tl.trans(b_dbkk), b_kb, allow_tf32=False)

        # reduce([BS0, BK] x [BS0, BK]) -> [BS0]
        b_db += tl.sum(b_dkb * b_k, axis=1)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


def chunkwise_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    g: torch.Tensor,
    l: torch.Tensor,
    head_first: bool,
    level_base: int,
    htype: HType,
    chunk_size: int,
    output_final_state: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    if head_first:
        B, G, T, K = k.shape
        _, H, _, V = v.shape
        _, _, _, L = l.shape
    else:
        B, T, G, K = k.shape
        _, _, H, V = v.shape
        _, _, _, L = l.shape

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)
    #print(BT)
    #print(NT)
    #print(T)
    assert H % G == 0
    if BT != chunk_size:
        raise NotImplementedError
    if K > 256:
        raise NotImplementedError("current kernel does not support head dimension larger than 256.")
    if NT * BT != T:
        raise ValueError
    if T % BT != 0:
        raise ValueError(f"T={T} and BT={BT}")
    for block_size_option in BLOCK_SIZE_OPTIONS:
        if K % block_size_option != 0:
            raise ValueError(f"K={K} and BK could be {block_size_option}")
        if V % block_size_option != 0:
            raise ValueError(f"V={V} and BV could be {block_size_option}")

    if head_first:
        shape_order_h  = (B, H, NT, K, V)
        shape_order_ht = (B, H,     K, V, L)
    else:
        shape_order_h  = (B, NT, H, K, V)
        shape_order_ht = (B,     H, K, V, L)

    # grouped
    if False:  # G == 1:
        using_groups = True
    else:
        using_groups = False
        if head_first:
            q = repeat(q, "b g t k -> b (g h) t k", g=G, h=H // G).contiguous()
            k = repeat(k, "b g t k -> b (g h) t k", g=G, h=H // G).contiguous()
        else:
            q = repeat(q, "b t g k -> b t (g h) k", g=G, h=H // G).contiguous()
            k = repeat(k, "b t g k -> b t (g h) k", g=G, h=H // G).contiguous()

    # temporary workaround
    gamma = rearrange(torch.exp(segsum(rearrange(g,
        "b  (cn ct) h -> b h  cn ct  ", ct=chunk_size))),
        "b h cn ct cs -> b cn ct cs h").to(dtype=k.dtype)
    #print('calculate llut')
    llut = make_levels_matrix(length=T, base=level_base, htype=htype, dtype=torch.int64, device=l.device, clamp_min=0, cached_length=None)
    #print(llut)
    g    = chunk_local_cumsum(g, chunk_size, offsets=None, head_first=head_first)
    o    = torch.zeros(v.shape, dtype=v.dtype, device=v.device)
    h    = torch.empty(shape_order_h , dtype=k.dtype, device=k.device)
    # zero-init as some levels might not get updated
    ht   = torch.zeros(shape_order_ht     , dtype=torch.float, device=k.device) if output_final_state else None
    _ht  = torch.zeros(shape_order_ht[:-1], dtype=torch.float, device=k.device) if output_final_state else None
    # values that include gatings
    k_new = torch.empty_like(k)
    w_new = torch.empty_like(k)
    g_new = torch.zeros_like(g)
    # we are done with accumulation, so should be fine to move it to possibly lower precision
    g_exp = torch.exp(g).to(dtype=k.dtype)

    def grid_qkw(meta): return (triton.cdiv(K, meta["BK"]), B * H, triton.cdiv(T, BT))
    def grid_h  (meta): return (triton.cdiv(V, meta["BV"]), B * H)
    def grid_o  (meta): return (triton.cdiv(V, meta["BV"]), NT, B * H)

    # obtain WY representation. u is actually the new v.
    Aw, Au = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=b,
        g_cumsum=g,
        cu_seqlens=None,
        output_dtype=torch.float32
    )
    Aw = solve_tril(
        A=Aw,
        cu_seqlens=None,
        output_dtype=k.dtype
    )
    Au = solve_tril(
        A=Au,
        cu_seqlens=None,
        output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=b,
        Aw=Aw,
        Au=Au,
        cu_seqlens=None,
    )
    assert_contiguous(preprocess_qkw[grid_qkw])(
        q=None,
        k=k,
        w=w,
        g=g,
        q_new=None,
        k_new=k_new,
        w_new=w_new,
        cu_seqlens=None,
        T=T,
        H=H,
        K=K,
        BT=BT,
    )

    # preprocess `q`
    q_new = (
        einsum(g_exp, q, "b t h, b t h d -> b t h d") -
        rearrange(einsum(
            rearrange(q    , "b (cn cs) h d -> b cn cs h d", cs=chunk_size),
            rearrange(k    , "b (cn cs) h d -> b cn cs h d", cs=chunk_size),
            gamma,
            rearrange(g_exp, "b (cn cs) h   -> b cn cs h  ", cs=chunk_size),
            rearrange(w    , "b (cn cs) h d -> b cn cs h d", cs=chunk_size),
            "b cn ct h d, b cn cs h d, b cn ct cs h, b cn cs h, b cn cs h dk -> b cn ct h dk"),
            "b cn ct h dk -> b (cn ct) h dk")
        )

    assert_contiguous(chunk_fwd_kernel_o_intra[grid_o])(
        q=q,
        k=k,
        v=v,
        b=b,
        g=g,
        l=l,
        o=o,
        Aw=Aw,
        llut=llut,
        T=T,
        H=H,
        K=K,
        V=V,
        L=L,
        BT=BT,
    )

    ell_h0 = get_num_levels(BT, level_base)
    num_chunk_levels = ceil_log(NT, level_base)
    if ell_h0 + num_chunk_levels > L:
        raise ValueError
    if htype == HType.WEAK:
        ell_init = 0
    else:
        ell_init = 1
    for ell in range(ell_init, num_chunk_levels):

        ts       = torch.arange(NT, device=k.device)
        skip_kv  = ((ts >> ell) & 1 == 1)
        # skip_q   = ~skip_kv
        period   = (1 << (ell + 1))
        skip_g   = ((ts % period) == (period - 1))
        if output_final_state:
            skip_g[-1] = False
        u_masked = u    .masked_fill(repeat(skip_kv, "cn -> 1 (cn cs) 1 1", cs=chunk_size), 0.)
        w_masked = w_new.masked_fill(repeat(skip_g , "cn -> 1 (cn cs) 1 1", cs=chunk_size), 0.)
        g_masked = g    .masked_fill(repeat(skip_g , "cn -> 1 (cn cs) 1  ", cs=chunk_size), -torch.inf)

        assert_contiguous(chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid_h])(
            k=k_new,
            v=u_masked,
            d=w_masked,
            v_new=None,
            g=g_masked,
            h=h,
            h0=None,
            ht=_ht,
            cu_seqlens=None,
            chunk_offsets=None,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
        )
        assert_contiguous(chunk_fwd_kernel_o[grid_o])(
            q=q_new,
            k=k,        # unused
            v=v,        # unused
            h=h,
            g=g_new,
            l=l,
            o=o,
            llut=llut,  # unused
            ell=ell,
            ell_h=ell_h0 + ell,
            offsets=None,
            indices=None,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            LB=level_base,
            BT=BT,
            HEAD_FIRST=head_first,
            USE_GROUPS=using_groups,
            USE_A=False,
            USE_A2=False,
            USE_H2=(htype == HType.STRONG),
        )

        if ht is not None and _ht is not None:
            ht[..., ell_h0 + ell] = _ht

    return g, o, Aw, Au, ht


@custom_autotune(
    configs=[
        triton.Config(kwargs={"block_size_K": BK, "block_size_V": BV})
        for BK in BLOCK_SIZE_OPTIONS
        for BV in BLOCK_SIZE_OPTIONS
    ],
    key=["head_first", "level_base", "chunk_size"],
)
def chunkwise_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    g: torch.Tensor,
    l: torch.Tensor,
    Aw: torch.Tensor,
    Au: torch.Tensor,
    do: torch.Tensor,
    head_first: bool,
    level_base: int,
    htype: HType,
    chunk_size: int,
    block_size_K: int,
    block_size_V: int,
    debug: bool = False,
    debug_intra: bool = False,
):
    if head_first:
        B, G, T, K = k.shape
        _, H, _, V = v.shape
        _, _, _, L = l.shape
    else:
        B, T, G, K = k.shape
        _, _, H, V = v.shape
        _, _, _, L = l.shape

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, block_size_K)
    NV = triton.cdiv(V, block_size_V)
    assert H % G == 0
    if BT != chunk_size:
        raise NotImplementedError
    if BT != 64:
        raise NotImplementedError
    if K > 256:
        raise NotImplementedError("current kernel does not support head dimension being larger than 256.")
    if NT * BT != T:
        raise ValueError
    if NK * block_size_K != K:
        raise ValueError
    if NV * block_size_V != V:
        raise ValueError
    if T % BT != 0:
        raise ValueError(f"T={T} and BT={BT}")
    for block_size_option in BLOCK_SIZE_OPTIONS:
        if K % block_size_option != 0:
            raise ValueError(f"K={K} and BK could be {block_size_option}")
        if V % block_size_option != 0:
            raise ValueError(f"V={V} and BV could be {block_size_option}")

    llut = make_levels_matrix(
        length=T,
        base=level_base,
        htype=htype,
        dtype=torch.int64,
        device=l.device,
        clamp_min=0)

    if head_first:
        shape_order_k = (B, H,  T, K)
        shape_order_v = (B, H,  T, V)
        shape_order_g = (B, H,  T)
        shape_order_l = (B, H,  T, L)
        shape_order_h = (B, H, NT, K, V)
    else:
        shape_order_k = (B,  T, H, K)
        shape_order_v = (B,  T, H, V)
        shape_order_g = (B,  T, H)
        shape_order_l = (B,  T, H, L)
        shape_order_h = (B, NT, H, K, V)

    # grouped
    if False:  # G == 1:
        using_groups = True
    else:
        using_groups = False
        if head_first:
            q = repeat(q, "b g t k -> b (g h) t k", g=G, h=H // G).contiguous()
            k = repeat(k, "b g t k -> b (g h) t k", g=G, h=H // G).contiguous()
        else:
            q = repeat(q, "b t g k -> b t (g h) k", g=G, h=H // G).contiguous()
            k = repeat(k, "b t g k -> b t (g h) k", g=G, h=H // G).contiguous()

    shape_k  = shape_order_k
    shape_v  = shape_order_v
    shape_b  = shape_order_g
    shape_g  = (NK, *shape_order_g)
    shape_l  = (NK, *shape_order_l)
    shape_h  = shape_order_h
    shape_hs = (*shape_order_h, L)

    # in backward pass, `q` and `k` do not use grouping
    dq  = torch.zeros(shape_k , dtype=q.dtype, device=q.device)
    dk  = torch.zeros(shape_k , dtype=k.dtype, device=k.device)
    dg  = torch.zeros(shape_g , dtype=torch.float, device=g.device)
    dl  = torch.zeros(shape_l , dtype=torch.float, device=l.device)
    dw  = torch.zeros(shape_k , dtype=k.dtype, device=k.device)
    du  = torch.zeros(shape_v , dtype=v.dtype, device=v.device)
    h   = torch.empty(shape_h , dtype=torch.float, device=k.device)
    dh  = torch.empty(shape_h , dtype=torch.float, device=k.device)
    hs  = torch.zeros(shape_hs, dtype=torch.float, device=k.device) if debug else None
    dhs = torch.zeros(shape_hs, dtype=torch.float, device=k.device) if debug else None

    # values that include gatings
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    w_new = torch.empty_like(k)
    qkgldo = torch.empty_like(v)
    qkgldo_kdh = torch.empty_like(v)

    # intra chunks
    dq_intra = torch.zeros(shape_k, dtype=q.dtype, device=q.device)
    dk_intra = torch.zeros(shape_k, dtype=k.dtype, device=k.device)
    dv_intra = torch.zeros(shape_v, dtype=v.dtype, device=v.device)
    db_intra = torch.zeros(shape_b, dtype=b.dtype, device=b.device)
    dg_intra = torch.zeros(shape_g, dtype=torch.float, device=g.device)
    dl_intra = torch.zeros(shape_l, dtype=torch.float, device=l.device)

    def grid_qkw(meta): return (triton.cdiv(K, meta["BK"]), B * H, triton.cdiv(T, BT))
    def grid_h  (meta): return (triton.cdiv(V, meta["BV"]), B * H)
    def grid_dh (meta): return (triton.cdiv(V, meta["BV"]), B * H)
    grid_dqkwg    = (NK, NT, B * H)
    grid_dv       = (NV, NT, B * H)
    grid_dv_local = (    NT, B * H)

    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=b,
        Aw=Aw,
        Au=Au,
        cu_seqlens=None,
    )
    assert_contiguous(preprocess_qkw[grid_qkw])(
        q=q,
        k=k,
        w=w,
        g=g,
        q_new=q_new,
        k_new=k_new,
        w_new=w_new,
        cu_seqlens=None,
        T=T,
        H=H,
        K=K,
        BT=BT,
    )

    # some hacks
    w_new_chunks = rearrange(w_new, "b (cn cs) h d -> b cn cs h d", cs=chunk_size)

    # intra chunk gradients
    # ------- dq, dk, dg, dl -------
    assert_contiguous(chunk_bwd_kernel_dqkgl_intra[grid_dqkwg])(
        q=q,
        k=k,
        v=v,
        b=b,
        g=g,
        l=l,
        Aw=Aw,
        do=do,
        dq=dq_intra,
        dk=dk_intra,
        dg=dg_intra,
        dl=dl_intra,
        llut=llut,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        L=L,
        BT=BT,
        BK=block_size_K,
        BV=block_size_V,
    )
    # ------- dk (from UT), dv, db (from UT) -------
    assert_contiguous(chunk_bwd_kernel_dv_local_intra[grid_dv_local])(
        q=q,
        k=k,
        v=v,
        b=b,
        g=g,
        l=l,
        Aw=Aw,
        do=do,
        dk=dk_intra,
        dv=dv_intra,
        db=db_intra,
        llut=llut,
        T=T,
        H=H,
        K=K,
        V=V,
        L=L,
        BT=BT,
        BK=block_size_K,
        BV=block_size_V,
    )
    if debug_intra:
        # separate reductions if grouping is used
        if head_first:
            dq_intra = reduce(dq_intra, "b (g h) t k -> b g t k", "sum", g=G, h=H // G)
            dk_intra = reduce(dk_intra, "b (g h) t k -> b g t k", "sum", g=G, h=H // G)
            # dv_intra = reduce(dv_intra, "b h t v -> b h t v", "sum")
            dg_intra = reduce(dg_intra, "nk b h t   -> b h t  ", "sum")
            dl_intra = reduce(dl_intra, "nk b h t l -> b h t l", "sum")
        else:
            dq_intra = reduce(dq_intra, "b t (g h) k -> b t g k", "sum", g=G, h=H // G)
            dk_intra = reduce(dk_intra, "b t (g h) k -> b t g k", "sum", g=G, h=H // G)
            # dv_intra = reduce(dv_intra, "b t h v -> b t h v", "sum")
            dg_intra = reduce(dg_intra, "nk b t h   -> b t h  ", "sum")
            dl_intra = reduce(dl_intra, "nk b t h l -> b t h l", "sum")
        dg_intra = chunk_local_cumsum(dg_intra, chunk_size, reverse=True, offsets=None, head_first=head_first).to(g.dtype)
        return dq_intra, dk_intra, dv_intra, db_intra, dg_intra, dl_intra

    ell_h0 = get_num_levels(BT, level_base)
    num_chunk_levels = ceil_log(NT, level_base)
    if ell_h0 + num_chunk_levels > L:
        raise ValueError
    if htype == HType.WEAK:
        ell_init = 0
    else:
        ell_init = 1
    for ell in range(ell_init, num_chunk_levels):

        ts       = torch.arange(NT, device=k.device)
        skip_kv  = ((ts >> ell) & 1 == 1)
        skip_q   = ~skip_kv
        period   = (1 << (ell + 1))
        skip_g   = ((ts % period) == (period - 1))  # forward `skip_g`
        skip_dg  = ((ts % period) == 0)             # backward `skip_g`

        # ------- h -------
        u_masked = u    .masked_fill(repeat(skip_kv, "cn -> 1 (cn cs) 1 1", cs=chunk_size), 0.)
        w_masked = w_new.masked_fill(repeat(skip_g , "cn -> 1 (cn cs) 1 1", cs=chunk_size), 0.)
        g_masked = g    .masked_fill(repeat(skip_g , "cn -> 1 (cn cs) 1  ", cs=chunk_size), -torch.inf)
        assert_contiguous(chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid_h])(
            k=k_new,
            v=u_masked,
            d=w_masked,
            v_new=None,
            g=g_masked,
            h=h,
            h0=None,
            ht=None,
            cu_seqlens=None,
            chunk_offsets=None,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
        )

        # ------- dh -------
        # we are hacking this function to compute (QK \odot G) dO
        g_dmasked   = g    .masked_fill(repeat(skip_dg, "cn -> 1 (cn cs) 1  ", cs=chunk_size), -torch.inf)
        k_dmasked   = k_new.masked_fill(repeat(skip_dg, "cn -> 1 (cn cs) 1 1", cs=chunk_size), 0.)
        do_dmasked  = do   .masked_fill(repeat(skip_q , "cn -> 1 (cn cs) 1 1", cs=chunk_size), 0.)
        ldo_dmasked = einsum(l[..., ell_h0 + ell], do_dmasked, "b t h, b t h d -> b t h d")
        assert_contiguous(chunk_bwd_kernel_dv_local[grid_dv_local])(
            q=q,
            k=k,
            g=g,
            do=ldo_dmasked,
            dv=qkgldo,
            offsets=None,
            indices=None,
            scale=1.0,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=block_size_K,
            BV=block_size_V,
            HEAD_FIRST=head_first,
        )
        assert_contiguous(chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid_dh])(
            q=q_new,
            k=k_dmasked,
            d=w_new,
            g=g_dmasked,
            dht=None,
            dh0=None,
            do=ldo_dmasked,
            dh=dh,
            dv=qkgldo,  # already `skip_q` masked from `do_masked`
            dv2=qkgldo_kdh,
            cu_seqlens=None,
            chunk_offsets=None,
            scale=1.0,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
        )

        # ------- dq, dw, dk, dg, dl -------
        mhw_new = -einsum(w_new_chunks.to(dtype=h.dtype), h, "b cn cs h k, b cn h k v -> b cn cs h v").to(dtype=v.dtype)
        mhw_new = rearrange(mhw_new, "b cn cs h d -> b (cn cs) h d")
        assert_contiguous(chunk_bwd_kernel_dqkwgl[grid_dqkwg])(
            q=q,
            k=k,
            v=mhw_new,
            h=h,
            g=g,
            l=l,
            w=w,
            u=u_masked,
            do=do_dmasked,
            dh=dh,
            dv=qkgldo_kdh,  # `skip_q` is included in `qkgdo` and `skip_g` is included in `kdh`
            dq=dq,
            dk=dk,
            dg=dg,
            dl=dl,
            dw=dw,
            ell_h=ell_h0 + ell,
            B=B,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            BT=BT,
            BK=block_size_K,
            BV=block_size_V,
        )

        # ------- du -------
        assert_contiguous(chunk_bwd_kernel_dv[grid_dv])(
            q=q,    # unused
            k=k,
            g=g,
            l=l,    # unused
            do=do,  # unused
            dv=du,
            dh=dh,
            llut=llut,
            ell=ell,
            offsets=None,
            indices=None,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            LB=level_base,
            BT=BT,
            BK=block_size_K,
            BV=block_size_V,
            HEAD_FIRST=head_first,
            USE_GROUPS=using_groups,
            USE_A=False,
            USE_A2=False,
            USE_H2=(htype == HType.STRONG),
        )

        # ------- dg -------
        # TODO: why is `dg` correct with `b_dg -= tl.sum(b_k * b_dk, axis=1)`???

        if debug:
            hs [..., ell_h0 + ell] = h .clone()
            dhs[..., ell_h0 + ell] = dh.clone()

    # inter-chunk UT transform gradients
    dk2, dv2, db2, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=b,
        g=g,
        Aw=Aw,
        Au=Au,
        dw=dw,
        du=du,
        cu_seqlens=None,
    )

    # combine the intra- and inter-chunk gradients
    dq.add_(dq_intra)
    dk.add_(dk_intra)
    dv = dv_intra
    db = db_intra
    dg.add_(dg_intra)
    dl.add_(dl_intra)

    # `dg2` does not have the leading `nk` dimensions, but
    # `dg_intra` and `dl_intra` share the same leading `nk` dimensions
    if head_first:
        dg = reduce(dg, "nk b h t   -> b h t  ", "sum")
        dl = reduce(dl, "nk b h t l -> b h t l", "sum")
    else:
        dg = reduce(dg, "nk b t h   -> b t h  ", "sum")
        dl = reduce(dl, "nk b t h l -> b t h l", "sum")

    # combine the additional inter-chunk UT transform gradients
    dk.add_(dk2)
    dv.add_(dv2)
    db.add_(db2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise TypeError("dg should be fp32")

    # separate reductions if grouping is used
    if head_first:
        dq = reduce(dq, "b (g h) t k -> b g t k", "sum", g=G, h=H // G)
        dk = reduce(dk, "b (g h) t k -> b g t k", "sum", g=G, h=H // G)
        # dv = reduce(dv, "b h t v -> b h t v", "sum")
    else:
        dq = reduce(dq, "b t (g h) k -> b t g k", "sum", g=G, h=H // G)
        dk = reduce(dk, "b t (g h) k -> b t g k", "sum", g=G, h=H // G)
        # dv = reduce(dv, "b t h v -> b t h v", "sum")

    dg = chunk_local_cumsum(dg, chunk_size, reverse=True, offsets=None, head_first=head_first).to(g.dtype)
    if debug:
        return dq, dk, dv, db, dg, dl, hs, dhs, dw, du
    return dq, dk, dv, db, dg, dl
