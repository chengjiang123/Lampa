# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl
from einops import repeat, reduce
from .cumsum import chunk_global_cumsum, chunk_local_cumsum
from .base import make_levels_matrix
from .autotune import autotune as custom_autotune
from ._generated import level_lut_block


@triton.jit
def get_lambda_block_indices(
    ptr_base,
    ptr_level_lut,
    index_target,  # _global_ target index
    index_source,  # _global_ source index
    stride_target,
    stride_level,
    LENGTH: tl.constexpr,
    LEVELS: tl.constexpr,
    LEVEL_BASE: tl.constexpr,
    BLOCK_TARGET: tl.constexpr,
    BLOCK_SOURCE: tl.constexpr,
    off_diagonal: tl.constexpr,
):
    if off_diagonal is False:
        # load a block of levels [BT, BS]
        ptrs_level_lut = tl.make_block_ptr(
            ptr_level_lut,
            (LENGTH, LENGTH),
            (LENGTH, 1),
            (index_target, index_source),
            (BLOCK_TARGET, BLOCK_SOURCE),
            (1, 0))
        # `level_lut` is [LENGTH, LENGTH]
        vals_level_lut = tl.load(ptrs_level_lut, boundary_check=(0, 1))
        indices_target = index_target + tl.arange(0, BLOCK_TARGET)[:, None]
        indices_target = tl.broadcast_to(indices_target, (BLOCK_TARGET, BLOCK_SOURCE))
        return ptr_base + indices_target * stride_target + vals_level_lut * stride_level

    else:
        # note that off-diagonal blocks share the same level index
        # so we can use the block-level lookup table to get the level index
        val_level_lut = level_lut_block(
            t=index_target,
            s=index_source,
            T=LENGTH,
            LB=LEVEL_BASE,
            BT=BLOCK_TARGET,
            BS=BLOCK_SOURCE)

        return tl.make_block_ptr(
            ptr_base,
            (LENGTH, LEVELS),
            (stride_target, stride_level),
            (index_target, val_level_lut),
            (BLOCK_TARGET, 1),
            (1, 0))


@triton.jit
def load_lambda_block(
    i_bh,
    i_t,
    i_s,
    l,
    llut,
    s_l_h,
    s_l_t,
    T: tl.constexpr,
    L: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    off_diagonal: tl.constexpr,
):
    p_l = get_lambda_block_indices(
        ptr_base=l + i_bh * s_l_h,
        ptr_level_lut=llut,
        # global target and source index
        index_target=i_t * BT,
        index_source=i_s,
        stride_target=s_l_t,
        stride_level=1,
        LENGTH=T,
        LEVELS=L,
        LEVEL_BASE=LB,
        BLOCK_TARGET=BT,
        BLOCK_SOURCE=BS,
        off_diagonal=off_diagonal)

    if off_diagonal:
        return tl.load(p_l, boundary_check=(0, 1))
    else:
        return tl.load(p_l)


@triton.jit
def store_lambda_block(
    i_kv,
    i_bh,
    i_t,
    i_s,
    val,
    dl,
    llut,
    s_dl_n,
    s_dl_h,
    s_dl_t,
    T: tl.constexpr,
    L: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    off_diagonal: tl.constexpr,
):
    p_l = get_lambda_block_indices(
        ptr_base=dl + i_kv * s_dl_n + i_bh * s_dl_h,
        ptr_level_lut=llut,
        # global target and source index
        index_target=i_t * BT,
        index_source=i_s,
        stride_target=s_dl_t,
        stride_level=1,
        LENGTH=T,
        LEVELS=L,
        LEVEL_BASE=LB,
        BLOCK_TARGET=BT,
        BLOCK_SOURCE=BS,
        off_diagonal=off_diagonal)

    if off_diagonal:
        val_old = tl.load(p_l, boundary_check=(0, 1))
        val_new = val_old + tl.sum(val, axis=1, keep_dims=True)
        tl.store(p_l, val_new, boundary_check=(0, 1))
    else:
        tl.atomic_add(p_l, val, scope="cta")


@triton.autotune(
    configs=[
        triton.Config(kwargs={}, num_warps=4, num_stages=1),
        triton.Config(kwargs={}, num_warps=4, num_stages=2),
        triton.Config(kwargs={}, num_warps=4, num_stages=3),
        triton.Config(kwargs={}, num_warps=4, num_stages=4),
        triton.Config(kwargs={}, num_warps=8, num_stages=1),
        triton.Config(kwargs={}, num_warps=8, num_stages=2),
        triton.Config(kwargs={}, num_warps=8, num_stages=3),
        triton.Config(kwargs={}, num_warps=8, num_stages=4),
    ],
    key=["B", "H", "T", "K", "V", "L", "LB", "BT", "BS", "BK", "BV", "HEAD_FIRST", "USING_GROUPS"],
)
@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV']),
})
@triton.jit
def parallel_simple_gla_fwd_kernel(
    q,
    k,
    v,
    g,
    l,
    o,
    llut,
    s_k_g,
    s_k_t,
    s_v_h,
    s_v_t,
    s_g_h,
    s_g_t,
    s_l_h,
    s_l_t,
    s_o_n,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USING_GROUPS: tl.constexpr,
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV

    if not HEAD_FIRST:
        # Example: k
        # head-first layout: (B, G, T, K), s_k_g = T * K
        # time-first layout: (B, T, G, K), s_k_g = K
        # head-first offset = i_b * G * T * K + i_g * T * K = (i_b * G + i_g) * T * K = i_bg * s_k_g
        # time-first offset = i_b * T * G * K + i_g * K     = (i_b * T * G + i_g) * K = i_bg * s_k_g

        # Example: v
        # head-first layout: (B, H, T, V), s_v_h = T * V
        # time-first layout: (B, T, H, V), s_v_h = V
        # head-first offset = i_b * H * T * V + i_h * T * V = (i_b * H + i_h) * T * V = i_bh * s_v_h
        # time-first offset = i_b * T * H * V + i_h * V     = (i_b * T * H + i_h) * V = i_bh * s_v_h
        i_b, i_h = i_bh // H, i_bh % H
        i_bh = i_b * T * H + i_h

    # only supporting `G == 1`
    if USING_GROUPS:
        i_bg = i_bh // H
    else:
        i_bg = i_bh

    # the Q block is kept in the shared memory throughout the whole kernel
    p_q = tl.make_block_ptr(q + i_bg * s_k_g, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + i_bg * s_k_g, (K, T), (1, s_k_t), (i_k * BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_g_h, (T,), (s_g_t,), (i_s,), (BS,), (0,))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS,]
        b_g = tl.load(p_g, boundary_check=(0,))

        b_gn = tl.load(g + i_bh * s_g_h + (min(i_s + BS, T) - 1) * s_g_t)
        b_gp = tl.load(g + i_bh * s_g_h + (i_s - 1) * s_g_t) if i_s % BT > 0 else 0.

        # [BT, BS]
        b_l = load_lambda_block(
            i_bh=i_bh,
            i_t=i_t,
            i_s=i_s,
            l=l,
            llut=llut,
            s_l_h=s_l_h,
            s_l_t=s_l_t,
            T=T,
            L=L,
            LB=LB,
            BT=BT,
            BS=BS,
            off_diagonal=True)
        b_s = tl.dot(b_q, b_k) * tl.exp(b_gn - b_g)[None, :] * b_l
        # do this check to avoid some layout bugs
        # [[BT, BV]
        if i_s > 0:
            b_o = b_o * tl.exp(b_gn - b_gp)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    p_gq = tl.make_block_ptr(g + i_bh * s_g_h, (T,), (s_g_t,), (i_t * BT,), (BT,), (0,))
    # [BT,]
    b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
    # rescale interchunk output
    b_o *= tl.exp(b_gq)[:, None]

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    # [BS]
    o_k = i_t * BT + tl.arange(0, BS)
    # Q block and K block have overlap.
    # masks required
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + i_bg * s_k_g, (K, T), (1, s_k_t), (i_k * BK, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bh * s_g_h, (T,), (s_g_t,), (i_s,), (BS,), (0,))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS,]
        b_gk = tl.load(p_gk, boundary_check=(0,))
        # [BT, BS]
        b_l = load_lambda_block(
            i_bh=i_bh,
            i_t=i_t,
            i_s=i_s,
            l=l,
            llut=llut,
            s_l_h=s_l_h,
            s_l_t=s_l_t,
            T=T,
            L=L,
            LB=LB,
            BT=BT,
            BS=BS,
            off_diagonal=False)
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.where(m_s, tl.dot(b_q, b_k) * tl.exp(b_gq[:, None] - b_gk[None, :]) * b_l, 0)
        # [BT, BV]
        if i_s >= 0:
            b_o += tl.dot(b_s.to(b_q.dtype), b_v)

        o_k += BS

    p_o = tl.make_block_ptr(o + i_k * s_o_n + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def parallel_simple_gla_bwd_kernel_dq(
    i_bh,
    i_bg,
    i_t,
    i_k,
    i_v,
    i_kv,
    q,
    k,
    v,
    g,
    cg,
    l,
    llut,
    do,
    dq,
    dg,
    dl,
    s_k_g,
    s_k_t,
    s_v_h,
    s_v_t,
    s_g_h,
    s_g_t,
    s_l_h,
    s_l_t,
    s_dk_n,
    s_dk_h,
    s_dk_t,
    s_dg_n,
    s_dg_h,
    s_dg_t,
    s_dl_n,
    s_dl_h,
    s_dl_t,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT, BK]
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)

    # [BT, BK]
    p_q = tl.make_block_ptr(q + i_bg * s_k_g, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))

    # [BT,]
    p_gq = tl.make_block_ptr(g + i_bh * s_g_h, (T,), (s_g_t,), (i_t * BT,), (BT,), (0,))
    b_gq = tl.load(p_gq, boundary_check=(0,))

    # scalar
    b_cgf = tl.load(cg + i_bh * s_g_h + (i_t * BT - 1) * s_g_t, mask=i_t > 0)

    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + i_bg * s_k_g, (T, K), (s_k_t, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (1, s_v_t), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_g = tl.make_block_ptr(g + i_bh * s_g_h, (T,), (s_g_t,), (i_s,), (BS,), (0,))
        # [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS]
        b_g = tl.load(p_g, boundary_check=(0,))

        b_gn = tl.load(g + i_bh * s_g_h + (min(i_s + BS, T) - 1) * s_g_t)
        b_gp = tl.load(g + i_bh * s_g_h + (i_s - 1) * s_g_t) if i_s % BT > 0 else 0.
        # [BT, BS]
        b_l = load_lambda_block(
            i_bh=i_bh,
            i_t=i_t,
            i_s=i_s,
            l=l,
            llut=llut,
            s_l_h=s_l_h,
            s_l_t=s_l_t,
            T=T,
            L=L,
            LB=LB,
            BT=BT,
            BS=BS,
            off_diagonal=True)
        b_ds_no_l = tl.dot(b_do, b_v) * tl.exp(b_gn - b_g)[None, :]
        b_ds = b_ds_no_l * b_l
        # [BT, BK]
        if i_s > 0:
            b_dq *= tl.exp(b_gn - b_gp)
        b_dq += tl.dot(b_ds.to(b_v.dtype), b_k)

        # scalar
        b_cgn = tl.load(cg + i_bh * s_g_h + (min(i_s + BS, T) - 1) * s_g_t)
        b_dl = (
            b_ds_no_l *
            tl.dot(b_q, tl.trans(b_k)) *
            # cumulative decay from future
            tl.exp(b_cgf - b_cgn) * tl.exp(b_gq)[:, None])
        # atomic accumulation, inefficient but simple
        store_lambda_block(
            i_kv=i_kv,
            i_bh=i_bh,
            i_t=i_t,
            i_s=i_s,
            val=b_dl,
            dl=dl,
            llut=llut,
            s_dl_n=s_dl_n,
            s_dl_h=s_dl_h,
            s_dl_t=s_dl_t,
            T=T,
            L=L,
            LB=LB,
            BT=BT,
            BS=BS,
            off_diagonal=True)

    # [BT, BK]
    b_dq *= tl.exp(b_gq)[:, None]

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    # [BS]
    o_k = i_t * BT + tl.arange(0, BS)
    # Q block and K block have overlap. masks required
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + i_bg * s_k_g, (T, K), (s_k_t, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (1, s_v_t), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * s_g_h, (T,), (s_g_t,), (i_s,), (BS,), (0,))
        # [BS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS]
        b_gk = tl.load(p_gk, boundary_check=(0,))
        # [BT, BS]
        b_l = load_lambda_block(
            i_bh=i_bh,
            i_t=i_t,
            i_s=i_s,
            l=l,
            llut=llut,
            s_l_h=s_l_h,
            s_l_t=s_l_t,
            T=T,
            L=L,
            LB=LB,
            BT=BT,
            BS=BS,
            off_diagonal=False)
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds_no_l = tl.where(m_s, tl.dot(b_do, b_v) * tl.exp((b_gq[:, None] - b_gk[None, :])), 0)
        b_ds = b_ds_no_l * b_l
        # [BT, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)
        o_k += BS

        b_dl = b_ds_no_l * tl.dot(b_q, tl.trans(b_k))
        # atomic accumulation, inefficient but simple
        store_lambda_block(
            i_kv=i_kv,
            i_bh=i_bh,
            i_t=i_t,
            i_s=i_s,
            val=b_dl,
            dl=dl,
            llut=llut,
            s_dl_n=s_dl_n,
            s_dl_h=s_dl_h,
            s_dl_t=s_dl_t,
            T=T,
            L=L,
            LB=LB,
            BT=BT,
            BS=BS,
            off_diagonal=False)

    # `dq` does not use grouping to avoid using atomics
    p_dq = tl.make_block_ptr(dq + i_v * s_dk_n + i_bh * s_dk_h, (T, K), (s_dk_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_kv * s_dg_n + i_bh * s_dg_h, (T,), (s_dg_t,), (i_t * BT,), (BT,), (0,))

    b_dg = tl.sum(b_dq * b_q, 1)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.jit
def parallel_simple_gla_bwd_kernel_dkv(
    i_bh,
    i_bg,
    i_t,
    i_k,
    i_v,
    i_kv,
    q,
    k,
    v,
    g,
    l,
    llut,
    do,
    dk,
    dv,
    dg,
    s_k_g,
    s_k_t,
    s_v_h,
    s_v_t,
    s_g_h,
    s_g_t,
    s_l_h,
    s_l_t,
    s_dk_n,
    s_dk_h,
    s_dk_t,
    s_dv_n,
    s_dv_h,
    s_dv_t,
    s_dg_n,
    s_dg_h,
    s_dg_t,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # compute dk dv
    p_k = tl.make_block_ptr(k + i_bg * s_k_g, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bh * s_g_h, (T,), (s_g_t,), (i_t * BT,), (BT,), (0,))
    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    # [BT,]
    b_gk = tl.load(p_gk, boundary_check=(0,))

    NTS = tl.cdiv(T, BS)

    for i_s in range(NTS * BS - BS, (i_t + 1) * BT - BS, -BS):
        p_q = tl.make_block_ptr(q + i_bg * s_k_g, (T, K), (s_k_t, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_gq = tl.make_block_ptr(g + i_bh * s_g_h, (T,), (s_g_t,), (i_s,), (BS,), (0,))
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BS,]
        b_gq = tl.load(p_gq, boundary_check=(0,))

        b_gp = tl.load(g + i_bh * s_g_h + (min(i_s + BS, T) - 1) * s_g_t)
        b_gn = tl.load(g + i_bh * s_g_h + (i_s - 1) * s_g_t) if i_s % BT > 0 else 0.
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # overall decay rate for an entire block
        tmp0 = tl.exp(b_gp - b_gn)
        tmp1 = tl.exp(b_gq - b_gn)
        # [BS, BK]
        b_dk *= tmp0
        # [BS, BV]
        b_dv *= tmp0
        # [BT, BS]
        # notice that we swap `t` and `s` here
        b_l = load_lambda_block(
            i_bh=i_bh,
            i_t=tl.cdiv(i_s, BS),
            i_s=i_t * BT,
            l=l,
            llut=llut,
            s_l_h=s_l_h,
            s_l_t=s_l_t,
            T=T,
            L=L,
            LB=LB,
            BT=BS,
            BS=BT,
            off_diagonal=True)
        b_ds = tl.dot(b_v, tl.trans(b_do)) * tmp1[None, :] * tl.trans(b_l)
        b_s = tl.dot(b_k, tl.trans(b_q)) * tmp1[None, :] * tl.trans(b_l)
        # [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        # [BT, BV]
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do)

    if i_t >= 0:
        tmp2 = tl.exp(tl.load(g + i_bh * s_g_h + (min(i_t * BT + BT, T) - 1) * s_g_t) - b_gk)[:, None]
        # [BT, BK]
        b_dk *= tmp2
        # [BT, BV]
        b_dv *= tmp2

    o_q = i_t * BT + tl.arange(0, BS)
    o_k = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q + i_bg * s_k_g, (T, K), (s_k_t, 1), (i_s, i_k * BK), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_gq = tl.make_block_ptr(g + i_bh * s_g_h, (T,), (s_g_t,), (i_s,), (BS,), (0,))
        # [BS, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BS, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BS]
        b_gq = tl.load(p_gq, boundary_check=(0,))
        # [BT, BS]
        # notice that we swap `t` and `s` here
        b_l = load_lambda_block(
            i_bh=i_bh,
            i_t=tl.cdiv(i_s, BS),
            i_s=i_t * BT,
            l=l,
            llut=llut,
            s_l_h=s_l_h,
            s_l_t=s_l_t,
            T=T,
            L=L,
            LB=LB,
            BT=BS,
            BS=BT,
            off_diagonal=False)
        m_s = o_k[:, None] <= o_q[None, :]
        d_s = tl.where(m_s, tl.exp(-b_gk[:, None] + b_gq[None, :]), 0)

        b_ds = tl.dot(b_v, tl.trans(b_do)) * d_s * tl.trans(b_l)
        b_s = tl.dot(b_k, tl.trans(b_q)) * d_s * tl.trans(b_l)
        # [BT, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
        b_dv += tl.dot(b_s.to(b_do.dtype), b_do)
        o_q += BS

    # `dk` does not use grouping to avoid using atomics
    p_dk = tl.make_block_ptr(dk + i_v * s_dk_n + i_bh * s_dk_h, (T, K), (s_dk_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_k * s_dv_n + i_bh * s_dv_h, (T, V), (s_dv_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_kv * s_dg_n + i_bh * s_dg_h, (T,), (s_dg_t,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    b_dg = tl.load(p_dg, boundary_check=(0,))
    b_dg -= tl.sum(b_dk * b_k, 1)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config(kwargs={}, num_warps=4, num_stages=1),
        triton.Config(kwargs={}, num_warps=4, num_stages=2),
        triton.Config(kwargs={}, num_warps=4, num_stages=3),
        triton.Config(kwargs={}, num_warps=4, num_stages=4),
        triton.Config(kwargs={}, num_warps=8, num_stages=1),
        triton.Config(kwargs={}, num_warps=8, num_stages=2),
        triton.Config(kwargs={}, num_warps=8, num_stages=3),
        triton.Config(kwargs={}, num_warps=8, num_stages=4),
    ],
    key=["B", "H", "T", "K", "V", "L", "LB", "BT", "BS", "BK", "BV", "HEAD_FIRST", "USING_GROUPS"],
)
@triton.heuristics({
    'NV': lambda args: triton.cdiv(args['V'], args['BV'])
})
@triton.jit
def parallel_simple_gla_bwd_kernel(
    q,
    k,
    v,
    g,
    cg,
    l,
    llut,
    do,
    dq,
    dk,
    dv,
    dg,
    dl,
    s_k_g,
    s_k_t,
    s_v_h,
    s_v_t,
    s_g_h,
    s_g_t,
    s_l_h,
    s_l_t,
    s_dk_n,
    s_dk_h,
    s_dk_t,
    s_dv_n,
    s_dv_h,
    s_dv_t,
    s_dg_n,
    s_dg_h,
    s_dg_t,
    s_dl_n,
    s_dl_h,
    s_dl_t,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USING_GROUPS: tl.constexpr,
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k, i_v = i_kv // NV, i_kv % NV

    if not HEAD_FIRST:
        # Example: k
        # head-first layout: (B, G, T, K), s_k_g = T * K
        # time-first layout: (B, T, G, K), s_k_g = K
        # head-first offset = i_b * G * T * K + i_g * T * K = (i_b * G + i_g) * T * K = i_bg * s_k_g
        # time-first offset = i_b * T * G * K + i_g * K     = (i_b * T * G + i_g) * K = i_bg * s_k_g

        # Example: v
        # head-first layout: (B, H, T, V), s_v_h = T * V
        # time-first layout: (B, T, H, V), s_v_h = V
        # head-first offset = i_b * H * T * V + i_h * T * V = (i_b * H + i_h) * T * V = i_bh * s_v_h
        # time-first offset = i_b * T * H * V + i_h * V     = (i_b * T * H + i_h) * V = i_bh * s_v_h
        i_b, i_h = i_bh // H, i_bh % H
        i_bh = i_b * T * H + i_h

    # only supporting `G == 1`
    if USING_GROUPS:
        i_bg = i_bh // H
    else:
        i_bg = i_bh

    parallel_simple_gla_bwd_kernel_dq(
        i_bh=i_bh,
        i_bg=i_bg,
        i_t=i_t,
        i_k=i_k,
        i_v=i_v,
        i_kv=i_kv,
        q=q,
        k=k,
        v=v,
        g=g,
        cg=cg,
        l=l,
        llut=llut,
        do=do,
        dq=dq,
        dg=dg,
        dl=dl,
        s_k_g=s_k_g,
        s_k_t=s_k_t,
        s_v_h=s_v_h,
        s_v_t=s_v_t,
        s_g_h=s_g_h,
        s_g_t=s_g_t,
        s_l_h=s_l_h,
        s_l_t=s_l_t,
        s_dk_n=s_dk_n,
        s_dk_h=s_dk_h,
        s_dk_t=s_dk_t,
        s_dg_n=s_dg_n,
        s_dg_h=s_dg_h,
        s_dg_t=s_dg_t,
        s_dl_n=s_dl_n,
        s_dl_h=s_dl_h,
        s_dl_t=s_dl_t,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        L=L,
        LB=LB,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    tl.debug_barrier()
    parallel_simple_gla_bwd_kernel_dkv(
        i_bh=i_bh,
        i_bg=i_bg,
        i_t=i_t,
        i_k=i_k,
        i_v=i_v,
        i_kv=i_kv,
        q=q,
        k=k,
        v=v,
        g=g,
        l=l,
        llut=llut,
        do=do,
        dk=dk,
        dv=dv,
        dg=dg,
        s_k_g=s_k_g,
        s_k_t=s_k_t,
        s_v_h=s_v_h,
        s_v_t=s_v_t,
        s_g_h=s_g_h,
        s_g_t=s_g_t,
        s_l_h=s_l_h,
        s_l_t=s_l_t,
        s_dk_n=s_dk_n,
        s_dk_h=s_dk_h,
        s_dk_t=s_dk_t,
        s_dv_n=s_dv_n,
        s_dv_h=s_dv_h,
        s_dv_t=s_dv_t,
        s_dg_n=s_dg_n,
        s_dg_h=s_dg_h,
        s_dg_t=s_dg_t,
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        L=L,
        LB=LB,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
    )


@custom_autotune(
    configs=[
        triton.Config(kwargs={"block_size_K": 128, "block_size_V": 128}),
        triton.Config(kwargs={"block_size_K": 128, "block_size_V": 64}),
        triton.Config(kwargs={"block_size_K": 64, "block_size_V": 128}),
        triton.Config(kwargs={"block_size_K": 64, "block_size_V": 64}),
    ],
    key=["head_first", "level_base", "chunk_size_T", "chunk_size_S"],
)
def parallel_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    l: torch.Tensor,
    head_first: bool,
    level_base: int,
    block_size_K: int,
    block_size_V: int,
    chunk_size_T: int,
    chunk_size_S: int,
):
    if head_first:
        B, G, T, K = k.shape
        _, H, _, V = v.shape
        _, _, _, L = l.shape
        head_dim = 1
        time_dim = 2
        shape_order = (B, H, T)
    else:
        B, T, G, K = k.shape
        _, _, H, V = v.shape
        _, _, _, L = l.shape
        head_dim = 2
        time_dim = 1
        shape_order = (B, T, H)

    BT = chunk_size_T
    BS = chunk_size_S
    BK = block_size_K
    BV = block_size_V
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert BT % BS == 0
    assert H % G == 0

    llut = make_levels_matrix(
        length=T,
        base=level_base,
        dtype=torch.int64,
        device=l.device,
        clamp_min=0)

    # grouped
    if G == 1:
        using_groups = True
        # https://github.com/pytorch/pytorch/issues/144711
        s_k_g = k.stride(head_dim + 1) * k.size(head_dim + 1)
    else:
        using_groups = False
        s_k_g = k.stride(head_dim)
        if head_first:
            q = repeat(q, "b g t k -> b (g h) t k", g=G, h=H // G).contiguous()
            k = repeat(k, "b g t k -> b (g h) t k", g=G, h=H // G).contiguous()
        else:
            q = repeat(q, "b t g k -> b t (g h) k", g=G, h=H // G).contiguous()
            k = repeat(k, "b t g k -> b t (g h) k", g=G, h=H // G).contiguous()

    # local cumulative decay in log space
    g = chunk_local_cumsum(g, BT, head_first=head_first)

    grid = (NK * NV, triton.cdiv(T, BT), B * H)
    o = torch.empty(NK, *shape_order, V, dtype=q.dtype, device=q.device)
    parallel_simple_gla_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        l=l,
        o=o,
        llut=llut,
        s_k_g=s_k_g,
        s_k_t=k.stride(time_dim),
        s_v_h=v.stride(head_dim),
        s_v_t=v.stride(time_dim),
        s_g_h=g.stride(head_dim),
        s_g_t=g.stride(time_dim),
        s_l_h=l.stride(head_dim),
        s_l_t=l.stride(time_dim),
        s_o_n=o.stride(0),
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        L=L,
        LB=level_base,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first,
        USING_GROUPS=using_groups,
    )
    o = o.sum(0)
    return o, g


@custom_autotune(
    configs=[
        triton.Config(kwargs={"block_size_K": 128, "block_size_V": 128}),
        triton.Config(kwargs={"block_size_K": 128, "block_size_V": 64}),
        triton.Config(kwargs={"block_size_K": 64, "block_size_V": 128}),
        triton.Config(kwargs={"block_size_K": 64, "block_size_V": 64}),
    ],
    key=["head_first", "level_base", "chunk_size_T", "chunk_size_S"],
)
def parallel_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    l: torch.Tensor,
    do: torch.Tensor,
    head_first: bool,
    level_base: int,
    block_size_K: int,
    block_size_V: int,
    chunk_size_T: int,
    chunk_size_S: int,
):
    if head_first:
        B, G, T, K = k.shape
        _, H, _, V = v.shape
        _, _, _, L = l.shape
        head_dim = 1
        time_dim = 2
        shape_order = (B, H, T)
    else:
        B, T, G, K = k.shape
        _, _, H, V = v.shape
        _, _, _, L = l.shape
        head_dim = 2
        time_dim = 1
        shape_order = (B, T, H)

    BT = chunk_size_T
    BS = chunk_size_S
    BK = block_size_K
    BV = block_size_V
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    assert BT % BS == 0
    assert H % G == 0

    llut = make_levels_matrix(
        length=T,
        base=level_base,
        dtype=torch.int64,
        device=l.device,
        clamp_min=0)

    # grouped
    if G == 1:
        using_groups = True
        # https://github.com/pytorch/pytorch/issues/144711
        s_k_g = k.stride(head_dim + 1) * k.size(head_dim + 1)
    else:
        using_groups = False
        s_k_g = k.stride(head_dim)
        if head_first:
            q = repeat(q, "b g t k -> b (g h) t k", g=G, h=H // G).contiguous()
            k = repeat(k, "b g t k -> b (g h) t k", g=G, h=H // G).contiguous()
        else:
            q = repeat(q, "b t g k -> b t (g h) k", g=G, h=H // G).contiguous()
            k = repeat(k, "b t g k -> b t (g h) k", g=G, h=H // G).contiguous()

    # global cumulative decay in log space
    cg = cumsum_from_local_cumsum(g, chunk_size=BT, head_first=head_first)

    # we do not use grouping for gradients to avoid atomics
    dq = torch.empty(NV, *shape_order, K, dtype=q.dtype, device=q.device)
    dk = torch.empty(NV, *shape_order, K, dtype=q.dtype, device=q.device)
    dv = torch.empty(NK, *shape_order, V, dtype=q.dtype, device=q.device)
    dg = torch.empty(NK * NV, *shape_order, dtype=torch.float, device=q.device)
    dl = torch.zeros(NK * NV, *shape_order, L, dtype=torch.float, device=q.device)
    grid = (NK * NV, NT, B * H)
    parallel_simple_gla_bwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        cg=cg,
        l=l,
        llut=llut,
        do=do,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        dl=dl,
        s_k_g=s_k_g,
        s_k_t=k.stride(time_dim),
        s_v_h=v.stride(head_dim),
        s_v_t=v.stride(time_dim),
        s_g_h=g.stride(head_dim),
        s_g_t=g.stride(time_dim),
        s_l_h=l.stride(head_dim),
        s_l_t=l.stride(time_dim),
        s_dk_n=dk.stride(0),
        s_dk_h=dk.stride(head_dim + 1),  # `+1` because of extra leading dimension
        s_dk_t=dk.stride(time_dim + 1),  # `+1` because of extra leading dimension
        s_dv_n=dv.stride(0),
        s_dv_h=dv.stride(head_dim + 1),  # `+1` because of extra leading dimension
        s_dv_t=dv.stride(time_dim + 1),  # `+1` because of extra leading dimension
        s_dg_n=dg.stride(0),
        s_dg_h=dg.stride(head_dim + 1),  # `+1` because of extra leading dimension
        s_dg_t=dg.stride(time_dim + 1),  # `+1` because of extra leading dimension
        s_dl_n=dl.stride(0),
        s_dl_h=dl.stride(head_dim + 1),  # `+1` because of extra leading dimension
        s_dl_t=dl.stride(time_dim + 1),  # `+1` because of extra leading dimension
        B=B,
        H=H,
        T=T,
        K=K,
        V=V,
        L=L,
        LB=level_base,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first,
        USING_GROUPS=using_groups,
    )
    # separate reductions if grouping is used
    if head_first:
        dq = reduce(dq, "nv b (g h) t k -> b g t k", "sum", g=G, h=H // G)
        dk = reduce(dk, "nv b (g h) t k -> b g t k", "sum", g=G, h=H // G)
        dv = reduce(dv, "nk b h t v -> b h t v", "sum")
        dg = reduce(dg, "nknv b h t -> b h t", "sum")
        dl = reduce(dl, "nknv b h t l -> b h t l", "sum")
    else:
        dq = reduce(dq, "nv b t (g h) k -> b t g k", "sum", g=G, h=H // G)
        dk = reduce(dk, "nv b t (g h) k -> b t g k", "sum", g=G, h=H // G)
        dv = reduce(dv, "nk b t h v -> b t h v", "sum")
        dg = reduce(dg, "nknv b t h -> b t h", "sum")
        dl = reduce(dl, "nknv b t h l -> b t h l", "sum")

    dg = chunk_global_cumsum(dg, reverse=True, head_first=head_first)
    return dq, dk, dv, dg, dl


@torch.compile(fullgraph=True, dynamic=False)
def cumsum_from_local_cumsum(tensor: torch.Tensor, chunk_size: int, head_first: bool) -> torch.Tensor:
    if tensor.ndim != 3:
        raise NotImplementedError

    # exclusive prefix sum
    # https://discuss.pytorch.org/t/pytorch-equivalent-of-exclusive-cumsum/107259
    if head_first:
        # [B, H, T]
        partials = tensor[:, :, chunk_size - 1::chunk_size]
        partials_cumsum = partials.cumsum(dim=-1) - partials
        partials_cumsum = repeat(partials_cumsum, "b h n -> b h (n c)", c=chunk_size)

    else:
        # [B, T, H]
        partials = tensor[:, chunk_size - 1::chunk_size, :]
        partials_cumsum = partials.cumsum(dim=-2) - partials
        partials_cumsum = repeat(partials_cumsum, "b n h -> b (n c) h", c=chunk_size)

    return partials_cumsum + tensor
