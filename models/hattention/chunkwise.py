# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import functools
import torch
import triton
import triton.language as tl
from einops import repeat, reduce
from .fla_op import safe_exp
from .cumsum import chunk_global_cumsum, chunk_local_cumsum
from .base import HType, ceil_log, make_levels_matrix, get_num_levels
from .autotune import autotune as custom_autotune
from typing import Optional, Tuple, Callable, Any

NUM_WARPS_8_COMPATIBLE = False
NUM_WARPS_OPTIONS = [1, 2, 4, 8] if NUM_WARPS_8_COMPATIBLE else [1, 2, 4]
BLOCK_SIZE_OPTIONS = [32, 64]  # [32, 64, 128]
NUM_STAGES_OPTIONS = [2, 3, 4]


def assert_contiguous(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        for arg in args:
            if isinstance(arg, torch.Tensor) and not arg.is_contiguous():
                raise ValueError(f"Tensor argument {arg} must be contiguous")
        for kwname, kwarg in kwargs.items():
            if isinstance(kwarg, torch.Tensor) and not kwarg.is_contiguous():
                raise ValueError(f"Tensor argument {kwname}={kwarg} must be contiguous")
        return fn(*args, **kwargs)
    return wrapper


@triton.jit
def skip_kv(i_t, ell, LB: tl.constexpr):
    tl.static_assert(LB == 2)
    # Level Masking for K and V
    # level 0: [0 1 0 1 0 1 0 1 ...] -> period of 2
    # level 1: [0 0 1 1 0 0 1 1 ...] -> period of 4
    # level 2: [0 0 0 0 1 1 1 1 ...] -> period of 8
    # ...
    return ((i_t >> ell) & 1 == 1)


@triton.jit
def skip_q(i_t, ell, LB: tl.constexpr):
    # for off-diagonal blocks, `skip_q` and `skip_k` are opposite
    return not skip_kv(i_t=i_t, ell=ell, LB=LB)


@triton.jit
def skip_g(i_t, ell, LB: tl.constexpr, BACKWARD: tl.constexpr, OFFSET: tl.constexpr):
    tl.static_assert(LB == 2)
    period = 1 << (ell + 1)

    if OFFSET:
        i_t = i_t + (1 << ell)

    if not BACKWARD:
        # Level Masking for G
        # level 0: [0 1 0 1 0 1 0 1 ...] -> period of 2
        # level 1: [0 0 0 1 0 0 0 1 ...] -> period of 4
        # level 2: [0 0 0 0 0 0 0 1 ...] -> period of 8
        # ...
        return i_t % period == period - 1
    else:
        # Level Masking for G
        # level 0: [1 0 1 0 1 0 1 0 ...] -> period of 2
        # level 1: [1 0 0 0 1 0 0 0 ...] -> period of 4
        # level 2: [1 0 0 0 0 0 0 0 ...] -> period of 8
        # ...
        return i_t % period == 0


@triton.jit
def get_phase_index(i_t, ell, LB: tl.constexpr, OFFSET: tl.constexpr):
    tl.static_assert(LB == 2)

    # the read and write phase has an offset
    if OFFSET:
        i_t = i_t + (1 << ell)

    # Level 1: [0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 ...]
    # Level 2: [0 0 1 1 2 2 3 3 0 0 1 1 2 2 3 3 ...]
    # Level 3: [0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 ...]
    period_0 = 1 << (ell    )
    period_1 = 1 << (ell - 1)
    period_2 = 1 << (ell + 1)
    bit_0 = (i_t % period_0 >= period_1).to(tl.int32)
    bit_1 = (i_t % period_2 >= period_0).to(tl.int32)
    return (bit_1 << 1) | bit_0


@triton.heuristics({
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "USE_OFFSETS": lambda args: args["offsets"] is not None
})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BLOCK_SIZE_OPTIONS
        for BV in BLOCK_SIZE_OPTIONS
        for num_warps in NUM_WARPS_OPTIONS
        for num_stages in NUM_STAGES_OPTIONS
    ],
    key=["BT", "LB", "USE_GROUPS", "USE_H2", "MODE_H2"]
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_h(
    k,
    v,
    h,
    g,
    h0,
    ht,
    hr,
    ell,
    ell_h,
    offsets,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_GROUPS: tl.constexpr,
    USE_H2: tl.constexpr,
    MODE_H2: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # only supporting `G == 1`
    i_ng = i_nh // H if USE_GROUPS else i_nh
    i_g  = i_h  // H if USE_GROUPS else i_h
    G    = 1         if USE_GROUPS else H

    # stride calculation
    s_qk = K if HEAD_FIRST else G * K
    s_vo = V if HEAD_FIRST else H * V

    # offset calculation
    k += (i_ng * T * K) if HEAD_FIRST else ((bos * G + i_g) * K)
    v += (i_nh * T * V) if HEAD_FIRST else ((bos * H + i_h) * V)

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_H2:
        b_h1 = tl.zeros([BK, BV], dtype=tl.float32)
        b_h2 = tl.zeros([BK, BV], dtype=tl.float32)
        b_h3 = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K * V * L + ell_h, (K, V), (V * L, L), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if USE_H2:
            b_h1 = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
            b_h2 = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
            b_h3 = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):

        last_idx = min((i_t + 1) * BT, T) - 1
        if HEAD_FIRST:
            o_h = (i_nh * NT + i_t).to(tl.int64) * K * V
            b_g_last = tl.load(g + i_nh * T + last_idx)
            p_g = g + i_nh * T + i_t * BT + tl.arange(0, BT)
            p_g = tl.max_contiguous(tl.multiple_of(p_g, BT), BT)
        else:
            o_h = ((boh + i_t) * H + i_h).to(tl.int64) * K * V
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + (i_t * BT + tl.arange(0, BT)) * H + i_h

        p_k = tl.make_block_ptr(k      , (K, T), (1, s_qk), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v      , (T, V), (s_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        if not USE_H2:
            p_h = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        if USE_H2 and (MODE_H2 == 1 or MODE_H2 == 3):  # b01, b11
            # in theory, we could write all `b_h`'s, and let the following output
            # computation function to decide how to combine them. That being said,
            # I don't think leaving this logic to other functions don't really save
            # us anything as the logic will happen somewhere. Doing so also introduce
            # unnecessary IOs. Hence we will do the state combination there.
            write_phase_index = get_phase_index(i_t=i_t, ell=ell, LB=LB, OFFSET=True)
            if write_phase_index == 0:
                b_h0123 = b_h
            elif write_phase_index == 1:
                b_h0123 = b_h + b_h1
            elif write_phase_index == 2:
                b_h0123 = b_h2
            else:
                b_h0123 = b_h2 + b_h3

            p_h = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_h, b_h0123.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        if USE_H2 and (MODE_H2 == 2 or MODE_H2 == 3):  # b10, b11
            # this join order is [0, 1, 2, 3] after reshape
            b_h02 = tl.join(b_h  , b_h2 )
            b_h13 = tl.join(b_h1 , b_h3 )
            b_hr  = tl.join(b_h02, b_h13)
            b_hr  = tl.reshape(b_hr, (BK, BV, 4), can_reorder=False)

            # we have four intermediate raw states
            p_hr = tl.make_block_ptr(hr + o_h * 4, (K, V, 4), (V * 4, 4, 1), (i_k * BK, i_v * BV, 0), (BK, BV, 4), (2, 1, 0))
            tl.store(p_hr, b_hr.to(p_hr.dtype.element_ty), boundary_check=(0, 1, 2))

        # if we want to store final state, we should not zero out the final state
        if STORE_FINAL_STATE:
            not_skip_last_g = (i_t == (NT - 1))
        else:
            not_skip_last_g = False

        if not USE_H2:
            if not skip_g(i_t=i_t, ell=ell, LB=LB, BACKWARD=False, OFFSET=False) or not_skip_last_g:
                b_h *= tl.exp(b_g_last)
            else:
                b_h = tl.zeros_like(b_h)
        else:
            b_h_decay = tl.exp(b_g_last)
            if not skip_g(i_t=i_t, ell=ell, LB=LB, BACKWARD=False, OFFSET=False) or not_skip_last_g:
                b_h  *= b_h_decay
                b_h1 *= b_h_decay
            else:
                b_h   = tl.zeros_like(b_h)
                b_h1  = tl.zeros_like(b_h1)
            if not skip_g(i_t=i_t, ell=ell, LB=LB, BACKWARD=False, OFFSET=True) or not_skip_last_g:
                b_h2 *= b_h_decay
                b_h3 *= b_h_decay
            else:
                b_h2  = tl.zeros_like(b_h2)
                b_h3  = tl.zeros_like(b_h3)

        # if we use strong H matrix, we use all `kv`
        if not skip_kv(i_t=i_t, ell=ell, LB=LB) or USE_H2:
            # [BK, BT]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BT, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            # [BT]
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_v = (b_v * tl.exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

            if not USE_H2:
                b_h  += tl.dot(b_k, b_v)
            else:
                read_phase_index = get_phase_index(i_t=i_t, ell=ell, LB=LB, OFFSET=False)
                if read_phase_index == 0:
                    b_h  += tl.dot(b_k, b_v)
                elif read_phase_index == 1:
                    b_h1 += tl.dot(b_k, b_v)
                elif read_phase_index == 2:
                    b_h2 += tl.dot(b_k, b_v)
                else:
                    b_h3 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K * V * L + ell_h, (K, V), (V * L, L), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    "STORE_INITIAL_STATE_GRADIENT": lambda args: args["dh0"] is not None,
    "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
    "USE_OFFSETS": lambda args: args["offsets"] is not None
})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BLOCK_SIZE_OPTIONS
        for BV in BLOCK_SIZE_OPTIONS
        for num_warps in NUM_WARPS_OPTIONS
        for num_stages in NUM_STAGES_OPTIONS
    ],
    key=["BT", "L", "LB", "USE_GROUPS", "USE_H2", "MODE_H2"]
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dh(
    q,
    g,
    l,
    do,
    dh,
    dht,
    dh0,
    dhr,
    ell,
    ell_h,
    offsets,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_GROUPS: tl.constexpr,
    USE_H2: tl.constexpr,
    MODE_H2: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # only supporting `G == 1`
    i_ng = i_nh // H if USE_GROUPS else i_nh
    i_g  = i_h  // H if USE_GROUPS else i_h
    G    = 1         if USE_GROUPS else H

    # stride calculation
    s_qk = K if HEAD_FIRST else G * K
    s_l  = L if HEAD_FIRST else H * L
    s_do = V if HEAD_FIRST else H * V

    # offset calculation
    q  += (i_ng * T * K) if HEAD_FIRST else ((bos * G + i_g) * K)
    l  += (i_nh * T * L) if HEAD_FIRST else ((bos * H + i_h) * L)
    do += (i_nh * T * V) if HEAD_FIRST else ((bos * H + i_h) * V)

    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_H2:
        b_dh1 = tl.zeros([BK, BV], dtype=tl.float32)
        b_dh2 = tl.zeros([BK, BV], dtype=tl.float32)
        b_dh3 = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)
        if USE_H2:
            b_dh1 += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)
            b_dh2 += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)
            b_dh3 += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        last_idx = min(i_t * BT + BT, T) - 1
        if HEAD_FIRST:
            o_dh = (i_nh * NT + i_t).to(tl.int64) * K * V
            p_g = g + i_nh * T + i_t * BT + tl.arange(0, BT)
            p_g = tl.max_contiguous(tl.multiple_of(p_g, BT), BT)
            b_g_last = tl.load(g + i_nh * T + last_idx)
        else:
            o_dh = ((boh + i_t) * H + i_h).to(tl.int64) * K * V
            p_g = g + (bos + i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)

        p_q  = tl.make_block_ptr(q        , (K, T), (1, s_qk), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_lh = tl.make_block_ptr(l + ell_h, (T,  ), (s_l ,  ), (i_t * BT,         ), (BT,   ), (0,  ))
        p_do = tl.make_block_ptr(do       , (T, V), (s_do, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        if not USE_H2:
            p_dh = tl.make_block_ptr(dh + o_dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))

        if USE_H2 and (MODE_H2 == 1 or MODE_H2 == 3):  # b01, b11
            # In the forward pass, we use the `write_phase_index` to decide how to combine
            # the states. In the backward pass, we use the `read_phase_index` to decide
            # which state to select instead. Similar to the discussion earlier in the
            # forward pass, we could write all the `b_dh`'s and let following functions
            # to decide which one to select. We do it here instead for IO considerations
            # and that we still have to write this logic somewhere regardless.
            read_phase_index = get_phase_index(i_t=i_t, ell=ell, LB=LB, OFFSET=False)
            if read_phase_index == 0:
                b_dh0123 = b_dh
            elif read_phase_index == 1:
                b_dh0123 = b_dh1
            elif read_phase_index == 2:
                b_dh0123 = b_dh2
            else:
                b_dh0123 = b_dh3

            p_dh = tl.make_block_ptr(dh + o_dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_dh, b_dh0123.to(p_dh.dtype.element_ty), boundary_check=(0, 1))

        if USE_H2 and (MODE_H2 == 2 or MODE_H2 == 3):  # b10, b11
            # this join order is [0, 1, 2, 3] after reshape
            b_dh02 = tl.join(b_dh  , b_dh2 )
            b_dh13 = tl.join(b_dh1 , b_dh3 )
            b_dhr  = tl.join(b_dh02, b_dh13)
            b_dhr  = tl.reshape(b_dhr, (BK, BV, 4), can_reorder=False)

            # we have four intermediate raw states
            p_dhr = tl.make_block_ptr(dhr + o_dh * 4, (K, V, 4), (V * 4, 4, 1), (i_k * BK, i_v * BV, 0), (BK, BV, 4), (2, 1, 0))
            tl.store(p_dhr, b_dhr.to(p_dhr.dtype.element_ty), boundary_check=(0, 1, 2))

        if not USE_H2:
            if not skip_g(i_t=i_t, ell=ell, LB=LB, BACKWARD=True, OFFSET=False):
                b_dh *= tl.exp(b_g_last)
            else:
                b_dh = tl.zeros_like(b_dh)
        else:
            b_dh_decay = tl.exp(b_g_last)
            if not skip_g(i_t=i_t, ell=ell, LB=LB, BACKWARD=True, OFFSET=False):
                b_dh  *= b_dh_decay
                b_dh1 *= b_dh_decay
            else:
                b_dh   = tl.zeros_like(b_dh)
                b_dh1  = tl.zeros_like(b_dh1)
            if not skip_g(i_t=i_t, ell=ell, LB=LB, BACKWARD=True, OFFSET=True):
                b_dh2 *= b_dh_decay
                b_dh3 *= b_dh_decay
            else:
                b_dh2  = tl.zeros_like(b_dh2)
                b_dh3  = tl.zeros_like(b_dh3)

        # if we use strong H matrix, we use all `q`
        if not skip_q(i_t=i_t, ell=ell, LB=LB) or USE_H2:
            # [BK, BT]
            b_q  = tl.load(p_q , boundary_check=(0, 1))
            # [BT, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_g  = tl.load(p_g , mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_l  = tl.load(p_lh, boundary_check=(0,))
            b_q = (b_q * (tl.exp(b_g) * b_l)[None, :]).to(b_q.dtype)

            if not USE_H2:
                b_dh += tl.dot(b_q, b_do)
            else:
                b_qdo = tl.dot(b_q, b_do)
                write_phase_index = get_phase_index(i_t=i_t, ell=ell, LB=LB, OFFSET=True)
                if write_phase_index == 0:
                    b_dh  += b_qdo
                elif write_phase_index == 1:
                    b_dh  += b_qdo
                    b_dh1 += b_qdo
                elif write_phase_index == 2:
                    b_dh2 += b_qdo
                else:
                    b_dh2 += b_qdo
                    b_dh3 += b_qdo

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    "USE_OFFSETS": lambda args: args["offsets"] is not None
})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BLOCK_SIZE_OPTIONS
        for BV in BLOCK_SIZE_OPTIONS
        for num_warps in NUM_WARPS_OPTIONS
        for num_stages in NUM_STAGES_OPTIONS
    ],
    key=["BT", "LB", "USE_GROUPS", "USE_A", "USE_A2", "USE_H2"],
    restore_value=["o"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    l,
    o,
    llut,
    ell,
    ell_h,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    USE_GROUPS: tl.constexpr,
    USE_A: tl.constexpr,
    USE_A2: tl.constexpr,
    USE_H2: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    # we skip the first subdiagonal chunk
    i_t2 = i_t - 1

    if not USE_A and not USE_H2:
        # if we only compute state -> output, then we skip the entire block if to be masked
        # but if we additionally use strong H matrix, then we never it.
        if skip_q(i_t=i_t, ell=ell, LB=LB):
            return

    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # only supporting `G == 1`
    i_bg = i_bh // H if USE_GROUPS else i_bh
    i_g  = i_h  // H if USE_GROUPS else i_h
    G    = 1         if USE_GROUPS else H

    # stride calculation
    s_qk = K if HEAD_FIRST else G * K
    s_vo = V if HEAD_FIRST else H * V
    s_g  = 1 if HEAD_FIRST else H
    s_l  = L if HEAD_FIRST else H * L

    # offset calculation
    q += (i_bg * T * K) if HEAD_FIRST else ((bos * G + i_g) * K)
    k += (i_bg * T * K) if HEAD_FIRST else ((bos * G + i_g) * K)
    v += (i_bh * T * V) if HEAD_FIRST else ((bos * H + i_h) * V)
    g += (i_bh * T    ) if HEAD_FIRST else ((bos * H + i_h)    )
    l += (i_bh * T * L) if HEAD_FIRST else ((bos * H + i_h) * L)
    o += (i_bh * T * V) if HEAD_FIRST else ((bos * H + i_h) * V)
    h += ((i_bh * NT + i_t).to(tl.int64) * K * V) if HEAD_FIRST else ((i_tg * H + i_h).to(tl.int64) * K * V)

    if not USE_A:
        b_o  = tl.zeros([BT, BV], dtype=tl.float32)
    else:
        b_A  = tl.zeros([BT, BT], dtype=tl.float32)
    if USE_A and USE_A2:
        b_A2 = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q  = tl.make_block_ptr(q, (T, K), (s_qk, 1), (i_t * BT, i_k  * BK), (BT, BK), (1, 0))
        p_k  = tl.make_block_ptr(k, (K, T), (1, s_qk), (i_k * BK, i_t  * BT), (BK, BT), (0, 1))
        p_k2 = tl.make_block_ptr(k, (K, T), (1, s_qk), (i_k * BK, i_t2 * BT), (BK, BT), (0, 1))
        p_h  = tl.make_block_ptr(h, (K, V), (V, 1   ), (i_k * BK, i_v  * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))

        if not USE_A:
            # [BK, BV]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            # [BT, BK] @ [BK, BV] -> [BT, BV]
            b_o += tl.dot(b_q, b_h)
        else:
            # [BK, BT]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BT, BK] @ [BK, BT] -> [BT, BT]
            b_A += tl.dot(b_q, b_k)

        if USE_A and USE_A2:
            if i_t2 >= 0:
                # [BK, BT]
                b_k2 = tl.load(p_k2, boundary_check=(0, 1))
                # [BT, BK] @ [BK, BT] -> [BT, BT]
                b_A2 += tl.dot(b_q, b_k2)

    p_g     = tl.make_block_ptr(g        , (T,  ), (s_g ,  ), (i_t  * BT,          ), (BT,   ), (0,  ))
    p_g2    = tl.make_block_ptr(g        , (T,  ), (s_g ,  ), (i_t2 * BT,          ), (BT,   ), (0,  ))
    p_lh    = tl.make_block_ptr(l + ell_h, (T,  ), (s_l ,  ), (i_t  * BT,          ), (BT,   ), (0,  ))
    p_v     = tl.make_block_ptr(v        , (T, V), (s_vo, 1), (i_t  * BT, i_v  * BV), (BT, BV), (1, 0))
    p_v2    = tl.make_block_ptr(v        , (T, V), (s_vo, 1), (i_t2 * BT, i_v  * BV), (BT, BV), (1, 0))
    p_o     = tl.make_block_ptr(o        , (T, V), (s_vo, 1), (i_t  * BT, i_v  * BV), (BT, BV), (1, 0))
    p_llut  = tl.make_block_ptr(llut     , (T, T), (T   , 1), (i_t  * BT, i_t  * BT), (BT, BT), (1, 0))
    p_llut2 = tl.make_block_ptr(llut     , (T, T), (T   , 1), (i_t  * BT, i_t2 * BT), (BT, BT), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0,))

    if not USE_A:
        b_l = tl.load(p_lh, boundary_check=(0,))
        b_o = b_o * (tl.exp(b_g) * b_l)[:, None]
    else:
        # diagonal level
        vals_level_lut = tl.load(p_llut, boundary_check=(0, 1))
        indices_target = i_t * BT + tl.arange(0, BT)[:, None]
        indices_target = tl.broadcast_to(indices_target, (BT, BT))
        p_la = l + indices_target * s_l + vals_level_lut
        b_l = tl.load(p_la)

        b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :]) * b_l

        # diagonal block masking
        o_i = tl.arange(0, BT)
        m_A = o_i[:, None] >= o_i[None, :]
        b_A = tl.where(m_A, b_A, 0)

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o = tl.dot(b_A.to(b_v.dtype), b_v)

    if USE_A and USE_A2:
        if i_t2 >= 0:
            b_g2 = tl.load(p_g2, boundary_check=(0,))
            b_g_last2 = tl.load(g + (i_t * BT - 1) * s_g)

            # the target indices are the same (`i_t`), but source indices
            # are different (`i_t2`), hence the level lookup is different
            vals_level_lut2 = tl.load(p_llut2, boundary_check=(0, 1))
            p_la2 = l + indices_target * s_l + vals_level_lut2
            b_l2 = tl.load(p_la2)

            # since this is the subdiagonal block, we need to include
            # decay terms from the last block
            b_A2 = b_A2 * safe_exp(b_g[:, None] - b_g2[None, :] + b_g_last2) * b_l2

            b_v2 = tl.load(p_v2, boundary_check=(0, 1))
            b_o += tl.dot(b_A2.to(b_v2.dtype), b_v2)

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o_old = tl.load(p_o, boundary_check=(0, 1))
    b_o += b_o_old
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    "USE_OFFSETS": lambda args: args["offsets"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_OPTIONS
        for num_stages in NUM_STAGES_OPTIONS
    ],
    key=["BT", "BK", "BV", "L", "LB", "USE_GROUPS", "USE_A", "USE_A2", "USE_H2"],
    restore_value=["dq", "dk", "dg", "dl"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    h,
    g,
    l,
    hr,
    do,
    dh,
    dq,
    dk,
    dg,
    dl,
    dhr,
    llut,
    ell,
    ell_h,
    offsets,
    indices,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_GROUPS: tl.constexpr,
    USE_A: tl.constexpr,
    USE_A2: tl.constexpr,
    USE_H2: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    # This is different from the forward pass. In the forward pass, we compute
    # the `contribution` of outputs at `t` from inputs `t, t - 1`. In the backward
    # pass, gradients contributions of outputs at `t` still come from `t, t - 1`, but
    # the gradient contributions of inputs at `t` come from `t, t + 1`. We choose to
    # compute both instead of just one of them, as otherwise we would only have partial
    # gradients for either inputs or outputs. As this function is parallelized across
    # time steps, we need to somehow reduce the partial gradients while avoid race
    # conditions. This means either atomics or separate reductions. As such, we pay
    # extra computations to save bandwidths.
    i_t2 = i_t - 1
    i_t3 = i_t + 1

    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # only supporting `G == 1`
    i_bg = i_bh // H if USE_GROUPS else i_bh
    i_g  = i_h  // H if USE_GROUPS else i_h
    G    = 1         if USE_GROUPS else H

    # stride calculation
    s_qk  = K if HEAD_FIRST else G * K
    s_dqk = K if HEAD_FIRST else H * K
    s_vo  = V if HEAD_FIRST else H * V
    s_g   = 1 if HEAD_FIRST else H
    s_l   = L if HEAD_FIRST else H * L

    # offset calculation
    q  += i_bg * T * K if HEAD_FIRST else (bos * G + i_g) * K
    k  += i_bg * T * K if HEAD_FIRST else (bos * G + i_g) * K
    v  += i_bh * T * V if HEAD_FIRST else (bos * H + i_h) * V
    g  += i_bh * T     if HEAD_FIRST else (bos * H + i_h)
    l  += i_bh * T * L if HEAD_FIRST else (bos * H + i_h) * L
    h  += (i_bh * NT + i_t).to(tl.int64) * K * V if HEAD_FIRST else (i_tg * H + i_h).to(tl.int64) * K * V

    # in backward pass, `q` and `k` do not use grouping
    dq += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    dk += i_bh * T * K if HEAD_FIRST else (bos * H + i_h) * K
    dg += i_k  * B * H * T
    dg += i_bh * T     if HEAD_FIRST else (bos * H + i_h)
    dl += i_k  * B * H * T * L
    dl += i_bh * T * L if HEAD_FIRST else (bos * H + i_h) * L
    do += i_bh * T * V if HEAD_FIRST else (bos * H + i_h) * V
    dh += (i_bh * NT + i_t).to(tl.int64) * K * V if HEAD_FIRST else (i_tg * H + i_h).to(tl.int64) * K * V

    if USE_H2:
        hr  += (i_bh * NT + i_t).to(tl.int64) * K * V * 4 if HEAD_FIRST else (i_tg * H + i_h).to(tl.int64) * K * V * 4
        dhr += (i_bh * NT + i_t).to(tl.int64) * K * V * 4 if HEAD_FIRST else (i_tg * H + i_h).to(tl.int64) * K * V * 4

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dg = tl.zeros([BT,   ], dtype=tl.float32)
    b_dg_last = tl.zeros([1,], dtype=tl.float32)
    if USE_A:
        b_ds  = tl.zeros([BT, BT], dtype=tl.float32)
    else:
        b_dlh = tl.zeros([BT,   ], dtype=tl.float32)
    if USE_A and USE_A2:
        b_ds2 = tl.zeros([BT, BT], dtype=tl.float32)
        b_ds3 = tl.zeros([BT, BT], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v   = tl.make_block_ptr(v , (T, V), (s_vo, 1), (i_t  * BT, i_v * BV), (BT, BV), (1, 0))
        p_v2  = tl.make_block_ptr(v , (T, V), (s_vo, 1), (i_t2 * BT, i_v * BV), (BT, BV), (1, 0))
        p_h   = tl.make_block_ptr(h , (V, K), (1   , V), (i_v  * BV, i_k * BK), (BV, BK), (0, 1))
        p_do  = tl.make_block_ptr(do, (T, V), (s_vo, 1), (i_t  * BT, i_v * BV), (BT, BV), (1, 0))
        p_do3 = tl.make_block_ptr(do, (T, V), (s_vo, 1), (i_t3 * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh  = tl.make_block_ptr(dh, (V, K), (1   , V), (i_v  * BV, i_k * BK), (BV, BK), (0, 1))

        # [BT, BV]
        b_v  = tl.load(p_v , boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        if USE_A:
            # [BT, BV] @ [BV, BT] -> [BT, BT]
            b_ds += tl.dot(b_do, tl.trans(b_v))
        else:
            # [BV, BK]
            b_h  = tl.load(p_h , boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))

            if not USE_H2:
                b_dg_last += (tl.sum(b_h * b_dh))
            else:
                # [4, BV, BK]
                p_hr  = tl.make_block_ptr(hr , (4, V, K), (1, 4, V * 4), (0, i_v * BV, i_k * BK), (4, BV, BK), (0, 1, 2))
                p_dhr = tl.make_block_ptr(dhr, (4, V, K), (1, 4, V * 4), (0, i_v * BV, i_k * BK), (4, BV, BK), (0, 1, 2))
                b_hr  = tl.load(p_hr , boundary_check=(0, 1, 2))
                b_dhr = tl.load(p_dhr, boundary_check=(0, 1, 2))
                b_dg_last += (tl.sum(b_hr * b_dhr))

            if not skip_q(i_t=i_t, ell=ell, LB=LB) or USE_H2:
                # [BT, BV] @ [BV, BK] -> [BT, BK]
                b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            if not skip_kv(i_t=i_t, ell=ell, LB=LB) or USE_H2:
                # [BT, BV] @ [BV, BK] -> [BT, BK]
                b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

        if USE_A and USE_A2:
            if i_t2 >= 0:
                # [BT, BV]
                b_v2 = tl.load(p_v2, boundary_check=(0, 1))
                # [BT, BV] @ [BV, BT] -> [BT, BT]
                b_ds2 += tl.dot(b_do, tl.trans(b_v2))
            if i_t3 < NT:
                # [BT, BV]
                b_do3 = tl.load(p_do3, boundary_check=(0, 1))
                # [BT, BV] @ [BV, BT] -> [BT, BT]
                b_ds3 += tl.dot(b_do3, tl.trans(b_v))

    tl.debug_barrier()
    o_i = tl.arange(0, BT)
    p_q     = tl.make_block_ptr(q         , (T, K), (s_qk , 1), (i_t  * BT, i_k  * BK), (BT, BK), (1, 0))
    p_q3    = tl.make_block_ptr(q         , (T, K), (s_qk , 1), (i_t3 * BT, i_k  * BK), (BT, BK), (1, 0))
    p_k     = tl.make_block_ptr(k         , (T, K), (s_qk , 1), (i_t  * BT, i_k  * BK), (BT, BK), (1, 0))
    p_k2    = tl.make_block_ptr(k         , (T, K), (s_qk , 1), (i_t2 * BT, i_k  * BK), (BT, BK), (1, 0))
    p_g     = tl.make_block_ptr(g         , (T,  ), (s_g  ,  ), (i_t  * BT,          ), (BT,   ), (0,  ))
    p_g2    = tl.make_block_ptr(g         , (T,  ), (s_g  ,  ), (i_t2 * BT,          ), (BT,   ), (0,  ))
    p_g3    = tl.make_block_ptr(g         , (T,  ), (s_g  ,  ), (i_t3 * BT,          ), (BT,   ), (0,  ))
    p_lh    = tl.make_block_ptr(l + ell_h , (T,  ), (s_l  ,  ), (i_t  * BT,          ), (BT,   ), (0,  ))
    p_dq    = tl.make_block_ptr(dq        , (T, K), (s_dqk, 1), (i_t  * BT, i_k  * BK), (BT, BK), (1, 0))
    p_dk    = tl.make_block_ptr(dk        , (T, K), (s_dqk, 1), (i_t  * BT, i_k  * BK), (BT, BK), (1, 0))
    p_dg    = tl.make_block_ptr(dg        , (T,  ), (s_g  ,  ), (i_t  * BT,          ), (BT,   ), (0,  ))
    p_dlh   = tl.make_block_ptr(dl + ell_h, (T,  ), (s_l  ,  ), (i_t  * BT,          ), (BT,   ), (0,  ))
    p_llut  = tl.make_block_ptr(llut      , (T, T), (T    , 1), (i_t  * BT, i_t  * BT), (BT, BT), (1, 0))
    p_llut2 = tl.make_block_ptr(llut      , (T, T), (T    , 1), (i_t  * BT, i_t2 * BT), (BT, BT), (1, 0))
    p_llut3 = tl.make_block_ptr(llut      , (T, T), (T    , 1), (i_t3 * BT, i_t  * BT), (BT, BT), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * s_g)
    b_dg_last *= tl.exp(b_g_last)

    if USE_A:
        # diagonal level
        vals_level_lut = tl.load(p_llut, boundary_check=(0, 1))
        indices_target = i_t * BT + tl.arange(0, BT)[:, None]
        indices_target = tl.broadcast_to(indices_target, (BT, BT))
        p_la  = l  + indices_target * s_l + vals_level_lut
        p_dla = dl + indices_target * s_l + vals_level_lut
        b_l = tl.load(p_la)

        b_ds  = tl.where(o_i[:, None] >= o_i[None, :], b_ds * safe_exp(b_g[:, None] - b_g[None, :]), 0)
        b_dla = b_ds * tl.dot(b_q, tl.trans(b_k))
        b_dsg = b_dla * b_l
        b_dg += tl.sum(b_dsg, axis=1)
        b_dg -= tl.sum(b_dsg, axis=0)

        b_dsl = (b_ds * b_l).to(b_k.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_dsl, b_k)
        b_dk += tl.dot(tl.trans(b_dsl), b_q)
    else:
        if not skip_q(i_t=i_t, ell=ell, LB=LB) or USE_H2:
            b_l = tl.load(p_lh, boundary_check=(0,))
            b_dq = b_dq * tl.exp(b_g)[:, None]
            b_dlh += tl.sum(b_dq * b_q, axis=1)
            b_dq = b_dq * b_l[:, None]
            b_dg += tl.sum(b_dq * b_q, axis=1)
        if not skip_kv(i_t=i_t, ell=ell, LB=LB) or USE_H2:
            b_dk = b_dk * safe_exp(-b_g + b_g_last)[:, None]
            b_dg -= tl.sum(b_k * b_dk, axis=1)
            b_dg_last += tl.sum(b_dk * b_k)

    if USE_A and USE_A2:
        if i_t2 >= 0:
            b_g2 = tl.load(p_g2, boundary_check=(0,))
            b_k2 = tl.load(p_k2, boundary_check=(0, 1))
            b_g_last2 = tl.load(g + (i_t * BT - 1) * s_g)

            # the target indices are the same (`i_t`), but source indices
            # are different (`i_t2`), hence the level lookup is different
            vals_level_lut2 = tl.load(p_llut2, boundary_check=(0, 1))
            p_la2  = l  + indices_target * s_l + vals_level_lut2
            p_dla2 = dl + indices_target * s_l + vals_level_lut2
            b_l2 = tl.load(p_la2)

            # since this is the subdiagonal block, we need to include
            # decay terms from the last block
            b_ds2  = b_ds2 * safe_exp(b_g[:, None] - b_g2[None, :] + b_g_last2)
            b_dla2 = b_ds2 * tl.dot(b_q, tl.trans(b_k2))
            b_dsg2 = b_dla2 * b_l2
            # for each subdiagonal block, the output depends on `g[t, :], g[t - 1, :], g[t - 1, -1]`
            # hence during the backward pass, each subdiagonal `dg[t, :]` block depends on stuff coming
            # from `t` (+ sign) and `t + 1` (- sign). We also need to update `dg[t, -1]` based on stuff
            # coming from `t + 1` (+ sign) as well. The following line computes the first component.
            b_dg += tl.sum(b_dsg2, axis=1)

            b_dsl2 = (b_ds2 * b_l2).to(b_k2.dtype)
            # [BT, BK]
            b_dq += tl.dot(b_dsl2, b_k2)
        else:
            # unused, but we need to define them
            p_dla2 = p_dla
            b_dla2 = b_dla

        if i_t3 < NT:
            b_g3 = tl.load(p_g3, boundary_check=(0,))
            b_q3 = tl.load(p_q3, boundary_check=(0, 1))
            b_g_last3 = tl.load(g + (i_t3 * BT - 1) * s_g)

            # the source indices are the same (`i_t`), but target indices
            # are different (`i_t3`), hence the level lookup is different
            vals_level_lut3 = tl.load(p_llut3, boundary_check=(0, 1))
            indices_target3 = i_t3 * BT + tl.arange(0, BT)[:, None]
            indices_target3 = tl.broadcast_to(indices_target3, (BT, BT))
            p_la3  = l  + indices_target3 * s_l + vals_level_lut3
            # p_dla3 = dl + indices_target3 * s_l + vals_level_lut3
            b_l3 = tl.load(p_la3)

            # note that this computes the `b_ds` of the next block
            b_ds3  = b_ds3 * safe_exp(b_g3[:, None] - b_g[None, :] + b_g_last3)
            b_dla3 = b_ds3 * tl.dot(b_q3, tl.trans(b_k))
            b_dsg3 = b_dla3 * b_l3
            # these lines compute the 2nd and 3rd components of subdiagonal `dg` mentioned above
            b_dg -= tl.sum(b_dsg3, axis=0)
            b_dg_last3 = tl.sum(b_dsg3, axis=None)
            # not sure if the `min(...)` is necessary or needed, but should be fine when `T % BT == 0`
            b_dg = tl.where(o_i < min(BT, T - i_t * BT) - 1, b_dg, b_dg + b_dg_last3)

            b_dsl3 = (b_ds3 * b_l3).to(b_k.dtype)
            # [BT, BK]
            b_dk += tl.dot(tl.trans(b_dsl3), b_q3)

    # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
    # b_dg = tl.dot(tl.where(o_i[:, None] <= o_i[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
    b_dg = tl.where(o_i < min(BT, T - i_t * BT) - 1, b_dg, b_dg + b_dg_last)

    b_dq_old = tl.load(p_dq, boundary_check=(0, 1))
    b_dk_old = tl.load(p_dk, boundary_check=(0, 1))
    b_dg_old = tl.load(p_dg, boundary_check=(0,))
    b_dq += b_dq_old
    b_dk += b_dk_old
    b_dg += b_dg_old
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

    if USE_A:
        tl.atomic_add(p_dla, b_dla, scope="cta")
    elif not skip_q(i_t=i_t, ell=ell, LB=LB) or USE_H2:
        b_dlh_old = tl.load(p_dlh, boundary_check=(0,))
        b_dlh += b_dlh_old
        tl.store(p_dlh, b_dlh.to(p_dlh.dtype.element_ty), boundary_check=(0,))
    if USE_A and USE_A2:
        if i_t2 >= 0:
            tl.atomic_add(p_dla2, b_dla2, scope="cta")


@triton.heuristics({
    "USE_OFFSETS": lambda args: args["offsets"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_OPTIONS
    ],
    key=["BT", "BK", "BV", "L", "LB", "USE_GROUPS", "USE_A", "USE_A2", "USE_H2"],
    restore_value=["dv"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dv(
    q,
    k,
    g,
    l,
    do,
    dv,
    dh,
    llut,
    ell,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    LB: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_GROUPS: tl.constexpr,
    USE_A: tl.constexpr,
    USE_A2: tl.constexpr,
    USE_H2: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    # for consistency, we will use `i_t3` to refer to future
    # time steps and `i_t2` to refer to past time steps
    i_t3 = i_t + 1

    if not USE_A and not USE_H2:
        # if we only compute state -> output, then we skip the entire block if to be masked
        # but if we additionally use strong H matrix, then we never it.
        if skip_kv(i_t=i_t, ell=ell, LB=LB):
            return

    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # only supporting `G == 1`
    i_bg = i_bh // H if USE_GROUPS else i_bh
    i_g  = i_h  // H if USE_GROUPS else i_h
    G    = 1         if USE_GROUPS else H

    # stride calculation
    s_qk = K if HEAD_FIRST else G * K
    s_vo = V if HEAD_FIRST else H * V
    s_g  = 1 if HEAD_FIRST else H
    s_l  = L if HEAD_FIRST else H * L

    # offset calculation
    q  += i_bg * T * K if HEAD_FIRST else (bos * G + i_g) * K
    k  += i_bg * T * K if HEAD_FIRST else (bos * G + i_g) * K
    g  += i_bh * T     if HEAD_FIRST else (bos * H + i_h)
    l  += i_bh * T * L if HEAD_FIRST else (bos * H + i_h) * L
    do += i_bh * T * V if HEAD_FIRST else (bos * H + i_h) * V
    dv += i_bh * T * V if HEAD_FIRST else (bos * H + i_h) * V
    dh += (i_bh * NT + i_t).to(tl.int64) * K * V if HEAD_FIRST else (i_tg * H + i_h).to(tl.int64) * K * V

    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    if USE_A:
        b_A  = tl.zeros([BT, BT], dtype=tl.float32)
    if USE_A and USE_A2:
        b_A3 = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k  = tl.make_block_ptr(k , (T, K), (s_qk, 1), (i_t * BT, i_k  * BK), (BT, BK), (1, 0))
        p_q  = tl.make_block_ptr(q , (K, T), (1, s_qk), (i_k * BK, i_t  * BT), (BK, BT), (0, 1))
        p_q3 = tl.make_block_ptr(q , (K, T), (1, s_qk), (i_k * BK, i_t3 * BT), (BK, BT), (0, 1))
        p_dh = tl.make_block_ptr(dh, (K, V), (V,    1), (i_k * BK, i_v  * BV), (BK, BV), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))

        if USE_A:
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_A += tl.dot(b_k, b_q)
        else:
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))

        if USE_A and USE_A2:
            if i_t3 < NT:
                b_q3 = tl.load(p_q3, boundary_check=(0, 1))
                b_A3 += tl.dot(b_k, b_q3)

    p_g     = tl.make_block_ptr(g   , (T,  ), (s_g,   ), (i_t  * BT,         ), (BT,   ), (0,  ))
    p_g3    = tl.make_block_ptr(g   , (T,  ), (s_g,   ), (i_t3 * BT,         ), (BT,   ), (0,  ))
    p_do    = tl.make_block_ptr(do  , (T, V), (s_vo, 1), (i_t  * BT, i_v * BV), (BT, BV), (1, 0))
    p_do3   = tl.make_block_ptr(do  , (T, V), (s_vo, 1), (i_t3 * BT, i_v * BV), (BT, BV), (1, 0))
    p_dv    = tl.make_block_ptr(dv  , (T, V), (s_vo, 1), (i_t  * BT, i_v * BV), (BT, BV), (1, 0))
    p_llut  = tl.make_block_ptr(llut, (T, T), (T   , 1), (i_t  * BT, i_t * BT), (BT, BT), (1, 0))
    p_llut3 = tl.make_block_ptr(llut, (T, T), (T   , 1), (i_t3 * BT, i_t * BT), (BT, BT), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0,))

    if not USE_A:
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * s_g)
        b_dv *= safe_exp(-b_g + b_g_last)[:, None]
    else:
        # diagonal level
        vals_level_lut = tl.load(p_llut, boundary_check=(0, 1))
        indices_target = i_t * BT + tl.arange(0, BT)[:, None]
        indices_target = tl.broadcast_to(indices_target, (BT, BT))
        p_la = l + indices_target * s_l + vals_level_lut
        b_l = tl.load(p_la)

        mask = (tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :])
        b_A = tl.where(mask, b_A * safe_exp(b_g[None, :] - b_g[:, None]), 0).to(do.dtype.element_ty)
        b_Al = b_A * tl.trans(b_l)
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv += tl.dot(b_Al.to(b_do.dtype), b_do)

    if USE_A and USE_A2:
        if i_t3 < NT:
            b_g3 = tl.load(p_g3, boundary_check=(0,))
            b_g_last3 = tl.load(g + (i_t3 * BT - 1) * s_g)

            vals_level_lut3 = tl.load(p_llut3, boundary_check=(0, 1))
            indices_target3 = i_t3 * BT + tl.arange(0, BT)[:, None]
            indices_target3 = tl.broadcast_to(indices_target3, (BT, BT))
            p_la3 = l + indices_target3 * s_l + vals_level_lut3
            b_l3 = tl.load(p_la3)

            b_A3  = b_A3 * safe_exp(b_g3[None, :] - b_g[:, None] + b_g_last3)
            b_Al3 = b_A3 * tl.trans(b_l3)
            b_do3 = tl.load(p_do3, boundary_check=(0, 1))
            b_dv += tl.dot(b_Al3.to(b_do3.dtype), b_do3)

    b_dv_old = tl.load(p_dv, boundary_check=(0, 1))
    b_dv += b_dv_old
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def chunkwise_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    l: torch.Tensor,
    head_first: bool,
    level_base: int,
    htype: HType,
    chunk_size: int,
    output_final_state: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

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
    assert H % G == 0
    if BT != chunk_size:
        raise NotImplementedError
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
    if G == 1:
        using_groups = True
    else:
        using_groups = False
        if head_first:
            q = repeat(q, "b g t k -> b (g h) t k", g=G, h=H // G).contiguous()
            k = repeat(k, "b g t k -> b (g h) t k", g=G, h=H // G).contiguous()
        else:
            q = repeat(q, "b t g k -> b t (g h) k", g=G, h=H // G).contiguous()
            k = repeat(k, "b t g k -> b t (g h) k", g=G, h=H // G).contiguous()

    llut = make_levels_matrix(length=T, base=level_base, htype=htype, dtype=torch.int64, device=l.device, clamp_min=0)
    g    = chunk_local_cumsum(g, chunk_size, offsets=None, head_first=head_first)
    o    = torch.zeros(v.shape, dtype=v.dtype, device=v.device)
    h    = torch.empty(shape_order_h , dtype=k.dtype, device=k.device)
    # zero-init as some levels might not get updated
    ht   = torch.zeros(shape_order_ht, dtype=torch.float, device=k.device) if output_final_state else None

    def grid_o(meta): return (triton.cdiv(V, meta["BV"]), NT, B * H)
    def grid_h(meta): return (triton.cdiv(K, meta["BK"]), triton.cdiv(V, meta["BV"]), B * H)

    assert_contiguous(chunk_fwd_kernel_o[grid_o])(
        q=q,
        k=k,
        v=v,
        h=h,      # unused
        g=g,
        l=l,
        o=o,
        llut=llut,
        ell=0,    # unused
        ell_h=0,  # unused
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
        USE_A=True,
        USE_A2=(htype == HType.STRONG),
        USE_H2=False,
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
        assert_contiguous(chunk_fwd_kernel_h[grid_h])(
            k=k,
            v=v,
            h=h,
            g=g,
            h0=None,
            ht=ht,
            hr=None,
            ell=ell,
            ell_h=ell_h0 + ell,
            offsets=None,
            chunk_offsets=None,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            LB=level_base,
            BT=BT,
            HEAD_FIRST=head_first,
            USE_GROUPS=using_groups,
            USE_H2=(htype == HType.STRONG),
            MODE_H2=1,  # b01, i.e., we save just the processed states not raw states
        )

        # to get the level corresponding to `ell`-th step, notice that
        # when `ell = num_chunk_levels`, it must be the case that `level = L`
        # hence we get tje level backwards
        assert_contiguous(chunk_fwd_kernel_o[grid_o])(
            q=q,
            k=k,        # unused
            v=v,        # unused
            h=h,
            g=g,
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

    return o, g, ht


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
    g: torch.Tensor,
    l: torch.Tensor,
    do: torch.Tensor,
    head_first: bool,
    level_base: int,
    htype: HType,
    chunk_size: int,
    block_size_K: int,
    block_size_V: int,
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
    if G == 1:
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
    shape_g  = (NK, *shape_order_g)
    shape_l  = (NK, *shape_order_l)
    shape_h  = shape_order_h
    shape_hr = (*shape_order_h, 4)

    # in backward pass, `q` and `k` do not use grouping
    dq  = torch.zeros(shape_k , dtype=q.dtype, device=q.device)
    dk  = torch.zeros(shape_k , dtype=k.dtype, device=k.device)
    dv  = torch.zeros(shape_v , dtype=v.dtype, device=v.device)
    dg  = torch.zeros(shape_g , dtype=torch.float, device=g.device)
    dl  = torch.zeros(shape_l , dtype=torch.float, device=l.device)

    if htype == HType.WEAK:
        MODE_H2 = 1  # b01, i.e., we save processed states only
    else:
        MODE_H2 = 3  # b11, i.e., we save both processed and raw states
    h   = torch.empty(shape_h , dtype=torch.float, device=k.device) if MODE_H2 in [1, 3] else None
    hr  = torch.empty(shape_hr, dtype=torch.float, device=k.device) if MODE_H2 in [2, 3] else None
    dh  = torch.empty(shape_h , dtype=torch.float, device=k.device) if MODE_H2 in [1, 3] else None
    dhr = torch.empty(shape_hr, dtype=torch.float, device=k.device) if MODE_H2 in [2, 3] else None

    def grid_h (meta): return (triton.cdiv(K, meta["BK"]), triton.cdiv(V, meta["BV"]), B * H)
    def grid_dh(meta): return (triton.cdiv(K, meta["BK"]), triton.cdiv(V, meta["BV"]), B * H)
    grid_dqkwg = (NK, NT, B * H)
    grid_dv    = (NV, NT, B * H)

    assert_contiguous(chunk_bwd_kernel_dqkwg[grid_dqkwg])(
        q=q,
        k=k,
        v=v,
        h=h,      # unused
        g=g,
        l=l,
        hr=None,
        do=do,
        dh=dh,    # unused
        dq=dq,
        dk=dk,
        dg=dg,
        dl=dl,
        dhr=None,
        llut=llut,
        ell=0,    # unused
        ell_h=0,  # unused
        offsets=None,
        indices=None,
        B=B,
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
        USE_A=True,
        USE_A2=(htype == HType.STRONG),
        USE_H2=False,
    )

    assert_contiguous(chunk_bwd_kernel_dv[grid_dv])(
        q=q,
        k=k,
        g=g,
        l=l,
        do=do,
        dv=dv,
        dh=dh,    # unused
        llut=llut,
        ell=0,    # unused
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
        USE_A=True,
        USE_A2=(htype == HType.STRONG),
        USE_H2=False,
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

        assert_contiguous(chunk_fwd_kernel_h[grid_h])(
            k=k,
            v=v,
            h=h,
            g=g,
            h0=None,
            ht=None,
            hr=hr,
            ell=ell,
            ell_h=ell_h0 + ell,
            offsets=None,
            chunk_offsets=None,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            LB=level_base,
            BT=BT,
            HEAD_FIRST=head_first,
            USE_GROUPS=using_groups,
            USE_H2=(htype == HType.STRONG),
            MODE_H2=MODE_H2,
        )

        assert_contiguous(chunk_bwd_kernel_dh[grid_dh])(
            q=q,
            g=g,
            l=l,
            do=do,
            dh=dh,
            dht=None,
            dh0=None,
            dhr=dhr,
            ell=ell,
            ell_h=ell_h0 + ell,
            offsets=None,
            chunk_offsets=None,
            T=T,
            H=H,
            K=K,
            V=V,
            L=L,
            LB=level_base,
            BT=BT,
            HEAD_FIRST=head_first,
            USE_GROUPS=using_groups,
            USE_H2=(htype == HType.STRONG),
            MODE_H2=MODE_H2,
        )

        assert_contiguous(chunk_bwd_kernel_dqkwg[grid_dqkwg])(
            q=q,
            k=k,
            v=v,
            h=h,
            g=g,
            l=l,
            hr=hr,
            do=do,
            dh=dh,
            dq=dq,
            dk=dk,
            dg=dg,
            dl=dl,
            dhr=dhr,
            llut=llut,
            ell=ell,
            ell_h=ell_h0 + ell,
            offsets=None,
            indices=None,
            B=B,
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

        assert_contiguous(chunk_bwd_kernel_dv[grid_dv])(
            q=q,
            k=k,
            g=g,
            l=l,
            do=do,
            dv=dv,
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

    # separate reductions if grouping is used
    if head_first:
        dq = reduce(dq, "b (g h) t k -> b g t k", "sum", g=G, h=H // G)
        dk = reduce(dk, "b (g h) t k -> b g t k", "sum", g=G, h=H // G)
        # dv = reduce(dv, "b h t v -> b h t v", "sum")
        dg = reduce(dg, "nk b h t -> b h t", "sum")
        dl = reduce(dl, "nk b h t l -> b h t l", "sum")
    else:
        dq = reduce(dq, "b t (g h) k -> b t g k", "sum", g=G, h=H // G)
        dk = reduce(dk, "b t (g h) k -> b t g k", "sum", g=G, h=H // G)
        # dv = reduce(dv, "b t h v -> b t h v", "sum")
        dg = reduce(dg, "nk b t h -> b t h", "sum")
        dl = reduce(dl, "nk b t h l -> b t h l", "sum")

    dg = chunk_local_cumsum(dg, chunk_size, reverse=True, offsets=None, head_first=head_first).to(g.dtype)
    return dq, dk, dv, dg, dl
