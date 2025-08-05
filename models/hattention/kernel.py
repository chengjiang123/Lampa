# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Tuple, Optional

import torch
from einops import einsum, repeat
from .fla_utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_bwd, l2norm_fwd
from .base import HType, HStruct, get_num_levels
from .parallel import parallel_fwd, parallel_bwd
from .chunkwise import chunkwise_fwd, chunkwise_bwd
from .recurrent import HState, hattention_recurrent, step_state, step_output
from .chunkwise_hgdn import (
    chunkwise_fwd as chunkwise_fwd_hgdn,
    chunkwise_bwd as chunkwise_bwd_hgdn)


def l2norm_and_scale_fwd(x: torch.Tensor, scale: float) -> torch.Tensor:
    y = l2norm_fwd(x)
    y = y * scale
    return y


def l2norm_and_scale_bwd(x: torch.Tensor, dy: torch.Tensor, scale: float) -> torch.Tensor:
    dx = l2norm_bwd(x, dy)
    dx = dx * scale
    return dx


class HAttentionFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: Optional[torch.Tensor],
        g: torch.Tensor,
        l: torch.Tensor,
        scale: Optional[float],
        head_first: bool,
        level_base: int,
        htype: HType,
        hstruct: HStruct,
        chunk_sizes: Tuple[int, int],
        chunkwise: bool,
        use_qk_l2norm_in_kernel: bool,
    ) -> torch.Tensor:

        if hstruct == HStruct.MAMBA2 and not chunkwise:
            if b is not None:
                raise ValueError
            if htype != HType.WEAK:
                raise NotImplementedError
            if use_qk_l2norm_in_kernel:
                raise NotImplementedError
            if scale is not None:
                raise NotImplementedError
            o, g = parallel_fwd(  # type: ignore
                q=q,
                k=k,
                v=v,
                g=g,
                l=l,
                head_first=head_first,
                level_base=level_base,
                chunk_size_T=chunk_sizes[0],
                chunk_size_S=chunk_sizes[1])
            Aw = None
            Au = None

        elif hstruct == HStruct.MAMBA2 and chunkwise:
            if b is not None:
                raise ValueError
            if chunk_sizes[0] != chunk_sizes[1]:
                raise NotImplementedError
            if use_qk_l2norm_in_kernel:
                raise NotImplementedError
            if scale is not None:
                raise NotImplementedError
            o, g, _ = chunkwise_fwd(  # type: ignore
                q=q,
                k=k,
                v=v,
                g=g,
                l=l,
                head_first=head_first,
                level_base=level_base,
                htype=htype,
                chunk_size=chunk_sizes[0],
                output_final_state=False)
            Aw = None
            Au = None

        elif hstruct == HStruct.GDELTA and chunkwise:
            if b is None:
                raise ValueError
            if chunk_sizes[0] != chunk_sizes[1]:
                raise NotImplementedError
            if not use_qk_l2norm_in_kernel:
                raise NotImplementedError
            if scale is None:
                raise NotImplementedError
            g, o, Aw, Au, _ = chunkwise_fwd_hgdn(
                q=l2norm_and_scale_fwd(q, scale=scale),
                k=l2norm_fwd(k),
                v=v,
                b=b,
                g=g,
                l=l,
                head_first=head_first,
                level_base=level_base,
                htype=htype,
                chunk_size=chunk_sizes[0],
                output_final_state=False)

        else:
            raise NotImplementedError

        ctx.save_for_backward(q, k, v, b, g, l, Aw, Au)
        ctx.scale = scale
        ctx.dtype = g.dtype
        ctx.head_first = head_first
        ctx.level_base = level_base
        ctx.htype = htype
        ctx.hstruct = hstruct
        ctx.chunk_sizes = chunk_sizes
        ctx.chunkwise = chunkwise
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        do: torch.Tensor,
    ) -> Tuple[torch.Tensor,
               torch.Tensor,
               torch.Tensor,
               Optional[torch.Tensor],
               torch.Tensor,
               torch.Tensor,
               None,
               None,
               None,
               None,
               None,
               None,
               None,
               None]:
        q, k, v, b, g, l, Aw, Au = ctx.saved_tensors

        if ctx.hstruct == HStruct.MAMBA2 and not ctx.chunkwise:
            if b is not None or Aw is not None or Au is not None:
                raise ValueError
            if ctx.htype != HType.WEAK:
                raise NotImplementedError
            if ctx.use_qk_l2norm_in_kernel:
                raise NotImplementedError
            if ctx.scale is not None:
                raise NotImplementedError
            dq, dk, dv, dg, dl = parallel_bwd(  # type: ignore
                q=q,
                k=k,
                v=v,
                g=g,
                l=l,
                do=do,
                head_first=ctx.head_first,
                level_base=ctx.level_base,
                chunk_size_T=ctx.chunk_sizes[0],
                chunk_size_S=ctx.chunk_sizes[1])

        elif ctx.hstruct == HStruct.MAMBA2 and ctx.chunkwise:
            if b is not None or Aw is not None or Au is not None:
                raise ValueError
            if ctx.chunk_sizes[0] != ctx.chunk_sizes[1]:
                raise NotImplementedError
            if ctx.use_qk_l2norm_in_kernel:
                raise NotImplementedError
            if ctx.scale is not None:
                raise NotImplementedError
            dq, dk, dv, dg, dl = chunkwise_bwd(  # type: ignore
                q=q,
                k=k,
                v=v,
                g=g,
                l=l,
                do=do,
                head_first=ctx.head_first,
                level_base=ctx.level_base,
                htype=ctx.htype,
                chunk_size=ctx.chunk_sizes[0])

        elif ctx.hstruct == HStruct.GDELTA and ctx.chunkwise:
            if b is None or Aw is None or Au is None:
                raise ValueError
            if ctx.chunk_sizes[0] != ctx.chunk_sizes[1]:
                raise NotImplementedError
            if not ctx.use_qk_l2norm_in_kernel:
                raise NotImplementedError
            if ctx.scale is None:
                raise NotImplementedError
            dq, dk, dv, db, dg, dl = chunkwise_bwd_hgdn(  # type: ignore
                q=l2norm_and_scale_fwd(q, scale=ctx.scale),
                k=l2norm_fwd(k),
                v=v,
                b=b,
                g=g,
                l=l,
                Aw=Aw,
                Au=Au,
                do=do,
                head_first=ctx.head_first,
                level_base=ctx.level_base,
                htype=ctx.htype,
                chunk_size=ctx.chunk_sizes[0])
            dq = l2norm_and_scale_bwd(q, dq, scale=ctx.scale)
            dk = l2norm_bwd(k, dk)

        else:
            raise NotImplementedError

        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            db.to(b) if b is not None else None,
            dg.to(ctx.dtype),
            dl.to(l),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None)


class HAttentionPrefillFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: Optional[torch.Tensor],
        g: torch.Tensor,
        l: torch.Tensor,
        scale: Optional[float],
        head_first: bool,
        level_base: int,
        htype: HType,
        hstruct: HStruct,
        chunk_size: int,
        use_qk_l2norm_in_kernel: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if hstruct == HStruct.MAMBA2:
            if b is not None:
                raise ValueError
            if use_qk_l2norm_in_kernel:
                raise NotImplementedError
            if scale is not None:
                raise NotImplementedError
            o, _, h = chunkwise_fwd(  # type: ignore
                q=q,
                k=k,
                v=v,
                g=g,
                l=l,
                head_first=head_first,
                level_base=level_base,
                htype=htype,
                chunk_size=chunk_size,
                output_final_state=True)

        elif hstruct == HStruct.GDELTA:
            if b is None:
                raise ValueError
            if not use_qk_l2norm_in_kernel:
                raise NotImplementedError
            if scale is None:
                raise NotImplementedError
            _, o, _, _, h = chunkwise_fwd_hgdn(
                q=l2norm_and_scale_fwd(q, scale=scale),
                k=l2norm_fwd(k),
                v=v,
                b=b,
                g=g,
                l=l,
                head_first=head_first,
                level_base=level_base,
                htype=htype,
                chunk_size=chunk_size,
                output_final_state=True)

        else:
            raise NotImplementedError

        if h is None:
            raise ValueError
        return o.to(q.dtype), h

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        do: torch.Tensor,
        dh: torch.Tensor,
    ) -> Tuple[torch.Tensor,
               torch.Tensor,
               torch.Tensor,
               Optional[torch.Tensor],
               torch.Tensor,
               torch.Tensor,
               None,
               None,
               None,
               None,
               None,
               None,
               None]:
        raise NotImplementedError


def hattention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: Optional[torch.Tensor],
    g: torch.Tensor,
    l: torch.Tensor,
    scale: Optional[float],
    head_first: bool,
    level_base: int,
    htype: HType,
    hstruct: HStruct,
    chunk_sizes: Optional[Tuple[int, int]] = None,
    chunkwise: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
) -> torch.Tensor:

    if chunk_sizes is None:
        chunk_sizes = (64, 64)

    o = HAttentionFunction.apply(
        q,
        k,
        v,
        b,
        g,
        l,
        scale,
        head_first,
        level_base,
        htype,
        hstruct,
        chunk_sizes,
        chunkwise,
        use_qk_l2norm_in_kernel)

    return o


@torch.no_grad()
def postprocess_prefill_hstate_(
    hstate: HState,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: Optional[torch.Tensor],
    g: torch.Tensor,
    l: torch.Tensor,
    scale: Optional[float],
    head_first: bool,
    level_base: int,
    htype: HType,
    hstruct: HStruct,
    chunk_size: int,
    use_qk_l2norm_in_kernel: bool = False,
) -> HState:
    if head_first:
        raise NotImplementedError
    else:
        B, T, G, K = k.shape
        _, _, H, V = v.shape
        _, _, _, L = l.shape

    if use_qk_l2norm_in_kernel:
        if scale is None:
            raise NotImplementedError
        q = l2norm_and_scale_fwd(q, scale=scale)
        k = l2norm_fwd(k)
    else:
        if scale is not None:
            raise NotImplementedError

    # do not use grouping for the non-kernel part
    q = repeat(q, "b t g k -> b t (g h) k", g=G, h=H // G).contiguous()
    k = repeat(k, "b t g k -> b t (g h) k", g=G, h=H // G).contiguous()
    level_chunk = get_num_levels(chunk_size, level_base)

    if hstruct == HStruct.MAMBA2:
        # fast path for H-Mamba-2
        gr = chunk_local_cumsum(g, chunk_size, reverse=True, offsets=None, head_first=head_first).to(g.dtype)

        t = T - 1
        hstate.states[..., 0] = einsum(
            k[:, T - 1: T, ...].to(dtype=hstate.dtype),
            v[:, T - 1: T, ...].to(dtype=hstate.dtype),
            "b t h k, b t h v -> b h k v")
        for level in range(1, level_chunk):
            counts = level_base ** (level - 1)
            hstate.states[..., level] = einsum(
                k [:, (t - counts    ): (t    ), ...].to(dtype=hstate.dtype),
                v [:, (t - counts    ): (t    ), ...].to(dtype=hstate.dtype),
                gr[:, (t - counts + 1): (t + 1), ...].exp(),
                "b t h k, b t h v, b t h -> b h k v")

            # walking backwards
            t = t - counts

    else:
        # fall back to generic algorithm
        _, _hstate = hattention_recurrent(
            Q=q[:, -chunk_size:, ...],
            K=k[:, -chunk_size:, ...],
            V=v[:, -chunk_size:, ...],
            A=g[:, -chunk_size:, ...].exp(),
            B=b[:, -chunk_size:, ...] if b is not None else None,
            L=l[:, -chunk_size:, ...],
            base=level_base,
            htype=htype,
            hstruct=hstruct)
        hstate.states[..., :level_chunk] = _hstate.states[..., :level_chunk]
        hstate.counts[..., :level_chunk] = _hstate.counts[..., :level_chunk]

    # handle artifact of how we saved state
    remainder = T - 1
    hstate.counts[0] = 1
    for level in range(L - 1, 0, -1):
        counts = level_base ** (level - 1)
        if remainder >= counts:
            remainder = remainder - counts
            hstate.counts[level] = counts
        else:
            hstate.states[..., level] = torch.zeros_like(hstate.states[..., level])

    if remainder != 0:
        raise ValueError

    return hstate


@torch.no_grad()
def hattention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: Optional[torch.Tensor],
    g: torch.Tensor,
    l: torch.Tensor,
    scale: Optional[float],
    head_first: bool,
    level_base: int,
    htype: HType,
    hstruct: HStruct,
    chunk_size: Optional[int] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, HState]:

    if chunk_size is None:
        chunk_size = 64

    if head_first:
        raise NotImplementedError
    else:
        B, T, G, K = k.shape
        _, _, H, V = v.shape
        _, _, _, L = l.shape

    # the last time step that is multiple of `chunk_size`
    if htype == HType.WEAK:
        T0 = T - (T % chunk_size)
    else:
        # do not support parallel prefilling yet
        T0 = 0

    if head_first:
        raise NotImplementedError
    else:
        q0 = q[:, :T0, ...]
        k0 = k[:, :T0, ...]
        v0 = v[:, :T0, ...]
        b0 = b[:, :T0, ...] if b is not None else None
        g0 = g[:, :T0, ...]
        l0 = l[:, :T0, ...]

    state_dtype = g.dtype
    state_device = g.device
    o = torch.zeros_like(v)
    hstate = HState(
        base=level_base,
        htype=htype,
        hstruct=hstruct,
        shape=(B, H, K, V, L),
        dtype=state_dtype,
        device=state_device)

    if T0 != 0:
        # parallel prefill
        o[:, :T0, ...], hstate.states = HAttentionPrefillFunction.apply(  # type: ignore
            q0,
            k0,
            v0,
            b0,
            g0,
            l0,
            scale,
            head_first,
            level_base,
            htype,
            hstruct,
            chunk_size,
            use_qk_l2norm_in_kernel)

        # the chunkwise prefill only computes the outputs
        # and state corresponding to `levels >= level_chunk`.
        # Hence we have to recompute the states corresponding
        # to `levels < level_chunk`
        postprocess_prefill_hstate_(
            hstate=hstate,
            q=q0,
            k=k0,
            v=v0,
            b=b0,
            g=g0,
            l=l0,
            scale=scale,
            head_first=head_first,
            level_base=level_base,
            htype=htype,
            hstruct=hstruct,
            chunk_size=chunk_size,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

    # recurrently prefill the remaining time steps
    for t in range(T0, T):
        o[:, t, ...], hstate = hattention_step(
            hstate=hstate,
            q=q[:, t, ...],
            k=k[:, t, ...],
            v=v[:, t, ...],
            b=b[:, t, ...] if b is not None else None,
            g=g[:, t, ...],
            l=l[:, t, ...],
            scale=scale,
            level_base=level_base,
            htype=htype,
            hstruct=hstruct,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
    return o, hstate


@torch.no_grad()
def hattention_step(
    hstate: HState,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: Optional[torch.Tensor],
    g: torch.Tensor,
    l: torch.Tensor,
    scale: Optional[float],
    level_base: int,
    htype: HType,
    hstruct: HStruct,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, HState]:
    if not all([
        q.ndim == 3,
        k.ndim == 3,
        v.ndim == 3,
        b.ndim == 2 if b is not None else True,
        g.ndim == 2,
        l.ndim == 3,
        g.dtype == hstate.dtype,
        q.dtype == k.dtype,
        q.dtype == v.dtype,
        q.dtype == l.dtype,
        hstate.base == level_base,
        hstate.htype == htype,
        hstate.hstruct == hstruct]):
        raise TypeError

    _, G, _ = k.shape
    _, H, _ = v.shape

    if use_qk_l2norm_in_kernel:
        if scale is None:
            raise NotImplementedError
        q = l2norm_and_scale_fwd(q, scale=scale)
        k = l2norm_fwd(k)
    else:
        if scale is not None:
            raise NotImplementedError

    q = repeat(q, "b g k -> b (g h) k", g=G, h=H // G).contiguous()
    k = repeat(k, "b g k -> b (g h) k", g=G, h=H // G).contiguous()

    hstate = step_state(
        S=hstate,
        k=k.to(dtype=hstate.dtype),
        v=v.to(dtype=hstate.dtype),
        a=g.exp(),
        b=b)
    o = step_output(
        S=hstate.to(dtype=v.dtype),
        q=q,
        l=l)
    return o, hstate
