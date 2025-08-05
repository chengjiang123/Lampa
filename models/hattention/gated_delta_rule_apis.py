# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional

import torch
import click
from einops import rearrange
from .base import HType, HStruct
from .recurrent import HState
from .kernel import (
    hattention_step,
    hattention_kernel,
    hattention_prefill)

SCALE_LAMBDA = True


@torch.compiler.disable
def chunk_h_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    l: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: Optional[HState] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False
):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
    assert cu_seqlens is None
    assert use_qk_l2norm_in_kernel is True

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta, g = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v, beta, g))
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if SCALE_LAMBDA:
        warnings.warn(click.style("[H-GDN] Scaling lambda", fg="yellow"))
        l = l / scale

    if output_final_state is False:
        if initial_state is not None:
            raise NotImplementedError
        final_state = None
        o = hattention_kernel(
            q=q,
            k=k,
            v=v,
            b=beta,
            g=g,
            l=l,
            scale=scale,
            head_first=False,
            level_base=2,
            htype=HType.WEAK,
            hstruct=HStruct.GDELTA,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    elif output_final_state is True and initial_state is None:
        o, final_state = hattention_prefill(
            q=q,
            k=k,
            v=v,
            b=beta,
            g=g,
            l=l,
            scale=scale,
            head_first=False,
            level_base=2,
            htype=HType.WEAK,
            hstruct=HStruct.GDELTA,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    elif output_final_state is True and initial_state is not None:
        if not all([
            q   .ndim == 4, q   .shape[1] == 1,
            k   .ndim == 4, k   .shape[1] == 1,
            v   .ndim == 4, v   .shape[1] == 1,
            beta.ndim == 3, beta.shape[1] == 1,
            g   .ndim == 3, g   .shape[1] == 1,
            l   .ndim == 4, l   .shape[1] == 1]):
            raise ValueError
        o, final_state = hattention_step(
            hstate=initial_state,
            q=q   .squeeze(dim=1),
            k=k   .squeeze(dim=1),
            v=v   .squeeze(dim=1),
            b=beta.squeeze(dim=1),
            g=g   .squeeze(dim=1),
            l=l   .squeeze(dim=1),
            scale=scale,
            level_base=2,
            htype=HType.WEAK,
            hstruct=HStruct.GDELTA,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        o = o.unsqueeze(dim=1)
    else:
        raise NotImplementedError

    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    return o, final_state
