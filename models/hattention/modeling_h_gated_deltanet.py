# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import click
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from torch.nn import functional as F
#from transformers.generation import GenerationMixin
#from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
#from transformers.modeling_utils import PreTrainedModel
#from transformers.utils import logging
#from transformers.utils.deprecation import deprecate_kwarg


from .fused_norm import FusedRMSNormGated, RMSNorm
from .short_convolution import ShortConvolution

from .base import HType, HStruct, get_num_levels
#from hattention.mamba_apis import compute_lambda_maybe_fixed
from .gated_delta_rule_apis import chunk_h_gated_delta_rule
#from hattention.configuration_h_gated_deltanet import HGatedDeltaNetConfig


from .fla_activations import swiglu, swiglu_linear


#logger = logging.get_logger(__name__)

MAX_SEQUENCE_LENGTH = 2048 * 2
LAMBDA_LEVEL_BASE = 2
LAMBDA_LEVEL_FIXED = True
MAX_NUM_LEVELS = get_num_levels(
    length=MAX_SEQUENCE_LENGTH,
    base=LAMBDA_LEVEL_BASE)


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class LambdaLevelMLP(torch.nn.Module):
    def __init__(self, dim: int, max_num_levels: int, **kwargs) -> None:
        super().__init__()
        self.dim = dim
        self.max_num_levels = max_num_levels
        self.mlp0 = torch.nn.Linear(in_features=dim, out_features=1)
        self.mlp1 = torch.nn.Linear(in_features=dim, out_features=1)
        self.rope = RotaryPositionalEmbeddings(dim=dim, max_seq_len=max_num_levels)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, num_levels: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0.ndim != 4 or x1.ndim != 4:
            raise ValueError
        if num_levels > self.max_num_levels:
            raise ValueError
        y0s = []
        y1s = []
        for level_index in range(num_levels):
            # [batch, seqlen, nheads, dim]
            lpos0 = torch.full(
                size=(x0.shape[1],),
                fill_value=level_index,
                dtype=torch.int32,
                device=x0.device)
            lpos1 = torch.full(
                size=(x1.shape[1],),
                fill_value=level_index,
                dtype=torch.int32,
                device=x1.device)
            # [batch, seqlen, nheads, 1]
            y0 = self.mlp0(self.rope(x0, input_pos=lpos0))
            y1 = self.mlp1(self.rope(x1, input_pos=lpos1))
            y0s.append(y0)
            y1s.append(y1)

        # [batch, seqlen, nheads, num_levels]
        return torch.concat(y0s, dim=-1), torch.concat(y1s, dim=-1)


def compute_lambda(
    L: torch.Tensor,
    dl: torch.Tensor,
    lambda_mode: Optional[str],
) -> torch.Tensor:
    if lambda_mode == "positive":
        warnings.warn(click.style("[HAttention] Using positive lambda mode", fg="yellow"))
        return torch.nn.functional.softplus(L * dl)
    elif lambda_mode == "bounded":
        warnings.warn(click.style("[HAttention] Using bounded lambda mode", fg="yellow"))
        return torch.exp(-torch.exp(L) * torch.nn.functional.softplus(dl))
    elif lambda_mode is None:
        warnings.warn(click.style("[HAttention] Using default lambda mode", fg="yellow"))
        return L * dl
    else:
        raise ValueError
    
def compute_lambda_maybe_fixed(
    L: torch.Tensor,
    dl: torch.Tensor,
    lambda_mode: Optional[str],
    lambda_level_max: int,
    lambda_level_fixed: bool,
    lambda_level_module: LambdaLevelMLP,
) -> torch.Tensor:
    if lambda_level_fixed:
        if not all([
            L.ndim in [3, 4],
            dl.ndim in [3, 4],
            L.shape[-1] == lambda_level_max,
            dl.shape[-1] == lambda_level_max]):
            raise ValueError
        return compute_lambda(L=L, dl=dl, lambda_mode=lambda_mode)
    else:
        if not all([
            L.ndim in [3, 4],
            dl.ndim in [3, 4],
            L.shape[-1] == lambda_level_module.dim,
            dl.shape[-1] == lambda_level_module.dim]):
            raise ValueError
        warnings.warn(click.style("[HAttention] Using non-fixed lambda mode", fg="yellow"))
        L_new, dl_new = lambda_level_module(L, dl, num_levels=lambda_level_max)
        return compute_lambda(L=L_new, dl=dl_new, lambda_mode=lambda_mode)







class GatedMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
        fuse_swiglu: bool = True
    ) -> GatedMLP:
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.fuse_swiglu = fuse_swiglu

        if hidden_act != 'swish':
            raise ValueError(f'Unsupported hidden_act: {hidden_act}')

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if self.fuse_swiglu:
            self.swiglu_linear = SwiGLULinear()

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        gate, y = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu:
            return self.swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)
        else:
            return self.down_proj(swiglu(gate, y))


class SwiGLULinear(nn.Module):

    def forward(self, x, y, weight, bias):
        return swiglu_linear(x, y, weight, bias)



class HGatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        **kwargs
    ) -> None:
        super().__init__()
        
        # Extract num_v_heads from kwargs if provided
        num_v_heads = kwargs.pop('num_v_heads', None)
        
        # Initialize parameters from GatedDeltaNet
        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Validate expand_v configuration
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} results in non-integer value_dim: "
                f"{self.num_v_heads * self.head_dim * expand_v}"
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}"
            )
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} results in non-integer head_v_dim: {head_dim * expand_v}"
            )
        assert mode in ['chunk', 'fused_recurrent'], f"Unsupported mode `{mode}`"

        # Projection layers
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        # DeltaNet parameters
        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # Time step parameters
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Short convolution layers
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial for performance. "
                "Only disable it if you understand the implications."
            )

        # Gating and output projections
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # HGatedDeltaNet specific parameters
        self.lambdas_dim = int(self.num_heads * MAX_NUM_LEVELS)
        self.l_proj = nn.Linear(hidden_size, self.lambdas_dim, bias=False)
        self.lambda_mode = "positive"
        L = torch.ones(self.num_heads, MAX_NUM_LEVELS)
        self.L = nn.Parameter(L)
        self.L._no_weight_decay = True

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        past_key_values = None,
        use_cache = False,
        output_attentions = False,
        **kwargs
    ):
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        mode = self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        beta = self.b_proj(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)
        l = compute_lambda_maybe_fixed(
            L=rearrange(self.L, "h ell -> 1 1 h ell"),
            dl=rearrange(self.l_proj(hidden_states), "b t (h ell) -> b t h ell", ell=MAX_NUM_LEVELS),
            lambda_mode=self.lambda_mode,
            lambda_level_max=MAX_NUM_LEVELS,
            lambda_level_fixed=LAMBDA_LEVEL_FIXED,
            lambda_level_module=None)

        # dealing with padding
        if attention_mask is not None:
            # we don't really support padding
            ####should uncomment this!!!!!!
            #if not (attention_mask == 1).all():
            #    raise NotImplementedError
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])
            g = g.mul(attention_mask[:, -g.shape[-2]:, None])
            l = l.mul(attention_mask[:, -l.shape[-3]:, None, None])
            if l.dtype != q.dtype:
                warnings.warn(click.style(
                    f"`l.dtype`: {l.dtype} -> {q.dtype} "
                    f"(`self.L.dtype` = {self.L.dtype})",
                    fg="red"))
                l = l.to(dtype=q.dtype)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state = chunk_h_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                l=l,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values


class HGatedDeltaNetBlock(nn.Module):
    def __init__(self, dim: int, head_dim:int,  layer_idx: int):
        super().__init__()

        #self.config = config
        self.layer_idx = layer_idx
        
        self.fuse_norm = True


        self.attn_norm = (RMSNorm if self.fuse_norm else nn.RMSNorm)(dim, eps=1e-6)


        self.attn = HGatedDeltaNet(
            mode="chunk",
            hidden_size=dim,
            expand_v=2,
            head_dim=head_dim,
            num_heads=int(dim/head_dim),  # or int(dim/head_dim)
            use_gate=True,
            use_short_conv=True,
            conv_size=4,
            norm_eps=1e-6,
            layer_idx=layer_idx
        )
        self.mlp_norm = (RMSNorm if self.fuse_norm else nn.RMSNorm)(dim, eps=1e-6)
        self.mlp = GatedMLP(
            hidden_size=dim,
            hidden_ratio=4,
            intermediate_size=None,
            hidden_act="swish",
            fuse_swiglu=True
        )

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        past_key_values = None,
        use_cache = False,
        output_attentions = False,
        **kwargs
    ):
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        if self.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)
        #outputs = hidden_states

        return outputs



