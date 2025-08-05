# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Code is adapted from flash-attn.bert_padding.py

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

#from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from .fused_norm import FusedRMSNormSwishGate, RMSNorm
from .short_convolution import ShortConvolution
from .gated_delta_rule_ori import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


from .index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
from .fla_utils import tensor_cache


from .fla_activations import swiglu, swiglu_linear

from ..model_utils.hash_utils import pad_to_multiple, get_regions, quantile_partition
from ..model_utils.window_utils import discretize_coords, FlattenedWindowMapping, get_pe_func
from torch_geometric.nn import MLP, MessagePassing




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



class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, indices):
        ctx.save_for_backward(indices)
        assert x.ndim >= 2
        ctx.first_axis_dim, other_shape = x.shape[0], x.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return x[indices]
        return torch.gather(
            rearrange(x, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, do):
        (indices,) = ctx.saved_tensors
        assert do.ndim >= 2
        other_shape = do.shape[1:]
        do = rearrange(do, "b ... -> b (...)")
        dx = torch.zeros(
            [ctx.first_axis_dim, do.shape[1]],
            device=do.device,
            dtype=do.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # dx[indices] = do
        dx.scatter_(0, repeat(indices, "z -> z d", d=do.shape[1]), do)
        return dx.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert x.ndim >= 2
        y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
        # TODO [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        y[indices] = x
        # y.scatter_(0, repeat(indices, 'z -> z d', d=x.shape[1]), x)
        return y

    @staticmethod
    def backward(ctx, do):
        (indices,) = ctx.saved_tensors
        # TODO [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        dx = do[indices]
        # dx = torch.gather(do, 0, repeat(indices, 'z -> z d', d=do.shape[1]))
        return dx, None, None


index_put_first_axis = IndexPutFirstAxis.apply


@tensor_cache
def get_unpad_data(
    attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Args:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors.
            `cu_seqlens` shape is [batch_size + 1].
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    lens = prepare_lens_from_mask(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
    return indices, cu_seqlens, max_seqlen_in_batch


def unpad_input(
    q: torch.Tensor,
    states: Tuple[torch.Tensor],
    attention_mask: torch.Tensor,
    q_len: int,
    keepdim: bool = False,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens
    even though they belong to different batches.


    Arguments:
        q (`torch.Tensor`):
            Query state with padding. Shape: [batch_size, q_len, ...].
        states (`Tuple[torch.Tensor]`):
            Attention state with padding. Shape: [batch_size, seq_len, ...].
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape [batch_size, sequence_length], 1 means valid and 0 means not valid.
        q_len (`int`):
            Target length.
        keepdim (`bool`):
            Whether to keep the batch dimension. Default: `False`.

    Return:
        q (`torch.Tensor`):
            Query state without padding.
            Shape: [1, total_target_length, ...] if `keepdim=True` else [total_target_length, ...].
        states (`Tuple[torch.Tensor]`):
            Attention state without padding.
            Shape: [1, total_source_length, ...] if `keepdim=True` else [total_source_length, ...].
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value),
            used to index into ragged (unpadded) tensors.
            `cu_seqlens` shape is [batch_size + 1].
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence
            i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(attention_mask)
    batch_size, seq_len, *_ = states[0].shape

    state = tuple(
        index_first_axis(rearrange(s, "b s ... -> (b s) ..."), indices_k)
        for s in states
    )

    if q_len == seq_len:
        q = index_first_axis(rearrange(q, "b s ... -> (b s) ..."), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif q_len == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
        indices_q = cu_seqlens_q[:-1]
        q = q.squeeze(1)
    else:
        raise NotImplementedError("We only support either q_len == k_len (prefilling) or q_len == 1 (decoding)")

    if keepdim:
        q = q.unsqueeze(0)
        state = tuple(s.unsqueeze(0) for s in state)

    return (
        q,
        state,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Args:
        hidden_states ([total_tokens, ...]):
            where total_tokens denotes the number of tokens in selected in attention_mask.
        indices ([total_tokens]):
            the indices that represent the non-masked tokens of the original padded input sequence.
        batch_size (int):
            batch_size size for the padded sequence.
        seq_len (int):
            maximum sequence length for the padded sequence.

    Return:
        hidden_states of shape [batch_size, seq_len, ...]
    """
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)



def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


def interleave_multiple_sequences(*sequences):
    """
    Interleave multiple sequences together.
    For example, with sequences [A1, A2], [B1, B2], [C1, C2],
    returns [A1, B1, C1, A2, B2, C2]
    """
    if isinstance(sequences[0], (list, tuple)):
        sequences = sequences[0]

    if len(sequences) == 1:
        return sequences[0]

    # All sequences should have the same shape
    assert all(s.shape == sequences[0].shape for s in sequences)

    # Get the original shape
    batch_size, seq_len, *rest = sequences[0].shape

    # Stack sequences along a new dimension
    stacked = torch.stack(sequences, dim=2)

    # Reshape to interleave
    reshaped = stacked.view(batch_size, seq_len * len(sequences), *rest)

    return reshaped


class LocalAggregation(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(LocalAggregation, self).__init__(aggr='mean')  # mean aggregation
        self.lin_msg = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        agg_out = self.propagate(edge_index=edge_index, x=x)
        out = self.lin_self(x) + agg_out  # residual-style fusion
        return F.gelu(out)

    def message(self, x_j):
        return self.lin_msg(x_j)



class GatedDeltaProduct(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_householder: int = 2,  # New parameter for number of householder transformations
        num_v_heads: int = None,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_forget_gate: bool = True,  # when true Gated DeltaProduct, when false DeltaProduct
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        allow_neg_eigval: bool = False,  # when true (Gated) DeltaProduct [-1, 1], when false (Gated) DeltaProduct 
        **kwargs
    ) -> GatedDeltaNet:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_householder = num_householder
        self.allow_neg_eigval = allow_neg_eigval
        self.use_forget_gate = use_forget_gate
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_qk_dim = head_dim
        self.head_v_dim = int(head_dim * self.expand_v)
        self.layer_idx = layer_idx
        self.silu = nn.SiLU()
        #assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."
        # Create multiple projection layers for each householder transformation
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)

        self.k_projs = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.key_dim, bias=False)
                for _ in range(num_householder)
            ]
        )
        self.v_projs = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.value_dim, bias=False)
                for _ in range(num_householder)
            ]
        )
        self.b_projs = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.num_heads, bias=False)
                for _ in range(num_householder)
            ]
        )
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1ds = nn.ModuleList(
                [
                    ShortConvolution(
                        hidden_size=self.key_dim,
                        kernel_size=conv_size,
                        bias=conv_bias,
                        activation="silu",
                    )
                    for _ in range(num_householder)
                ]
            )
            self.v_conv1ds = nn.ModuleList(
                [
                    ShortConvolution(
                        hidden_size=self.value_dim,
                        kernel_size=conv_size,
                        bias=conv_bias,
                        activation="silu",
                    )
                    for _ in range(num_householder)
                ]
            )

        if self.use_forget_gate:
            self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
            A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
            A_log = torch.log(A)
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True

            # Initialize dt parameters
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.k_id = torch.nn.Identity()
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        #pe = None,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        
        mode = (
            "chunk"  # 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        )
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."


        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]
            
            
        ### add pe if not None:
        #if pe is not None:
        #    if pe.dim() == 2:
        #        pe = pe.unsqueeze(0)  
        #    hidden_states = hidden_states + pe
        
        ###

        # Process each householder transformation
        ks, vs, betas = [], [], []
        conv_states = []

        for i in range(self.num_householder):
            if self.use_short_conv:
                conv_state_q, conv_state_k, conv_state_v = None, None, None
                if last_state is not None:
                    conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"][
                        i
                    ]
                conv_mask = (
                    attention_mask[:, -hidden_states.shape[1]:]
                    if attention_mask is not None
                    else None
                )
                if i == self.num_householder - 1:
                    q, conv_state_q = self.q_conv1d(
                        x=self.q_proj(hidden_states),
                        mask=conv_mask,
                        cache=conv_state_q,
                        output_final_state=use_cache,
                    )

                k, conv_state_k = self.k_conv1ds[i](
                    x=self.k_projs[i](hidden_states),
                    mask=conv_mask,
                    cache=conv_state_k,
                    output_final_state=use_cache,
                )
                v, conv_state_v = self.v_conv1ds[i](
                    x=self.v_projs[i](hidden_states),
                    mask=conv_mask,
                    cache=conv_state_v,
                    output_final_state=use_cache,
                )
                conv_states.append((conv_state_q, conv_state_k, conv_state_v))
            else:
                k = self.silu(self.k_projs[i](hidden_states))
                v = self.silu(self.v_projs[i](hidden_states))
                if i == self.num_householder - 1:
                    q = self.silu(self.q_proj(hidden_states))

            ks.append(k)
            vs.append(v)

            beta = self.b_projs[i](
                hidden_states
            ).sigmoid()  # bs, sequence_length, num_heads
            if attention_mask is not None:
                beta = beta.mul(attention_mask[:, -hidden_states.shape[1]:, None])
            if self.allow_neg_eigval:
                beta = beta * 2
            betas.append(beta)

        q = interleave_multiple_sequences(
            [torch.zeros_like(q)] * (self.num_householder - 1) + [q]
        )
        # Interleave all sequences
        k = interleave_multiple_sequences(ks)
        v = interleave_multiple_sequences(vs)
        beta = interleave_multiple_sequences(betas)

        q, k, v = (
            rearrange(x, "b t (h d) -> b t h d", h=self.num_heads) for x in (q, k, v)
        )

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        offsets = kwargs.get("offsets")



        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(hidden_states).float() + self.dt_bias
        )
        if attention_mask is not None:
            g = g.mul(attention_mask[:, -g.shape[-2]:, None])

        # Interleave g with zeros for non-first transformations
        g = interleave_multiple_sequences(
            [g] + [torch.zeros_like(g)] * (self.num_householder - 1)
        )
        
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        g = g.to(torch.float16)
        beta = beta.to(torch.float16)

        o, recurrent_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=offsets,
            use_qk_l2norm_in_kernel=True
        )

        # Take every nth element for n householder transformations
        o = o[:, self.num_householder - 1:: self.num_householder, :]

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=conv_states if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[2],
            )

        if self.use_gate:
            g = rearrange(
                self.g_proj(hidden_states),
                "... (h d) -> ... h d",
                h=self.num_heads,
            )
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = o.to(torch.float32)
        o = self.o_proj(o)

        return o, None, past_key_values
    
    
    
    
class LSHGatedDeltaProduct(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_householder: int = 2,  # New parameter for number of householder transformations
        num_v_heads: int = None,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_forget_gate: bool = True,  # when true Gated DeltaProduct, when false DeltaProduct
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        allow_neg_eigval: bool = False,  # when true (Gated) DeltaProduct [-1, 1], when false (Gated) DeltaProduct 
        **kwargs
    ) -> GatedDeltaNet:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_householder = num_householder
        self.allow_neg_eigval = allow_neg_eigval
        self.use_forget_gate = use_forget_gate
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_qk_dim = head_dim
        self.head_v_dim = int(head_dim * self.expand_v)
        self.layer_idx = layer_idx
        self.silu = nn.SiLU()
        
        self.dim_per_head = kwargs["h_dim"]
        self.block_size = kwargs["block_size"]
        self.n_hashes = kwargs["n_hashes"]
        self.num_w_per_dist = kwargs["num_w_per_dist"]
        self.e2lsh = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=hash_dim)
        #assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."
        # Create multiple projection layers for each householder transformation
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)

        self.k_projs = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.key_dim, bias=False)
                for _ in range(num_householder)
            ]
        )
        self.v_projs = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.value_dim, bias=False)
                for _ in range(num_householder)
            ]
        )
        self.b_projs = nn.ModuleList(
            [
                nn.Linear(hidden_size, self.num_heads, bias=False)
                for _ in range(num_householder)
            ]
        )
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1ds = nn.ModuleList(
                [
                    ShortConvolution(
                        hidden_size=self.key_dim,
                        kernel_size=conv_size,
                        bias=conv_bias,
                        activation="silu",
                    )
                    for _ in range(num_householder)
                ]
            )
            self.v_conv1ds = nn.ModuleList(
                [
                    ShortConvolution(
                        hidden_size=self.value_dim,
                        kernel_size=conv_size,
                        bias=conv_bias,
                        activation="silu",
                    )
                    for _ in range(num_householder)
                ]
            )

        if self.use_forget_gate:
            self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
            A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
            A_log = torch.log(A)
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True

            # Initialize dt parameters
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.k_id = torch.nn.Identity()
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        
        mode = (
            "chunk"  # 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        )
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."


        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # Process each householder transformation
        ks, vs, betas = [], [], []
        conv_states = []

        for i in range(self.num_householder):
            if self.use_short_conv:
                conv_state_q, conv_state_k, conv_state_v = None, None, None
                if last_state is not None:
                    conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"][
                        i
                    ]
                conv_mask = (
                    attention_mask[:, -hidden_states.shape[1]:]
                    if attention_mask is not None
                    else None
                )
                if i == self.num_householder - 1:
                    q, conv_state_q = self.q_conv1d(
                        x=self.q_proj(hidden_states),
                        mask=conv_mask,
                        cache=conv_state_q,
                        output_final_state=use_cache,
                    )

                k, conv_state_k = self.k_conv1ds[i](
                    x=self.k_projs[i](hidden_states),
                    mask=conv_mask,
                    cache=conv_state_k,
                    output_final_state=use_cache,
                )
                v, conv_state_v = self.v_conv1ds[i](
                    x=self.v_projs[i](hidden_states),
                    mask=conv_mask,
                    cache=conv_state_v,
                    output_final_state=use_cache,
                )
                conv_states.append((conv_state_q, conv_state_k, conv_state_v))
            else:
                k = self.silu(self.k_projs[i](hidden_states))
                v = self.silu(self.v_projs[i](hidden_states))
                if i == self.num_householder - 1:
                    q = self.silu(self.q_proj(hidden_states))

            ks.append(k)
            vs.append(v)

            beta = self.b_projs[i](
                hidden_states
            ).sigmoid()  # bs, sequence_length, num_heads
            if attention_mask is not None:
                beta = beta.mul(attention_mask[:, -hidden_states.shape[1]:, None])
            if self.allow_neg_eigval:
                beta = beta * 2
            betas.append(beta)

        q = interleave_multiple_sequences(
            [torch.zeros_like(q)] * (self.num_householder - 1) + [q]
        )
        # Interleave all sequences
        k = interleave_multiple_sequences(ks)
        v = interleave_multiple_sequences(vs)
        beta = interleave_multiple_sequences(betas)

        q, k, v = (
            rearrange(x, "b t (h d) -> b t h d", h=self.num_heads) for x in (q, k, v)
        )


        # ---- BEGIN LSH Integration ----

        B, T, H, D = q.shape
        query = q.reshape(-1, H, D)   # shape: (B*T, H, D)
        key   = k.reshape(-1, H, D)
        value = v.reshape(-1, H, D)

        # Compute relative-pos embeddings
        w = rearrange(
            kwargs["w_rpe"].weight,
            "(h d) (r k) -> h d r k",
            h=self.num_heads,
            d=self.dim_per_head,
            k=self.num_w_per_dist,
        )
        q_hat, k_hat = prep_qk(query, key, w, kwargs["coords"])

        q_hat = rearrange(q_hat, "n h d -> (n // T) * h, T, d")
        k_hat = rearrange(k_hat, "n h d -> (n // T) * h, T, d")
        value = rearrange(value, "n h d -> (n // T) * h, T, d")
        
        q_hat = q_hat.clone()
        k_hat = k_hat.clone()
        value = value.clone()
        q_hat[..., kwargs["raw_size"]:] = 0.0
        k_hat[..., kwargs["raw_size"]:] = 0.0
        value[..., kwargs["raw_size"]:] = 0.0

        q_hashed, k_hashed, hash_shift = lsh_mapping(self.e2lsh, q_hat, k_hat)
        
        hash_shift = rearrange(hash_shift, "c h d -> (c h) d")

        q_hashed[..., kwargs["raw_size"]:] = float("inf")
        k_hashed[..., kwargs["raw_size"]:] = float("inf")

        # compute and apply the geo shifts for each region/hash‐round
        q_shifts, k_shifts = get_geo_shift(
            kwargs["regions_h"],
            hash_shift,
            kwargs["region_indices"],
            self.n_hashes
        )
        
        q_hashed = q_hashed + q_shifts
        k_hashed = k_hashed + k_shifts
        
        q_positions = q_hashed.argsort(dim=-1)
        k_positions = k_hashed.argsort(dim=-1)

        q_buckets = sort_to_buckets(q_hat, q_positions, block_size=self.block_size)
        k_buckets = sort_to_buckets(k_hat, k_positions, block_size=self.block_size)
        v_buckets = sort_to_buckets(value,  k_positions, block_size=self.block_size)

        q = rearrange(q_buckets, "(b h) t d -> b t h d", b=B)
        k = rearrange(k_buckets, "(b h) t d -> b t h d", b=B)
        v = rearrange(v_buckets, "(b h) t d -> b t h d", b=B)
        # ---- END LSH Integration ----

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        offsets = kwargs.get("offsets")



        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(hidden_states).float() + self.dt_bias
        )
        if attention_mask is not None:
            g = g.mul(attention_mask[:, -g.shape[-2]:, None])

        # Interleave g with zeros for non-first transformations
        g = interleave_multiple_sequences(
            [g] + [torch.zeros_like(g)] * (self.num_householder - 1)
        )
        
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        g = g.to(torch.float16)
        beta = beta.to(torch.float16)

        o, recurrent_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=offsets,
            use_qk_l2norm_in_kernel=True
        )
        
        # ---- UNSORT OUTPUT FROM BUCKETS ----
        # Flatten output and unsort
        B, T, H, D = o.shape
        o_flat = rearrange(o, "b t h d -> (b h) t d")
        q_rev_positions = invert_permutation(q_positions)
        o_unsorted = unsort_from_buckets(o_flat, q_rev_positions)
        o = rearrange(o_unsorted, "(b h) t d -> b t h d", b=B)
        # ---- END UNSORT ----

        # Take every nth element for n householder transformations
        o = o[:, self.num_householder - 1:: self.num_householder, :]

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=conv_states if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[2],
            )

        if self.use_gate:
            g = rearrange(
                self.g_proj(hidden_states),
                "... (h d) -> ... h d",
                h=self.num_heads,
            )
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = o.to(torch.float32)
        o = self.o_proj(o)

        return o, None, past_key_values

    
class GatedDeltaProductBlock(nn.Module):
    def __init__(self, dim: int, head_dim:int,  layer_idx: int):
        super().__init__()

        #self.config = config
        self.layer_idx = layer_idx
        
        self.fuse_norm = True


        self.attn_norm = (RMSNorm if self.fuse_norm else nn.RMSNorm)(dim, eps=1e-6)


        self.attn = GatedDeltaProduct(
            mode="chunk",
            hidden_size=dim,
            expand_v=2,
            head_dim=head_dim,
            num_heads=6,  # or int(dim/head_dim)
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

        return outputs
    
class GatedDeltaProductBlock_local1(nn.Module):
    def __init__(self, dim: int, head_dim:int,  layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx
        
        self.fuse_norm = True


        self.attn_norm = (RMSNorm if self.fuse_norm else nn.RMSNorm)(dim, eps=1e-6)


        self.attn = GatedDeltaProduct(
            mode="chunk",
            hidden_size=dim,
            expand_v=2,
            head_dim=head_dim,
            num_heads=6,  # or int(dim/head_dim)
            use_gate=True,
            use_short_conv=True,
            conv_size=4,
            norm_eps=1e-6,
            layer_idx=layer_idx
        )
        self.mlp_norm = (RMSNorm if self.fuse_norm else nn.RMSNorm)(dim, eps=1e-6)
        self.local_agg_norm = (RMSNorm if self.fuse_norm else nn.RMSNorm)(dim, eps=1e-6)
        self.mlp = GatedMLP(
            hidden_size=dim,
            hidden_ratio=4,
            intermediate_size=None,
            hidden_act="swish",
            fuse_swiglu=True
        )
        self.local_agg = LocalAggregation(dim, dim)

    def forward(
        self,
        hidden_states,
        edge_index,
        attention_mask = None,
        past_key_values = None,
        use_cache = False,
        output_attentions = False,
        **kwargs
    ):
        residual = hidden_states
        
        hidden_states = hidden_states + self.local_agg_norm(
            self.local_agg(hidden_states, edge_index)
        )
        
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

        return outputs    
    
class LSHGatedDeltaProductBlock(nn.Module):
    def __init__(self, dim: int, head_dim:int,  layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx
        
        self.fuse_norm = True


        self.attn_norm = (RMSNorm if self.fuse_norm else nn.RMSNorm)(dim, eps=1e-6)


        self.attn = GatedDeltaProduct(
            mode="chunk",
            hidden_size=dim,
            expand_v=2,
            head_dim=head_dim,
            num_heads=6,  # or int(dim/head_dim)
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

        return outputs
    
def prepare_input(x, edge_index, coords, batch, helper_funcs):
    kwargs = {}
    assert batch.max() == 0
    key_padding_mask = None
    mask = None
    kwargs["key_padding_mask"] = key_padding_mask
    kwargs["edge_index"] = edge_index
    kwargs["coords"] = coords
    with torch.no_grad():
        block_size = helper_funcs["block_size"]
        kwargs["raw_size"] = x.shape[0]
        x = pad_to_multiple(x, block_size, dims=0)
        kwargs["coords"] = pad_to_multiple(kwargs["coords"], block_size, dims=0, value=float("inf"))
        sorted_eta_idx = torch.argsort(kwargs["coords"][..., 0], dim=-1)
        sorted_phi_idx = torch.argsort(kwargs["coords"][..., 1], dim=-1)
        regions = helper_funcs["regions"]
        regions_h = rearrange(regions, "c a h -> a (c h)")
        region_indices_eta = quantile_partition(sorted_eta_idx, regions_h[0][:, None])
        region_indices_phi = quantile_partition(sorted_phi_idx, regions_h[1][:, None])
        kwargs["region_indices"] = [region_indices_eta, region_indices_phi]
        kwargs["regions_h"] = regions_h
        kwargs["coords"][kwargs["raw_size"] :] = 0.0
    return x, mask, kwargs

def prepare_input1(x, edge_index, coords, batch, helper_funcs):
    kwargs = {}
    assert batch.max() == 0
    key_padding_mask = None
    mask = None
    kwargs["key_padding_mask"] = key_padding_mask
    kwargs["edge_index"] = edge_index
    kwargs["coords"] = coords
    with torch.no_grad():
        kwargs["raw_size"] = x.shape[0]
        discretized_coords = torch.zeros((x.shape[0], 4), device=x.device)
        discretized_coords[:, -2:] = discretize_coords(coords[:, :2], B=helper_funcs["B"])
        mappings = helper_funcs["mapping"](discretized_coords, batch_size=1)
        kwargs["mappings"] = mappings
    return x, mask, kwargs

class GatedDelta(nn.Module):
    def __init__(self, in_dim, task, **kwargs):
        super(GatedDelta, self).__init__()

        self.depth = kwargs["n_layers"]
        self.mlpdepth = kwargs["mlp_layers"]
        self.hidden_dim = kwargs["h_dim"]
        self.head_size = kwargs["head_size"]

        device = torch.device('cuda:0')

        self.task = task
            
        self.feature_encoder = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            


        self.drop_path = kwargs["drop_path"]
        self.rms_norm = kwargs["rms_norm"]
        self.drop_out_in_block = kwargs["drop_out_in_block"]
        self.cls_dim = 2
        

        self.W = nn.Linear(int(self.hidden_dim // 2)* (self.mlpdepth + 1), int(self.hidden_dim), bias=False)
        self.W0 = nn.Linear(int(self.hidden_dim), int(self.hidden_dim // 2), bias=False)
        self.mlp_out = MLP(
            in_channels=int(self.hidden_dim // 2),
            out_channels=int(self.hidden_dim // 2),
            hidden_channels=256,
            num_layers=4,
            norm="layer_norm",
            act="relu",
            norm_kwargs={"mode": "node"},
        )


        
        self.backbone = nn.ModuleDict(
            dict(
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=GatedDeltaProductBlock(self.hidden_dim,self.head_size, i)
                            )
                        )
                        for i in range(self.depth)
                    ]
                ),
                norm_f=RMSNorm(int(self.hidden_dim)),
            )
        )
        


        self.norm = nn.LayerNorm(int(self.hidden_dim // 2))
        self.rmsnorm = RMSNorm(int(self.hidden_dim // 2))

        self.HEAD_CHANEL = 1


        self.drop_out = nn.Dropout(kwargs["drop_out"]) # else nn.Dropout(0)
        
        self.helper_funcs = {}

        self.helper_funcs["block_size"] = kwargs["block_size"]
        self.regions = nn.Parameter(
            get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False
        )
        self.helper_funcs["regions"] = self.regions
        self.helper_funcs["num_heads"] = kwargs["num_heads"]
        
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: Optional[str] = 'rescale',
        num_residuals_per_layer: int = 2
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Normal init
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                if prenorm_residual_strategy == 'rescale':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        scale = math.sqrt(num_residuals_per_layer * 1)
                        p.div_(scale)
                elif prenorm_residual_strategy == 'zero':
                    nn.init.zeros_(p)
                else:
                    raise ValueError(f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}")
        


    def forward(self, data):

        if isinstance(data, dict):
            input_features, edge_index, coords, batch = data["x"], data["edge_index"], data["coords"], data["batch"]
        else:
            input_features, edge_index, coords, batch = data.x, data.edge_index, data.coords , data["batch"]


        input_features, mask, kwargs = prepare_input(input_features, edge_index, coords, batch, self.helper_funcs)

   
            
        encoded_features = self.feature_encoder(input_features)   
        x = encoded_features
        
        
        

        past_key_values = None
        
        n = x.size()[-2]
        d = x.size()[-1]
        x = x.view(1,n,d)
        
        for i, layer in enumerate(self.backbone.layers):
            
            
            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.mixer.__call__,
                    x,
                    None,
                    past_key_values,
                    False,
                    False
                )
            else:
                hidden_states, attentions, past_key_values = layer.mixer(
                    x,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=False,
                    output_attentions=False
                )
        
        x = hidden_states.view(n,d)
        x = self.backbone.norm_f(x)
        
        x = self.W0(x)
        
        out = x + self.drop_out(self.mlp_out(x))
 
        out = out.view(-1,self.hidden_dim // 2)
        out = out[: kwargs["raw_size"],:]
            
        if mask is not None:
            out = out[mask]


        return out
    
    
class GatedDelta_local1(nn.Module):
    def __init__(self, in_dim, task, **kwargs):
        super(GatedDelta_local1, self).__init__()
        #self.config = config

        self.depth = kwargs["n_layers"]
        self.mlpdepth = kwargs["mlp_layers"]
        self.hidden_dim = kwargs["h_dim"]
        self.head_size = kwargs["head_size"]
        #self.cls_dim = config.cls_dim

        device = torch.device('cuda:0')

        self.task = task

            
        self.feature_encoder = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            


        self.drop_path = kwargs["drop_path"]
        self.rms_norm = kwargs["rms_norm"]
        self.drop_out_in_block = kwargs["drop_out_in_block"]
        self.cls_dim = 2
        

        self.W = nn.Linear(int(self.hidden_dim // 2)* (self.mlpdepth + 1), int(self.hidden_dim), bias=False)
        self.W0 = nn.Linear(int(self.hidden_dim), int(self.hidden_dim // 2), bias=False)
        self.mlp_out = MLP(
            in_channels=int(self.hidden_dim // 2),
            out_channels=int(self.hidden_dim // 2),
            hidden_channels=256,
            num_layers=4,
            norm="layer_norm",
            act="relu",
            norm_kwargs={"mode": "node"},
        )


        
        self.backbone = nn.ModuleDict(
            dict(
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=GatedDeltaProductBlock_local1(self.hidden_dim,self.head_size, i)
                            )
                        )
                        for i in range(self.depth)
                    ]
                ),
                norm_f=RMSNorm(int(self.hidden_dim)),
            )
        )
        


        self.norm = nn.LayerNorm(int(self.hidden_dim // 2))
        self.rmsnorm = RMSNorm(int(self.hidden_dim // 2))

        self.HEAD_CHANEL = 1



        self.drop_out = nn.Dropout(kwargs["drop_out"]) # else nn.Dropout(0)
        
        self.helper_funcs = {}

        self.helper_funcs["block_size"] = kwargs["block_size"]
        self.regions = nn.Parameter(
            get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False
        )
        self.helper_funcs["regions"] = self.regions
        self.helper_funcs["num_heads"] = kwargs["num_heads"]
        
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: Optional[str] = 'rescale',
        num_residuals_per_layer: int = 2
    ):
        #print(module)
        # Custom initialization for HGatedDeltaNet parameters
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Normal init
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        # Prenorm residual scaling
        if prenorm_residual_strategy is not None:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                if prenorm_residual_strategy == 'rescale':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        scale = math.sqrt(num_residuals_per_layer * 1)
                        p.div_(scale)
                elif prenorm_residual_strategy == 'zero':
                    nn.init.zeros_(p)
                else:
                    raise ValueError(f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}")
        




    def forward(self, data):

        if isinstance(data, dict):
            input_features, edge_index, coords, batch = data["x"], data["edge_index"], data["coords"], data["batch"]
        else:
            input_features, edge_index, coords, batch = data.x, data.edge_index, data.coords , data["batch"]


        input_features, mask, kwargs = prepare_input(input_features, edge_index, coords, batch, self.helper_funcs)

   
            
        encoded_features = self.feature_encoder(input_features)   
        x = encoded_features
        
        
        

        past_key_values = None
        
        n = x.size()[-2]
        d = x.size()[-1]
        x = x.view(1,n,d)
        
        for i, layer in enumerate(self.backbone.layers):
            
            
            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.mixer.__call__,
                    x,
                    edge_index,
                    None,
                    past_key_values,
                    False,
                    False
                )
            else:
                hidden_states, attentions, past_key_values = layer.mixer(
                    x,
                    edge_index,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=False,
                    output_attentions=False
                )
        
        x = hidden_states.view(n,d)
        x = self.backbone.norm_f(x)
        
        x = self.W0(x)
        
        out = x + self.drop_out(self.mlp_out(x))
        
        out = out.view(-1,self.hidden_dim // 2)
        out = out[: kwargs["raw_size"],:]
            
        if mask is not None:
            out = out[mask]


        return out
    
    
class LSHGatedDelta(nn.Module):
    def __init__(self, in_dim, task, **kwargs):
        super(LSHGatedDelta, self).__init__()
        #self.config = config

        self.depth = kwargs["n_layers"]
        self.mlpdepth = kwargs["mlp_layers"]
        self.hidden_dim = kwargs["h_dim"]
        self.head_size = kwargs["head_size"]
        #self.cls_dim = config.cls_dim

        device = torch.device('cuda:0')

        self.task = task
 
            
        self.feature_encoder = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            


        self.drop_path = kwargs["drop_path"]
        self.rms_norm = kwargs["rms_norm"]
        self.drop_out_in_block = kwargs["drop_out_in_block"]
        self.cls_dim = 2
        

        self.W = nn.Linear(int(self.hidden_dim // 2)* (self.mlpdepth + 1), int(self.hidden_dim), bias=False)
        self.W0 = nn.Linear(int(self.hidden_dim), int(self.hidden_dim // 2), bias=False)
        self.mlp_out = MLP(
            in_channels=int(self.hidden_dim // 2),
            out_channels=int(self.hidden_dim // 2),
            hidden_channels=256,
            num_layers=4,
            norm="layer_norm",
            act="relu",
            norm_kwargs={"mode": "node"},
        )


        
        self.backbone = nn.ModuleDict(
            dict(
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=LSHGatedDeltaProductBlock(self.hidden_dim,self.head_size, i)
                            )
                        )
                        for i in range(self.depth)
                    ]
                ),
                norm_f=RMSNorm(int(self.hidden_dim)),
            )
        )
        


        self.norm = nn.LayerNorm(int(self.hidden_dim // 2))
        self.rmsnorm = RMSNorm(int(self.hidden_dim // 2))

        self.HEAD_CHANEL = 1


        self.drop_out = nn.Dropout(kwargs["drop_out"]) # else nn.Dropout(0)
        
        self.helper_funcs = {}

        self.helper_funcs["block_size"] = kwargs["block_size"]
        self.regions = nn.Parameter(
            get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False
        )
        self.helper_funcs["regions"] = self.regions
        self.helper_funcs["num_heads"] = kwargs["num_heads"]
        
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: Optional[str] = 'rescale',
        num_residuals_per_layer: int = 2
    ):
        #print(module)
        # Custom initialization for HGatedDeltaNet parameters
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Normal init
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        # Prenorm residual scaling
        if prenorm_residual_strategy is not None:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                if prenorm_residual_strategy == 'rescale':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        scale = math.sqrt(num_residuals_per_layer * 1)
                        p.div_(scale)
                elif prenorm_residual_strategy == 'zero':
                    nn.init.zeros_(p)
                else:
                    raise ValueError(f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}")



    def forward(self, data):

        if isinstance(data, dict):
            input_features, edge_index, coords, batch = data["x"], data["edge_index"], data["coords"], data["batch"]
        else:
            input_features, edge_index, coords, batch = data.x, data.edge_index, data.coords , data["batch"]


        input_features, mask, kwargs = prepare_input(input_features, edge_index, coords, batch, self.helper_funcs)


   
            
        encoded_features = self.feature_encoder(input_features)   
        x = encoded_features
        
        
        

        past_key_values = None
        
        n = x.size()[-2]
        d = x.size()[-1]
        x = x.view(1,n,d)
        
        for i, layer in enumerate(self.backbone.layers):
            
            
            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.mixer.__call__,
                    x,
                    None,
                    past_key_values,
                    False,
                    False,
                    **kwargs
                )
            else:
                hidden_states, attentions, past_key_values = layer.mixer(
                    x,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=False,
                    output_attentions=False,
                    **kwargs
                )
        
        x = hidden_states.view(n,d)
        x = self.backbone.norm_f(x)
        
        x = self.W0(x)
        
        out = x + self.drop_out(self.mlp_out(x))


        out = out[: kwargs["raw_size"],:]
            
        if mask is not None:
            out = out[mask]


        return out
    
    
    
    
