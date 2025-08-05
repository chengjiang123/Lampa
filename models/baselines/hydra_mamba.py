from typing import Union, Optional
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MLP

from ..model_utils.mamba_utils import *
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

#from knn_cuda import KNN
import json
from dataclasses import dataclass
from typing import Iterable, NamedTuple
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import LongTensor, Tensor, nn


from ..attention import (
    HEPTAttention,
)
from einops import rearrange
from ..model_utils.hash_utils import pad_to_multiple, get_regions, quantile_partition
from ..model_utils.hash_utils import lsh_mapping, batched_index_select, invert_permutation, E2LSH

def bit_shift(base, shift_idx):
    max_base = base.max(dim=1, keepdim=True).values
    num_bits = torch.ceil(torch.log2(max_base + 1)).long()
    return (shift_idx << num_bits) | base


def pad_and_unpad(batch, block_size, region_indices, raw_sizes):
    padded_sizes = ((raw_sizes + block_size - 1) // block_size) * block_size
    pad_sizes = padded_sizes - raw_sizes

    pad_cumsum = padded_sizes.cumsum(0)
    pad_seq = torch.arange(pad_cumsum[-1], device=batch.device)
    unpad_seq = torch.ones(pad_cumsum[-1], device=batch.device).bool()

    sorted_region_indices = region_indices.argsort()
    for i in range(len(raw_sizes)):
        idx_to_fill = pad_cumsum[i] - block_size - pad_sizes[i] + torch.arange(pad_sizes[i], device=batch.device)
        if i >= 1:
            pad_seq[pad_cumsum[i - 1] :] -= pad_sizes[i - 1]
            idx_to_fill -= pad_sizes[:i].sum()
        pad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = sorted_region_indices[idx_to_fill]
        unpad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = False
    return pad_seq, unpad_seq


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


@dataclass
class Mamba2Config:
    d_model: int = 48  # model dimension (D)
    n_layer: int = 8  # number of Mamba-2 layers in the language model
    d_state: int = 16  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 12  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
        
    A_init_range: tuple = (1, 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: tuple = (0.0, float("inf"))
    conv_init = None

    learnable_init_states: bool = False
    activation: str = "swish" # "swish" or "silu"
    
    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True

    mup: bool = False
    mup_base_width: float = 96 # width=d_model

    dtype=None
    device=None

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim

class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2* (2 * args.d_state), 2*args.d_conv, device=device
            ),
            torch.zeros(
                2*batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )      
        #return InferenceCache(
        #    torch.zeros(
        #        batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
        #    ),
        #    torch.zeros(
        #        batch_size, args.nheads, args.headdim, args.d_state, device=device
        #    ),
        #)        
        

    

class HydraMamba(nn.Module):
    def __init__(self, d_model, args: Mamba2Config, device = None):
        super().__init__()
        self.args = args
        self.device = device
        self.d_model = d_model
        args.d_model = d_model
        # Order: (z, x, B, C, dt)
        #d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        d_in_proj = 2 * args.d_inner + 2 * (2 * args.d_state) + 2 * args.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, device=device)

        conv_dim = args.d_inner + 2 * (2 * args.d_state)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )

        #self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device))
        #self.A_log = nn.Parameter(torch.empty(args.nheads, device=device))
        #self.D = nn.Parameter(torch.empty(args.nheads, device=device))
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(args.d_inner, self.d_model, bias=False, device=device)
        
        ##
        self.fc_D = nn.Linear(args.d_inner, args.nheads, bias=False, device=device)
        self.act = nn.SiLU()
        
        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(args.nheads, device=device) * (math.log(args.dt_max) - math.log(args.dt_min))
            + math.log(args.dt_min)
        )
        dt = torch.clamp(dt, min=args.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert self.args.A_init_range[0] > 0 and args.A_init_range[1] >= args.A_init_range[0]
        A = torch.empty(args.nheads, dtype=torch.float32, device=device).uniform_(*args.A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True
        
         # D "skip" parameter
        self.D = nn.Parameter(torch.ones(args.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection

    def forward(self, u: Tensor, h: InferenceCache):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing `u`
        """
        if h:
            return self.step(u, h)

        A = -torch.exp(self.A_log)  # (nheads,)
        
        dt_limit_kwargs = {} if self.args.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.args.dt_limit)

        
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)

        
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2* (2 * self.args.d_state),
                2* self.args.nheads,
            ],
            dim=-1,
        )
        

        
        ######quasisep
        dt = torch.cat((dt[:, :, :self.args.nheads], torch.flip(dt[:, :, self.args.nheads:], (1,))), dim=0) 
        
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Pad or truncate xBC seqlen to d_conv
        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )

        #xBC = silu(
        #    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        #)  # (batch, seqlen, d_inner + 2 * d_state))
        
        
        if causal_conv1d_fn is None or self.args.activation not in ["silu", "swish"]:
            xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))[:, : u.shape[1], :] # (B, L, self.d_inner + 2 * n_groups * d_state)
        else:
            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.args.activation,
            ).transpose(1, 2)[:, : u.shape[1], :]
        
        
        
        #x, B, C = torch.split(
        #    xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        #)
        
        ######quasisep
        
        x, BC = torch.split(xBC, [self.args.d_inner, 2 * (2 * self.args.d_state)], dim=-1)
        x_og = x
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)

        BC = torch.cat((BC[:, :, :2 * self.args.d_state], torch.flip(BC[:, :, 2 * self.args.d_state:], (1,))), dim=0)
        B, C = torch.split(BC, [self.args.d_state, self.args.d_state], dim=-1) 
        
        ######
        
        
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=self.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        
        ######quasisep
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y_fw, y_bw = y[:1], torch.flip(y[1:], (1,))
        y = y_fw + y_bw + x_og * repeat(F.linear(x_og, self.fc_D.weight, bias=self.D), "b l h -> b l (h p)", p=self.args.headdim)
        ######
        
        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)
        return y, h

    def step(self, u: Tensor, h: InferenceCache) -> tuple[Tensor, InferenceCache]:
        """Take a single inference step for the current input and hidden state

        Unlike attention-based models, RNN-based models (eg Mamba) does not need
        to look back at all the past tokens to generate a new token. Instead a
        hidden state (initialized to 0s initially) is updated for each input and
        passed to the next inference step. This means that the total inference
        time is linear with respect to the sequence length instead of quadratic
        in attention's case.

        Arguments
            u: (batch, 1, d_model)
            h: initial/running hidden state

        Return (y, h)
            y: (batch, 1, d_model)
            h: updated hidden state
        """
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        #z, xBC, dt = torch.split(
        #    zxbcdt,
        #    [
        #        self.args.d_inner,
        #        self.args.d_inner + 2 * self.args.d_state,
        #        self.args.nheads,
        #    ],
        #    dim=-1,
        #)
        
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2* (2 * self.args.d_state),
                2* self.args.nheads,
            ],
            dim=-1,
        )

        # Advance convolution input
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        # Convolution step
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)

        #x, B, C = torch.split(
        #    xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        #)
        
        ######quasisep
        
        x, BC = torch.split(xBC, [self.args.d_inner, 2 * (2 * self.args.d_state)], dim=-1)
        x_og = x
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)

        BC = torch.cat((BC[:, :, :2 * self.args.d_state], torch.flip(BC[:, :, 2 * self.args.d_state:], (1,))), dim=0)
        B, C = torch.split(BC, [self.args.d_state, self.args.d_state], dim=-1) 
        
        ######
        
        A = -torch.exp(self.A_log)  # (nheads,)

        # SSM step
        
        ##
        dt = torch.cat((dt[:, :, :self.args.nheads], torch.flip(dt[:, :, self.args.nheads:], (1,))), dim=0) 
        
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h


def segsum(x: Tensor, device = None) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

###
def pad_to_multiple0(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pads the sequence length of tensor x to be a multiple of 'multiple'."""
    seqlen = x.shape[1]
    pad_len = (multiple - seqlen % multiple) % multiple
    if pad_len > 0:
        pad = (0, 0) * (x.dim() - 2) + (0, pad_len)  # Only pad the sequence length dimension
        x = F.pad(x, pad)
    return x, seqlen

def ssd(x, A, B, C, chunk_size, initial_states=None, device = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    
    
    ###
    x, orig_seqlen = pad_to_multiple0(x, chunk_size)
    A, _ = pad_to_multiple0(A, chunk_size)
    B, _ = pad_to_multiple0(B, chunk_size)
    C, _ = pad_to_multiple0(C, chunk_size)
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader 😜
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    
    Y = Y[:, :orig_seqlen, :, :]

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device = None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * F.sigmoid(x)


class HydraMamba2(nn.Module):
    def __init__(self, in_dim, task,  args: Mamba2Config, **kwargs):
        super(HydraMamba2, self).__init__()
        self.trans_dim = kwargs["trans_dim"]
        self.depth = kwargs["n_layers"]
        self.mlpdepth = kwargs["mlp_layers"]
        self.hidden_dim = kwargs["h_dim"]

        self.group_size = kwargs["group_size"]
        self.num_group = kwargs["num_group"]
        self.encoder_dims = kwargs["encoder_dim"]
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
        self.W0 = nn.Linear(int(self.trans_dim // 2), int(self.hidden_dim // 2), bias=False)
        self.mlp_out = MLP(
            in_channels=int(self.hidden_dim // 2),
            out_channels=int(self.hidden_dim // 2),
            hidden_channels=256,
            num_layers=5,
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
                                mixer=HydraMamba(int(self.trans_dim // 2),args, device=device),
                                norm=RMSNorm(args.d_model, device=device),
                            )
                        )
                        for _ in range(self.depth)
                    ]
                ),
                norm_f=RMSNorm(int(self.trans_dim // 2), device=device),
            )
        )

        self.norm = nn.LayerNorm(int(self.hidden_dim // 2))
        self.rmsnorm = RMSNorm(int(self.hidden_dim // 2), device=device)

        self.HEAD_CHANEL = 1

        self.drop_out = nn.Dropout(kwargs["drop_out"]) # else nn.Dropout(0)
        
        self.helper_funcs = {}

        self.helper_funcs["block_size"] = kwargs["block_size"]
        self.regions = nn.Parameter(
            get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False
        )
        self.helper_funcs["regions"] = self.regions
        self.helper_funcs["num_heads"] = kwargs["num_heads"]
        
        self.lsh_attention = LSHAtt(**kwargs)



    def forward(self, data):

        if isinstance(data, dict):
            input_features, edge_index, coords, batch = data["x"], data["edge_index"], data["coords"], data["batch"]
        else:
            input_features, edge_index, coords, batch = data.x, data.edge_index, data.coords , data["batch"]

        ## reshape to B,N,D = 1,N,D
        input_features, mask, kwargs = prepare_input(input_features, edge_index, coords, batch, self.helper_funcs)


        encoded_features = self.feature_encoder(input_features)   
        x = encoded_features
    
        h = None
        if h is None:
            h = [None for _ in range(self.depth)]
        
        for i, layer in enumerate(self.backbone.layers):
            n = x.size()[-2]
            d = x.size()[-1]

            x = x.view(1,n,d)
            
            y, h[i] = layer.mixer(layer.norm(x), h[i])
            
            x = x.view(n,d)
            x = self.lsh_attention(x, kwargs)

            
            x = y + x

        x = self.backbone.norm_f(x)
        
        x = self.W0(x)
        
        out = x + self.drop_out(self.mlp_out(x))
 
        out = out.view(-1,int(self.hidden_dim // 2))
        out = out[: kwargs["raw_size"],:]
            
        if mask is not None:
            out = out[mask]


        return out
    
    
class FF(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        


        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(int(self.dim_per_head // 2))
        self.norm2 = nn.LayerNorm(int(self.dim_per_head // 2))
        self.ff = nn.Sequential(
            nn.Linear(int(self.dim_per_head // 2), int(self.dim_per_head // 2)),
            nn.ReLU(),
            nn.Linear(int(self.dim_per_head // 2), int(self.dim_per_head // 2)),
        )


    def forward(self, x):
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)


        return x
    

    
class LSHAtt(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]

        self.w_q = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_k = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_v = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        coords_dim = 6
        #coords_dim = 3

        # +2 for data.pos
        self.attn = HEPTAttention(self.dim_per_head + coords_dim, **kwargs)

        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(self.dim_per_head)
        self.norm2 = nn.LayerNorm(self.dim_per_head)
        self.ff = nn.Sequential(
            nn.Linear(self.dim_per_head, self.dim_per_head),
            nn.ReLU(),
            nn.Linear(self.dim_per_head, self.dim_per_head),
        )

        # eta/phi from data.pos use the same weights as they are used to calc dR
        self.w_rpe = nn.Linear(kwargs["num_w_per_dist"] * (coords_dim - 1), self.num_heads * self.dim_per_head)

    def forward(self, x, kwargs):
        x_normed = self.norm1(x)
        q, k, v = self.w_q(x_normed), self.w_k(x_normed), self.w_v(x_normed)
        aggr_out = self.attn(q, k, v, pe=kwargs["coords"], w_rpe=self.w_rpe, **kwargs)
        
        x = aggr_out
        x = x + self.dropout(aggr_out)
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)


        return x
 

