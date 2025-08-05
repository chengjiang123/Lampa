from typing import Union, Optional, List
import math
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MLP

from ..model_utils.mamba_utils import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath


from ..attention import (
    HEPTAttention,
)
from einops import rearrange
from ..model_utils.hash_utils import pad_to_multiple, get_regions, quantile_partition
from ..model_utils.hash_utils import lsh_mapping, batched_index_select, invert_permutation, E2LSH

##to do, try mamba2 
from mamba_ssm.modules.mamba_simple import Mamba
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

#from knn_cuda import KNN
from .mblocks import Block
from .mbuild import MODELS



def sort_to_buckets(x, perm, bucketsz):
    return rearrange(
        batched_index_select(rearrange(x, "b s d -> 1 b s d"), perm),
        "h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d",
        bucketsz=bucketsz,
    )


def unsort_from_buckets(s_x, perm_inverse):
    b_x = rearrange(s_x, "h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d")
    return batched_index_select(b_x, perm_inverse)


def qkv_res(s_query, s_key, s_value):
    q_sq_05 = -0.5 * (s_query**2).sum(dim=-1, keepdim=True)
    k_sq_05 = -0.5 * (s_key**2).sum(dim=-1, keepdim=True)

    clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key)
    clustered_dists = (clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)).clamp(max=0.0).exp()

    denom = clustered_dists.sum(dim=-1, keepdim=True) + (1e-20)
    qk = clustered_dists

    so = torch.einsum("...ij,...jd->...id", qk, s_value)
    return denom, so


def prep_qk(query, key, w, coords):
    qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
    new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)

    sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim)[None] * coords[:, None]

    
    q_hat = torch.cat([query, sqrt_w_r], dim=-1)
    k_hat = torch.cat([key, sqrt_w_r], dim=-1)
    return q_hat, k_hat


@torch.no_grad()
def get_geo_shift(regions_h: List[List[int]], hash_shift, region_indices, num_or_hashes):
    region_indices_eta, region_indices_phi = region_indices

    q_hash_shift_eta = region_indices_eta * hash_shift
    k_hash_shift_eta = region_indices_eta * hash_shift

    q_hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    k_hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    res = torch.stack([q_hash_shift_phi + q_hash_shift_eta, k_hash_shift_phi + k_hash_shift_eta], dim=0)
    return rearrange(res, "a (c h) n -> a c h n", c=num_or_hashes)


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
        coords = pad_to_multiple(coords, block_size, dims=0, value=float("inf"))
        kwargs["coords"] = pad_to_multiple(kwargs["coords"], block_size, dims=0, value=float("inf"))
        sorted_eta_idx = torch.argsort(kwargs["coords"][..., 0], dim=-1)
        sorted_phi_idx = torch.argsort(kwargs["coords"][..., 1], dim=-1)
        regions = helper_funcs["regions"]
        regions_h = rearrange(regions, "c a h -> a (c h)")
        
        #print(f'sorted_eta_idx {sorted_eta_idx.shape}')
        
        
        region_indices_eta = quantile_partition(sorted_eta_idx, regions_h[0][:, None])
        
        #print(f'region_indices_eta {region_indices_eta.shape}')
        region_indices_phi = quantile_partition(sorted_phi_idx, regions_h[1][:, None])
        kwargs["region_indices"] = [region_indices_eta, region_indices_phi]
        kwargs["regions_h"] = regions_h
        kwargs["coords"][kwargs["raw_size"] :] = 0.0
        coords[kwargs["raw_size"] :] = 0.0
    return x, coords, mask, kwargs


class LSHBucketing:
    def __init__(self, window_size):
        self.window_size = window_size

    def sort_by_proximity(self, coords):
        """
        Sort tokens based on their proximity (distance) to each other.
        This ensures that tokens closer together spatially are processed together.
        Input:
            coords: (batch_size, seq_length, coord_dim)
        Output:
            sorted_indices: (batch_size, seq_length)
        """
        batch_size, seq_length, _ = coords.size()
        sorted_indices = torch.argsort(coords.norm(dim=2), dim=1)
        return sorted_indices

    def sliding_window(self, vectors, sorted_indices):
        """
        Apply a sliding window to group tokens for enhanced locality.
        Input:
            vectors: (batch_size, seq_length, h_dim)
            sorted_indices: (batch_size, seq_length)
        Output:
            windowed_vectors: (batch_size, seq_length, h_dim)
        """
        batch_size, seq_length, h_dim = vectors.size()
        windowed_vectors = torch.zeros_like(vectors)

        # Apply sliding window processing
        for i in range(0, seq_length, self.window_size):
            window_indices = sorted_indices[:, i:i+self.window_size]
            windowed_vectors[:, i:i+self.window_size, :] = vectors.gather(1, window_indices.unsqueeze(-1).expand(-1, -1, h_dim))

        return windowed_vectors

    def preprocess(self, vectors, coords):
        """
        Main preprocessing function to enhance locality.
        Input:
            vectors: (batch_size, seq_length, h_dim)
            coords: (batch_size, seq_length, coord_dim)
        Output:
            processed_vectors: (batch_size, seq_length, h_dim)
        """
        sorted_indices = self.sort_by_proximity(coords)
        processed_vectors = self.sliding_window(vectors, sorted_indices)
        return processed_vectors



    
# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454

def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:

                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block

@torch.no_grad()
def lsh_mapping(e2lsh, queries, keys):
    """
    queries, keys: each (n_hashes, N, dim)
    returns:
      q_hashed, k_hashed: (n_hashes, N, dim)
      hash_shift:         (n_hashes, N, 1)
    """
    q_h = e2lsh(queries)
    k_h = e2lsh(keys)
    max_h = torch.maximum(
        q_h.max(-1, keepdim=True).values,
        k_h.max(-1, keepdim=True).values
    )
    min_h = torch.minimum(
        q_h.min(-1, keepdim=True).values,
        k_h.min(-1, keepdim=True).values
    )
    shift = max_h - min_h
    return q_h, k_h, shift


class E2LSH(nn.Module):
    def __init__(self, n_hashes, n_heads, dim, r=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.normal(0,1,(n_heads, dim, n_hashes)))
        self.beta  = nn.Parameter(torch.rand(1, n_hashes) * r)
        self.alpha.requires_grad = False
        self.beta .requires_grad = False

    def forward(self, vecs):
        proj = torch.bmm(vecs, self.alpha)     
        return proj.permute(2, 1, 0)           


class LSHOrdering(nn.Module):
    def __init__(self, input_dim: int, num_hashes: int = 3):
        """
        input_dim:   feature‐dim of each token
        num_hashes:  how many independent LSH functions to use
        """
        super().__init__()
        self.num_hashes = num_hashes
        # Set up a single‐head E2LSH (so we flatten B·L → N)
        self.e2lsh = E2LSH(self.num_hashes, n_heads=1, dim=input_dim)

    def forward(self, points: torch.Tensor, coords: torch.Tensor = None):
        """
        points: (B, L, D)      — embeddings you want to reorder
        coords: (B, L, C) or None  — positions used for hashing
        returns:
          ordered_points: (B, L, D)
          indices:        (B, L)
        """
        B, L, D = points.shape

        if coords is None:
            coords_for_hash = points         # use embeddings
            C = D
        else:
            coords_for_hash = coords         # use spatial coords
            _, _, C = coords.shape

        flat = coords_for_hash.reshape(B * L, C).unsqueeze(0)

        q_hashed, _, shift = lsh_mapping(self.e2lsh, flat, flat)

        q_shifted = q_hashed + shift          # (H, N, C)
        scores    = q_shifted.sum(dim=-1)     # (H, N)

        bucket_matrix = (
            scores.view(self.num_hashes, B, L)
                  .permute(1, 2, 0)          # → (B, L, H)
        )

        exponents   = torch.tensor(
            [2**i for i in range(self.num_hashes)],
            device=points.device,
            dtype=torch.long
        )
        bucket_keys = (bucket_matrix * exponents).sum(dim=-1)  # (B, L)

        indices = []
        for b in range(B):
            # a) spatial pre-sort by norm
            norm      = coords_for_hash[b].norm(dim=1)  # (L,)
            norm_idx  = norm.argsort()                  # (L,)

            # b) stable sort by bucket key
            bk        = bucket_keys[b][norm_idx]        # (L,)
            _, bk_idx = bk.sort(stable=True)            # (L,)
            indices.append(norm_idx[bk_idx])

        indices = torch.stack(indices, dim=0)  # (B, L)

        ordered_points = points.gather(
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, D)
        )

        return ordered_points, indices


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        drop_out_in_block: float = 0.0,
        drop_path: float = 0.1,
        device=None,
        dtype=None,
        **kwargs
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        # build Mamba/SSM blocks
        self.layers = nn.ModuleList([
            create_block(
                d_model,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                drop_path=drop_path,
                **factory_kwargs,
            )
            for i in range(n_layer)
        ])

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg or {}),
            )
        )

        print('using LSH ordering')
        #self.lsh_order = LSHOrdering(input_dim=6, num_hashes=4, num_buckets=64)
        self.lsh_order = LSHOrdering(input_dim=6, num_hashes=3)

        self.drop_out_in_block = (
            nn.Dropout(drop_out_in_block) if drop_out_in_block > 0 else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, input_ids, coords, kwargs=None, inference_params=None):
        """
        input_ids: (B, L, D_model)
        coords:    (B, L, coord_dim)
        """

        hidden_states = input_ids  # (B, L, D_model)
        
        n = hidden_states.size()[-2]
        d = hidden_states.size()[-1]
        d1 = coords.size()[-1]
        hidden_states = hidden_states.view(1,n,d)
        coords = coords.view(1,n,d1)

        ordered, perm = self.lsh_order(hidden_states, coords)  # (B, L, D), (B, L)

        x = ordered
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual, inference_params=inference_params)
            x = self.drop_out_in_block(x)

        if not self.fused_add_norm:
            res = (x + residual) if residual is not None else x
            x = self.norm_f(res.to(dtype=self.norm_f.weight.dtype))
        else:
            norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            x = norm_fn(
                x,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        inv_perm = perm.argsort(dim=1)  # (B, L)
        output = x.gather(
            1,
            inv_perm.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )

        return output

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

class MixerModel0(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
            **kwargs
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        
        
        ###attention
        self.lsh_attention = LSHAtt(**kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids,coords, kwargs, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        n = hidden_states.size()[-2]
        d = hidden_states.size()[-1]
 
        hidden_states = hidden_states.view(n,d)
        hidden_states = self.lsh_attention(hidden_states, kwargs)
        
        for layer in self.layers:
            hidden_states = hidden_states.view(1,n,d)
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
            #hidden_states, residual = layer(bucketed_hidden_states, residual, inference_params=inference_params)
            hidden_states = self.drop_out_in_block(hidden_states)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states
    
@MODELS.register_module()
class HMamba_v1(nn.Module):
    def __init__(self, in_dim, task, **kwargs):
        super(HMamba_v1, self).__init__()

        self.trans_dim = kwargs["trans_dim"]
        self.depth = kwargs["n_layers"]
        self.hidden_dim = kwargs["h_dim"]

        self.group_size = kwargs["group_size"]
        self.num_group = kwargs["num_group"]
        self.encoder_dims = kwargs["encoder_dim"]

        self.task = task        
        self.feature_encoder = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            


        self.drop_path = kwargs["drop_path"]
        self.rms_norm = kwargs["rms_norm"]
        ## try 0 or 0.1
        self.drop_out_in_block = kwargs["drop_out_in_block"]
        self.cls_dim = 2
        
        self.attns = nn.ModuleList()
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


        self.blocks = MixerModel(d_model=int(self.trans_dim // 2),
                                 n_layer=self.depth,
                                 **kwargs)

        self.norm = nn.LayerNorm(int(self.hidden_dim // 2))

        self.HEAD_CHANEL = 1

        self.drop_out = nn.Dropout(kwargs["drop_out"]) # else nn.Dropout(0)
        
        self.helper_funcs = {}

        self.helper_funcs["block_size"] = kwargs["block_size"]
        self.regions = nn.Parameter(
            get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False
        )
        self.helper_funcs["regions"] = self.regions
        self.helper_funcs["num_heads"] = kwargs["num_heads"]


    def forward(self, data):

        if isinstance(data, dict):
            input_features, edge_index, coords, batch = data["x"], data["edge_index"], data["coords"], data["batch"]
        else:
            input_features, edge_index, coords, batch = data.x, data.edge_index, data.coords , data["batch"]

        input_features, coords, mask, kwargs = prepare_input(input_features, edge_index, coords, batch, self.helper_funcs)
        
        
        encoded_features = self.feature_encoder(input_features)

        
        
        x = encoded_features

        x = self.drop_out(x)
        x = self.blocks(x, coords, kwargs)
        x = x.squeeze(0)
        x = self.W0(x)
        x = self.norm(x)
        
        out = x + self.drop_out(self.mlp_out(x))

        out = out.view(-1,int(self.hidden_dim // 2))           
        out = out[: kwargs["raw_size"],:]
            
        if mask is not None:
            out = out[mask]
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return out


@MODELS.register_module()
class HMamba_v2(nn.Module):
    def __init__(self, in_dim, task, **kwargs):
        super(HMamba_v2, self).__init__()

        self.trans_dim = kwargs["trans_dim"]
        self.depth = kwargs["n_layers"]
        self.hidden_dim = kwargs["h_dim"]

        self.group_size = kwargs["group_size"]
        self.num_group = kwargs["num_group"]
        self.encoder_dims = kwargs["encoder_dim"]

        self.task = task        
        self.feature_encoder = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            


        self.drop_path = kwargs["drop_path"]
        self.rms_norm = kwargs["rms_norm"]
        ## try 0 or 0.1
        self.drop_out_in_block = kwargs["drop_out_in_block"]
        self.cls_dim = 2
        
        self.attns = nn.ModuleList()
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


        self.blocks = MixerModel0(d_model=int(self.trans_dim // 2),
                                 n_layer=self.depth,
                                 **kwargs)

        self.norm = nn.LayerNorm(int(self.hidden_dim // 2))

        self.HEAD_CHANEL = 1

        self.drop_out = nn.Dropout(kwargs["drop_out"]) # else nn.Dropout(0)
        
        self.helper_funcs = {}

        self.helper_funcs["block_size"] = kwargs["block_size"]
        self.regions = nn.Parameter(
            get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False
        )
        self.helper_funcs["regions"] = self.regions
        self.helper_funcs["num_heads"] = kwargs["num_heads"]


    def forward(self, data):

        if isinstance(data, dict):
            input_features, edge_index, coords, batch = data["x"], data["edge_index"], data["coords"], data["batch"]
        else:
            input_features, edge_index, coords, batch = data.x, data.edge_index, data.coords , data["batch"]

        input_features, coords, mask, kwargs = prepare_input(input_features, edge_index, coords, batch, self.helper_funcs)
        
        
        encoded_features = self.feature_encoder(input_features)

        
        
        x = encoded_features

        x = self.drop_out(x)
        x = self.blocks(x, coords, kwargs)
        x = x.squeeze(0)
        x = self.W0(x)
        x = self.norm(x)
        
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

        return x
    
