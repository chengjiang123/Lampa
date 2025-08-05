# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update


from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from torch_geometric.nn import MLP

from ..attention import (
    HEPTAttention,
)
from ..model_utils.hash_utils import pad_to_multiple, get_regions, quantile_partition
from ..model_utils.hash_utils import lsh_mapping, batched_index_select, invert_permutation, E2LSH
from ..model_utils.window_utils import discretize_coords, FlattenedWindowMapping, get_pe_func
from typing import Any, Dict, Optional, Tuple

chunk_size = 64

def prepare_input(x, coords, edge_index=None, chunk_sizes=chunk_size, no_pad = False):
    kwargs = {}

    key_padding_mask = None
    mask = None
    kwargs["key_padding_mask"] = key_padding_mask
    kwargs["edge_index"] = edge_index
    
    
    if not no_pad:
        kwargs["raw_size"] = x.shape[0]

        x = pad_to_multiple(x, chunk_sizes, dims=0)
        coords = pad_to_multiple(coords, chunk_sizes, dims=0)
        coords[kwargs["raw_size"] :] = 0.0
    else:
        x = x
    kwargs["coords"] = coords
    return x, mask,coords, kwargs

def prepare_input1(x, edge_index, coords, batch, helper_funcs):
    kwargs = {}
    assert batch.max() == 0
    key_padding_mask = None
    mask = None
    kwargs["key_padding_mask"] = key_padding_mask
    kwargs["edge_index"] = edge_index
    kwargs["coords"] = coords
    #x, _ = pad_to_multiple0(x, multiple=chunk_size)
    with torch.no_grad():
        kwargs["raw_size"] = x.shape[0]
        discretized_coords = torch.zeros((x.shape[0], 4), device=x.device)
        discretized_coords[:, -2:] = discretize_coords(coords[:, :2], B=helper_funcs["B"])
        mappings = helper_funcs["mapping"](discretized_coords, batch_size=1)
        kwargs["mappings"] = mappings
    return x, mask, kwargs


def prepare_input2(x, edge_index, coords, batch, helper_funcs):
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




def pad_to_multiple0(x: torch.Tensor, multiple: int, dim=1) -> torch.Tensor:
    """Pads the sequence length of tensor x to be a multiple of 'multiple'."""
    #seqlen = x.shape[1]
    seqlen = x.shape[dim]
    pad_len = (multiple - seqlen % multiple) % multiple
    if pad_len > 0:
        pad = (0, 0) * (x.dim() - 2) + (0, pad_len)  # Only pad the sequence length dimension
        x = F.pad(x, pad)
    return x, seqlen


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
        # vecs: (n_heads, N, dim)
        # returns (n_hashes, N, dim)
        #print(f'vecs {vecs.shape}')
        proj = torch.bmm(vecs, self.alpha)      # (n_heads, N, n_hashes)
        return proj.permute(2, 1, 0)            # (n_hashes, N, dim=heads)


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

        # --- 1) Prepare for hashing ---
        if coords is None:
            coords_for_hash = points         # use embeddings
            C = D
        else:
            coords_for_hash = coords         # use spatial coords
            _, _, C = coords.shape

        # Flatten to (1, B*L, C)
        flat = coords_for_hash.reshape(B * L, C).unsqueeze(0)

        # --- 2) LSH-map ---
        q_hashed, _, shift = lsh_mapping(self.e2lsh, flat, flat)
        #   q_hashed: (H, N, C), shift: (H, N, 1)

        # --- 3) Reduce to scalar scores per hash ---
        q_shifted = q_hashed + shift          # (H, N, C)
        scores    = q_shifted.sum(dim=-1)     # (H, N)

        # --- 4) Build bucket_matrix (B, L, H) ---
        bucket_matrix = (
            scores.view(self.num_hashes, B, L)
                  .permute(1, 2, 0)          # → (B, L, H)
        )

        # --- 5) Pack to single integer keys (B, L) ---
        exponents   = torch.tensor(
            [2**i for i in range(self.num_hashes)],
            device=points.device,
            dtype=torch.long
        )
        bucket_keys = (bucket_matrix * exponents).sum(dim=-1)  # (B, L)

        # --- 6) Two-stage stable sort per batch ---
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

        # --- 7) Reorder the original point embeddings (B, L, D) ---
        ordered_points = points.gather(
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, D)
        )

        return ordered_points, indices


class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=16,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=chunk_size,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
        custom = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx


        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, pe=None, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None, no_pad=False):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            if u.dim() == 2:
                u = u.unsqueeze(0)  
            if not no_pad:
                u, _ = pad_to_multiple0(u, multiple=chunk_size)
            batch, seqlen, dim = u.shape
            
            
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen
        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out
            
            
        if pe is not None:
            origlen = pe.size(0)
            pe = F.pad(pe, (0, 0, 0, seqlen - origlen))
            u = u + pe

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        
        
        
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
                
        else:
            print('use chunk')
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            assert self.activation in ["silu", "swish"]
            xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        if pe is not None:
            out = out[:,: origlen,:]    
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state
    
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

    
class HMamba2(nn.Module):
    """
    Experimental version (Remove NaN before using)
    """
    def __init__(self, in_dim, task, **kwargs):
   
        super(HMamba2, self).__init__()
        self.depth = kwargs["n_layers"]
        self.mlpdepth = kwargs["mlp_layers"]
        self.hidden_dim = kwargs["h_dim"]
        self.use_mem_eff_path = kwargs.get("use_mem_eff_path", True)
        self.freeze_backbone = kwargs.get("freeze_backbone", False)
        self.freeze_encoder = kwargs.get("freeze_encoder", False)
        self.freeze_head = kwargs.get("freeze_head", False)

        device = torch.device('cuda:0')

        self.task = task
            
        self.feature_encoder = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        self.drop_path = kwargs["drop_path"]
        self.drop_out_in_block = kwargs["drop_out_in_block"]
        self.cls_dim = 2
        
        self.attns = nn.ModuleList()

        self.W = nn.Linear(int(self.hidden_dim // 2)* (self.mlpdepth + 1), int(self.hidden_dim // 2), bias=False)
        self.W0 = nn.Linear(int(self.hidden_dim), int(self.hidden_dim // 2), bias=False)
        self.mlp_out = MLP(
            in_channels=int(self.hidden_dim // 2),
            out_channels=int(self.hidden_dim // 2),
            hidden_channels=256,
            num_layers=3,
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
                                mixer=Mamba2(int(self.hidden_dim), use_mem_eff_path=self.use_mem_eff_path, device=device),
                                norm=RMSNorm(self.hidden_dim, device=device),
                            )
                        )
                        for _ in range(self.depth)
                    ]
                ),
                norm_f=RMSNorm(int(self.hidden_dim), device=device),
            )
        )

        self.norm = nn.LayerNorm(int(self.hidden_dim // 2))
        self.rmsnorm = RMSNorm(int(self.hidden_dim // 2), device=device)

        self.HEAD_CHANEL = 1
        print('using LSH ordering')
        self.lsh_order = LSHOrdering(input_dim=6, num_hashes=3)

        self.drop_out = nn.Dropout(kwargs["drop_out"]) # else nn.Dropout(0)
        
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.freeze_encoder:
            for p in self.feature_encoder.parameters():
                p.requires_grad = False
        if self.freeze_head:
            head_params = list(self.W.parameters()) + list(self.W0.parameters()) + list(self.mlp_out.parameters())
            for p in head_params:
                p.requires_grad = False

        # Print module freeze status
        if self.freeze_backbone or self.freeze_encoder or self.freeze_head:
            print("Module freeze status:")
            for name, flag in [
                ("Feature Encoder", self.freeze_encoder),
                ("Backbone", self.freeze_backbone),
                ("Head (W, W0, MLP)", self.freeze_head)
            ]:
                emoji = "🔒" if flag else "✅"
                print(f"{emoji} {name}")



    def forward(self, data):

        if isinstance(data, dict):
            input_features, edge_index, coords = data["x"], data["edge_index"], data["coords"]
        else:
            input_features, edge_index, coords = data.x, data.edge_index, data.coords
            
        encoded_features = self.feature_encoder(input_features)
        encoded_features      = torch.nan_to_num(encoded_features,     nan=0.0, posinf=0.0, neginf=0.0)
        x = encoded_features
        x, mask,coords, kwargs = prepare_input(x, coords, edge_index)
        x      = torch.nan_to_num(x,     nan=0.0, posinf=0.0, neginf=0.0)

        n = x.size()[-2]
        d = x.size()[-1]
        x = x.view(1,n,d)


        coords = coords.unsqueeze(0)

        
        x, perm = self.lsh_order(x, coords)  # (B, L, D), (B, L)

        
        x      = torch.nan_to_num(x,     nan=0.0, posinf=0.0, neginf=0.0)
        coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

        
        
        h = None
        if h is None:
            h = [None for _ in range(self.depth)]
        
        
        ## to-do: try different combination for attention conv+mamba, the most computing expensive way is to add per layer, but could do jamba way, add 1 or 2 attention and mambamoe.
        for i, layer in enumerate(self.backbone.layers):

            y = layer.mixer(layer.norm(x))
            x = y + x
            
        inv_perm = perm.argsort(dim=1)  # (B, L)
        x = x.gather(
            1,
            inv_perm.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )

        x = self.backbone.norm_f(x)
        
        x = x.view(n,d)
        
        x = self.W0(x)
        
        
        
        out = x + self.drop_out(self.mlp_out(x))

        out = out[: kwargs["raw_size"],:]
        return out

    
    
class FullMamba2(nn.Module):
    def __init__(self, in_dim, task, **kwargs):
        super(FullMamba2, self).__init__()
        self.depth = kwargs["n_layers"]
        self.mlpdepth = kwargs["mlp_layers"]
        self.hidden_dim = kwargs["h_dim"]
        self.use_mem_eff_path = kwargs.get("use_mem_eff_path", True)
        self.freeze_backbone = kwargs.get("freeze_backbone", False)
        self.freeze_encoder = kwargs.get("freeze_encoder", False)
        self.freeze_head = kwargs.get("freeze_head", False)

        device = torch.device('cuda:0')

        self.task = task
            
        self.feature_encoder = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            

        self.drop_path = kwargs["drop_path"]
        self.drop_out_in_block = kwargs["drop_out_in_block"]
        self.cls_dim = 2
        
        self.attns = nn.ModuleList()

        self.W = nn.Linear(int(self.hidden_dim // 2)* (self.mlpdepth + 1), int(self.hidden_dim // 2), bias=False)
        self.W0 = nn.Linear(int(self.hidden_dim), int(self.hidden_dim // 2), bias=False)
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
                                mixer=Mamba2(int(self.hidden_dim), use_mem_eff_path=self.use_mem_eff_path, device=device),
                                norm=RMSNorm(self.hidden_dim, device=device),
                            )
                        )
                        for _ in range(self.depth)
                    ]
                ),
                norm_f=RMSNorm(int(self.hidden_dim), device=device),
            )
        )

        self.norm = nn.LayerNorm(int(self.hidden_dim // 2))
        self.rmsnorm = RMSNorm(int(self.hidden_dim // 2), device=device)

        self.HEAD_CHANEL = 1


        self.drop_out = nn.Dropout(kwargs["drop_out"]) # else nn.Dropout(0)
        
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.freeze_encoder:
            for p in self.feature_encoder.parameters():
                p.requires_grad = False
        if self.freeze_head:
            head_params = list(self.W.parameters()) + list(self.W0.parameters()) + list(self.mlp_out.parameters())

            for p in head_params:
                p.requires_grad = False

        # Print module freeze status
        if self.freeze_backbone or self.freeze_encoder or self.freeze_head:
            print("Module freeze status:")
            for name, flag in [
                ("Feature Encoder", self.freeze_encoder),
                ("Backbone", self.freeze_backbone),
                ("Head (W, W0, MLP)", self.freeze_head)
            ]:
                emoji = "🔒" if flag else "✅"
                print(f"{emoji} {name}")
        



    def forward(self, data):

        if isinstance(data, dict):
            input_features, edge_index, coords = data["x"], data["edge_index"], data["coords"]
        else:
            input_features, edge_index, coords = data.x, data.edge_index, data.coords
     
        encoded_features = self.feature_encoder(input_features)
        x = encoded_features  
        x, mask,coords, kwargs = prepare_input(x, coords, edge_index)
  
        h = None
        if h is None:
            h = [None for _ in range(self.depth)]
        
        n = x.size()[-2]
        d = x.size()[-1]
        
        for i, layer in enumerate(self.backbone.layers):

            x = x.view(1,n,d)
            
            y = layer.mixer(layer.norm(x))
            x = y + x

        x = self.backbone.norm_f(x)
        
        x = x.view(n,d)
        
        x = self.W0(x)
   
        out = x + self.drop_out(self.mlp_out(x))

        out = out[: kwargs["raw_size"],:]
        return out    
    
    
class FlatMamba(nn.Module):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_dim = kwargs["h_dim"]
        self.d_state = kwargs["d_state"]
        self.headdim = kwargs["headdim"]
        self.use_mem_eff_path = kwargs.get("use_mem_eff_path", True)

        self.block = nn.ModuleList()
        for _ in range(4):
            layer = Mamba2(d_model=int(self.hidden_dim),d_state=self.d_state,headdim=self.headdim, use_mem_eff_path=self.use_mem_eff_path)
            self.block.append(layer)

    def forward(self, x: torch.Tensor, pe: torch.Tensor, mappings: Dict[str, Any], **kwargs) -> torch.Tensor:
        all_x = []
        for k, name in enumerate(["x", "x_shift", "y", "y_shift"]):
            indices = mappings[name]
            x[indices] = self.block[k](
                u=x[indices][mappings["flat2win"]],
                pe=pe[indices][mappings["flat2win"]],
            ).squeeze(0)[mappings["win2flat"]]
            all_x.append(x)
        return x, all_x
    
class PEFullMamba2(nn.Module):
    def __init__(self, in_dim,coords_dim, task, **kwargs):
        super(PEFullMamba2, self).__init__()
        self.depth = kwargs["n_layers"]
        self.mlpdepth = kwargs["mlp_layers"]
        self.hidden_dim = kwargs["h_dim"]
        device = torch.device('cuda:0')

        self.task = task

        self.feature_encoder = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        self.drop_path = kwargs["drop_path"]
        self.drop_out_in_block = kwargs["drop_out_in_block"]
        self.cls_dim = 2
        
        self.attns = nn.ModuleList()
        
        self.helper_funcs = {}
        self.helper_funcs["B"] = kwargs["B"]
        self.helper_funcs["mapping"] = FlattenedWindowMapping(**kwargs)
        self.W = nn.Linear(self.hidden_dim * (self.depth * 4 + 1), int(self.hidden_dim // 2), bias=False)
        
        self.W0 = nn.Linear(int(self.hidden_dim), int(self.hidden_dim), bias=False)
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
                                mixer=FlatMamba(**kwargs),
                                norm=RMSNorm(self.hidden_dim, device=device),
                            )
                        )
                        for _ in range(self.depth)
                    ]
                ),
                norm_f=RMSNorm(int(self.hidden_dim // 2), device=device),
            )
        )

        self.norm = nn.LayerNorm(int(self.hidden_dim // 2))
        self.rmsnorm = RMSNorm(int(self.hidden_dim // 2), device=device)

        self.HEAD_CHANEL = 1

        self.drop_out = nn.Dropout(kwargs["drop_out"]) # else nn.Dropout(0)
        self.pe_func = get_pe_func(kwargs["pe_type"], coords_dim, kwargs)



    def forward(self, data):

        if isinstance(data, dict):
            input_features, edge_index, coords, batch = data["x"], data["edge_index"], data["coords"], data["batch"]
        else:
            input_features, edge_index, coords, batch = data.x, data.edge_index, data.coords , data["batch"]

        input_features, mask, kwargs = prepare_input1(input_features, edge_index, coords, batch, self.helper_funcs)
            
        
        encoded_features = self.feature_encoder(input_features)


        x = encoded_features
        
        
        
        pe = kwargs["coords"] if self.pe_func is None else self.pe_func(kwargs["coords"])
        

        
        n = x.size()[-2]
        d = x.size()[-1]
        
        all_encoded_x = [x]  

        for i, layer in enumerate(self.backbone.layers):                          

            encoded, shift_list = layer.mixer(x, pe=pe, **kwargs)
            all_encoded_x = all_encoded_x + shift_list     
            y = encoded             

            x = x + y
        
        
        x = torch.cat(all_encoded_x, dim=-1)
        x = self.W(x) 
        
        x = self.backbone.norm_f(x)

        
        
        out = x + self.drop_out(self.mlp_out(x))


        out = out.view(-1,self.hidden_dim//2)
        out = out[: kwargs["raw_size"],:]
        


        return out


class FullHybridMamba2(nn.Module):
    def __init__(self, in_dim, task, **kwargs):
        super(FullHybridMamba2, self).__init__()
        self.depth = kwargs["n_layers"]
        self.mlpdepth = kwargs["mlp_layers"]
        self.hidden_dim = kwargs["h_dim"]
        device = torch.device('cuda:0')
        
        self.use_mem_eff_path = kwargs.get("use_mem_eff_path", True)

        self.task = task

            
        self.feature_encoder = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        self.drop_path = kwargs["drop_path"]
        self.drop_out_in_block = kwargs["drop_out_in_block"]
        self.cls_dim = 2
        
        self.attns = nn.ModuleList()

        self.W = nn.Linear(int(self.hidden_dim // 2)* (self.mlpdepth + 1), int(self.hidden_dim // 2), bias=False)
        self.W0 = nn.Linear(int(self.hidden_dim), int(self.hidden_dim // 2), bias=False)
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
                                mixer=Mamba2(int(self.hidden_dim), use_mem_eff_path=self.use_mem_eff_path,device=device),
                                norm=RMSNorm(self.hidden_dim, device=device),
                            )
                        )
                        for _ in range(self.depth)
                    ]
                ),
                norm_f=RMSNorm(int(self.hidden_dim), device=device),
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

        input_features, mask, kwargs = prepare_input2(input_features, edge_index, coords, batch, self.helper_funcs)

         
        encoded_features = self.feature_encoder(input_features)   
        x = encoded_features
        
        
        h = None
        if h is None:
            h = [None for _ in range(self.depth)]
        
        n = x.size()[-2]
        d = x.size()[-1]
        ## to-do: try different combination for attention conv+mamba, the most computing expensive way is to add per layer, but could do jamba way, add 1 or 2 attention and mambamoe.
        total = len(self.backbone.layers)
        mid = total // 2
        if total % 2 == 0:
            hybrid_idxs = {mid - 1, mid}
        else:
            hybrid_idxs = {mid}

        for i, layer in enumerate(self.backbone.layers):
            x = x.view(1, n, d)
            y = layer.mixer(layer.norm(x), no_pad=True)

            if i in hybrid_idxs:
                x = x.view(n, d)
                x = self.lsh_attention(x, kwargs).view(1, n, d)
                x = y + x
            else:
                x = y + x



        x = self.backbone.norm_f(x)
        
        x = x.view(n,d)
        
        x = self.W0(x)

        
        out = x + self.drop_out(self.mlp_out(x))
 
        out = out[: kwargs["raw_size"],:]

        return out
    
class FullFullHybridMamba2(nn.Module):
    def __init__(self, in_dim, task, **kwargs):
        super(FullFullHybridMamba2, self).__init__()
        self.depth = kwargs["n_layers"]
        self.mlpdepth = kwargs["mlp_layers"]
        self.hidden_dim = kwargs["h_dim"]
        device = torch.device('cuda:0')
        
        self.use_mem_eff_path = kwargs.get("use_mem_eff_path", True)

        self.task = task
            
        self.feature_encoder = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        self.drop_path = kwargs["drop_path"]
        self.drop_out_in_block = kwargs["drop_out_in_block"]
        self.cls_dim = 2
        
        self.attns = nn.ModuleList()

        self.W = nn.Linear(int(self.hidden_dim // 2)* (self.mlpdepth + 1), int(self.hidden_dim // 2), bias=False)
        self.W0 = nn.Linear(int(self.hidden_dim), int(self.hidden_dim // 2), bias=False)
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
                                mixer=Mamba2(int(self.hidden_dim), use_mem_eff_path=self.use_mem_eff_path,device=device),
                                norm=RMSNorm(self.hidden_dim, device=device),
                            )
                        )
                        for _ in range(self.depth)
                    ]
                ),
                norm_f=RMSNorm(int(self.hidden_dim), device=device),
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

        input_features, mask, kwargs = prepare_input2(input_features, edge_index, coords, batch, self.helper_funcs)

         
        encoded_features = self.feature_encoder(input_features)   
        x = encoded_features
        
        
        h = None
        if h is None:
            h = [None for _ in range(self.depth)]
        
        n = x.size()[-2]
        d = x.size()[-1]
       
        for i, layer in enumerate(self.backbone.layers):
            x = x.view(1, n, d)
            y = layer.mixer(layer.norm(x), no_pad=True)

            x = x.view(n, d)
            x = self.lsh_attention(x, kwargs).view(1, n, d)
            x = y + x



        x = self.backbone.norm_f(x)
        
        x = x.view(n,d)
        
        x = self.W0(x)

        out = x + self.drop_out(self.mlp_out(x))
 
        out = out[: kwargs["raw_size"],:]

        return out
    
class LSHAtt(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]

        self.w_q = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_k = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_v = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        coords_dim = kwargs.get("coords_dim", 6)
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
    