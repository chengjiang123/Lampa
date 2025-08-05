import torch, types, os, gc, math
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


from torch.utils.cpp_extension import load
from torch_geometric.nn import MLP
from ..attention import (
    PCTAttention,HEPTAttention,
)
from einops import rearrange
from ..model_utils.hash_utils import pad_to_multiple, get_regions, quantile_partition
from ..model_utils.hash_utils import lsh_mapping, batched_index_select, invert_permutation, E2LSH
import yaml
from pathlib import Path

def __nop(ob):
    return ob

def prepare_input0(x, coords, edge_index):
    
    n, d = x.shape
    
    kwargs = {}
    
    key_padding_mask = None
    mask = None
    kwargs["key_padding_mask"] = key_padding_mask
    kwargs["edge_index"] = edge_index
    kwargs["coords"] = coords
    with torch.no_grad():
        kwargs["raw_size"] = x.shape[0]
    
    # Calculate the number of rows to add to make n % 16 == 0
    # 16 is the chunk_size which rwkv_cuda requires
    pad_rows = (16 - n % 16) % 16  # This ensures no extra padding if already divisible by 16

    if pad_rows > 0:
        # Pad x with rows of zeros
        padding = torch.zeros((pad_rows, d), device=x.device, dtype=x.dtype)
        x = torch.cat([x, padding], dim=0)
    

    return x, mask, kwargs

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
        region_indices_eta = quantile_partition(sorted_eta_idx, regions_h[0][:, None])
        region_indices_phi = quantile_partition(sorted_phi_idx, regions_h[1][:, None])
        kwargs["region_indices"] = [region_indices_eta, region_indices_phi]
        kwargs["regions_h"] = regions_h
        kwargs["coords"][kwargs["raw_size"] :] = 0.0
        coords[kwargs["raw_size"] :] = 0.0
    return x, coords, mask, kwargs



head_size_a = 16 # don't change  # change from 64 to 32 need to be hidden%4 == 0 ? every time need to change the namespace in wkv5.cpp to compile unique
head_size_divisor = 8 # don't change
MyFunction = __nop
base_path = "../SWAN_projects/trackmamba/HEPT-main/src/models/rwkv/"
## prepare input

#wkv6_cuda = load(name="wkv6", sources=["/eos/user/c/chjiang/SWAN_projects/trackmamba/HEPT-main/src/models/baselines/cuda/wkv6_op.cpp", f"/eos/user/c/chjiang/SWAN_projects/trackmamba/HEPT-main/src/models/baselines/cuda/wkv6_cuda.cu"],
#                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={head_size_a}", f"-D_T_={1024}"])

# Cache for compiled modules
compiled_modules = {}

def get_wkv6_cuda(ctx_len, head_size):
    key = (ctx_len, head_size)
    if key in compiled_modules:
        return compiled_modules[key]

    extra_cuda_cflags = [
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
        f"-D_N_={head_size}",
        f"-D_T_={ctx_len}"
    ]
    
    module = load(
        name="wkv6",
        sources=["../SWAN_projects/trackmamba/HEPT-main/src/models/rwkv/cuda/wkv6_op.cpp", "../SWAN_projects/trackmamba/HEPT-main/src/models/rwkv/cuda/wkv6_cuda.cu"],
        verbose=True,
        extra_cuda_cflags=extra_cuda_cflags
    )
    
    compiled_modules[key] = module
    return module

class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u): # forward: r, k, v, w, u => y
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert head_size_a == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ctx.save_for_backward(r, k, v, w, u)

            #wkv6_cuda = get_wkv6_cuda(T, head_size_a)

            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            torch.ops.wkv6.forward(B, T, C, H, r, k, v, w, u, y)
            return y

    @staticmethod
    def backward(ctx, gy): # backward: gy => gr, gk, gv, gw, gu
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, w, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            #wkv6_cuda = get_wkv6_cuda(T, head_size_a)
            torch.ops.wkv6.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu) # return gradients for r,k,v,w,u

def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)



wkv5_cuda = load(name="wkv5", sources=["../SWAN_projects/trackmamba/HEPT-main/src/models/rwkv/cuda/wkv5_op.cpp", f"../SWAN_projects/trackmamba/HEPT-main/src/models/rwkv/cuda/wkv5_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={head_size_a}"])
        
class WKV_5(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert head_size_a == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            eew = (torch.exp(ew)).contiguous()
            ctx.save_for_backward(r, k, v, eew, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, eew, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
            wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
            gw = torch.sum(gw, 0).view(H, C//H)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
    return WKV_5.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################
# RWKV TimeMix
########################################################################################################

class RWKV_Tmix_x060(nn.Module):
    def __init__(self, layer_id, **kwargs):
        super().__init__()
        #self.args = args
        self.layer_id = layer_id

        self.n_embd = kwargs['h_dim'] # 128 larger?
        self.dim_att = self.n_embd
        self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)
        self.n_layer = kwargs["n_layers"]

        self.head_size = kwargs["head_size_a"]
        self.n_head = self.dim_att // self.head_size
        assert self.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (self.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                ddd[0, 0, i] = i / self.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            #self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            #self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            #self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))
            self.time_maa_w1 = nn.Parameter(torch.zeros(self.n_embd, D_MIX_LORA*4))
            self.time_maa_w2 = nn.Parameter(torch.zeros(4, D_MIX_LORA, self.n_embd).uniform_(-0.01, 0.01))
            # fancy time_decay
            decay_speed = torch.ones(self.dim_att)
            for n in range(self.dim_att):
                decay_speed[n] = -6 + 5 * (n / (self.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, self.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(self.dim_att)
            for n in range(self.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (self.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.key = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.value = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.output = nn.Linear(self.dim_att, self.n_embd, bias=False)
        #self.gate = nn.Linear(self.n_embd, self.dim_att, bias=False)
        #self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))
        self.ln_x = nn.LayerNorm(self.dim_att)

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        #xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        #xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        #mw, mk, mv, mr, mg = xxx.unbind(dim=0)
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(4, B, T, -1)
        mw, mk, mv, mr = xxx.unbind(dim=0)


        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        #xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr).to(torch.bfloat16)
        k = self.key(xk).to(torch.bfloat16)
        v = self.value(xv).to(torch.bfloat16)
        #g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = (self.time_decay + ww).to(torch.bfloat16)
        
        u=self.time_faaaa.to(torch.bfloat16)

        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u)
        #x = RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u)
        
        x = x.to(torch.float32)
        x = self.ln_x(x)
        x = self.output(x)	

        return x



    
class RWKV_Tmix_x052(nn.Module):
    def __init__(self, layer_id, **kwargs):
        super().__init__()
        #self.args = args
        self.layer_id = layer_id

        self.n_embd = kwargs['h_dim'] # 128 larger?
        self.dim_att = self.n_embd
        self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)
        self.n_layer = kwargs["n_layers"]

        self.head_size = kwargs["head_size_a"]
        #assert head_size == self.head_size # change HEAD_SIZE to match self.head_size_a
        self.n_head = self.dim_att // self.head_size
        assert self.dim_att % self.n_head == 0
        self.head_size_divisor = 8

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (self.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                ddd[0, 0, i] = i / self.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            
            ###
            self.time_mix_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            D_MIX_LORA = 16 # generate TIME_MIX for w,k,v,r,g
            self.time_mix_w1 = nn.Parameter(torch.zeros(self.n_embd, D_MIX_LORA*5))
            self.time_mix_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, self.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(self.dim_att)
            for n in range(self.dim_att):
                decay_speed[n] = -6 + 5 * (n / (self.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            #self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.dim_att))
        
            
            ###
            D_DECAY_LORA = 32
            self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, self.head_size).uniform_(-0.01, 0.01))

            tmp = torch.zeros(self.dim_att)
            for n in range(self.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (self.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.key = nn.Linear(self.n_embd, self.dim_att, bias=False)

        self.value = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.output = nn.Linear(self.dim_att, self.n_embd, bias=False)
        self.gate = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, self.dim_att)
        #self.ln_x = nn.LayerNorm(self.dim_att)

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        #xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        #xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        #xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        #xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        #xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)
        
        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_mix_x
        xxx = torch.tanh(xxx @ self.time_mix_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_mix_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)


        xw = x + xx * (self.time_mix_w + mw)
        xk = x + xx * (self.time_mix_k + mk)
        xv = x + xx * (self.time_mix_v + mv)
        xr = x + xx * (self.time_mix_r + mr)
        xg = x + xx * (self.time_mix_g + mg)

        r = self.receptance(xr).to(torch.bfloat16)
        k = self.key(xk).to(torch.bfloat16)
        v = self.value(xv).to(torch.bfloat16)
        g = F.silu(self.gate(xg)).to(torch.bfloat16)
        
        #ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        
        #w = self.time_decay + ww
        #w = w.to(torch.bfloat16)

        return r, k, v, g

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        x = x.to(torch.float32)
        x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g = self.jit_func(x)

        x = RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w=self.time_decay.to(torch.bfloat16), u=self.time_faaaa.to(torch.bfloat16))

        return self.jit_func_2(x, g)    
########################################################################################################
# RWKV ChannelMix
########################################################################################################

class RWKV_CMix_x060(nn.Module):
    def __init__(self, layer_id,  **kwargs):
        super().__init__()
        #self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.n_embd = kwargs['h_dim'] # 128 larger?
        self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)
        self.n_layer = kwargs["n_layers"]

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                ddd[0, 0, i] = i / self.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(self.n_embd, self.dim_ffn, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.dim_ffn, self.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
    


########################################################################################################
# RWKV Block
########################################################################################################

class Block(nn.Module):
    def __init__(self, layer_id, **kwargs):
        super().__init__()
        #self.args = args
        self.layer_id = layer_id

        self.n_embd = kwargs['h_dim']

        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(self.n_embd)

        #self.att = RWKV_Tmix_x060(layer_id, **kwargs)
        self.att = RWKV_Tmix_x052(layer_id, **kwargs)
        self.ffn = RWKV_CMix_x060(layer_id, **kwargs)
        
    def forward(self, x, kwargs):

        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x

########################################################################################################
# RWKV Model
########################################################################################################

class RWKV(nn.Module):
    def __init__(self, in_dim, task, **kwargs):
        super().__init__()
        #self.args = args
        self.n_embd = kwargs['h_dim'] # 256 larger?
        self.dim_att = self.n_embd
        self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)

        assert self.n_embd % 32 == 0
        assert self.dim_att % 32 == 0
        assert self.dim_ffn % 32 == 0
        
        self.vocab_size = in_dim
        self.task = task

        self.n_layer = kwargs["n_layers"]

        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, self.n_embd),
            nn.ReLU(),
            nn.Linear(self.n_embd, self.n_embd),
        )
        
        
        
        self.blocks = nn.ModuleList([Block(i,**kwargs) for i in range(self.n_layer)])

        self.ln_out = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, int(self.n_embd // 2), bias=False)

        self.init_params() # !!! When you train RWKV from scratch, try my initialization for best performance !!!
        
        self.dropout = nn.Dropout(0.1)
        self.mlp_out = MLP(
            in_channels=int(self.n_embd // 2),
            out_channels=int(self.n_embd // 2),
            hidden_channels=256,
            num_layers=4,
            norm="layer_norm",
            act="relu",
            norm_kwargs={"mode": "node"},
        )

        self.helper_funcs = {}


    def forward(self, data):

        if isinstance(data, dict):
            x, edge_index, coords, batch, self.use_ckpt = data["x"], data["edge_index"], data["coords"], data["batch"], False
        else:
            x, edge_index, coords, batch = data.x, data.edge_index, data.coords, data.batch


        x, mask, kwargs = prepare_input0(x, coords, edge_index)
        x = self.feat_encoder(x)
  
        
        x = x.unsqueeze(0)


        for block in self.blocks:
            x = block(x,kwargs)
            
        b,n,d = x.shape

        x = self.ln_out(x)
        x = self.head(x)

        
        out = x + self.dropout(self.mlp_out(x))

 
        out = out.view(-1,int(self.n_embd // 2))
            
            
        out = out[: kwargs["raw_size"],:]

        return out
    
    def init_params(self):
        m = self.state_dict()
        n_params = 0
        for n in self.state_dict():
            p = m[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias'):
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                print()
            #elif n == "emb.weight":
            #    m[n] = p
            #    scale = -1e-4
            #    nn.init.uniform_(m[n], a=scale, b=-scale) # !!! If you are using positional embedding, maybe it's better to remove block.0.ln0, and use default initialization for emb.weight instead of my uniform_(a=-1e-4, b=1e-4) !!!
            #    print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.vocab_size > self.n_embd:
                    scale = 0.5 * math.sqrt(self.vocab_size / self.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                #assert n.endswith('.weight') # should always be true

                for kk in [".att.output.", ".ffn.value.", ".ffn.receptance."]:
                    if kk in n:
                        scale = 0
                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                #m[n] = torch.empty((shape[0], shape[1]), device=p.device)
                #if scale == 0:
                #    nn.init.zeros_(m[n])
                #else:
                #    nn.init.orthogonal_(m[n], gain=scale)
                if len(shape) == 2:
                    m[n] = torch.empty((shape[0], shape[1]), device=p.device)
                    if scale == 0:
                        nn.init.zeros_(m[n])
                    else:
                        nn.init.orthogonal_(m[n], gain=scale)
                elif len(shape) == 1:
                    m[n] = torch.empty((shape[0],), device=p.device)
                    if scale == 0:
                        nn.init.zeros_(m[n])
                else:
                    m[n] = torch.empty(shape, device=p.device)
                    nn.init.zeros_(m[n])  # Default initialization for unexpected shapes


            n_params += m[n].numel()
        
        print('model params', n_params)
        gc.collect()
        torch.cuda.empty_cache()
        
        
        
