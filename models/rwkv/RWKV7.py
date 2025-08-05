import torch, types, os, gc, math
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


from torch.utils.cpp_extension import load
from torch_geometric.nn import MLP

import yaml
from pathlib import Path

def __nop(ob):
    return ob

def prepare_input(x, coords, edge_index):
    
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



head_size_a = 16 # don't change  # change from 64 to 32 need to be hidden%4 == 0 ? every time need to change the namespace in wkv5.cpp to compile unique
head_size_divisor = 8 # don't change
MyFunction = __nop
base_path = "../SWAN_projects/trackmamba/HEPT-main/src/models/rwkv/"

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
        sources=[" ../SWAN_projects/trackmamba/HEPT-main/src/models/rwkv/cuda/wkv6_op.cpp", "../SWAN_projects/trackmamba/HEPT-main/src/models/rwkv/cuda/wkv6_cuda.cu"],
        verbose=True,
        extra_cuda_cflags=extra_cuda_cflags
    )
    
    compiled_modules[key] = module
    return module



DTYPE = torch.half
#load(name="wkv7", sources=["../SWAN_projects/trackmamba/HEPT-main/src/models/rwkv/cuda/wkv7_op.cpp", f"../SWAN_projects/trackmamba/HEPT-main/src/models/rwkv/cuda/wkv7.cu"], is_python_module=False,
#                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={head_size_a}"])

load(name="wind",sources=['../SWAN_projects/trackmamba/HEPT-main/src/models/rwkv/cuda/wind_rwkv7.cu','../SWAN_projects/trackmamba/HEPT-main/src/models/rwkv/cuda/wind_rwkv7.cpp'],is_python_module=False,verbose=True,extra_cuda_cflags=[f'-D_C_={head_size_a}',"-res-usage", "-gencode=arch=compute_80,code=sm_80", "--use_fast_math","-O3","-Xptxas=-O3","--extra-device-vectorization"])

        
class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // head_size_a
            N = head_size_a
            assert head_size_a == C // H
            assert r.dtype == DTYPE
            assert w.dtype == DTYPE
            assert k.dtype == DTYPE
            assert v.dtype == DTYPE
            assert a.dtype == DTYPE
            assert b.dtype == DTYPE
            assert r.is_contiguous()
            assert w.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert a.is_contiguous()
            assert b.is_contiguous()
            ctx.save_for_backward(r, k, v, w, a, b)
            y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
            torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
            return y
        
    @staticmethod
    def backward(ctx, gy): # backward: gy => gr, gk, gv, gw, gu
        with torch.no_grad():
            assert gy.dtype == torch.half
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, w, a, b = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.half, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.half, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.half, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.half, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            ga = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.half, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gb = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.half, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            torch.ops.wkv7.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, ga, gb)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu) # return gradients for r,k,v,w,u
        
        
class WindRWKV7(torch.autograd.Function):
    @staticmethod
    def forward(ctx,w,q,k,v,a,b):
        B,T,H,C = w.shape
        s0 = torch.zeros(B,H,C,C,dtype=w.dtype,device=w.device)
        assert T%16 == 0
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,a,b,s0])
        w,q,k,v,a,b,s0 = [i.contiguous() for i in [w,q,k,v,a,b,s0]]
        y = torch.empty_like(v)
        sT = torch.empty_like(s0)
        s = torch.zeros(B,H,T//16,C,C, dtype=w.dtype,device=w.device)
        torch.ops.wind.forward(w,q,k,v,a,b, s0,y,s,sT)
        ctx.save_for_backward(w,q,k,v,a,b,s)
        return y

    @staticmethod
    def backward(ctx,dy):
        w,q,k,v,a,b,s = ctx.saved_tensors
        B,T,H,C = w.shape
        dsT = torch.zeros(B,H,C,C,dtype=dy.dtype,device=dy.device)
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        dy,dsT = [i.contiguous() for i in [dy,dsT]]
        dw,dq,dk,dv,da,db,ds0 = [torch.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
        torch.ops.wind.backward(w,q,k,v,a,b, dy,s,dsT, dw,dq,dk,dv,da,db,ds0)
        return dw,dq,dk,dv,da,db

def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//head_size_a,head_size_a) for i in [q,w,k,v,a,b]]
    return WindRWKV7.apply(w,q,k,v,a,b).view(B,T,HC)

#def RUN_CUDA_RWKV7(r, w, k, v, a, b):
#    return WKV_7.apply(r, w, k, v, a, b)

########################################################################################################
# RWKV TimeMix
########################################################################################################

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, layer_id, **kwargs):
        super().__init__()
        #self.args = args
        self.n_embd = kwargs['h_dim'] # 128 larger?
        self.dim_att = self.n_embd
        #self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)
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
                
                
                
            # initialization comes from fitting my RWKV-6 7B runs
            # merging r&g w&a to save params
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0 ** 0.9))
            self.time_maa_rg = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.time_maa_wa = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))

            decay_speed = torch.ones(self.dim_att)
            for n in range(self.dim_att):
                decay_speed[n] = -7 + 5 * (n / (self.dim_att - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.dim_att) + 0.5) # !!! 0.5 comes from F.softplus !!!

            self.time_faaaa = nn.Parameter(torch.zeros(1,1,self.n_head,self.head_size))
            self.time_aaaaa = nn.Parameter(torch.zeros(1,1,self.dim_att))
            
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x
            
            

            D_MIX_LORA = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(self.n_embd, D_MIX_LORA*4))
            self.time_maa_w2 = nn.Parameter(ortho_init(torch.zeros(4, D_MIX_LORA, self.n_embd), 0.1))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, self.dim_att), 0.1))

            D_AAA_LORA = 16
            self.time_aaa_w1 = nn.Parameter(torch.zeros(self.n_embd, D_AAA_LORA))
            self.time_aaa_w2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, self.dim_att), 0.1))

            D_KKK_LORA = 16
            self.time_kkk_w1 = nn.Parameter(torch.zeros(self.n_embd, D_KKK_LORA))
            self.time_kkk_w2 = nn.Parameter(ortho_init(torch.zeros(D_KKK_LORA, self.dim_att), 0.1))

            D_GATE_LORA = 128
            self.gate_w1 = nn.Parameter(ortho_init(torch.zeros(self.n_embd, D_GATE_LORA), 0.1))
            self.gate_w2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, self.dim_att), 0.1))

            D_MA_LORA = 16
            self.ma_w1 = nn.Parameter(torch.zeros(self.n_embd, D_MA_LORA))
            self.ma_w2 = nn.Parameter(ortho_init(torch.zeros(D_MA_LORA, self.dim_att), 0.1))
            self.time_misc_a = nn.Parameter(torch.zeros(1,1,self.n_embd))
            D_MK_LORA = 16
            self.mk_w1 = nn.Parameter(torch.zeros(self.n_embd, D_MK_LORA))
            self.mk_w2 = nn.Parameter(ortho_init(torch.zeros(D_MK_LORA, self.dim_att), 0.1))
            self.time_misc_k = nn.Parameter(torch.zeros(1,1,self.n_embd))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.key = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.value = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.output = nn.Linear(self.dim_att, self.n_embd, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, self.dim_att, eps=64e-5)


    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(4, B, T, -1)
        mrg, mwa, mk, mv = xxx.unbind(dim=0)

        xrg = x + xx * (self.time_maa_rg + mrg)
        xwa = x + xx * (self.time_maa_wa + mwa)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)

        r = self.receptance(xrg).to(torch.bfloat16)
        w = (-F.softplus(-(self.time_decay + torch.tanh(xwa @ self.time_decay_w1) @ self.time_decay_w2)) - 0.5).to(torch.bfloat16)
        k = self.key(xk).to(torch.bfloat16)
        v = self.value(xv).to(torch.bfloat16)
        g = (torch.tanh(xrg @ self.gate_w1) @ self.gate_w2).to(torch.bfloat16)

        kk = k + torch.tanh(xk @ self.time_kkk_w1) @ self.time_kkk_w2
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C).to(torch.bfloat16)
        a = torch.sigmoid( self.time_aaaaa + (xwa @ self.time_aaa_w1) @ self.time_aaa_w2 ).to(torch.bfloat16) # a is "in-context learning rate"

        ma = torch.sigmoid(self.time_misc_a + (xwa @ self.ma_w1) @ self.ma_w2)
        k = k * ma + k*a * (1 - ma)
        mk = torch.sigmoid(self.time_misc_k + (xk @ self.mk_w1) @ self.mk_w2)
        k = (k * torch.clamp(w*mk, max=0).exp()).to(torch.bfloat16)
        
        
        return r, k, w, v, g, kk, a
    
    
    @MyFunction
    def jit_func_2(self, x, g, r, k, v, H):
        B, T, C = x.size()
        x = x.view(B * T, C)
        x = x.to(torch.float32)
        #x = x.bfloat16()


        try:
            x = self.ln_x(x).view(B, T, C)
        except Exception as e:
            print(f"Error in GroupNorm: {e}")
            raise
            
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x
        
    
    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, w, v, g, kk, a = self.jit_func(x)
        


        x = RUN_CUDA_RWKV7g(r.bfloat16(), w.bfloat16(), k.bfloat16(), v.bfloat16(), -kk.bfloat16(), (kk*a).bfloat16())

        return self.jit_func_2(x, g, r, k, v, H)   
        



    

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

        self.att = RWKV_Tmix_x070(layer_id, **kwargs)
        #self.att = RWKV_Tmix_x052(layer_id, **kwargs)
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

class RWKV7(nn.Module):
    def __init__(self, in_dim, task, **kwargs):
        super().__init__()
        #self.args = args
        self.n_embd = kwargs['h_dim'] # 256 larger?
        self.dim_att = self.n_embd
        #self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)

        assert self.n_embd % 32 == 0
        assert self.dim_att % 32 == 0
        #assert self.dim_ffn % 32 == 0
        
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
            num_layers=3,
            norm="layer_norm",
            act="relu",
            norm_kwargs={"mode": "node"},
        )

    def forward(self, data):

        if isinstance(data, dict):
            x, edge_index, coords, batch, self.use_ckpt = data["x"], data["edge_index"], data["coords"], data["batch"], False
        else:
            x, edge_index, coords, batch = data.x, data.edge_index, data.coords, data.batch

        x = self.feat_encoder(x)
        
        x, mask, kwargs = prepare_input(x, coords, edge_index)
        
        
        x = x.unsqueeze(0)
        

        for block in self.blocks:
            x = block(x,kwargs)

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
        
    
    
