import torch
#from models.baselines import GNNStack
from models.baselines import Transformer,HMamba_v1, HMamba_v2, HydraMamba2, FullMamba2, FullHybridMamba2, FullFullHybridMamba2, PEFullMamba2, HMamba2
from models.hattention import GatedDelta, GatedDelta_local1, LSHGatedDelta
#from models.baselines import Transformer,SegTransformer
#from models.rwkv import RWKV, RWKV7
from dataclasses import dataclass

def get_model(model_name, model_kwargs, dataset, test_N=10000, test_k=100):
    model_type = model_name.split("_")[0]
    print(f'data x shape {dataset.x_dim}')
    print(f'data coord shape {dataset.coords_dim}')
    print(f'model_type {model_type}')
    if model_type == "trans":
        model = Transformer(
            attn_type=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            coords_dim=dataset.coords_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "gnn":
        model = GNNStack(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            h_dim=model_kwargs["hidden_dim"],
            n_layers=model_kwargs["num_layers"],
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "rwkv":
        model = RWKV(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "rwkv7":
        model = RWKV7(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "gatedelta":
        model = GatedDelta(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "gdlocal1":
        model = GatedDelta_local1(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "lshgd":
        model = LSHGatedDelta(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "fullmamba2":
        model = FullMamba2(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "pemamba2":
        model = PEFullMamba2(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            coords_dim=dataset.coords_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "fullhybrid2":
        model = FullHybridMamba2(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "fullfullhybrid2":
        model = FullFullHybridMamba2(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "hmamba2":
        model = HMamba2(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "hydra":
        hydra_config = Mamba2Config()
        model = HydraMamba2(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            task=dataset.dataset_name,
            args=hydra_config,
            **model_kwargs,
        )
    elif model_type == "hmambav1":
        model = HMamba_v1(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    elif model_type == "hmambav2":
        model = HMamba_v2(
            model_name=model_name.split("_")[1],
            in_dim=dataset.x_dim,
            task=dataset.dataset_name,
            **model_kwargs,
        )
    else:
        raise NotImplementedError
    model.model_name = model_name
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    return model

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
