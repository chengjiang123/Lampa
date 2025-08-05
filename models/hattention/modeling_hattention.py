# Copyright 2024 state-spaces/mamba2 org and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MAMBA2 model (v0.2.1)."""

import math
import click
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.deprecation import deprecate_kwarg
from fla.modules import GatedMLP
from fla.models.mamba2.modeling_mamba2 import (
    logger,
    RMSNorm,
    RMSNormGated,
    Mamba2Cache,
    Mamba2Output,
    Mamba2CausalLMOutput,
    FusedCrossEntropyLoss,
    FusedLinearCrossEntropyLoss,
    causal_conv1d_fn,
    causal_conv1d_update,
    pad_tensor_by_size,
    is_fast_path_available)

from hattention.base import HType, HStruct, get_num_levels
from hattention.recurrent import HState
from hattention.mamba_apis import (
    LambdaLevelMLP,
    hselective_state_update,
    hmamba_chunk_scan_combined,
    hmamba_split_conv1d_scan_combined)
from hattention.configuration_hattention import HAttentionConfig

MAX_SEQUENCE_LENGTH = 2048 * 8
LAMBDA_LEVEL_BASE = 2
LAMBDA_HTYPE = HType.WEAK
LAMBDA_HSTRUCT = HStruct.MAMBA2
LAMBDA_LEVEL_FIXED = True
MAX_NUM_LEVELS = get_num_levels(
    length=MAX_SEQUENCE_LENGTH,
    base=LAMBDA_LEVEL_BASE)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


class HAttentionCache(Mamba2Cache):

    def __init__(
        self,
        config: HAttentionConfig,
        batch_size: int,
        dtype: torch.dtype = torch.float16,
        device: Optional[str] = None,
    ):
        self.dtype = dtype
        self.conv_kernel_size = config.conv_kernel
        self.n_groups = config.n_groups
        self.state_size = config.state_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.intermediate_size = int(config.expand * config.hidden_size)

        self.conv_states = {
            i: torch.zeros(
                batch_size,
                self.intermediate_size + 2 * config.n_groups * config.state_size,
                self.conv_kernel_size,
                device=device,
                dtype=dtype,
            )
            for i in range(config.num_hidden_layers)
        }
        self.hssm_states = {
            i: HState(
                base=LAMBDA_LEVEL_BASE,
                htype=LAMBDA_HTYPE,
                hstruct=LAMBDA_HSTRUCT,
                shape=(
                    batch_size,
                    config.num_heads,
                    config.state_size,
                    config.head_dim,
                    MAX_NUM_LEVELS,
                ),
                # we keep HSSM state in high precision
                dtype=torch.float32,
                device=device,
            )
            for i in range(config.num_hidden_layers)
        }

    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: torch.Tensor,
        cache_init: bool = False
    ) -> torch.Tensor:
        if new_conv_state.dtype != self.conv_states[layer_idx].dtype:
            warnings.warn(click.style(
                f"`new_conv_state.dtype` = {new_conv_state.dtype} -> "
                f"{self.conv_states[layer_idx].dtype}", fg="blue"))
            new_conv_state = new_conv_state.to(dtype=self.conv_states[layer_idx].dtype)

        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states[layer_idx].device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(self.conv_states[layer_idx].device)
        return self.conv_states[layer_idx]

    def update_hssm_state(self, layer_idx: int, new_hssm_state: HState) -> HState:
        self.hssm_states[layer_idx].replace(new_hssm_state)
        return self.hssm_states[layer_idx]

    def reset(self) -> None:
        for k in self.conv_states.keys():
            self.conv_states[k].zero_()
        for k in self.hssm_states.keys():
            self.hssm_states[k].reset_states()


class HAttentionMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: HAttentionConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        if LAMBDA_LEVEL_FIXED:
            self.num_lambda_dims = MAX_NUM_LEVELS
            self.lambda_level_module = None
        else:
            self.num_lambda_dims = 16
            self.lambda_level_module = LambdaLevelMLP(
                dim=self.num_lambda_dims,
                max_num_levels=MAX_NUM_LEVELS)

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads * (self.num_lambda_dims + 1)
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.lambda_mode = "positive"
        L = torch.ones(self.num_heads, self.num_lambda_dims)
        self.L = nn.Parameter(L)
        self.L._no_weight_decay = True

        self.norm = RMSNormGated(
            self.intermediate_size, eps=self.layer_norm_epsilon, norm_before_gate=False
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of "
                "`(selective_state_update, causal_conv1d_fn, causal_conv1d_update)` is None. "
                "Falling back to the naive implementation. "
                "To install follow https://github.com/state-spaces/mamba/#installation and"
                "https://github.com/Dao-AILab/causal-conv1d"
            )

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HAttentionCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if attention_mask is not None:
            # only supporting this in decoding
            if cache_params is None:
                raise NotImplementedError
        if self.activation not in ["silu", "swish"]:
            raise ValueError

        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        projected_states = self.in_proj(hidden_states)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads * (self.num_lambda_dims + 1)
        ) // 2
        if d_mlp != 0:
            raise ValueError

        # Single step calculations via cache
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            if hidden_states.shape[1] != 1:
                raise ValueError

            gate, xBC, dt, dl = torch.split(
                projected_states.squeeze(1),
                [
                    self.intermediate_size,
                    self.conv_dim,
                    self.num_heads,
                    self.num_heads * self.num_lambda_dims,
                ],
                dim=-1,
            )

            # 2. Convolution sequence transformation
            xBC = causal_conv1d_update(
                xBC,
                cache_params.conv_states[self.layer_idx],
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

            x, B, C = torch.split(
                xBC,
                [
                    self.intermediate_size,
                    groups_time_state_size,
                    groups_time_state_size,
                ],
                dim=-1,
            )

            # 3. SSM transformation
            A = -torch.exp(self.A_log.float())  # (nheads,)
            B = rearrange(
                B,
                "b (g n) -> b g n",
                b=batch_size,
                g=self.n_groups,
                n=self.ssm_state_size,
            )
            C = rearrange(
                C,
                "b (g n) -> b g n",
                b=batch_size,
                g=self.n_groups,
                n=self.ssm_state_size,
            )
            x_reshaped = rearrange(
                x,
                "b (h p) -> b h p",
                b=batch_size,
                h=self.num_heads,
                p=self.head_dim,
            )
            dl_reshaped = rearrange(
                dl,
                "b (h ell) -> b h ell",
                b=batch_size,
                h=self.num_heads,
                ell=self.num_lambda_dims,
            )
            y, hssm_state = hselective_state_update(
                cache_params.hssm_states[self.layer_idx],
                x_reshaped,
                dt=dt,
                A=A,
                B=B,
                C=C,
                dl=dl_reshaped,
                L=self.L,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                lambda_mode=self.lambda_mode,
                lambda_level_max=MAX_NUM_LEVELS,
                lambda_level_base=LAMBDA_LEVEL_BASE,
                lambda_htype=LAMBDA_HTYPE,
                lambda_hstruct=LAMBDA_HSTRUCT,
                lambda_level_fixed=LAMBDA_LEVEL_FIXED,
                lambda_level_module=self.lambda_level_module,
            )
            cache_params.update_hssm_state(
                layer_idx=self.layer_idx,
                new_hssm_state=hssm_state,
            )
            y = rearrange(
                y,
                "b h p -> b (h p)",
                b=batch_size,
                h=self.num_heads,
                p=self.head_dim,
            )
            y = self.norm(y, gate)

            # 4. Final linear projection
            out = self.out_proj(y)[:, None, ...]

        # Fused calculations or step by step if no initialized cache is found
        else:
            A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

            # 2-4. Fused kernel for conv1d, SSM, and the final projection
            if self.training and cache_params is None:
                out = torch.utils.checkpoint.checkpoint(
                    hmamba_split_conv1d_scan_combined,
                    use_reentrant=False,
                    # function arguments
                    zxbcdtdl=projected_states,
                    conv1d_weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    conv1d_bias=self.conv1d.bias,
                    dt_bias=self.dt_bias,
                    A=A,
                    L=self.L,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,  # was seq_idx
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.eps,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=False,
                    **dt_limit_kwargs,
                    lambda_mode=self.lambda_mode,
                    lambda_level_max=MAX_NUM_LEVELS,
                    lambda_level_base=LAMBDA_LEVEL_BASE,
                    lambda_htype=LAMBDA_HTYPE,
                    lambda_hstruct=LAMBDA_HSTRUCT,
                    lambda_level_fixed=LAMBDA_LEVEL_FIXED,
                    lambda_level_module=self.lambda_level_module,
                )

            else:
                gate, xBC, dt, dl = torch.split(
                    projected_states,
                    [
                        self.intermediate_size,
                        self.conv_dim,
                        self.num_heads,
                        self.num_heads * self.num_lambda_dims,
                    ],
                    dim=-1,
                )

                # 2. Convolution sequence transformation
                # Init cache
                if cache_params is not None:
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_states = torch.nn.functional.pad(
                        xBC_t,
                        (cache_params.conv_kernel_size - xBC_t.shape[-1], 0),
                    )
                    cache_params.update_conv_state(
                        layer_idx=self.layer_idx,
                        new_conv_state=conv_states,
                        cache_init=True,
                    )

                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

                xBC = apply_mask_to_padding_states(
                    hidden_states=xBC,
                    attention_mask=attention_mask,
                )

                x, B, C = torch.split(
                    xBC,
                    [
                        self.intermediate_size,
                        groups_time_state_size,
                        groups_time_state_size,
                    ],
                    dim=-1,
                )

                # 3. SSM transformation
                y, hssm_state = hmamba_chunk_scan_combined(
                    rearrange(x, "b l (h p) -> b l h p", b=batch_size, l=seq_len, p=self.head_dim),
                    dt=dt,
                    A=A,
                    B=rearrange(B, "b l (g n) -> b l g n", b=batch_size, l=seq_len, g=self.n_groups),
                    C=rearrange(C, "b l (g n) -> b l g n", b=batch_size, l=seq_len, g=self.n_groups),
                    dl=rearrange(dl, "b l (h ell) -> b l h ell", b=batch_size, h=self.num_heads, ell=self.num_lambda_dims),
                    L=self.L,
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                    lambda_mode=self.lambda_mode,
                    lambda_level_max=MAX_NUM_LEVELS,
                    lambda_level_base=LAMBDA_LEVEL_BASE,
                    lambda_htype=LAMBDA_HTYPE,
                    lambda_hstruct=LAMBDA_HSTRUCT,
                    lambda_level_fixed=LAMBDA_LEVEL_FIXED,
                    lambda_level_module=self.lambda_level_module,
                )

                # Init cache
                if hssm_state is not None and cache_params is not None:
                    cache_params.update_hssm_state(
                        layer_idx=self.layer_idx,
                        new_hssm_state=hssm_state,
                    )

                y = rearrange(
                    y,
                    "b l h p -> b l (h p)",
                    b=batch_size,
                    l=seq_len,
                    h=self.num_heads,
                    p=self.head_dim,
                )
                # Multiply "gate" branch and apply extra normalization layer
                y = self.norm(y, gate)

                # 4. Final linear projection
                out = self.out_proj(y)

        return out

    def forward(
        self,
        hidden_states,
        cache_params: Optional[HAttentionCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(
                hidden_states=hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask)
        raise NotImplementedError


class HAttentionBlock(nn.Module):
    def __init__(self, config: HAttentionConfig, layer_idx: int) -> None:
        super().__init__()
        if config.residual_in_fp32:
            raise NotImplementedError
        self.config = config
        self.layer_idx = layer_idx
        self.mixer_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = HAttentionMixer(config, layer_idx=layer_idx)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=4,
            intermediate_size=None,
            hidden_act="swish",
            fuse_swiglu=True
        )

    def forward(
        self,
        hidden_states,
        cache_params: Optional[HAttentionCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.mixer_norm(hidden_states)
        hidden_states = self.mixer(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(
                hidden_states,
                residual=residual,
                prenorm=True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HAttentionPreTrainedModel(PreTrainedModel, GenerationMixin):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HAttentionConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["HAttentionBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(
        self,
        module: nn.Module,
        num_residuals_per_layer: int = 2,  # HAttention + MLP
    ):
        """Initialize the weights."""
        if isinstance(module, HAttentionMixer):

            # --- A_log ---
            A = torch.arange(1, module.num_heads + 1)
            with torch.no_grad():
                if not isinstance(module.A_log, torch.distributed.tensor.DTensor):
                    module.A_log.copy_(torch.log(A))
                else:
                    logger.warning_once("`A_log` is a DTensor, skipping initialization")
            module.A_log._no_weight_decay = True

            # --- D ---
            nn.init.ones_(module.D)
            module.D._no_weight_decay = True

            # --- L ---
            nn.init.ones_(module.L)
            module.L._no_weight_decay = True

            # --- dt_bias ---
            dt = torch.exp(
                torch.rand(self.config.num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)

            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                if not isinstance(module.dt_bias, torch.distributed.tensor.DTensor):
                    module.dt_bias.copy_(inv_dt)
                else:
                    logger.warning_once("`dt_bias` is a DTensor, skipping initialization")
            module.dt_bias._no_reinit = True

        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                # guard against deprecated behavior
                if hasattr(module.bias, "_no_reinit"):
                    raise ValueError("This is not supposed to happen")
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, 'o_proj'):
                # p = module.o_proj.weight
                # guard against deprecated behavior
                raise ValueError("This is not supposed to happen")
            elif hasattr(module, 'out_proj'):
                p = module.out_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class HAttentionModel(HAttentionPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HAttentionBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[HAttentionCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Mamba2Output]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = HAttentionCache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__,
                    hidden_states,
                    cache_params,
                    cache_position,
                    attention_mask,
                )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class HAttentionForCausalLM(HAttentionPreTrainedModel):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.backbone = HAttentionModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[HAttentionCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[int] = None,
        **kwargs,
    ):
        if use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            if cache_position[0] > 0:
                input_ids = input_ids[:, -1][..., None]

                if attention_mask is not None:
                    attention_mask = None
            else:
                # we initialize the `cache_position` to full size of `conv_states` at prefill stage
                # considering padding will be applied when input length is shorter, and truncation
                # will be applied when it is longer, so it will be equivalent to always have it match
                # the length of `cache_params.conv_states`, which is `config.conv_kernel`
                cache_position = torch.arange(0, self.config.conv_kernel, device=input_ids.device)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'attention_mask': attention_mask,
            'cache_params': cache_params,
            'use_cache': use_cache,
            'cache_position': cache_position,
            'logits_to_keep': logits_to_keep
        })
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[HAttentionCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, Mamba2CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training

        loss, logits = None, None
        if not fuse_linear_and_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Mamba2CausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states,
        )
