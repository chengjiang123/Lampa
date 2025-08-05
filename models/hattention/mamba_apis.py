import click
import torch
import warnings
from tqdm.auto import trange
from einops import rearrange, repeat
from typing import Optional, Tuple
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.triton.ssd_combined import (
    mamba_chunk_scan_combined,
    ssd_chunk_scan_combined_ref,
    mamba_split_conv1d_scan_combined)
from fla.modules.layernorm_gated import rmsnorm_fn

from hattention.base import (
    HType,
    HStruct)
from hattention.kernel import (
    hattention_kernel,
    hattention_prefill)
from hattention.recurrent import (
    HState,
    step_state,
    step_output,
    hattention_recurrent)

try:
    from torchtune.modules import RotaryPositionalEmbeddings
except ImportError:
    RotaryPositionalEmbeddings = None


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


def hmamba_split_conv1d_scan_combined(
    zxbcdtdl: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    L: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
    activation: str = "silu",
    rmsnorm_weight: Optional[torch.Tensor] = None,
    rmsnorm_eps: float = 1e-6,
    outproj_weight: Optional[torch.Tensor] = None,
    outproj_bias: Optional[torch.Tensor] = None,
    headdim: Optional[int] = None,
    ngroups: int = 1,
    norm_before_gate: bool = True,
    lambda_mode: Optional[str] = None,
    lambda_level_max: Optional[int] = None,
    lambda_level_base: Optional[int] = None,
    lambda_htype: Optional[HType] = None,
    lambda_hstruct: Optional[HStruct] = None,
    lambda_level_fixed: Optional[bool] = None,
    lambda_level_module: Optional[LambdaLevelMLP] = None,
) -> torch.Tensor:
    """
    Argument:
        zxbcdtdl: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        L: (nheads, nlevels)
        D: (nheads, headdim) or (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen), int32
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    if initial_states is not None:
        raise NotImplementedError
    if seq_idx is not None:
        raise NotImplementedError
    if dt_limit != (0.0, float("inf")):
        raise NotImplementedError
    if return_final_states is not False:
        raise NotImplementedError
    if norm_before_gate is not False:
        raise NotImplementedError
    if rmsnorm_weight is None:
        raise NotImplementedError
    if activation not in ["silu", "swish"]:
        raise NotImplementedError

    batch, seqlen, _ = zxbcdtdl.shape
    dlambda = L.shape[-1]
    nheads, = D.shape
    dim = nheads * headdim
    dstate = (zxbcdtdl.shape[-1] - 2 * dim - nheads - nheads * dlambda) // ngroups // 2

    if D.dim() != 1:
        raise ValueError
    if headdim is None:
        raise ValueError
    if nheads % ngroups != 0:
        raise ValueError
    if zxbcdtdl.shape != (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads + nheads * dlambda):
        raise ValueError
    if dt_bias.shape != (nheads,):
        raise ValueError
    if A.shape != (nheads,):
        raise ValueError
    if L.shape != (nheads, dlambda):
        raise ValueError
    if D.shape != (nheads,):
        raise ValueError
    if rmsnorm_weight is None:
        raise ValueError

    zxBCdtl_splits = [dim, dim + 2 * ngroups * dstate, nheads, nheads * dlambda]
    xBC_splits = [dim, ngroups * dstate, ngroups * dstate]
    z, xBC, dt, dl = torch.split(zxbcdtdl, zxBCdtl_splits, dim=-1)
    xBC = rearrange(
        causal_conv1d_fn(
            rearrange(xBC, "b s d -> b d s"),
            conv1d_weight,
            bias=conv1d_bias,
            activation=activation,
            seq_idx=seq_idx),
        "b d s -> b s d")
    x, B, C = torch.split(xBC, xBC_splits, dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads, p=headdim)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups, n=dstate)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups, n=dstate)
    dl = rearrange(dl, "b l (h ell) -> b l h ell", h=nheads, ell=dlambda)
    y, _ = hmamba_chunk_scan_combined(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        dl=dl,
        L=L,
        chunk_size=chunk_size,
        D=D,
        z=z if rmsnorm_weight is None else None,
        dt_bias=dt_bias,
        dt_softplus=True,
        seq_idx=seq_idx,
        cu_seqlens=None,
        dt_limit=dt_limit,
        return_final_states=return_final_states,
        return_varlen_states=False,
        lambda_mode=lambda_mode,
        lambda_level_max=lambda_level_max,
        lambda_level_base=lambda_level_base,
        lambda_htype=lambda_htype,
        lambda_hstruct=lambda_hstruct,
        lambda_level_fixed=lambda_level_fixed,
        lambda_level_module=lambda_level_module)

    y = rearrange(y, "b l h p -> b l (h p)")
    if rmsnorm_weight is not None:
        y = rmsnorm_fn(
            x=y,
            weight=rmsnorm_weight,
            bias=None,
            z=z,
            eps=rmsnorm_eps,
            group_size=None,
            norm_before_gate=False)
    out = torch.nn.functional.linear(
        y,
        outproj_weight,
        outproj_bias)
    return out


def hmamba_chunk_scan_combined(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dl: torch.Tensor,
    L: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
    return_varlen_states: bool = False,
    lambda_mode: Optional[str] = None,
    lambda_level_max: Optional[int] = None,
    lambda_level_base: Optional[int] = None,
    lambda_htype: Optional[HType] = None,
    lambda_hstruct: Optional[HStruct] = None,
    lambda_level_fixed: Optional[bool] = None,
    lambda_level_module: Optional[LambdaLevelMLP] = None,
) -> Tuple[torch.Tensor, Optional[HState]]:
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        dl: (batch, seqlen, nheads, nlevels)
        L: (nheads, nlevels)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
        dt_softplus: Whether to apply softplus to dt
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    if z is not None:
        raise NotImplementedError
    if initial_states is not None:
        raise NotImplementedError
    if seq_idx is not None:
        raise NotImplementedError
    if cu_seqlens is not None:
        raise NotImplementedError
    if dt_softplus is not True:
        raise NotImplementedError
    if tuple(dt_limit) != (0.0, float("inf")):
        raise NotImplementedError
    if return_varlen_states is not False:
        raise NotImplementedError

    # if chunk_size != 128:
    #     raise NotImplementedError
    if not B.shape == C.shape:
        raise ValueError("B and C must have the same shape")

    if lambda_level_max is None:
        raise ValueError
    if lambda_level_base is None:
        raise ValueError
    if lambda_htype is None:
        raise ValueError
    if lambda_hstruct is None:
        raise ValueError
    if lambda_level_fixed is None:
        raise ValueError
    warnings.warn(click.style(f"[HAttention] Using level base = {lambda_level_base}, htype = {lambda_htype}, hstruct = {lambda_hstruct}", fg="yellow"))

    seqlen = x.shape[-3]
    nheads = x.shape[-2]
    ngroups = B.shape[-2]

    if D is not None:
        if D.dim() != 1:
            raise ValueError
        D = rearrange(D, "h -> 1 1 h 1")
        D_residual = x * D

    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> 1 1 h")
    if dt_softplus:
        dt = torch.nn.functional.softplus(dt)
    if dt_limit != (0.0, float("inf")):
        dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1])
    x = x * rearrange(dt, "b l h -> b l h 1")
    A = rearrange(A, "h -> 1 1 h") * dt
    L = compute_lambda_maybe_fixed(
        L=rearrange(L, "h ell -> 1 1 h ell"),
        dl=dl,
        lambda_mode=lambda_mode,
        lambda_level_max=lambda_level_max,
        lambda_level_fixed=lambda_level_fixed,
        lambda_level_module=lambda_level_module)
    if return_final_states is False:
        state = None
        y = hattention_kernel(
            q=C,
            k=B,
            v=x,
            b=None,
            g=A,
            l=L,
            scale=None,
            head_first=False,
            level_base=lambda_level_base,
            htype=lambda_htype,
            hstruct=lambda_hstruct)
    else:
        if x.dtype != C.dtype:
            warnings.warn(click.style(f"`x.dtype` = {x.dtype} -> {C.dtype}", fg="blue"))
            x = x.to(dtype=C.dtype)
        if L.dtype != dl.dtype:
            warnings.warn(click.style(f"`L.dtype` = {L.dtype} -> {dl.dtype}", fg="blue"))
            L = L.to(dtype=dl.dtype)
        y, state = hattention_prefill(
            q=C,
            k=B,
            v=x,
            b=None,
            g=A,
            l=L,
            scale=None,
            head_first=False,
            level_base=lambda_level_base,
            htype=lambda_htype,
            hstruct=lambda_hstruct)

    if D is not None:
        y = y + D_residual

    return y, state


def hselective_state_update(
    state: HState,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dl: torch.Tensor,
    L: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    lambda_mode: Optional[str] = None,
    lambda_level_max: Optional[int] = None,
    lambda_level_base: Optional[int] = None,
    lambda_htype: Optional[HType] = None,
    lambda_hstruct: Optional[HStruct] = None,
    lambda_level_fixed: Optional[bool] = None,
    lambda_level_module: Optional[LambdaLevelMLP] = None,
) -> Tuple[torch.Tensor, HState]:
    """
    Argument:
        state
        x: (batch, nheads, headdim)
        dt: (batch, nheads)
        A: (nheads)
        B: (batch, ngroups, dstate)
        C: (batch, ngroups, dstate)
        dl: (batch, nheads, nlevels)
        L: (nheads, nlevels)
        D: (nheads, headdim) or (nheads,)
        z: (batch, nheads, headdim)
        dt_bias: (nheads,)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    if state_batch_indices is not None:
        raise NotImplementedError
    if z is not None:
        raise NotImplementedError

    if x.dim() != 3:
        raise ValueError
    if dt.dim() != 2:
        raise ValueError
    if A.dim() != 1:
        raise ValueError
    if B.dim() != 3:
        raise ValueError
    if C.dim() != 3:
        raise ValueError
    if dl.dim() != 3:
        raise ValueError
    if L.dim() != 2:
        raise ValueError
    if D is not None and D.dim() != 1:
        raise ValueError
    if z is not None and z.dim() != 3:
        raise ValueError
    if dt_bias is not None and dt_bias.dim() != 1:
        raise ValueError

    batch, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    dlambda = L.shape[-1]

    if x.shape != (batch, nheads, headdim):
        raise ValueError
    if dt.shape != x.shape[:2]:
        raise ValueError
    if A.shape != (nheads,):
        raise ValueError
    if nheads % ngroups != 0:
        raise ValueError
    if B.shape != (batch, ngroups, dstate):
        raise ValueError
    if C.shape != B.shape:
        raise ValueError
    if dl.shape != (batch, nheads, dlambda):
        raise ValueError
    if L.shape != (nheads, dlambda):
        raise ValueError
    if D is not None:
        if D.shape != (nheads,):
            raise ValueError
    if z is not None:
        if z.shape != x.shape:
            raise ValueError
    if state_batch_indices is not None:
        if state_batch_indices.shape != (batch,):
            raise ValueError

    if lambda_level_max is None:
        raise ValueError
    if lambda_level_base is None:
        raise ValueError
    if lambda_htype is None:
        raise ValueError
    if lambda_hstruct is None:
        raise ValueError
    if lambda_level_fixed is None:
        raise ValueError
    warnings.warn(click.style(f"[HAttention] Using level base = {lambda_level_base}, htype = {lambda_htype}, hstruct = {lambda_hstruct}", fg="yellow"))

    if D is not None:
        if D.dim() != 1:
            raise ValueError
        D = rearrange(D, "h -> 1 h 1")
        D_residual = x * D

    if dt_bias is not None:
        if dt_bias.shape != (nheads,):
            raise ValueError
        dt = dt + dt_bias
    if dt_softplus:
        dt = torch.nn.functional.softplus(dt)

    x = x * rearrange(dt, "b h -> b h 1")
    A = rearrange(A, "h -> 1 h") * dt
    B = repeat(B, "b g d -> b (g h) d", g=ngroups, h=nheads // ngroups)
    C = repeat(C, "b g d -> b (g h) d", g=ngroups, h=nheads // ngroups)
    L = compute_lambda_maybe_fixed(
        L=rearrange(L, "h ell -> 1 h ell", h=nheads, ell=dlambda),
        dl=dl,
        lambda_mode=lambda_mode,
        lambda_level_max=lambda_level_max,
        lambda_level_fixed=lambda_level_fixed,
        lambda_level_module=lambda_level_module)

    if x.dtype != C.dtype:
        warnings.warn(click.style(f"`x.dtype` = {x.dtype}, `C.dtype` = {C.dtype}", fg="blue"))
    if L.dtype != dl.dtype:
        warnings.warn(click.style(f"`L.dtype` = {L.dtype}, `dl.dtype` = {dl.dtype}", fg="blue"))

    if not all([
        A.dtype == state.dtype,
        C.dtype == B.dtype,
        C.dtype == x.dtype or x.dtype == state.dtype,  # not sure, temporary
        C.dtype == L.dtype or L.dtype == state.dtype,  # not sure, temporary
        state.base == lambda_level_base,
        state.htype == lambda_htype,
        state.hstruct == lambda_hstruct]):
        raise TypeError

    state = step_state(
        state,
        k=B.to(dtype=state.dtype),
        v=x.to(dtype=state.dtype),
        a=A.exp())
    out = step_output(
        state.to(dtype=x.dtype),
        q=C,
        l=L)

    if D is not None:
        out = out + D_residual

    return out, state


def checks():
    batch_size = 2
    seqlen = 2048
    dim = 1024 * 2
    headdim = 64
    dstate = 128
    nheads = dim // headdim
    ngroups = 4
    conv_dim = 4
    num_levels = 12
    chunk_size = 128
    dtype = torch.float32
    device = "cuda"
    lambda_mode = None
    lambda_level_base = 2

    zxbcdt = torch.randn(batch_size, seqlen, 2 * dim + 2 * ngroups * dstate + nheads, device=device, dtype=dtype)
    dl = torch.ones(batch_size, seqlen, nheads * num_levels, device=device, dtype=dtype)
    zxbcdtdl = torch.cat([zxbcdt, dl], dim=-1)
    conv1d_weight = torch.randn(dim + 2 * ngroups * dstate, conv_dim, device=device, dtype=dtype)
    conv1d_bias = torch.randn(dim + 2 * ngroups * dstate, device=device, dtype=dtype)
    dt_bias = torch.randn(nheads, device=device, dtype=dtype)
    A = (-torch.exp(torch.randn(nheads, device=device, dtype=dtype)))
    L = torch.ones(nheads, num_levels, device=device, dtype=dtype)
    D = torch.randn(nheads, device=device, dtype=dtype)
    rmsnorm_weight = torch.randn(dim, device=device, dtype=dtype)
    rmsnorm_eps = 1e-5
    outproj_weight = torch.randn(dim, dim, device=device, dtype=dtype)
    outproj_bias = torch.randn(dim, device=device, dtype=dtype)

    out = hmamba_split_conv1d_scan_combined(
        zxbcdtdl=zxbcdtdl,
        conv1d_weight=conv1d_weight,
        conv1d_bias=conv1d_bias,
        dt_bias=dt_bias,
        A=A,
        L=L,
        D=D,
        chunk_size=chunk_size,
        initial_states=None,
        seq_idx=None,
        dt_limit=(0.0, float("inf")),
        return_final_states=False,
        activation="silu",
        rmsnorm_weight=rmsnorm_weight,
        rmsnorm_eps=rmsnorm_eps,
        outproj_weight=outproj_weight,
        outproj_bias=outproj_bias,
        headdim=headdim,
        ngroups=ngroups,
        norm_before_gate=False,
        lambda_mode=lambda_mode,
        lambda_level_base=lambda_level_base)

    out_ref1 = mamba_split_conv1d_scan_combined(
        zxbcdt=zxbcdt,
        conv1d_weight=conv1d_weight,
        conv1d_bias=conv1d_bias,
        dt_bias=dt_bias,
        A=A,
        D=D,
        chunk_size=chunk_size,
        initial_states=None,
        seq_idx=None,
        dt_limit=(0.0, float("inf")),
        return_final_states=False,
        activation="silu",
        rmsnorm_weight=rmsnorm_weight,
        rmsnorm_eps=rmsnorm_eps,
        outproj_weight=outproj_weight,
        outproj_bias=outproj_bias,
        headdim=headdim,
        ngroups=ngroups,
        norm_before_gate=False)

    out_ref2 = mamba_split_conv1d_scan_ref(
        zxbcdt=zxbcdt,
        conv1d_weight=conv1d_weight,
        conv1d_bias=conv1d_bias,
        dt_bias=dt_bias,
        A=A,
        D=D,
        chunk_size=chunk_size,
        # initial_states=None,
        # seq_idx=None,
        dt_limit=(0.0, float("inf")),
        # return_final_states=False,
        activation="silu",
        rmsnorm_weight=rmsnorm_weight,
        rmsnorm_eps=rmsnorm_eps,
        outproj_weight=outproj_weight,
        outproj_bias=outproj_bias,
        headdim=headdim,
        ngroups=ngroups,
        norm_before_gate=False)

    # torch.testing.assert_close(out, out_ref2)
    print(f"{(out - out_ref1).norm() / out_ref1.norm():.2e}")
    print(f"{(out - out_ref2).norm() / out_ref2.norm():.2e}")

    x = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    dt = torch.randn(batch_size, seqlen, nheads, device=device, dtype=dtype)
    B = torch.randn(batch_size, seqlen, ngroups, dstate, device=device, dtype=dtype)
    C = torch.randn(batch_size, seqlen, ngroups, dstate, device=device, dtype=dtype)
    dl = torch.ones(batch_size, seqlen, nheads, num_levels, device=device, dtype=dtype)

    out, _ = hmamba_chunk_scan_combined(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        dl=dl,
        L=L,
        chunk_size=chunk_size,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=None,
        seq_idx=None,
        cu_seqlens=None,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        return_final_states=True,
        return_varlen_states=False,
        lambda_mode=lambda_mode,
        lambda_level_base=lambda_level_base)

    out_ref1, _ = mamba_chunk_scan_combined(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        chunk_size=chunk_size,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=None,
        seq_idx=None,
        cu_seqlens=None,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        return_final_states=True,
        return_varlen_states=False)

    out_ref2 = ssd_chunk_scan_combined_ref(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        chunk_size=chunk_size,
        D=D,
        z=None,
        dt_bias=dt_bias,
        dt_softplus=True)

    # torch.testing.assert_close(out, out_ref2)
    print(f"{(out - out_ref1).norm() / out_ref1.norm():.2e}")
    print(f"{(out - out_ref2).norm() / out_ref2.norm():.2e}")


def mamba_split_conv1d_scan_ref(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, dt_limit=(0.0, float("inf")), activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    import torch.nn.functional as F

    if D.dim() == 1:
        assert headdim is not None
        nheads, = D.shape
    else:
        nheads, headdim = D.shape
    assert nheads % ngroups == 0
    batch, seqlen, _ = zxbcdt.shape
    dim = nheads * headdim
    dstate = (zxbcdt.shape[-1] - 2 * dim - nheads) // ngroups // 2
    assert zxbcdt.shape == (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads)
    assert dt_bias.shape == (nheads,)
    assert A.shape == (nheads,)
    if rmsnorm_weight is not None:
        assert rmsnorm_weight.shape == (dim,)
    z, xBC, dt = torch.split(zxbcdt, [dim, dim + 2 * ngroups * dstate, nheads], dim=-1)
    xBC = rearrange(causal_conv1d_fn(rearrange(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias, activation=activation),
                    "b d s -> b s d")
    x, B, C = torch.split(xBC, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
    out = ssd_selective_scan(x, dt.to(x.dtype), A, B, C, D=D.float(),
                             z=z if rmsnorm_weight is None else None, dt_bias=dt_bias, dt_softplus=True, dt_limit=dt_limit)
    out = rearrange(out, "b s h p -> b s (h p)")
    if rmsnorm_weight is not None:
        out = rmsnorm_fn(out, rmsnorm_weight, None, z=rearrange(z, "b l h p -> b l (h p)"), eps=rmsnorm_eps,
                         norm_before_gate=norm_before_gate)
    if outproj_weight is not None:
        out = F.linear(out, outproj_weight, outproj_bias)
    return out



def ssd_selective_scan(x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads) or (batch, seqlen, nheads, headdim)
        A: (nheads) or (dim, dstate)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,) or (nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    import torch.nn.functional as F
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref as selective_scan_fn

    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    x = rearrange(x, "b l h p -> b (h p) l")
    if dt.dim() == 3:
        dt = repeat(dt, "b l h -> b l h p", p=headdim)
    dt = rearrange(dt, "b l h p -> b (h p) l")
    if A.dim() == 1:
        A = repeat(A, "h -> (h p) n", p=headdim, n=dstate).to(dtype=torch.float32)
    else:
        A = A.to(dtype=torch.float32)
    B = rearrange(B, "b l g n -> b g n l")
    C = rearrange(C, "b l g n -> b g n l")
    if D is not None:
        if D.dim() == 2:
            D = rearrange(D, "h p -> (h p)")
        else:
            D = repeat(D, "h -> (h p)", p=headdim)
    if z is not None:
        z = rearrange(z, "b l h p -> b (h p) l")
    if dt_bias is not None:
        if dt_bias.dim() == 1:
            dt_bias = repeat(dt_bias, "h -> h p", p=headdim)
        dt_bias = rearrange(dt_bias, "h p -> (h p)")
    if dt_limit != (0.0, float("inf")):
        if dt_bias is not None:
            dt = dt + rearrange(dt_bias, "d -> d 1")
        if dt_softplus:
            dt = F.softplus(dt)
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1]).to(x.dtype)
        dt_bias = None
        dt_softplus = None
    out = selective_scan_fn(x, dt, A, B, C, D=D, z=z, delta_bias=dt_bias, delta_softplus=dt_softplus)
    return rearrange(out, "b (h p) l -> b l h p", p=headdim)


def check_parallel_vs_step():
    batch_size = 2
    seqlen = 2048
    dim = 1024 * 2
    headdim = 64
    dstate = 128
    nheads = dim // headdim
    ngroups = 4
    num_levels = 12
    chunk_size = 128
    dtype = torch.float32
    device = "cuda"
    lambda_mode = "bounded"
    lambda_level_base = 2

    x = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    dt = torch.randn(batch_size, seqlen, nheads, device=device, dtype=dtype)
    A = (-torch.exp(torch.randn(nheads, device=device, dtype=dtype)))
    B = torch.randn(batch_size, seqlen, ngroups, dstate, device=device, dtype=dtype)
    C = torch.randn(batch_size, seqlen, ngroups, dstate, device=device, dtype=dtype)
    dl = torch.randn(batch_size, seqlen, nheads, num_levels, device=device, dtype=dtype)
    L = torch.randn(nheads, num_levels, device=device, dtype=dtype)
    D = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = torch.randn(nheads, device=device, dtype=dtype)

    out_ref1, _ = hmamba_chunk_scan_combined(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        dl=dl,
        L=L,
        chunk_size=chunk_size,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=None,
        seq_idx=None,
        cu_seqlens=None,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        return_final_states=False,
        return_varlen_states=False,
        lambda_mode=lambda_mode,
        lambda_level_base=lambda_level_base)

    out_ref2, _ = hmamba_chunk_scan_combined(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        dl=dl,
        L=L,
        chunk_size=chunk_size,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=None,
        seq_idx=None,
        cu_seqlens=None,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        return_final_states=True,
        return_varlen_states=False,
        lambda_mode=lambda_mode,
        lambda_level_base=lambda_level_base)

    out1 = torch.zeros_like(out_ref1)
    state1 = HState(
        base=lambda_level_base,
        shape=(batch_size, nheads, dstate, headdim, num_levels),
        dtype=dtype,
        device=device)
    for t in trange(seqlen):
        out1[:, t], state1 = hselective_state_update(
            state=state1,
            x=x[:, t],
            dt=dt[:, t],
            A=A,
            B=B[:, t],
            C=C[:, t],
            dl=dl[:, t],
            L=L,
            D=D,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=True,
            lambda_mode=lambda_mode,
            lambda_level_base=lambda_level_base)

    init_t = seqlen // 2
    out2 = torch.zeros_like(out_ref1)
    out2[:, :init_t], state2 = hmamba_chunk_scan_combined(
        x=x[:, :init_t],
        dt=dt[:, :init_t],
        A=A,
        B=B[:, :init_t],
        C=C[:, :init_t],
        dl=dl[:, :init_t],
        L=L,
        chunk_size=chunk_size,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=None,
        seq_idx=None,
        cu_seqlens=None,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        return_final_states=True,
        return_varlen_states=False,
        lambda_mode=lambda_mode,
        lambda_level_base=lambda_level_base)
    for t in trange(init_t, seqlen):
        out2[:, t], state2 = hselective_state_update(
            state=state2,
            x=x[:, t],
            dt=dt[:, t],
            A=A,
            B=B[:, t],
            C=C[:, t],
            dl=dl[:, t],
            L=L,
            D=D,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=True,
            lambda_mode=lambda_mode,
            lambda_level_base=lambda_level_base)

    print(f"{(out1 - out_ref1).norm() / out_ref1.norm():.2e} {(out1 - out_ref2).norm() / out_ref2.norm():.2e}")
    print(f"{(out2 - out_ref1).norm() / out_ref1.norm():.2e} {(out2 - out_ref2).norm() / out_ref2.norm():.2e}")
