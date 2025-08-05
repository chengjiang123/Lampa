# Copyright (c) 2024, Albert Gu and Tri Dao.
"""Minimal implementation of SSD.

This is the same as Listing 1 from the paper.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from typing import Tuple, Optional
from einops import rearrange, repeat, einsum, reduce
from .cumsum import chunk_local_cumsum
from .base import (
    HType,
    ceil_log,
    get_num_levels,
    make_levels_matrix)


def make_masked_H_matrix(
    L: Float[torch.Tensor, "batch num_heads num_chunks chunk_length num_levels"],
    level_base: int,
    htype: HType,
    subdiagonal: bool,
) -> Float[torch.Tensor, "batch num_heads num_chunks chunk_length chunk_length"]:
    num_chunks = L.shape[2]
    chunk_length = L.shape[3]
    length = num_chunks * chunk_length

    levels = make_levels_matrix(
        length=length,
        base=level_base,
        htype=htype,
        dtype=torch.int64,
        device=L.device)
    lengths = torch.arange(length, device=L.device)
    lengths = repeat(lengths, "t -> t s", s=length)

    levels = rearrange(
        levels,
        "(tc tl) (sc sl) -> tl sl tc sc",
        tc=num_chunks,
        sc=num_chunks,
        tl=chunk_length,
        sl=chunk_length)
    lengths = rearrange(
        lengths,
        "(tc tl) (sc sl) -> tl sl tc sc",
        tc=num_chunks,
        sc=num_chunks,
        tl=chunk_length,
        sl=chunk_length)

    if not subdiagonal:
        levels = torch.diagonal(levels, dim1=-2, dim2=-1)
        lengths = torch.diagonal(lengths, dim1=-2, dim2=-1)
    else:
        levels = torch.diagonal(levels, offset=-1, dim1=-2, dim2=-1)
        lengths = torch.diagonal(lengths, offset=-1, dim1=-2, dim2=-1)

    levels = rearrange(levels, "tl sl c -> c tl sl")
    lengths = rearrange(lengths, "tl sl c -> c tl sl")

    L = rearrange(L, "b h c l ell -> b h (c l) ell")
    HD = L[..., lengths, levels]

    if subdiagonal:
        return HD

    # these are the diagonal blocks,
    # so we only keep the lower triangular part
    mask = torch.ones(
        chunk_length,
        chunk_length,
        dtype=torch.bool,
        device=HD.device)
    mask = torch.tril(
        mask,
        diagonal=0)
    return HD.masked_fill(~mask, 0.)


def get_level_from_chunk_levels(
    chunk_mask: Float[torch.Tensor, "length length"],
    levels_matrix: Float[torch.Tensor, "length length ..."],
) -> int:
    indices = chunk_mask.nonzero()
    indices_0 = indices[:, 0]
    indices_1 = indices[:, 1]
    levels = levels_matrix[indices_0, indices_1, ...]
    return torch.unique(levels).item()


def segsum(
    x: Float[torch.Tensor, "batch num_heads num_chunks chunk_length"],
) -> Float[torch.Tensor, "batch num_heads num_chunks chunk_length chunk_length"]:
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def hssd_minimal_discrete(
    X: Float[torch.Tensor, "batch length num_heads d_head"],
    A: Float[torch.Tensor, "batch length num_heads"],
    B: Float[torch.Tensor, "batch length num_heads d_state"],
    C: Float[torch.Tensor, "batch length num_heads d_state"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    level_base: int,
    htype: HType,
    block_len: int,
    initial_states: Optional[Float[torch.Tensor, "batch num_heads d_state"]] = None,
) -> Float[torch.Tensor, "batch length num_heads d_head"]:
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C, L = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C, L)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 0. Make H matrix
    num_blocks = L.shape[1]
    num_block_levels = get_num_levels(
        num_blocks,
        base=level_base)
    L = rearrange(L, "b c l h ell -> b h c l ell")
    HD = make_masked_H_matrix(
        L,
        level_base=level_base,
        htype=htype,
        subdiagonal=False)
    levels_matrix = make_levels_matrix(
        length=(num_blocks * block_len),
        base=level_base,
        htype=htype,
        dtype=torch.int64,
        device=L.device)
    levels_matrix = rearrange(
        levels_matrix,
        "(tc tl) (sc sl) -> tc sc tl sl",
        tl=block_len,
        sl=block_len)
    chunk_levels_matrix = make_levels_matrix(
        length=num_blocks,
        base=level_base,
        htype=htype,
        dtype=torch.int64,
        device=L.device)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    D = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bhcls,bcshp->bclhp", C, B, D, HD, X)

    if htype == HType.STRONG:
        X2 = X[:, :-1 , ...]
        A2 = torch.cat((A[..., :-1, :], A[..., 1:, :]), dim=-1)
        B2 = B[:, :-1 , ...]
        C2 = C[:,   1:, ...]
        D2 = torch.exp(segsum(A2)[..., block_len:, :block_len])
        HD2 = make_masked_H_matrix(
            L,
            level_base=level_base,
            htype=htype,
            subdiagonal=True)
        Y_diag[:, 1:, ...] += torch.einsum("bclhn,bcshn,bhcls,bhcls,bcshp->bclhp", C2, B2, D2, HD2, X2)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    begin_states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))

    begin_states = begin_states[:, 1:, ...]
    decay_chunk = decay_chunk[:, :, :-1, 1:]

    Y_off = torch.zeros_like(Y_diag)
    # we ignore the diagonal blocks
    if htype == HType.WEAK:
        ell_0 = 1
    else:
        ell_0 = 2
    for ell in range(ell_0, num_block_levels):
        chunk_mask = chunk_levels_matrix == ell
        level = get_level_from_chunk_levels(
            chunk_mask=chunk_mask,
            levels_matrix=levels_matrix)

        decay_chunk_masked = decay_chunk.masked_fill(~chunk_mask, 0.)
        end_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk_masked, begin_states)

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        state_decay_out = torch.exp(A_cumsum)
        Y_off = Y_off + torch.einsum("bclhn,bchpn,bhcl,bhcl->bclhp", C, end_states, state_decay_out, L[..., level])

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y


def make_T_matrix(
    K: Float[torch.Tensor, "batch num_heads num_chunks chunk_length d_state"],
    B: Float[torch.Tensor, "batch num_heads num_chunks chunk_length"],
) -> Float[torch.Tensor, "batch num_heads num_chunks chunk_length chunk_length"]:
    dtype = K.dtype
    device = K.device
    chunk_length = K.shape[3]

    mask = torch.ones(chunk_length, chunk_length, dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=0)
    mask = rearrange(mask, "cs0 cs1 -> 1 1 1 cs0 cs1")
    T = einsum(B, K, K, "b h cn cs0, b h cn cs0 d, b h cn cs1 d -> b h cn cs0 cs1")
    T = -T.masked_fill(mask, 0)
    for t in range(1, chunk_length):
        T[..., t, :t] = (
            T[..., t, :t].clone() +
            (
                T[..., t, :, None].clone() *
                T[...,    :,   :t].clone()
            ).sum(dim=-2)
        )
    I = torch.eye(chunk_length, dtype=dtype, device=device)
    I = rearrange(I, "cs0 cs1 -> 1 1 1 cs0 cs1")
    T = T + I
    return T


def make_T_matrix_v2(
    K: Float[torch.Tensor, "batch num_heads num_chunks chunk_length d_state"],
    B: Float[torch.Tensor, "batch num_heads num_chunks chunk_length"],
    G: Optional[Float[torch.Tensor, "batch num_heads num_chunks chunk_length"]],
    fla_compatible: bool = False,
) -> Float[torch.Tensor, "batch num_heads num_chunks chunk_length chunk_length"]:
    dtype = K.dtype
    device = K.device
    chunk_length = K.shape[3]

    mask = torch.ones(chunk_length, chunk_length, dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=0)
    mask = rearrange(mask, "cs0 cs1 -> 1 1 1 cs0 cs1")

    if G is None:
        T = einsum(B, K, K, "b h cn cs0, b h cn cs0 d, b h cn cs1 d -> b h cn cs0 cs1")
    else:
        G_segsum = torch.exp(segsum(G))
        T = einsum(B, G_segsum, K, K, "b h cn cs0, b h cn cs0 cs1, b h cn cs0 d, b h cn cs1 d -> b h cn cs0 cs1")

    T = -T.masked_fill(mask, 0)
    for t in range(1, chunk_length):
        T[..., t, :t] = (
            T[..., t, :t].clone() +
            (
                T[..., t, :, None].clone() *
                T[...,    :,   :t].clone()
            ).sum(dim=-2)
        )
    I = torch.eye(chunk_length, dtype=dtype, device=device)
    I = rearrange(I, "cs0 cs1 -> 1 1 1 cs0 cs1")
    T = T + I
    if not fla_compatible:
        T = einsum(T, B, "b h cn cs0 cs1, b h cn cs1 -> b h cn cs0 cs1")
    return T


def hgdn_minimal_diagonal_chunks(
    Q: Float[torch.Tensor, "batch length num_heads d_state"],
    K: Float[torch.Tensor, "batch length num_heads d_state"],
    V: Float[torch.Tensor, "batch length num_heads d_head"],
    B: Float[torch.Tensor, "batch length num_heads"],
    G: Float[torch.Tensor, "batch length num_heads"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    level_base: int,
    htype: HType,
    block_len: int,
) -> Float[torch.Tensor, "batch length num_heads d_head"]:
    assert V.dtype == K.dtype == Q.dtype
    assert G.dtype == torch.float32
    assert V.shape[1] % block_len == 0
    assert htype == HType.WEAK

    # Rearrange into blocks/chunks
    Q, K, V, B, G, L = [rearrange(x, "b (cn ct) ... -> b cn ct ...", ct=block_len) for x in (Q, K, V, B, G, L)]

    T = make_T_matrix(
        K=rearrange(K, "b cn cs h d -> b h cn cs d"),
        B=rearrange(B, "b cn ct h   -> b h cn ct  "))
    D = torch.exp(segsum(
        rearrange(G, "b cn ct h -> b h cn ct")))
    HD = make_masked_H_matrix(
        rearrange(L, "b cn ct h ell -> b h cn ct ell"),
        level_base=level_base,
        htype=htype,
        subdiagonal=False)
    T  = rearrange(T , "b h cn cs0 cs1 -> b cn cs0 cs1 h")
    D  = rearrange(D , "b h cn ct  cs  -> b cn ct  cs  h")
    HD = rearrange(HD, "b h cn ct  cs  -> b cn ct  cs  h")

    mask = torch.ones(block_len, block_len, dtype=torch.bool, device=K.device)
    mask = torch.tril(mask, diagonal=0)
    mask = rearrange(mask, "ct cs  -> 1 1 ct cs 1")
    QKT = einsum(Q, K, "b cn ct h d, b cn cs h d -> b cn ct cs h")
    QKT = torch.masked_fill(QKT, ~mask, 0.)
    QKTT = einsum(QKT, T, "b cn ct cs0 h, b cn cs0 cs1 h -> b cn ct cs1 h")
    QKTTH = QKTT * D.to(dtype=HD.dtype) * HD
    BV = einsum(B, V, "b cn cs h, b cn cs h d -> b cn cs h d")
    Y_diag = einsum(QKTTH, BV, "b cn ct cs h, b cn cs h d -> b cn ct h d")
    return rearrange(Y_diag, "b cn ct h d -> b (cn ct) h d")


def hgdn_minimal_diagonal_chunks_v2(
    Q: Float[torch.Tensor, "batch length num_heads d_state"],
    K: Float[torch.Tensor, "batch length num_heads d_state"],
    V: Float[torch.Tensor, "batch length num_heads d_head"],
    B: Float[torch.Tensor, "batch length num_heads"],
    G: Float[torch.Tensor, "batch length num_heads"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    level_base: int,
    htype: HType,
    block_len: int,
) -> Float[torch.Tensor, "batch length num_heads d_head"]:
    assert V.dtype == G.dtype == K.dtype == Q.dtype
    assert V.shape[1] % block_len == 0
    assert htype == HType.WEAK

    # Rearrange into blocks/chunks
    Q, K, V, B, G, L = [rearrange(x, "b (cn ct) ... -> b cn ct ...", ct=block_len) for x in (Q, K, V, B, G, L)]

    T2 = make_T_matrix_v2(
        K=rearrange(K, "b cn cs h d -> b h cn cs d"),
        B=rearrange(B, "b cn ct h   -> b h cn ct  "),
        G=rearrange(G, "b cn ct h   -> b h cn ct  "))
    D = torch.exp(segsum(
        rearrange(G, "b cn ct h -> b h cn ct")))
    T2 = rearrange(T2, "b h cn cs0 cs1 -> b cn cs0 cs1 h")
    D  = rearrange(D , "b h cn ct  cs  -> b cn ct  cs  h")
    U2 = einsum(T2, V, "b cn cs0 cs1 h, b cn cs1 h d -> b cn cs0 h d")

    mask = torch.ones(block_len, block_len, dtype=torch.bool, device=K.device)
    mask = torch.tril(mask, diagonal=0)
    mask = rearrange(mask, "ct cs  -> 1 1 ct cs 1")
    QKT = einsum(Q, K, D, "b cn ct h d, b cn cs h d, b cn ct cs h -> b cn ct cs h")
    QKT = torch.masked_fill(QKT, ~mask, 0.)
    Y_diag = einsum(QKT, U2, "b cn ct cs h, b cn cs h d -> b cn ct h d")
    return rearrange(Y_diag, "b cn ct h d -> b (cn ct) h d")


def hgdn_minimal_offdiagonal_chunks_v2(
    Q: Float[torch.Tensor, "batch length num_heads d_state"],
    K: Float[torch.Tensor, "batch length num_heads d_state"],
    V: Float[torch.Tensor, "batch length num_heads d_head"],
    B: Float[torch.Tensor, "batch length num_heads"],
    G: Float[torch.Tensor, "batch length num_heads"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    level_base: int,
    htype: HType,
    block_len: int,
) -> Float[torch.Tensor, "batch length num_heads d_head"]:
    assert V.dtype == G.dtype == K.dtype == Q.dtype
    assert V.shape[1] % block_len == 0
    assert htype == HType.WEAK

    # Rearrange into blocks/chunks
    Q, K, V, B, G, L = [rearrange(x, "b (cn ct) ... -> b cn ct ...", ct=block_len) for x in (Q, K, V, B, G, L)]

    T  = make_T_matrix_v2(
        K=rearrange(K, "b cn cs h d -> b h cn cs d"),
        B=rearrange(B, "b cn ct h   -> b h cn ct  "),
        G=None)
    T2 = make_T_matrix_v2(
        K=rearrange(K, "b cn cs h d -> b h cn cs d"),
        B=rearrange(B, "b cn ct h   -> b h cn ct  "),
        G=rearrange(G, "b cn ct h   -> b h cn ct  "))
    D = torch.exp(segsum(
        rearrange(G, "b cn ct h -> b h cn ct")))
    T  = rearrange(T , "b h cn cs0 cs1 -> b cn cs0 cs1 h")
    T2 = rearrange(T2, "b h cn cs0 cs1 -> b cn cs0 cs1 h")
    D  = rearrange(D , "b h cn ct  cs  -> b cn ct  cs  h")
    W  = einsum(T , K, "b cn cs0 cs1 h, b cn cs1 h d -> b cn cs0 h d")
    U2 = einsum(T2, V, "b cn cs0 cs1 h, b cn cs1 h d -> b cn cs0 h d")

    mask = torch.ones(block_len, block_len, dtype=torch.bool, device=K.device)
    mask = torch.tril(mask, diagonal=0)
    mask = rearrange(mask, "ct cs  -> 1 1 ct cs 1")
    D = torch.masked_fill(D, ~mask, 0.)

    G_cumsum      = torch.cumsum(G, dim=-2)
    G_cumsum_last = G_cumsum[..., -1, :]
    G_cumsum_diff = rearrange(G_cumsum_last, "b cn h -> b cn 1 h") - G_cumsum

    Q2 = einsum(torch.exp(G_cumsum     ), Q, "b cn ct h, b cn ct h d -> b cn ct h d")
    K2 = einsum(torch.exp(G_cumsum_diff), K, "b cn ct h, b cn ct h d -> b cn ct h d")
    W2 = einsum(torch.exp(G_cumsum     ), W, "b cn ct h, b cn ct h d -> b cn ct h d")

    Y = torch.zeros_like(V)
    ell_h0 = get_num_levels(K.shape[2], level_base)
    num_chunk_levels = ceil_log(K.shape[1], level_base)
    if ell_h0 + num_chunk_levels > L.shape[-1]:
        raise ValueError
    for ell in range(num_chunk_levels):

        S = torch.zeros(
            (K.shape[0], K.shape[3], K.shape[4], V.shape[4]),
            dtype=K.dtype,
            device=K.device)

        for t in range(K.shape[1]):
            Q_t  = Q [:, t, ...]
            K_t  = K [:, t, ...]
            L_t  = L [:, t, ..., ell_h0 + ell]
            D_t  = D [:, t, ...]
            Q2_t = Q2[:, t, ...]
            K2_t = K2[:, t, ...]
            U2_t = U2[:, t, ...]
            W2_t = W2[:, t, ...]
            G_cumsum_last_t = G_cumsum_last[:, t, ...]

            skip_kv = ((t >> ell) & 1 == 1)
            skip_q  = not skip_kv
            period  = (1 << (ell + 1))
            skip_g  = ((t % period) == (period - 1))

            if not skip_q:
                pQ_t = Q2_t - einsum(Q_t, K_t, D_t, W2_t, "b ct h d, b cs h d, b ct cs h, b cs h dk -> b ct h dk")
                Y[:, t, ...] = Y[:, t, ...] + einsum(pQ_t, L_t, S, "b ct h dk, b ct h, b h dk dv -> b ct h dv")

            if not skip_g:
                S2 = einsum(torch.exp(G_cumsum_last_t), S, "b h, b h dk dv -> b h dk dv")
                S3 = einsum(K2_t, W2_t, S, "b ct h dk0, b ct h dk1, b h dk1 dv -> b h dk0 dv")
                S  = S2 - S3
            else:
                S = torch.zeros_like(S)

            if not skip_kv:
                KV_t = einsum(K2_t, U2_t, "b ct h dk, b ct h dv -> b h dk dv")
                S = S + KV_t

    return rearrange(Y, "b cn ct h d -> b (cn ct) h d")


def hgdn_minimal(
    Q: Float[torch.Tensor, "batch length num_heads d_state"],
    K: Float[torch.Tensor, "batch length num_heads d_state"],
    V: Float[torch.Tensor, "batch length num_heads d_head"],
    B: Float[torch.Tensor, "batch length num_heads"],
    G: Float[torch.Tensor, "batch length num_heads"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    level_base: int,
    htype: HType,
    block_len: int,
) -> Float[torch.Tensor, "batch length num_heads d_head"]:
    Y_diag = hgdn_minimal_diagonal_chunks(
        Q=Q,
        K=K,
        V=V,
        B=B,
        G=G,
        L=L,
        level_base=level_base,
        htype=htype,
        block_len=block_len)
    Y_off = hgdn_minimal_offdiagonal_chunks_v2(
        Q=Q,
        K=K,
        V=V,
        B=B,
        G=G,
        L=L,
        level_base=level_base,
        htype=htype,
        block_len=block_len)
    return Y_diag + Y_off


def hgdn_minimal_bwd_state_passing(
    Q: Float[torch.Tensor, "batch length num_heads d_state"],
    K: Float[torch.Tensor, "batch length num_heads d_state"],
    V: Float[torch.Tensor, "batch length num_heads d_head"],
    B: Float[torch.Tensor, "batch length num_heads"],
    G: Float[torch.Tensor, "batch length num_heads"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    dO: Float[torch.Tensor, "batch length num_heads d_head"],
    level_base: int,
    htype: HType,
    block_len: int,
    no_levels: bool = False,
) -> Tuple[Float[torch.Tensor, "batch num_chunks num_heads d_state d_head num_levels"],
           Float[torch.Tensor, "batch num_chunks num_heads d_state d_head num_levels"],
           Float[torch.Tensor, "batch length num_heads d_state"],
           Float[torch.Tensor, "batch length num_heads d_state"],
           Float[torch.Tensor, "batch length num_heads d_head"],
           Float[torch.Tensor, "batch length num_heads"],
           Float[torch.Tensor, "batch length num_heads"],
           Float[torch.Tensor, "batch length num_heads num_levels"],
           Float[torch.Tensor, "batch length num_heads d_state"],
           Float[torch.Tensor, "batch length num_heads d_head"]]:
    assert V.dtype == G.dtype == K.dtype == Q.dtype
    assert V.shape[1] % block_len == 0
    assert htype == HType.WEAK

    # Rearrange into blocks/chunks
    Q, K, V, B, G, L, dO = [rearrange(x, "b (cn ct) ... -> b cn ct ...", ct=block_len) for x in (Q, K, V, B, G, L, dO)]

    T  = make_T_matrix_v2(
        K=rearrange(K, "b cn cs h d -> b h cn cs d"),
        B=rearrange(B, "b cn ct h   -> b h cn ct  "),
        G=None)
    T2 = make_T_matrix_v2(
        K=rearrange(K, "b cn cs h d -> b h cn cs d"),
        B=rearrange(B, "b cn ct h   -> b h cn ct  "),
        G=rearrange(G, "b cn ct h   -> b h cn ct  "))
    D = torch.exp(segsum(
        rearrange(G, "b cn ct h -> b h cn ct")))
    T  = rearrange(T , "b h cn cs0 cs1 -> b cn cs0 cs1 h")
    T2 = rearrange(T2, "b h cn cs0 cs1 -> b cn cs0 cs1 h")
    D  = rearrange(D , "b h cn ct  cs  -> b cn ct  cs  h")
    W  = einsum(T , K, "b cn cs0 cs1 h, b cn cs1 h d -> b cn cs0 h d")
    U2 = einsum(T2, V, "b cn cs0 cs1 h, b cn cs1 h d -> b cn cs0 h d")

    mask = torch.ones(block_len, block_len, dtype=torch.bool, device=K.device)
    mask = torch.tril(mask, diagonal=0)
    mask = rearrange(mask, "ct cs  -> 1 1 ct cs 1")
    D = torch.masked_fill(D, ~mask, 0.)

    G_cumsum      = torch.cumsum(G, dim=-2)
    G_cumsum_last = G_cumsum[..., -1, :]
    G_cumsum_diff = rearrange(G_cumsum_last, "b cn h -> b cn 1 h") - G_cumsum

    Q2 = einsum(torch.exp(G_cumsum     ), Q, "b cn ct h, b cn ct h d -> b cn ct h d")
    K2 = einsum(torch.exp(G_cumsum_diff), K, "b cn ct h, b cn ct h d -> b cn ct h d")
    W2 = einsum(torch.exp(G_cumsum     ), W, "b cn ct h, b cn ct h d -> b cn ct h d")

    S = torch.zeros(
        (K.shape[0], K.shape[1], K.shape[3], K.shape[4], V.shape[4], L.shape[-1]),
        dtype=torch.float32,
        device=Q.device)
    dS = torch.zeros(
        (K.shape[0], K.shape[1], K.shape[3], K.shape[4], V.shape[4], L.shape[-1]),
        dtype=torch.float32,
        device=Q.device)
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dG = torch.zeros_like(G)
    dL = torch.zeros_like(L)
    dW = torch.zeros_like(W)
    dU2 = torch.zeros_like(U2)

    ell_h0 = get_num_levels(K.shape[2], level_base)
    num_chunk_levels = ceil_log(K.shape[1], level_base)
    if ell_h0 + num_chunk_levels > L.shape[-1]:
        raise ValueError
    for ell in range(num_chunk_levels):

        S_t = torch.zeros(
            (K.shape[0], K.shape[3], K.shape[4], V.shape[4]),
            dtype=torch.float32,
            device=K.device)
        dS_dt = torch.zeros(
            (K.shape[0], K.shape[3], K.shape[4], V.shape[4]),
            dtype=torch.float32,
            device=Q.device)

        for t in range(K.shape[1]):
            Q_t  = Q [:, t, ...]
            K_t  = K [:, t, ...]
            L_t  = L [:, t, ..., ell_h0 + ell]
            D_t  = D [:, t, ...]
            Q2_t = Q2[:, t, ...]
            K2_t = K2[:, t, ...]
            U2_t = U2[:, t, ...]
            W2_t = W2[:, t, ...]
            dO_t = dO[:, t, ...]
            G_cumsum_t      = G_cumsum     [:, t, ...]
            G_cumsum_last_t = G_cumsum_last[:, t, ...]

            skip_kv = ((t >> ell) & 1 == 1)
            skip_q  = not skip_kv
            period  = (1 << (ell + 1))
            skip_g  = ((t % period) == (period - 1))  # forward

            # Store the current state of hidden gradient for this chunk
            S[:, t, ..., ell_h0 + ell] = S_t.clone()

            if (not skip_g) or no_levels:
                S_t2 = einsum(torch.exp(G_cumsum_last_t), S_t, "b h, b h dk dv -> b h dk dv")
                S_t3 = einsum(K2_t, W2_t, S_t, "b ct h dk0, b ct h dk1, b h dk1 dv -> b h dk0 dv")
                S_t  = S_t2 - S_t3
            else:
                S_t = torch.zeros_like(S_t)

            if (not skip_kv) or no_levels:
                KV_t = einsum(K2_t, U2_t, "b ct h dk, b ct h dv -> b h dk dv")
                S_t = S_t + KV_t

            # gradients
            if not skip_q:
                dOW2S_t   = einsum(dO_t, W2_t, S[:, t, ..., ell_h0 + ell], "b ct h dv, b cs h dk, b h dk dv -> b ct cs h")
                dOW2SDL_t = einsum(dOW2S_t, D_t, L_t, "b ct cs h, b ct cs h, b ct h -> b ct cs h")
                dOS_t     = einsum(dO_t, S[:, t, ..., ell_h0 + ell], "b ct h dv, b h dk dv -> b ct h dk")
                QKD_t     = einsum(Q_t, K_t, D_t, "b ct h d, b cs h d, b ct cs h -> b ct cs h")

                LGdOS_t = einsum(L_t, torch.exp(G_cumsum_t), dOS_t, "b ct h, b ct h, b ct h dk -> b ct h dk")
                dQ[:, t, ...] = dQ[:, t, ...] + LGdOS_t - einsum(dOW2SDL_t, K_t, "b ct cs h, b cs h dk -> b ct h dk")
                dK[:, t, ...] = dK[:, t, ...]           - einsum(dOW2SDL_t, Q_t, "b ct cs h, b ct h dk -> b cs h dk")

                pQ_t = Q2_t - einsum(QKD_t, W2_t, "b ct cs h, b cs h dk -> b ct h dk")
                dOpQS_t = einsum(pQ_t, dOS_t, "b ct h dk, b ct h dk -> b ct h")
                dL[:, t, ..., ell_h0 + ell] = dL[:, t, ..., ell_h0 + ell] + dOpQS_t

                LQKDdOS_t  = einsum(L_t, QKD_t, dOS_t, "b ct h, b ct cs h, b ct h dk -> b cs h dk")
                LGQKDdOS_t = einsum(torch.exp(G_cumsum_t), LQKDdOS_t, "b cs h, b cs h dk -> b cs h dk")
                dW[:, t, ...] = dW[:, t, ...] - LGQKDdOS_t

        for dt in range(K.shape[1] - 1, -1, -1):
            Q_dt  = Q [:, dt, ...]
            K_dt  = K [:, dt, ...]
            L_dt  = L [:, dt, ..., ell_h0 + ell]
            D_dt  = D [:, dt, ...]
            Q2_dt = Q2[:, dt, ...]
            K2_dt = K2[:, dt, ...]
            U2_dt = U2[:, dt, ...]
            W2_dt = W2[:, dt, ...]
            dO_dt = dO[:, dt, ...]
            G_cumsum_dt      = G_cumsum     [:, dt, ...]
            G_cumsum_diff_dt = G_cumsum_diff[:, dt, ...]
            G_cumsum_last_dt = G_cumsum_last[:, dt, ...]

            dskip_kv = ((dt >> ell) & 1 == 1)
            dskip_q  = not dskip_kv
            period   = (1 << (ell + 1))
            dskip_g  = (dt % period == 0)  # backward

            # Store the current state of hidden gradient for this chunk
            dS[:, dt, ..., ell_h0 + ell] = dS_dt.clone()

            if (not dskip_g) or no_levels:
                dS_dt2 = einsum(torch.exp(G_cumsum_last_dt), dS_dt, "b h, b h dk dv -> b h dk dv")
                dS_dt3 = einsum(K2_dt, W2_dt, dS_dt, "b ct h dk0, b ct h dk1, b h dk1 dv -> b h dk0 dv")
                dS_dt  = dS_dt2 - dS_dt3
            else:
                dS_dt = torch.zeros_like(dS_dt)

            if (not dskip_q) or no_levels:
                pQ_dt   = Q2_dt - einsum(Q_dt, K_dt, D_dt, W2_dt, "b ct h d, b cs h d, b ct cs h, b cs h dk -> b ct h dk")
                QLdO_dt = einsum(pQ_dt, L_dt, dO_dt, "b ct h dk, b ct h, b ct h dv -> b h dk dv")
                dS_dt   = dS_dt + QLdO_dt

            # gradients
            dG[:, dt, -1, ...] = dG[:, dt, -1, ...] + einsum(torch.exp(G_cumsum_last_dt), S[:, dt, ..., ell_h0 + ell], dS[:, dt, ..., ell_h0 + ell], "b h, b h dk dv, b h dk dv -> b h")
            dG[:, dt, -1, ...] = dG[:, dt, -1, ...] - einsum(K2_dt, W2_dt, S[:, dt, ..., ell_h0 + ell], dS[:, dt, ..., ell_h0 + ell], "b ct h dk0, b ct h dk1, b h dk1 dv, b h dk0 dv -> b h")

            if not dskip_q:
                dOS_dt    = einsum(dO_dt, S[:, dt, ..., ell_h0 + ell], "b ct h dv, b h dk dv -> b ct h dk")
                LdOpQS_dt = einsum(L_dt, pQ_dt, dOS_dt, "b ct h, b ct h dk, b ct h dk -> b ct h")
                dG[:, dt, ...] = dG[:, dt, ...] + LdOpQS_dt

                # W2dOS_dt     = einsum(W2_dt, dOS_dt, "b cs h dk, b ct h dk -> b ct cs h")
                # LQKD_dt      = einsum(L_dt, Q_dt, K_dt, D_dt, "b ct h, b ct h d, b cs h d, b ct cs h -> b ct cs h")
                # LQKDW2dOS_dt = einsum(LQKD_dt, W2dOS_dt, "b ct cs h, b ct cs h -> b ct cs h")
                # dG[:, dt, ...] = dG[:, dt, ...] + reduce(LQKDW2dOS_dt, "b ct cs h -> b ct h", "sum")
                # dG[:, dt, ...] = dG[:, dt, ...] - reduce(LQKDW2dOS_dt, "b ct cs h -> b cs h", "sum")

            if not dskip_kv:
                U2dS_dt = einsum(U2_dt, dS[:, dt, ..., ell_h0 + ell], "b ct h dv, b h dk dv -> b ct h dk")
                K2dS_dt = einsum(K2_dt, dS[:, dt, ..., ell_h0 + ell], "b ct h dk, b h dk dv -> b ct h dv")
                U2dS_dt = einsum(torch.exp(G_cumsum_diff_dt), U2dS_dt, "b ct h, b ct h dk -> b ct h dk")
                dK [:, dt, ...] = dK [:, dt, ...] + U2dS_dt
                dU2[:, dt, ...] = dU2[:, dt, ...] + K2dS_dt

                dG[:, dt,     ...] = dG[:, dt,     ...] - einsum(K_dt, U2dS_dt, "b ct h dk, b ct h dk -> b ct h")
                dG[:, dt, -1, ...] = dG[:, dt, -1, ...] + einsum(K_dt, U2dS_dt, "b ct h dk, b ct h dk -> b    h")

            if not dskip_g:
                SdS_dt   = einsum(S[:, dt, ..., ell_h0 + ell], dS[:, dt, ..., ell_h0 + ell], "b h dk1 dv, b h dk0 dv -> b h dk1 dk0")
                W2SdS_dt = einsum(W2_dt, SdS_dt, "b ct h dk1, b h dk1 dk0 -> b ct h dk0")
                K2SdS_dt = einsum(K2_dt, SdS_dt, "b ct h dk0, b h dk1 dk0 -> b ct h dk1")
                W2SdS_dt = einsum(torch.exp(G_cumsum_diff_dt), W2SdS_dt, "b ct h, b ct h dk0 -> b ct h dk0")
                K2SdS_dt = einsum(torch.exp(G_cumsum_dt)     , K2SdS_dt, "b ct h, b ct h dk1 -> b ct h dk1")
                dK[:, dt, ...] = dK[:, dt, ...] - W2SdS_dt
                dW[:, dt, ...] = dW[:, dt, ...] - K2SdS_dt

        if no_levels:
            break

    # dK2 = einsum(dW , T , "b cn cs0 h d, b cn cs0 cs1 h -> b cn cs1 h d")
    # dV  = einsum(dU2, T2, "b cn cs0 h d, b cn cs0 cs1 h -> b cn cs1 h d")
    # dK  = dK + dK2

    # temporary get-arounds to compute the gradients of UT transform
    gK = K.clone().detach().requires_grad_()
    gV = V.clone().detach().requires_grad_()
    gB = B.clone().detach().requires_grad_()
    gG = G.clone().detach().requires_grad_()
    gT  = make_T_matrix_v2(
        K=rearrange(gK, "b cn cs h d -> b h cn cs d"),
        B=rearrange(gB, "b cn ct h   -> b h cn ct  "),
        G=None)
    gT2 = make_T_matrix_v2(
        K=rearrange(gK, "b cn cs h d -> b h cn cs d"),
        B=rearrange(gB, "b cn ct h   -> b h cn ct  "),
        G=rearrange(gG, "b cn ct h   -> b h cn ct  "))
    gT  = rearrange(gT , "b h cn cs0 cs1 -> b cn cs0 cs1 h")
    gT2 = rearrange(gT2, "b h cn cs0 cs1 -> b cn cs0 cs1 h")
    gW  = einsum(gT , gK, "b cn cs0 cs1 h, b cn cs1 h d -> b cn cs0 h d")
    gU2 = einsum(gT2, gV, "b cn cs0 cs1 h, b cn cs1 h d -> b cn cs0 h d")
    dK2,     dB2      = torch.autograd.grad(outputs=gW , inputs=(gK,     gB    ), grad_outputs=dW)
    dK3, dV, dB3, dG3 = torch.autograd.grad(outputs=gU2, inputs=(gK, gV, gB, gG), grad_outputs=dU2)
    dK = dK + dK2 + dK3
    dB =      dB2 + dB3

    dQ = rearrange(dQ, "b cn ct ... -> b (cn ct) ...")
    dK = rearrange(dK, "b cn ct ... -> b (cn ct) ...")
    dV = rearrange(dV, "b cn ct ... -> b (cn ct) ...")
    dB = rearrange(dB, "b cn ct ... -> b (cn ct) ...")
    dG = rearrange(dG, "b cn ct ... -> b (cn ct) ...")
    dL = rearrange(dL, "b cn ct ... -> b (cn ct) ...")
    dG = chunk_local_cumsum(dG, chunk_size=block_len, reverse=True, head_first=False)
    dG = dG + rearrange(dG3, "b cn ct ... -> b (cn ct) ...")
    dW = rearrange(dW, "b cn ct ... -> b (cn ct) ...")
    dU2 = rearrange(dU2, "b cn ct ... -> b (cn ct) ...")
    return S, dS, dQ, dK, dV, dB, dG, dL, dW, dU2


def hgdn_minimal_bwd_diagonal_chunks(
    Q: Float[torch.Tensor, "batch length num_heads d_state"],
    K: Float[torch.Tensor, "batch length num_heads d_state"],
    V: Float[torch.Tensor, "batch length num_heads d_head"],
    B: Float[torch.Tensor, "batch length num_heads"],
    G: Float[torch.Tensor, "batch length num_heads"],
    L: Float[torch.Tensor, "batch length num_heads num_levels"],
    dO: Float[torch.Tensor, "batch length num_heads d_head"],
    level_base: int,
    htype: HType,
    block_len: int,
) -> Tuple[Float[torch.Tensor, "batch length num_heads d_state"],
           Float[torch.Tensor, "batch length num_heads d_state"],
           Float[torch.Tensor, "batch length num_heads d_head"],
           Float[torch.Tensor, "batch length num_heads"],
           Float[torch.Tensor, "batch length num_heads"],
           Float[torch.Tensor, "batch length num_heads num_levels"]]:
    assert V.dtype == K.dtype == Q.dtype
    assert G.dtype == torch.float32
    assert V.shape[1] % block_len == 0
    assert htype == HType.WEAK

    # Rearrange into blocks/chunks
    Q, K, V, B, G, L, dO = [rearrange(x, "b (cn ct) ... -> b cn ct ...", ct=block_len) for x in (Q, K, V, B, G, L, dO)]

    T = make_T_matrix_v2(
        K=rearrange(K, "b cn cs h d -> b h cn cs d"),
        B=rearrange(B, "b cn ct h   -> b h cn ct  "),
        G=None)
    D = torch.exp(segsum(
        rearrange(G, "b cn ct h -> b h cn ct")))
    HD = make_masked_H_matrix(
        rearrange(L, "b cn ct h ell -> b h cn ct ell"),
        level_base=level_base,
        htype=htype,
        subdiagonal=False)
    T  = rearrange(T , "b h cn cs0 cs1 -> b cn cs0 cs1 h")
    D  = rearrange(D , "b h cn ct  cs  -> b cn ct  cs  h")
    HD = rearrange(HD, "b h cn ct  cs  -> b cn ct  cs  h")

    mask = torch.ones(block_len, block_len, dtype=torch.bool, device=K.device)
    mask = torch.tril(mask, diagonal=0)
    mask = rearrange(mask, "ct cs  -> 1 1 ct cs 1")
    QKT = einsum(Q, K, "b cn ct h d, b cn cs h d -> b cn ct cs h")
    QKT = torch.masked_fill(QKT, ~mask, 0.)
    QKTT = einsum(QKT, T, "b cn ct cs0 h, b cn cs0 cs1 h -> b cn ct cs1 h")
    QKTTH = QKTT * D.to(dtype=HD.dtype) * HD
    dV = einsum(QKTTH, dO, "b cn ct cs h, b cn ct h d -> b cn cs h d")

    dOV = einsum(dO, V, "b cn ct h d, b cn cs h d -> b cn ct cs h")
    dOVDL = dOV * D.to(dtype=HD.dtype) * HD
    dOVDLT = einsum(dOVDL, T, "b cn ct cs1 h, b cn cs0 cs1 h -> b cn ct cs0 h")
    dOVDLT = torch.masked_fill(dOVDLT, ~mask, 0.)
    dQ = einsum(dOVDLT, K, "b cn ct cs h, b cn cs h d -> b cn ct h d")
    dK = einsum(dOVDLT, Q, "b cn ct cs h, b cn ct h d -> b cn cs h d")

    dG_raw = einsum(QKTTH, dOV, "b cn ct cs h, b cn ct cs h -> b cn ct cs h")
    dG = reduce(dG_raw, "b cn ct cs h -> b cn ct h", "sum") - reduce(dG_raw, "b cn ct cs h -> b cn cs h", "sum")

    levels_matrix = make_levels_matrix(
        length=block_len,
        base=level_base,
        htype=htype,
        dtype=torch.int64,
        device=L.device)
    dL_raw = einsum(QKTT, D.to(dtype=HD.dtype), dOV, "b cn ct cs h, b cn ct cs h, b cn ct cs h -> b cn ct cs h")
    dL = torch.zeros_like(L)
    for ell in range(levels_matrix.max().item() + 1):
        level_mask = (levels_matrix == ell)
        level_mask = rearrange(level_mask, "ct cs  -> 1 1 ct cs 1")
        dL_raw_ell = torch.masked_fill(dL_raw, ~level_mask, 0.)
        dL[..., ell] = reduce(dL_raw_ell, "b cn ct cs h -> b cn ct h", "sum")

    # temporary get-arounds to compute the gradients of UT transform
    dT = einsum(dOVDL, QKT, "b cn ct cs1 h, b cn ct cs0 h -> b cn cs0 cs1 h")
    gK = K.clone().detach().requires_grad_()
    gB = B.clone().detach().requires_grad_()
    gT  = make_T_matrix_v2(
        K=rearrange(gK, "b cn cs h d -> b h cn cs d"),
        B=rearrange(gB, "b cn ct h   -> b h cn ct  "),
        G=None)
    gT  = rearrange(gT , "b h cn cs0 cs1 -> b cn cs0 cs1 h")
    dK2, dB = torch.autograd.grad(outputs=gT, inputs=(gK, gB), grad_outputs=dT)
    dK = dK + dK2

    dQ = rearrange(dQ, "b cn cs h d   -> b (cn cs) h d  ")
    dK = rearrange(dK, "b cn cs h d   -> b (cn cs) h d  ")
    dV = rearrange(dV, "b cn cs h d   -> b (cn cs) h d  ")
    dB = rearrange(dB, "b cn cs h     -> b (cn cs) h    ")
    dG = rearrange(dG, "b cn cs h     -> b (cn cs) h    ")
    dL = rearrange(dL, "b cn cs h ell -> b (cn cs) h ell")
    dG = chunk_local_cumsum(dG, chunk_size=block_len, reverse=True, head_first=False)
    return dQ, dK, dV, dB, dG, dL


def hgdn_minimal_bwd_ut(
    K: Float[torch.Tensor, "batch length num_heads d_state"],
    B: Float[torch.Tensor, "batch length num_heads"],
    dT: Float[torch.Tensor, "batch num_chunks chunk_length chunk_length num_heads"],
    block_len: int,
) -> Tuple[Float[torch.Tensor, "batch length num_heads d_state"],
           Float[torch.Tensor, "batch length num_heads"],
           Float[torch.Tensor, "batch length num_heads d_state"],
           Float[torch.Tensor, "batch length num_heads"]]:

    # Rearrange into blocks/chunks
    K, B = [rearrange(x, "b (cn ct) ... -> b cn ct ...", ct=block_len) for x in (K, B)]

    T = make_T_matrix_v2(
        K=rearrange(K, "b cn cs h d -> b h cn cs d"),
        B=rearrange(B, "b cn ct h   -> b h cn ct  "),
        G=None,
        fla_compatible=True)  # do not apply additional `B` scaling
    T = rearrange(T, "b h cn cs0 cs1 -> b cn cs0 cs1 h")
    mask = torch.ones(block_len, block_len, dtype=torch.bool, device=K.device)
    mask = torch.triu(mask, diagonal=0)
    mask = rearrange(mask, "cs0 cs1 -> 1 1 cs0 cs1 1")

    dTB = einsum(dT, B, "b cn cs0 cs1 h, b cn cs1 h -> b cn cs0 cs1 h")
    # section 2.2 in https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
    TdTBT = -einsum(T, dTB, T, "b cn cs00 cs1 h, b cn cs00 cs11 h, b cn cs0 cs11 h -> b cn cs1 cs0 h")
    TdTBT = torch.masked_fill(TdTBT, mask, 0.)
    # `cs1 cs0` in `TdTBT` will be remapped into `cs0 cs1` instead
    dK0 = einsum(TdTBT, B, K, "b cn cs0 cs1 h, b cn cs0 h  , b cn cs1 h d -> b cn cs0 h d")
    dK1 = einsum(TdTBT, B, K, "b cn cs0 cs1 h, b cn cs0 h  , b cn cs0 h d -> b cn cs1 h d")
    dB0 = einsum(TdTBT, K, K, "b cn cs0 cs1 h, b cn cs0 h d, b cn cs1 h d -> b cn cs0 h  ")
    dB1 = einsum(dT, T, "b cn cs0 cs1 h, b cn cs0 cs1 h -> b cn cs1 h")

    dK = dK0 + dK1
    dB = dB0 + dB1

    # reference gradients using PyTorch autograd
    gK = K.clone().detach().requires_grad_()
    gB = B.clone().detach().requires_grad_()
    gT  = make_T_matrix_v2(
        K=rearrange(gK, "b cn cs h d -> b h cn cs d"),
        B=rearrange(gB, "b cn ct h   -> b h cn ct  "),
        G=None)
    gT  = rearrange(gT , "b h cn cs0 cs1 -> b cn cs0 cs1 h")
    dK_, dB_ = torch.autograd.grad(outputs=gT, inputs=(gK, gB), grad_outputs=dT)

    dK  = rearrange(dK , "b cn cs h d -> b (cn cs) h d")
    dB  = rearrange(dB , "b cn cs h   -> b (cn cs) h  ")
    dK_ = rearrange(dK_, "b cn cs h d -> b (cn cs) h d")
    dB_ = rearrange(dB_, "b cn cs h   -> b (cn cs) h  ")
    return dK, dB, dK_, dB_
