import math
import torch
import torch.nn as nn

from einops import rearrange


# Adapted from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/full_attention.py
class FullAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, softmax_temp=None, attention_dropout=0.0, device=None, dtype=None,**kwargs):
        super().__init__()
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)

    def forward(self, query, key, value, **kwargs):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            query: (B, T, H, E) The tensor containing the query
            key: (B, S, H, E) The tensor containing the key
            value: (B, S, H, D) The tensor containing the value
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            key_padding_mask: An implementation of BaseMask that encodes how
                         many query each sequence in the batch consists of
        """
        key_padding_mask = kwargs["key_padding_mask"]
        query = rearrange(query, "b n (h d) -> b h n d", h=self.num_heads, d=self.dim_per_head)
        key = rearrange(key, "b n (h d) -> b h n d", h=self.num_heads, d=self.dim_per_head)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.num_heads, d=self.dim_per_head)
        
        
        B, T, H, E = query.shape
        _, S, _, D = value.shape
        softmax_temp = self.softmax_temp or 1 / math.sqrt(E)

        query = query * softmax_temp

        QK = torch.einsum("b h t e, b h s e -> b h t s", query, key)
        #if attn_mask is not None and not attn_mask.all_ones:
        #    QK.masked_fill_(~attn_mask.bool_matrix, float('-inf'))
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            QK.masked_fill_(rearrange(~key_padding_mask.bool_matrix, 'b s -> b 1 1 s'),
                            float('-inf'))

        attn = torch.softmax(QK, dim=-1)
        A = self.dropout(attn)


        output = torch.einsum("bhts,bhsd->bthd", A, value)

        output = rearrange(output, "b t h d -> (b t) (h d)")

        output = self.out_linear(output)
        return output
