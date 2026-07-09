"""Transformer decoder + position encodings for the RABBiT ROI query stack.

Architecture: one learned query embedding per ROI; queries (optionally)
self-attend to one another, then cross-attend to the speech-token sequence,
then pass through a position-wise feed-forward block. ``num_layers`` of these
blocks are stacked. The output is one 256-dim token per ROI.

Pre-norm DETR convention: query positions are added to the inputs of attention
(not residual paths) so position information does not leak into ``value``
projections.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Temporal position encodings for speech tokens
# ─────────────────────────────────────────────────────────────────────────────


class TemporalPositionEmbeddingSine(nn.Module):
    """1-D sinusoidal positional encoding for temporal audio tokens.

    Honours an optional ``valid_mask`` so padded frames do not advance the
    position counter.
    """

    def __init__(
        self,
        d_model: int,
        temperature: float = 10_000.0,
        normalize: bool = True,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, dim], got {tuple(x.shape)}")

        batch_size, time_steps, _ = x.shape
        device = x.device
        dtype = x.dtype

        if valid_mask is None:
            valid_mask = torch.ones(batch_size, time_steps, device=device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)

        pos = valid_mask.to(torch.float32).cumsum(dim=1)
        if self.normalize:
            pos = pos / (pos[:, -1:].clamp_min(1.0)) * self.scale

        half_dim = max(1, self.d_model // 2)
        dim_t = torch.arange(half_dim, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / half_dim)
        angles = pos[..., None] / dim_t
        pos_embed = torch.cat((angles.sin(), angles.cos()), dim=-1)

        if pos_embed.shape[-1] < self.d_model:
            pad = self.d_model - pos_embed.shape[-1]
            pos_embed = torch.nn.functional.pad(pos_embed, (0, pad))
        elif pos_embed.shape[-1] > self.d_model:
            pos_embed = pos_embed[..., : self.d_model]

        return pos_embed.to(dtype=dtype)


class TemporalPositionEmbeddingLearned(nn.Module):
    """Learned absolute 1-D positional encoding for temporal audio tokens."""

    def __init__(self, d_model: int, max_positions: int = 4096) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_positions = max_positions
        self.position_embed = nn.Embedding(max_positions, d_model)
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, dim], got {tuple(x.shape)}")

        batch_size, time_steps, _ = x.shape
        if time_steps > self.max_positions:
            raise ValueError(
                f"Sequence length {time_steps} exceeds learned positional table "
                f"size {self.max_positions}."
            )

        position_ids = torch.arange(time_steps, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embed = self.position_embed(position_ids)
        if valid_mask is not None:
            pos_embed = pos_embed * valid_mask.to(device=x.device, dtype=pos_embed.dtype).unsqueeze(-1)
        return pos_embed.to(dtype=x.dtype)


def build_temporal_position_encoding(
    position_embedding_type: str,
    hidden_dim: int,
    max_positions: int = 4096,
) -> nn.Module:
    if position_embedding_type in {"sine", "sinusoidal", "v2"}:
        return TemporalPositionEmbeddingSine(hidden_dim)
    if position_embedding_type in {"learned", "v3"}:
        return TemporalPositionEmbeddingLearned(hidden_dim, max_positions=max_positions)
    raise ValueError(f"Unsupported temporal position encoding '{position_embedding_type}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Decoder block + stack
# ─────────────────────────────────────────────────────────────────────────────


class TemporalCrossAttentionBlock(nn.Module):
    """One decoder layer: optional self-attn over ROI queries → cross-attn to
    speech tokens → position-wise FFN. Pre-norm residuals throughout.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_self_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_self_attention = use_self_attention

        if use_self_attention:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_sa = nn.LayerNorm(d_model)
            self.dropout_sa = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    @staticmethod
    def _add_pos(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        query_tokens: Tensor,
        memory_tokens: Tensor,
        *,
        query_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        memory_valid_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if self.use_self_attention:
            q_norm = self.norm_sa(query_tokens)
            q_with_pos = self._add_pos(q_norm, query_pos)
            sa_out, _ = self.self_attn(
                query=q_with_pos,
                key=q_with_pos,
                value=q_norm,
                need_weights=False,
            )
            query_tokens = query_tokens + self.dropout_sa(sa_out)

        query_norm = self.norm1(query_tokens)
        query = self._add_pos(query_norm, query_pos)
        key = self._add_pos(memory_tokens, memory_pos)
        key_padding_mask = None
        if memory_valid_mask is not None:
            key_padding_mask = ~memory_valid_mask.to(torch.bool)

        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=key,
            value=memory_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        query_tokens = query_tokens + self.dropout1(attn_out)
        query_tokens = query_tokens + self.dropout2(self.ffn(self.norm2(query_tokens)))
        return query_tokens, attn_weights


class TemporalROIDecoder(nn.Module):
    """Stack of cross-attention blocks. Initial queries are zeros; each
    block sees the per-ROI ``query_pos`` (the learned ROI embedding).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_self_attention: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TemporalCrossAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    use_self_attention=use_self_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query_embed: Tensor,
        memory_tokens: Tensor,
        *,
        memory_pos: Optional[Tensor] = None,
        memory_valid_mask: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        batch_size, _, hidden_dim = memory_tokens.shape
        query_pos = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        query_tokens = torch.zeros(
            batch_size, query_embed.shape[0], hidden_dim,
            device=memory_tokens.device, dtype=memory_tokens.dtype,
        )

        attn_maps: list[Tensor] = []
        for layer in self.layers:
            query_tokens, attn = layer(
                query_tokens,
                memory_tokens,
                query_pos=query_pos,
                memory_pos=memory_pos,
                memory_valid_mask=memory_valid_mask,
                need_weights=return_attn,
            )
            if return_attn and attn is not None:
                attn_maps.append(attn)

        return self.norm(query_tokens), attn_maps
