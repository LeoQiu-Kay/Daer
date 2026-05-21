"""Pointer-style policy network for 泸州大贰 (V2 · 容量增强版).

主要改动相对 V1：
  - CardEncoder：3 个残差 ConvBlock + GroupNorm，channel 384
  - HistoryEncoder：Transformer 4 层 / 8 头
  - 所有 MLP 加 Dropout
  - 默认 hidden=384（V1 默认 256）

注意：旧 checkpoint (V1 架构) 无法直接加载到 V2，需要重新训练。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from data import (
    ACTION_DIM, CARD_CHANNELS, HISTORY_DIM, MAX_HISTORY, NUM_CARDS, SCALAR_DIM,
)


class ConvBlock(nn.Module):
    """残差 1D conv block：Conv-GN-GELU-Dropout-Conv-GN  +  skip."""

    def __init__(self, dim: int, dropout: float = 0.1, groups: int = 8):
        super().__init__()
        gn_groups = max(1, min(groups, dim))
        # 找到能整除 dim 的最大 groups（不超过 8）
        while dim % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(gn_groups, dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(gn_groups, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(x + h)


class CardEncoder(nn.Module):
    """(B, CARD_CHANNELS, NUM_CARDS=20) -> (B, 2*hidden)，mean + max 双池化."""

    def __init__(self, in_channels: int = CARD_CHANNELS, hidden: int = 384,
                 n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([ConvBlock(hidden, dropout) for _ in range(n_blocks)])
        self.out_dim = hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for blk in self.blocks:
            h = blk(h)
        return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)


class HistoryEncoder(nn.Module):
    """近 N 步动作序列 -> 池化向量."""

    def __init__(self, input_dim: int = HISTORY_DIM, hidden: int = 384,
                 n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, MAX_HISTORY, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            batch_first=True, dropout=dropout, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_dim = hidden

    def forward(self, hist: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(hist) + self.pos[:, : hist.shape[1]]
        kpm = ~hist_mask.bool()  # True = padded
        all_pad = kpm.all(dim=-1)
        if all_pad.any():
            kpm = kpm.clone()
            kpm[all_pad, 0] = False
        out = self.encoder(x, src_key_padding_mask=kpm)
        m = hist_mask.float().unsqueeze(-1)
        return (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class StateEncoder(nn.Module):
    def __init__(self, hidden: int = 384, dropout: float = 0.1,
                 conv_blocks: int = 3, hist_layers: int = 4, hist_heads: int = 8):
        super().__init__()
        self.card = CardEncoder(hidden=hidden, n_blocks=conv_blocks, dropout=dropout)
        self.scalar = nn.Sequential(
            nn.Linear(SCALAR_DIM, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.history = HistoryEncoder(
            hidden=hidden, n_layers=hist_layers, n_heads=hist_heads, dropout=dropout,
        )
        fused_in = self.card.out_dim + hidden + self.history.out_dim  # 2H + H + H
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.out_dim = hidden

    def forward(self, batch: dict) -> torch.Tensor:
        B = batch["hand"].shape[0]
        card_in = torch.cat([
            batch["hand"].unsqueeze(1),
            batch["table"].view(B, -1, NUM_CARDS),
            batch["discard"],
            batch["pending"].unsqueeze(1),
            batch["last_disc"].unsqueeze(1),
        ], dim=1)
        card_vec = self.card(card_in)

        scalar_in = torch.cat([
            batch["last_disc_pos"], batch["phase"], batch["pnum"],
            batch["zhuang_rel"], batch["remain"], batch["rules"],
        ], dim=-1)
        scalar_vec = self.scalar(scalar_in)

        hist_vec = self.history(batch["hist"], batch["hist_mask"])

        return self.fuse(torch.cat([card_vec, scalar_vec, hist_vec], dim=-1))


class PointerPolicy(nn.Module):
    def __init__(self, hidden: int = 384, action_dim: int = ACTION_DIM, dropout: float = 0.1,
                 conv_blocks: int = 3, hist_layers: int = 4, hist_heads: int = 8):
        super().__init__()
        self.hidden = hidden
        self.encoder = StateEncoder(
            hidden=hidden, dropout=dropout, conv_blocks=conv_blocks,
            hist_layers=hist_layers, hist_heads=hist_heads,
        )
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.win_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encoder(batch)
        actions = batch["actions"]
        a_emb = self.action_proj(actions)
        s_exp = state.unsqueeze(1).expand(-1, a_emb.shape[1], -1)
        logits = self.scorer(torch.cat([s_exp, a_emb], dim=-1)).squeeze(-1)
        mask = batch["action_mask"]
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits, state

    def aux_win_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.win_head(state).squeeze(-1)
