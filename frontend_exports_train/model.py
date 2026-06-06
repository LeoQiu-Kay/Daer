"""Pointer-style 策略网络 for 泸州大贰 人类反馈 BC（frontend_exports 版）.

结构沿用 qc_report 版 V2 的思路，但针对新数据做了调整：
  - CardEncoder 输入通道改为 CARD_CHANNELS（新增 passed / dead 通道）
  - StateEncoder 的 scalar 拼接含 task one-hot
  - 去掉胜负辅助头（新数据无 final_result）

变长合法动作：每个 legal_action 独立特征化，state ⊕ action_i 过 MLP 输出标量 logit，
行内 mask + softmax，损失只对 label_index 求 CE。CHU 出牌（多候选）与 response_choice
（吃/碰/招/胡/过/爆，少候选）共用同一套结构。
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
            batch["hand"].unsqueeze(1),               # (B,1,20)
            batch["table"].view(B, -1, NUM_CARDS),    # (B,16,20)
            batch["discard"],                         # (B,4,20)
            batch["passed"],                          # (B,4,20)
            batch["dead"].unsqueeze(1),               # (B,1,20)
            batch["pending"].unsqueeze(1),            # (B,1,20)
            batch["last_disc"].unsqueeze(1),          # (B,1,20)
        ], dim=1)
        card_vec = self.card(card_in)

        scalar_in = torch.cat([
            batch["last_disc_pos"], batch["phase"], batch["pnum"],
            batch["zhuang_rel"], batch["remain"], batch["task"], batch["rules"],
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

    def forward(self, batch: dict) -> torch.Tensor:
        state = self.encoder(batch)
        a_emb = self.action_proj(batch["actions"])
        s_exp = state.unsqueeze(1).expand(-1, a_emb.shape[1], -1)
        logits = self.scorer(torch.cat([s_exp, a_emb], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~batch["action_mask"], float("-inf"))
        return logits
