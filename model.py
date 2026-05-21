"""Pointer-style policy network for 泸州大贰.

输入：见 data.featurize 的输出
输出：每条合法动作的 logit（带 mask） + 辅助胜负预测
"""

from __future__ import annotations

import torch
import torch.nn as nn

from data import (
    ACTION_DIM, CARD_CHANNELS, HISTORY_DIM, MAX_HISTORY, NUM_CARDS, SCALAR_DIM,
)


class CardEncoder(nn.Module):
    """Conv1D over (B, CARD_CHANNELS, NUM_CARDS=20)."""

    def __init__(self, in_channels: int = CARD_CHANNELS, hidden: int = 128, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, out_dim, kernel_size=1),
        )
        # 同时保留 max 和 mean 两种池化
        self.out_dim = out_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)  # (B, out, 20)
        return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)


class HistoryEncoder(nn.Module):
    def __init__(self, input_dim: int = HISTORY_DIM, hidden: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, MAX_HISTORY, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            batch_first=True, dropout=0.1, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_dim = hidden

    def forward(self, hist: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(hist) + self.pos[:, : hist.shape[1]]
        kpm = ~hist_mask.bool()  # True == padded
        # 防止整行全 padded 时报错
        all_pad = kpm.all(dim=-1)
        if all_pad.any():
            kpm = kpm.clone()
            kpm[all_pad, 0] = False
        out = self.encoder(x, src_key_padding_mask=kpm)
        m = hist_mask.float().unsqueeze(-1)
        pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return pooled


class StateEncoder(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.card = CardEncoder(out_dim=hidden)
        self.scalar = nn.Sequential(
            nn.Linear(SCALAR_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.history = HistoryEncoder(hidden=hidden)
        fused_in = self.card.out_dim + hidden + self.history.out_dim
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.out_dim = hidden

    def forward(self, batch: dict) -> torch.Tensor:
        B = batch["hand"].shape[0]
        card_in = torch.cat([
            batch["hand"].unsqueeze(1),                         # (B, 1, 20)
            batch["table"].view(B, -1, NUM_CARDS),              # (B, MAX_P*4, 20)
            batch["discard"],                                   # (B, MAX_P, 20)
            batch["pending"].unsqueeze(1),                      # (B, 1, 20)
            batch["last_disc"].unsqueeze(1),                    # (B, 1, 20)
        ], dim=1)
        card_vec = self.card(card_in)

        scalar_in = torch.cat([
            batch["last_disc_pos"],
            batch["phase"],
            batch["pnum"],
            batch["zhuang_rel"],
            batch["remain"],
            batch["rules"],
        ], dim=-1)
        scalar_vec = self.scalar(scalar_in)

        hist_vec = self.history(batch["hist"], batch["hist_mask"])

        return self.fuse(torch.cat([card_vec, scalar_vec, hist_vec], dim=-1))


class PointerPolicy(nn.Module):
    """对每个合法动作单独打分 -> softmax over legal actions."""

    def __init__(self, hidden: int = 256, action_dim: int = ACTION_DIM):
        super().__init__()
        self.encoder = StateEncoder(hidden=hidden)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # 辅助：胜负预测
        self.win_head = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encoder(batch)                                    # (B, H)
        actions = batch["actions"]                                     # (B, N, A)
        a_emb = self.action_proj(actions)                              # (B, N, H)
        s_exp = state.unsqueeze(1).expand(-1, a_emb.shape[1], -1)
        logits = self.scorer(torch.cat([s_exp, a_emb], dim=-1)).squeeze(-1)  # (B, N)
        mask = batch["action_mask"]
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits, state

    def aux_win_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.win_head(state).squeeze(-1)
