"""Dataset + featurization for 泸州大贰 行为克隆 (BC).

样本来自 ../qc_report_with_guo_balanced_12/splits/{train,valid,test}.jsonl
每行 1 个决策节点 (CHU 出牌 或 CHI 吃/过)。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (HERE / ".." / "qc_report_with_guo_balanced_12" / "splits").resolve()

NUM_CARDS = 20
MAX_PLAYERS = 4
MAX_HISTORY = 16

CHUNK_TYPES = ["CHUNK_PENG", "CHUNK_CHI", "CHUNK_ZHAO", "CHUNK_LONG"]
CHUNK_IDX = {t: i for i, t in enumerate(CHUNK_TYPES)}

HISTORY_TYPES = ["DRAW", "MO", "PLAY", "CHI", "CHUNK_PENG", "CHUNK_ZHAO", "CHUNK_LONG", "PENG", "ZHAO", "LONG"]
HISTORY_IDX = {t: i for i, t in enumerate(HISTORY_TYPES)}

ACTION_TYPES = ["PLAY", "CHI", "GUO", "PENG", "ZHAO", "LONG", "HU"]
ACTION_IDX = {t: i for i, t in enumerate(ACTION_TYPES)}

# specialOptions 枚举（覆盖训练集中观察到的值，OOV 直接忽略）
SPECIAL_OPTIONS = [
    "QUAN", "ZI_MO_DUBBLE", "SAN_ZHAO_HU", "LONG_3_KAN_4", "KONG_ZHONG_FEI",
    "GUI", "KUN_4FAN", "HEI_BAI", "HONG_DI", "TANG_DI_GUI", "QR_CODE",
    "START_DRAW", "FANG_PAO_BAO_PEI", "BA_GANG", "ER_WEI_MA", "QIA_LUAN",
    "JIA_DI", "HONG_DI_GUI",
]
SPECIAL_IDX = {t: i for i, t in enumerate(SPECIAL_OPTIONS)}

HISTORY_DIM = len(HISTORY_TYPES) + MAX_PLAYERS + NUM_CARDS
ACTION_DIM = len(ACTION_TYPES) + 2 * NUM_CARDS  # type + cards + bi_cards
SCALAR_DIM = MAX_PLAYERS + 2 + 3 + MAX_PLAYERS + 1 + len(SPECIAL_OPTIONS) + 5
CARD_CHANNELS = 1 + MAX_PLAYERS * len(CHUNK_TYPES) + MAX_PLAYERS + 1 + 1


# ------------------------- 单条样本特征化 ------------------------- #

def _put_one(v: torch.Tensor, c: int) -> None:
    """1..20 -> v[c-1] += 1，越界直接忽略。"""
    if isinstance(c, int) and 1 <= c <= NUM_CARDS:
        v[c - 1] += 1.0


def featurize(sample: dict[str, Any]) -> dict[str, torch.Tensor | int | float | str]:
    obs = sample["observation"]
    self_id = obs["player_id"]
    order = list(obs["player_order"])
    n_players = obs.get("player_num", len(order))
    zhuang_id = obs.get("zhuang_player_id")

    if self_id in order:
        self_idx = order.index(self_id)
    else:
        self_idx = 0
    # 相对位：self=0，下家=1，……
    pid2rel: dict[int, int] = {}
    for i, pid in enumerate(order):
        rel = (i - self_idx) % max(n_players, 1)
        pid2rel[int(pid)] = rel

    def rel_of(pid) -> int | None:
        if pid is None:
            return None
        try:
            r = pid2rel.get(int(pid))
        except (TypeError, ValueError):
            return None
        if r is None or r >= MAX_PLAYERS:
            return None
        return r

    # 1) 自家手牌 (20,)
    hand = torch.tensor(obs.get("hand_cards", [0] * NUM_CARDS), dtype=torch.float32)
    if hand.numel() != NUM_CARDS:
        # 兜底，理论上不会发生
        tmp = torch.zeros(NUM_CARDS)
        tmp[: hand.numel()] = hand[:NUM_CARDS]
        hand = tmp

    # 2) 桌面牌组 (MAX_PLAYERS, 4, NUM_CARDS)
    table = torch.zeros(MAX_PLAYERS, len(CHUNK_TYPES), NUM_CARDS)
    for pid_str, groups in (obs.get("table_groups") or {}).items():
        rel = rel_of(pid_str)
        if rel is None:
            continue
        for g in groups:
            ci = CHUNK_IDX.get(g.get("type"))
            if ci is None:
                continue
            for c in g.get("cards", []):
                _put_one(table[rel, ci], c)

    # 3) 弃牌 (MAX_PLAYERS, NUM_CARDS)
    discard = torch.zeros(MAX_PLAYERS, NUM_CARDS)
    for pid_str, cards in (obs.get("discard_history") or {}).items():
        rel = rel_of(pid_str)
        if rel is None:
            continue
        for c in cards:
            _put_one(discard[rel], c)

    # 4) pending / last_discard
    pending = torch.zeros(NUM_CARDS)
    pc = obs.get("pending_card")
    _put_one(pending, pc) if pc is not None else None

    last_disc = torch.zeros(NUM_CARDS)
    last_disc_pos = torch.zeros(MAX_PLAYERS)
    ld = obs.get("last_discard") or {}
    if ld:
        _put_one(last_disc, ld.get("card"))
        rel = rel_of(ld.get("player_id"))
        if rel is not None:
            last_disc_pos[rel] = 1.0

    # 5) phase / pnum / zhuang
    phase = torch.zeros(2)
    phase[0 if obs.get("phase") == "CHU" else 1] = 1.0
    pnum = torch.zeros(3)
    pnum[max(0, min(2, n_players - 2))] = 1.0
    zhuang_rel = torch.zeros(MAX_PLAYERS)
    zrel = rel_of(zhuang_id)
    if zrel is not None:
        zhuang_rel[zrel] = 1.0

    # 6) 牌堆剩余（仅 CHU 有数值）
    rc = obs.get("remain_card_count")
    remain = torch.tensor([float(rc) / 40.0 if rc is not None else 0.0], dtype=torch.float32)

    # 7) 规则
    rules = obs.get("game_rules") or {}
    rule_feat = torch.zeros(len(SPECIAL_OPTIONS) + 5)
    for opt in rules.get("specialOptions", []) or []:
        i = SPECIAL_IDX.get(opt)
        if i is not None:
            rule_feat[i] = 1.0
    rule_feat[-5] = float(rules.get("baoZiScore", 0) or 0)
    rule_feat[-4] = 1.0 if rules.get("canDouble") else 0.0
    rule_feat[-3] = 1.0 if rules.get("chiOtherPai") else 0.0
    rule_feat[-2] = 1.0 if rules.get("xiaojiaKanPai") else 0.0
    rule_feat[-1] = float(rules.get("topLimit", 0) or 0) / 128.0

    # 8) 近期历史
    hist_rows: list[torch.Tensor] = []
    for h in (obs.get("recent_history") or [])[-MAX_HISTORY:]:
        row = torch.zeros(HISTORY_DIM)
        hi = HISTORY_IDX.get(h.get("type"))
        if hi is not None:
            row[hi] = 1.0
        rel = rel_of(h.get("player_id"))
        if rel is not None:
            row[len(HISTORY_TYPES) + rel] = 1.0
        # card / cards
        base = len(HISTORY_TYPES) + MAX_PLAYERS
        if h.get("card") is not None:
            c = h["card"]
            if isinstance(c, int) and 1 <= c <= NUM_CARDS:
                row[base + c - 1] += 1.0
        for c in h.get("cards", []) or []:
            if isinstance(c, int) and 1 <= c <= NUM_CARDS:
                row[base + c - 1] += 1.0
        hist_rows.append(row)

    if hist_rows:
        hist = torch.stack(hist_rows)
    else:
        hist = torch.zeros(0, HISTORY_DIM)
    hist_mask = torch.zeros(MAX_HISTORY)
    hist_mask[: hist.shape[0]] = 1.0
    if hist.shape[0] < MAX_HISTORY:
        hist = torch.cat([hist, torch.zeros(MAX_HISTORY - hist.shape[0], HISTORY_DIM)], dim=0)

    # 9) 合法动作
    legal = sample.get("legal_actions") or []
    if legal:
        rows = []
        for a in legal:
            at = torch.zeros(len(ACTION_TYPES))
            ai = ACTION_IDX.get(a.get("action_type"))
            if ai is not None:
                at[ai] = 1.0
            cards = torch.zeros(NUM_CARDS)
            _put_one(cards, a.get("card")) if a.get("card") is not None else None
            _put_one(cards, a.get("target_card")) if a.get("target_card") is not None else None
            for c in a.get("use_cards", []) or []:
                _put_one(cards, c)
            bi = torch.zeros(NUM_CARDS)
            for c in a.get("bi_cards", []) or []:
                _put_one(bi, c)
            rows.append(torch.cat([at, cards, bi]))
        actions = torch.stack(rows)
    else:
        actions = torch.zeros(0, ACTION_DIM)

    # final_result -> aux: 玩家是否胡牌
    fr = sample.get("final_result") or {}
    self_fr = fr.get(str(self_id)) or fr.get(self_id) or {}
    win = 1.0 if self_fr.get("is_win") else 0.0
    score = float(self_fr.get("score", 0) or 0)

    return {
        "hand": hand,
        "table": table,
        "discard": discard,
        "pending": pending,
        "last_disc": last_disc,
        "last_disc_pos": last_disc_pos,
        "phase": phase,
        "pnum": pnum,
        "zhuang_rel": zhuang_rel,
        "remain": remain,
        "rules": rule_feat,
        "hist": hist,
        "hist_mask": hist_mask,
        "actions": actions,
        "num_actions": actions.shape[0],
        "label": int(sample.get("label_index", 0)),
        "weight": float(sample.get("sample_weight", 1.0)),
        "phase_id": 0 if obs.get("phase") == "CHU" else 1,
        "source": sample.get("source", "unknown"),
        "expert_type": (sample.get("expert_action") or {}).get("action_type", ""),
        "n_players": int(n_players),
        "win": win,
        "score": score,
    }


# ------------------------- Dataset ------------------------- #

class JsonlDataset(Dataset):
    """JSONL 数据集，使用字节偏移惰性读取，避免一次性把 ~1.5GB JSON 解析进内存。"""

    def __init__(self, path: str | os.PathLike, limit: int | None = None):
        self.path = str(Path(path))
        self.offsets: list[int] = []
        with open(self.path, "rb") as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line)
                if limit is not None and len(self.offsets) >= limit:
                    break
        self._fh: "Any | None" = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _ensure_open(self):
        if self._fh is None:
            self._fh = open(self.path, "rb")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    def __getitem__(self, idx: int):
        self._ensure_open()
        self._fh.seek(self.offsets[idx])
        line = self._fh.readline()
        sample = json.loads(line)
        return featurize(sample)


# ------------------------- Collate ------------------------- #

STACK_KEYS = (
    "hand", "table", "discard", "pending", "last_disc", "last_disc_pos",
    "phase", "pnum", "zhuang_rel", "remain", "rules", "hist", "hist_mask",
)


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch]) for k in STACK_KEYS}
    max_n = max(b["num_actions"] for b in batch)
    max_n = max(max_n, 1)
    actions = torch.zeros(B, max_n, ACTION_DIM)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["num_actions"]
        if n > 0:
            actions[i, :n] = b["actions"]
            mask[i, :n] = True
    out["actions"] = actions
    out["action_mask"] = mask
    out["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    out["weights"] = torch.tensor([b["weight"] for b in batch], dtype=torch.float32)
    out["phase_ids"] = torch.tensor([b["phase_id"] for b in batch], dtype=torch.long)
    out["n_players"] = torch.tensor([b["n_players"] for b in batch], dtype=torch.long)
    out["win"] = torch.tensor([b["win"] for b in batch], dtype=torch.float32)
    out["sources"] = [b["source"] for b in batch]
    out["expert_types"] = [b["expert_type"] for b in batch]
    return out


def move_to(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ------------------------- 辅助：构造 split 路径 ------------------------- #

def split_path(split: str, data_dir: str | os.PathLike | None = None) -> Path:
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return (data_dir / f"{split}.jsonl").resolve()
