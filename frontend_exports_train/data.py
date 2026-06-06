"""Dataset + 特征化 for 泸州大贰 人类反馈行为克隆（frontend_exports 版）.

数据来源：`/root/autodl-tmp/luzhoudaer/frontend_exports/{YYYYMMDD}/{game_id}_r{round}.json`
每个文件是 `List[Sample]`（一回合内多个决策点）。先用 `prepare_dataset.py` 把全量样本
按 `game_id` 划分、去重并展平成 `{train,valid,test}.jsonl`（每行一条样本），再用本模块的
`JsonlDataset`（字节偏移惰性读取）训练。

与旧 `qc_report` 版的主要差异：
  - `observation.table_groups`：dict[pid_str -> [{type, cards}]]，type ∈ {PENG,CHI,ZHAO,LONG}
  - `observation.discards` / `discard_history`：**list**[{player_id, cards/pass_cards}]
  - `observation.dead_cards`：全局已现牌的扁平 list
  - `observation.game_rules.special_way`：规则枚举（key 不再是 specialOptions）
  - phase ∈ {CHU, CHI, GUO, HU, BAO}；任务 feedback_task ∈ {play_rank, response_choice}
  - 没有 final_result（故无胜负辅助任务）
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
# 真实数据根（AutoDL）。本机不可见，prepare_dataset.py 在远端跑。
DATA_ROOT = Path("/root/autodl-tmp/luzhoudaer/frontend_exports")
# 展平/划分后的 jsonl 输出目录（prepare_dataset.py 写出，train/eval 读取）
DEFAULT_DATA_DIR = Path("/root/autodl-tmp/luzhoudaer/frontend_exports_splits")

NUM_CARDS = 20
MAX_PLAYERS = 4
MAX_HISTORY = 16

# 桌面牌组类型（碰 / 吃 / 招 / 龙）
CHUNK_TYPES = ["PENG", "CHI", "ZHAO", "LONG"]
CHUNK_IDX = {t: i for i, t in enumerate(CHUNK_TYPES)}

# 决策阶段
PHASES = ["CHU", "CHI", "GUO", "HU", "BAO"]
PHASE_IDX = {t: i for i, t in enumerate(PHASES)}

# 反馈任务
TASKS = ["play_rank", "response_choice"]
TASK_IDX = {t: i for i, t in enumerate(TASKS)}

# 动作类型（覆盖 PLAY 出牌 + 各类响应；OOV 忽略）
ACTION_TYPES = ["PLAY", "CHI", "PENG", "ZHAO", "LONG", "GUO", "HU", "BAO"]
ACTION_IDX = {t: i for i, t in enumerate(ACTION_TYPES)}

# recent_history 动作类型（多数样本为空，留作扩展/未来全量数据）
HISTORY_TYPES = ["DRAW", "MO", "PLAY", "CHI", "PENG", "ZHAO", "LONG", "GUO", "HU", "BAO"]
HISTORY_IDX = {t: i for i, t in enumerate(HISTORY_TYPES)}

# game_rules.special_way 枚举（覆盖数据中观察到的值，OOV 直接忽略）
SPECIAL_WAYS = [
    "ZI_MO_DUBBLE", "SAN_ZHAO_HU", "LONG_3_KAN_4", "KONG_ZHONG_FEI", "GUI",
    "HEI_BAI", "FANG_PAO_BAO_PEI", "SHABAO", "DINGBAO", "BAOHU", "KUN_HU",
    "TIAN_HU", "DI_HU", "HONG_HU", "HEI_HU", "WU_HU", "SHUI_SHANG_PIAO",
    "HAI_DI_LAO", "HAI_DI_PAO", "ZHENG_HU_DUBBLE", "QUAN", "ER_WEI_MA",
]
SPECIAL_IDX = {t: i for i, t in enumerate(SPECIAL_WAYS)}

HISTORY_DIM = len(HISTORY_TYPES) + MAX_PLAYERS + NUM_CARDS
ACTION_DIM = len(ACTION_TYPES) + 2 * NUM_CARDS  # type + cards + bi_cards
# scalar: last_disc_pos + phase + pnum + zhuang_rel + remain + task + rules
SCALAR_DIM = MAX_PLAYERS + len(PHASES) + 3 + MAX_PLAYERS + 1 + len(TASKS) + len(SPECIAL_WAYS)
# card 通道：hand + 桌面组(4 player ×4 type) + discards(4) + pass(4) + dead + pending + last_disc
CARD_CHANNELS = 1 + MAX_PLAYERS * len(CHUNK_TYPES) + MAX_PLAYERS + MAX_PLAYERS + 1 + 1 + 1


# ------------------------- 单条样本特征化 ------------------------- #

def _put_one(v: torch.Tensor, c: Any) -> None:
    """1..20 -> v[c-1] += 1，越界/非整数直接忽略。"""
    if isinstance(c, int) and 1 <= c <= NUM_CARDS:
        v[c - 1] += 1.0


def _put_many(v: torch.Tensor, cards: Any) -> None:
    for c in cards or []:
        _put_one(v, c)


def featurize(sample: dict[str, Any]) -> dict[str, Any]:
    obs = sample["observation"]
    self_id = obs.get("player_id")
    order = list(obs.get("player_order") or [])
    n_players = obs.get("player_num", len(order) or 1)
    zhuang_id = obs.get("zhuang_player_id")

    self_idx = order.index(self_id) if self_id in order else 0
    # 相对位：self=0，下家=1，……（按 player_order 顺序）
    pid2rel: dict[int, int] = {}
    for i, pid in enumerate(order):
        try:
            pid2rel[int(pid)] = (i - self_idx) % max(int(n_players), 1)
        except (TypeError, ValueError):
            continue

    def rel_of(pid: Any) -> int | None:
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
    hand = torch.zeros(NUM_CARDS)
    raw_hand = obs.get("hand_cards") or []
    for i in range(min(len(raw_hand), NUM_CARDS)):
        hand[i] = float(raw_hand[i])

    # 2) 桌面牌组 (MAX_PLAYERS, len(CHUNK_TYPES), NUM_CARDS)
    table = torch.zeros(MAX_PLAYERS, len(CHUNK_TYPES), NUM_CARDS)
    for pid_str, groups in (obs.get("table_groups") or {}).items():
        rel = rel_of(pid_str)
        if rel is None:
            continue
        for g in groups or []:
            ci = CHUNK_IDX.get(g.get("type"))
            if ci is None:
                continue
            _put_many(table[rel, ci], g.get("cards"))

    # 3) 各家弃牌 (MAX_PLAYERS, NUM_CARDS)（discards 为 list[{player_id, cards}]）
    discard = torch.zeros(MAX_PLAYERS, NUM_CARDS)
    for d in obs.get("discards") or []:
        rel = rel_of(d.get("player_id"))
        if rel is not None:
            _put_many(discard[rel], d.get("cards"))

    # 4) 各家“过”掉的牌 (MAX_PLAYERS, NUM_CARDS)（discard_history[].pass_cards）
    passed = torch.zeros(MAX_PLAYERS, NUM_CARDS)
    for d in obs.get("discard_history") or []:
        rel = rel_of(d.get("player_id"))
        if rel is not None:
            _put_many(passed[rel], d.get("pass_cards"))

    # 5) 全局已现牌 dead_cards (20,)
    dead = torch.zeros(NUM_CARDS)
    _put_many(dead, obs.get("dead_cards"))

    # 6) pending / last_discard
    pending = torch.zeros(NUM_CARDS)
    _put_one(pending, obs.get("pending_card"))

    last_disc = torch.zeros(NUM_CARDS)
    last_disc_pos = torch.zeros(MAX_PLAYERS)
    ld = obs.get("last_discard") or {}
    if ld:
        _put_one(last_disc, ld.get("card"))
        rel = rel_of(ld.get("player_id"))
        if rel is not None:
            last_disc_pos[rel] = 1.0

    # 7) phase / pnum / zhuang
    phase = torch.zeros(len(PHASES))
    pi = PHASE_IDX.get(obs.get("phase"))
    if pi is not None:
        phase[pi] = 1.0
    pnum = torch.zeros(3)
    pnum[max(0, min(2, int(n_players) - 2))] = 1.0
    zhuang_rel = torch.zeros(MAX_PLAYERS)
    zrel = rel_of(zhuang_id)
    if zrel is not None:
        zhuang_rel[zrel] = 1.0

    # 8) 牌堆剩余
    rc = obs.get("remain_card_count")
    remain = torch.tensor([float(rc) / 40.0 if rc is not None else 0.0], dtype=torch.float32)

    # 9) 任务 one-hot
    task = torch.zeros(len(TASKS))
    ti = TASK_IDX.get(sample.get("feedback_task"))
    if ti is not None:
        task[ti] = 1.0

    # 10) 规则 special_way multi-hot
    rule_feat = torch.zeros(len(SPECIAL_WAYS))
    for opt in (obs.get("game_rules") or {}).get("special_way") or []:
        i = SPECIAL_IDX.get(opt)
        if i is not None:
            rule_feat[i] = 1.0

    # 11) 近期历史（多数样本为空，编码器对全 padding 安全）
    hist_rows: list[torch.Tensor] = []
    for h in (obs.get("recent_history") or [])[-MAX_HISTORY:]:
        row = torch.zeros(HISTORY_DIM)
        hi = HISTORY_IDX.get(h.get("type"))
        if hi is not None:
            row[hi] = 1.0
        rel = rel_of(h.get("player_id"))
        if rel is not None:
            row[len(HISTORY_TYPES) + rel] = 1.0
        base = len(HISTORY_TYPES) + MAX_PLAYERS
        _put_one(row[base:], h.get("card"))
        _put_many(row[base:], h.get("cards"))
        hist_rows.append(row)

    hist = torch.stack(hist_rows) if hist_rows else torch.zeros(0, HISTORY_DIM)
    hist_mask = torch.zeros(MAX_HISTORY)
    hist_mask[: hist.shape[0]] = 1.0
    if hist.shape[0] < MAX_HISTORY:
        hist = torch.cat([hist, torch.zeros(MAX_HISTORY - hist.shape[0], HISTORY_DIM)], dim=0)

    # 12) 合法动作
    legal = sample.get("legal_actions") or []
    if legal:
        rows = []
        for a in legal:
            at = torch.zeros(len(ACTION_TYPES))
            ai = ACTION_IDX.get(a.get("action_type"))
            if ai is not None:
                at[ai] = 1.0
            cards = torch.zeros(NUM_CARDS)
            _put_one(cards, a.get("card"))
            _put_one(cards, a.get("target_card"))
            _put_many(cards, a.get("use_cards"))
            _put_many(cards, a.get("chi_cards"))
            bi = torch.zeros(NUM_CARDS)
            _put_many(bi, a.get("bi_cards"))
            rows.append(torch.cat([at, cards, bi]))
        actions = torch.stack(rows)
    else:
        actions = torch.zeros(0, ACTION_DIM)

    label = sample.get("label_index")
    if label is None:
        label = sample.get("human_selected_index", 0)

    return {
        "hand": hand,
        "table": table,
        "discard": discard,
        "passed": passed,
        "dead": dead,
        "pending": pending,
        "last_disc": last_disc,
        "last_disc_pos": last_disc_pos,
        "phase": phase,
        "pnum": pnum,
        "zhuang_rel": zhuang_rel,
        "remain": remain,
        "task": task,
        "rules": rule_feat,
        "hist": hist,
        "hist_mask": hist_mask,
        "actions": actions,
        "num_actions": actions.shape[0],
        "label": int(label),
        "weight": float(sample.get("sample_weight", 1.0)),
        "phase_id": PHASE_IDX.get(obs.get("phase"), 0),
        "task_id": TASK_IDX.get(sample.get("feedback_task"), 0),
        "expert_type": (sample.get("expert_action") or {}).get("action_type", ""),
        "n_players": int(n_players),
    }


# ------------------------- Dataset ------------------------- #

class JsonlDataset(Dataset):
    """JSONL 数据集（每行一条样本），用字节偏移惰性读取，避免一次性解析进内存。"""

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
        self._fh: Any | None = None

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
        return featurize(json.loads(line))


# ------------------------- Collate ------------------------- #

STACK_KEYS = (
    "hand", "table", "discard", "passed", "dead", "pending", "last_disc",
    "last_disc_pos", "phase", "pnum", "zhuang_rel", "remain", "task", "rules",
    "hist", "hist_mask",
)


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch]) for k in STACK_KEYS}
    max_n = max(max(b["num_actions"] for b in batch), 1)
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
    out["task_ids"] = torch.tensor([b["task_id"] for b in batch], dtype=torch.long)
    out["n_players"] = torch.tensor([b["n_players"] for b in batch], dtype=torch.long)
    out["expert_types"] = [b["expert_type"] for b in batch]
    return out


def move_to(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


# ------------------------- 辅助：构造 split 路径 ------------------------- #

def split_path(split: str, data_dir: str | os.PathLike | None = None) -> Path:
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return (data_dir / f"{split}.jsonl").resolve()
