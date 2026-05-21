"""Demo / 单条推理：加载 best.pt，对若干样本逐条打印状态、合法动作概率与专家动作对比。

示例：
    # 默认从 test.jsonl 随机抽 5 条
    python demo.py

    # 指定 ckpt / 数据集 / 数量
    python demo.py --ckpt runs/bc/best.pt --split test --n 8

    # 只看 CHI 阶段（吃/过）样本
    python demo.py --phase CHI --n 5

    # 指定单条样本的下标
    python demo.py --sample_index 1234
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from data import (
    DEFAULT_DATA_DIR, JsonlDataset, collate_fn, featurize, split_path,
)
from model import PointerPolicy


# ---------- 牌名 / 渲染 ---------- #
# 小写一…十 = 1..10，大写壹…拾 = 11..20
CARD_NAMES = [
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "玖", "拾",
]
# 红字牌：二七十 (2,7,10) 与 贰柒拾 (12,17,20)
RED_CARDS = {2, 7, 10, 12, 17, 20}


def card_name(c) -> str:
    if not isinstance(c, int) or not (1 <= c <= 20):
        return "?"
    name = CARD_NAMES[c - 1]
    return name + ("*" if c in RED_CARDS else " ")


def fmt_hand(counts) -> str:
    parts = []
    for i, n in enumerate(counts or []):
        if n and n > 0:
            parts.append(card_name(i + 1).strip() * int(n))
    return "  ".join(parts) if parts else "(空)"


def fmt_cards(cards) -> str:
    return "".join(card_name(c).strip() for c in (cards or []))


def fmt_action(a: dict) -> str:
    t = a.get("action_type")
    if t == "PLAY":
        return f"PLAY 打 {card_name(a.get('card')).strip()}"
    if t == "CHI":
        tgt = a.get("target_card")
        use = a.get("use_cards", []) or []
        bi = a.get("bi_cards", []) or []
        s = f"CHI 吃 {card_name(tgt).strip()} 用 {fmt_cards(use)}"
        if bi:
            s += f"（比 {fmt_cards(bi)}）"
        return s
    if t == "GUO":
        tgt = a.get("target_card")
        miss = a.get("missed_action", "")
        return f"GUO 过 (放弃 {miss} {card_name(tgt).strip()})"
    return f"{t} {a}"


def print_sample(sample: dict, probs: torch.Tensor, pred_idx: int) -> None:
    obs = sample["observation"]
    expert = sample["expert_action"]
    label = sample["label_index"]
    phase = obs.get("phase", "?")
    pid = obs.get("player_id")
    zhuang = obs.get("zhuang_player_id")
    n_players = obs.get("player_num")

    print("=" * 88)
    print(f"sample_id : {sample.get('sample_id')}")
    print(f"replay_id : {sample.get('replay_id')}   turn={sample.get('turn_index')}   "
          f"source={sample.get('source')}")
    self_tag = "(自己当庄)" if pid == zhuang else ""
    print(f"phase     : {phase}    人数={n_players}    自家={pid}    庄家={zhuang} {self_tag}")
    print(f"手牌      : {fmt_hand(obs.get('hand_cards', []))}    (* 表示红字)")

    # 桌面牌组
    tg = obs.get("table_groups") or {}
    if tg:
        print("桌面牌组  :")
        for tg_pid, groups in tg.items():
            tag = "[自家]" if int(tg_pid) == pid else "      "
            for g in groups:
                print(f"  {tag} pid={tg_pid:>10s}  {g.get('type','?'):11s} {fmt_cards(g.get('cards', []))}")

    # 弃牌
    dh = obs.get("discard_history") or {}
    if dh:
        print("弃牌历史  :")
        for d_pid, cs in dh.items():
            tag = "[自家]" if int(d_pid) == pid else "      "
            print(f"  {tag} pid={d_pid:>10s}  {fmt_cards(cs)}")

    ld = obs.get("last_discard") or {}
    if ld:
        print(f"上一弃牌  : pid={ld.get('player_id')} -> {card_name(ld.get('card')).strip()}")
    if obs.get("pending_card") is not None:
        print(f"待决策牌  : {card_name(obs['pending_card']).strip()}")
    if obs.get("remain_card_count") is not None:
        print(f"牌堆剩余  : {obs['remain_card_count']}")

    # 历史最近 8 步
    rh = obs.get("recent_history") or []
    if rh:
        print("近期动作 (最近 8 步) :")
        for h in rh[-8:]:
            hp = h.get("player_id")
            tag = "[自家]" if hp == pid else "      "
            ht = h.get("type", "?")
            if "cards" in h:
                print(f"  {tag} pid={hp}  {ht:11s} {fmt_cards(h.get('cards', []))}")
            else:
                print(f"  {tag} pid={hp}  {ht:11s} {card_name(h.get('card')).strip()}")

    # 概率排序
    n_acts = len(sample.get("legal_actions") or [])
    print()
    print(f"合法动作 ({n_acts} 个)  按模型概率降序：")
    rows = [(i, float(probs[i]), a) for i, a in enumerate(sample["legal_actions"])]
    rows.sort(key=lambda r: -r[1])
    for i, p, a in rows:
        marks = ""
        if i == label:
            marks += "  ★expert"
        if i == pred_idx:
            marks += "  ←pred"
        bar = "█" * max(1, int(round(p * 30)))
        print(f"  [{i:2d}] {p*100:6.2f}%  {bar:30s}  {fmt_action(a)}{marks}")

    pred = sample["legal_actions"][pred_idx]
    correct = "✓ 正确" if pred_idx == label else "✗ 错误"
    print()
    print(f"结论     : pred = {fmt_action(pred)}")
    print(f"           expert = {fmt_action(expert)}    →  {correct}")


# ---------- 主流程 ---------- #

def load_raw(path: Path, offset: int) -> dict:
    with open(path, "rb") as f:
        f.seek(offset)
        return json.loads(f.readline())


def pick_indices(ds: JsonlDataset, path: Path, n: int, phase: str | None, seed: int) -> list[int]:
    rng = random.Random(seed)
    if phase is None:
        return [rng.randrange(0, len(ds)) for _ in range(n)]
    picked: list[int] = []
    tried = 0
    while len(picked) < n and tried < n * 200:
        tried += 1
        idx = rng.randrange(0, len(ds))
        raw = load_raw(path, ds.offsets[idx])
        if raw.get("phase") == phase:
            picked.append(idx)
    if not picked:
        raise RuntimeError(f"找不到 phase={phase} 的样本")
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/bc/best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--n", type=int, default=5, help="抽取样本数")
    parser.add_argument("--sample_index", type=int, default=None, help="指定单条样本的下标，覆盖 --n / --phase")
    parser.add_argument("--phase", type=str, default=None, choices=["CHU", "CHI"], help="只展示指定阶段的样本")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Windows 控制台中文输出
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    device = torch.device(args.device)
    print(f"[ckpt] {args.ckpt}   device={device}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ck_args = ckpt.get("args", {})
    hidden = ck_args.get("hidden", 256)
    val = ckpt.get("val_metrics", {})
    print(f"[ckpt] hidden={hidden}   epoch={ckpt.get('epoch')}   "
          f"saved_valid_acc={val.get('acc')}")

    model = PointerPolicy(hidden=hidden).to(device).eval()
    model.load_state_dict(ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params = {n_params/1e6:.2f}M")

    path = split_path(args.split, Path(args.data_dir))
    print(f"[data] {path}")
    ds = JsonlDataset(path)
    print(f"[data] {args.split} = {len(ds)} samples\n")

    if args.sample_index is not None:
        idxs = [args.sample_index]
    else:
        idxs = pick_indices(ds, path, args.n, args.phase, args.seed)

    n_correct = 0
    for k, idx in enumerate(idxs):
        raw = load_raw(path, ds.offsets[idx])
        feat = featurize(raw)
        batch = collate_fn([feat])
        batch = {kk: vv.to(device) if torch.is_tensor(vv) else vv for kk, vv in batch.items()}
        with torch.no_grad():
            logits, _ = model(batch)
            probs = F.softmax(logits, dim=-1)[0].cpu()
        pred_idx = int(probs.argmax().item())

        print(f"\n########## {k+1}/{len(idxs)}   sample idx={idx} ##########")
        print_sample(raw, probs, pred_idx)
        if pred_idx == raw["label_index"]:
            n_correct += 1

    print()
    print("=" * 88)
    print(f"汇总: {n_correct}/{len(idxs)} 与专家一致")


if __name__ == "__main__":
    main()
