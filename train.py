"""训练入口：BC 训练 Pointer-style 出牌/吃过策略网络。

用法（在 D:\\Data\\Card\\code 目录下执行）：

    python train.py                              # 默认配置，~10 epoch
    python train.py --batch_size 1024 --epochs 5
    python train.py --limit 50000 --epochs 2     # 小数据快速 smoke

数据集相对路径：../qc_report_with_guo_balanced_12/splits/{train,valid,test}.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DEFAULT_DATA_DIR, JsonlDataset, collate_fn, move_to, split_path
from model import PointerPolicy


HERE = Path(__file__).resolve().parent


# ------------------------- 评估 ------------------------- #

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None = None):
    model.eval()
    total = 0
    correct = 0
    correct_w = 0.0
    weight_sum = 0.0
    per_phase = defaultdict(lambda: [0, 0])          # phase -> [correct, total]
    per_source = defaultdict(lambda: [0, 0])         # source -> [correct, total]
    per_np = defaultdict(lambda: [0, 0])             # n_players -> [correct, total]
    # CHI vs GUO 混淆
    chi_guo_conf = torch.zeros(2, 2, dtype=torch.long)  # rows=expert(CHI/GUO), cols=pred
    aux_loss_sum = 0.0
    aux_n = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_to(batch, device)
        logits, state = model(batch)
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]
        weights = batch["weights"]
        ok = (preds == labels)
        correct += int(ok.sum())
        correct_w += float((ok.float() * weights).sum())
        weight_sum += float(weights.sum())
        total += labels.numel()

        # aux
        win_logits = model.aux_win_logits(state)
        aux_loss_sum += float(F.binary_cross_entropy_with_logits(win_logits, batch["win"], reduction="sum"))
        aux_n += labels.numel()

        # 分组
        phase_ids = batch["phase_ids"].tolist()
        sources = batch["sources"]
        nps = batch["n_players"].tolist()
        ok_cpu = ok.cpu().tolist()
        expert_types = batch["expert_types"]
        # 还原 CHI 阶段预测的 action_type
        pred_types: list[str] = []
        actions = batch["actions"]
        for b_idx in range(actions.shape[0]):
            ai = int(preds[b_idx])
            row = actions[b_idx, ai]
            # 前 7 维是 action_type one-hot；这里只关心 CHI / GUO
            t_idx = int(row[:7].argmax().item())
            from data import ACTION_TYPES
            pred_types.append(ACTION_TYPES[t_idx])

        for j, c in enumerate(ok_cpu):
            phase = "CHU" if phase_ids[j] == 0 else "CHI"
            per_phase[phase][0] += int(c)
            per_phase[phase][1] += 1
            per_source[sources[j]][0] += int(c)
            per_source[sources[j]][1] += 1
            per_np[nps[j]][0] += int(c)
            per_np[nps[j]][1] += 1
            if phase == "CHI":
                er = 0 if expert_types[j] == "CHI" else 1
                pr = 0 if pred_types[j] == "CHI" else 1
                chi_guo_conf[er, pr] += 1

    metrics = {
        "acc": correct / max(total, 1),
        "acc_weighted": correct_w / max(weight_sum, 1e-6),
        "aux_win_bce": aux_loss_sum / max(aux_n, 1),
        "total": total,
        "per_phase": {k: v[0] / max(v[1], 1) for k, v in per_phase.items()},
        "per_phase_count": {k: v[1] for k, v in per_phase.items()},
        "per_source": {k: v[0] / max(v[1], 1) for k, v in per_source.items()},
        "per_source_count": {k: v[1] for k, v in per_source.items()},
        "per_np": {int(k): v[0] / max(v[1], 1) for k, v in per_np.items()},
        "chi_guo_confusion": chi_guo_conf.tolist(),
    }
    return metrics


# ------------------------- 训练 ------------------------- #

def build_loader(split: str, batch_size: int, workers: int, shuffle: bool, data_dir: Path,
                 limit: int | None = None):
    ds = JsonlDataset(split_path(split, data_dir), limit=limit)
    return ds, DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        collate_fn=collate_fn, pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="JSONL 数据目录，默认 ../qc_report_with_guo_balanced_12/splits")
    parser.add_argument("--out_dir", type=str, default=str(HERE / "runs" / "bc"),
                        help="checkpoint 输出目录")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--aux_weight", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="只读取每个 split 的前 N 行（debug 用）")
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()

    print(f"[cfg] data_dir = {data_dir}")
    print(f"[cfg] out_dir  = {out_dir}")
    print(f"[cfg] device   = {args.device}")
    print(f"[cfg] amp      = {not args.no_amp}")

    train_ds, train_loader = build_loader("train", args.batch_size, args.workers, True, data_dir, args.limit)
    valid_ds, valid_loader = build_loader("valid", args.eval_batch_size, args.workers, False, data_dir, args.limit)
    print(f"[data] train={len(train_ds)}  valid={len(valid_ds)}")

    device = torch.device(args.device)
    model = PointerPolicy(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params = {n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    use_amp = (not args.no_amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    best_acc = -1.0
    step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_n = 0
        epoch_correct = 0
        for batch in train_loader:
            batch = move_to(batch, device)
            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits, state = model(batch)
                # 主任务：weighted CE，行内合法动作做 softmax
                logp = F.log_softmax(logits, dim=-1)
                nll = -logp.gather(1, batch["labels"].unsqueeze(1)).squeeze(1)
                w = batch["weights"]
                loss_main = (nll * w).sum() / w.sum().clamp(min=1e-6)
                # 辅助：胜负 BCE
                win_logits = model.aux_win_logits(state)
                loss_aux = F.binary_cross_entropy_with_logits(win_logits, batch["win"])
                loss = loss_main + args.aux_weight * loss_aux

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()
            scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                bsz = preds.numel()
                epoch_correct += int((preds == batch["labels"]).sum())
                epoch_loss += float(loss.detach()) * bsz
                epoch_n += bsz

            step += 1
            if step % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / max(epoch_n, 1)
                acc = epoch_correct / max(epoch_n, 1)
                elapsed = time.time() - t0
                print(f"[train] ep={epoch} step={step}/{total_steps} "
                      f"loss={avg_loss:.4f} acc={acc:.4f} lr={lr_now:.2e} "
                      f"elapsed={elapsed:.0f}s")

        # 每轮末评估
        val_metrics = evaluate(model, valid_loader, device, max_batches=args.max_eval_batches)
        print(f"[eval ] ep={epoch} acc={val_metrics['acc']:.4f} "
              f"acc_w={val_metrics['acc_weighted']:.4f} "
              f"aux_bce={val_metrics['aux_win_bce']:.4f}")
        print(f"        per_phase   = {val_metrics['per_phase']}")
        print(f"        per_source  = {val_metrics['per_source']}")
        print(f"        per_np      = {val_metrics['per_np']}")
        print(f"        chi_guo_cm  = {val_metrics['chi_guo_confusion']}  (rows=expert[CHI,GUO], cols=pred)")

        # 保存
        last_path = out_dir / "last.pt"
        torch.save({
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
        }, last_path)
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, out_dir / "best.pt")
            print(f"        [best] acc={best_acc:.4f} -> {out_dir / 'best.pt'}")

        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "val": val_metrics}, ensure_ascii=False) + "\n")

    print(f"[done] best valid acc = {best_acc:.4f}")


if __name__ == "__main__":
    main()
