"""训练入口：BC 训练 Pointer 策略网络，监督人类反馈（frontend_exports 版）.

任务：play_rank（出牌排序）+ response_choice（吃/碰/招/胡/过/爆 响应选择），混合训练。
损失：合法动作上的加权 label-smoothed CE（mask-safe）。无胜负辅助任务。

默认配置：hidden=384 conv=3 hist=4x8 dropout=0.1 bs=1024 lr=5e-4 epochs=25 warmup=2000 ls=0.05

用法：
    # 先在远端 prepare（一次）：
    python prepare_dataset.py --out_dir /root/autodl-tmp/luzhoudaer/frontend_exports_splits
    # 再训练：
    python train.py
    python train.py --epochs 30 --batch_size 2048 --hidden 512
    python train.py --limit 20000 --epochs 2 --workers 0   # smoke
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    ACTION_TYPES, DEFAULT_DATA_DIR, PHASES, TASKS,
    JsonlDataset, collate_fn, move_to, split_path,
)
from model import PointerPolicy


HERE = Path(__file__).resolve().parent


# ------------------------- 损失：合法动作上的 label-smoothed CE ------------------------- #

def masked_label_smoothed_nll(logits: torch.Tensor, labels: torch.Tensor,
                              mask: torch.Tensor, weights: torch.Tensor,
                              smoothing: float = 0.0) -> torch.Tensor:
    """logits (B,N) masked 位置已是 -inf；labels (B,)；mask (B,N) True=合法；weights (B,)."""
    B, N = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)
    valid = mask.float()
    n_valid = valid.sum(dim=-1).clamp(min=1.0)
    onehot = F.one_hot(labels, num_classes=N).float() * valid

    if smoothing > 0:
        target = (1.0 - smoothing) * onehot + smoothing * valid / n_valid.unsqueeze(-1)
    else:
        target = onehot

    log_probs_safe = log_probs.masked_fill(~mask, 0.0)
    nll = -(target * log_probs_safe).sum(dim=-1)
    return (nll * weights).sum() / weights.sum().clamp(min=1e-6)


# ------------------------- 评估 ------------------------- #

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             max_batches: int | None = None):
    model.eval()
    total = 0
    correct = 0
    correct_w = 0.0
    weight_sum = 0.0
    top3_correct = 0
    per_phase = defaultdict(lambda: [0, 0])
    per_task = defaultdict(lambda: [0, 0])
    per_np = defaultdict(lambda: [0, 0])
    # response_choice 里“是否选择放弃(GUO)”的 2×2 混淆矩阵：rows=expert[ACT,GUO], cols=pred
    act_guo_conf = torch.zeros(2, 2, dtype=torch.long)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_to(batch, device)
        logits = model(batch)
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]
        weights = batch["weights"]
        ok = (preds == labels)
        correct += int(ok.sum())
        correct_w += float((ok.float() * weights).sum())
        weight_sum += float(weights.sum())
        total += labels.numel()

        k = min(3, logits.shape[-1])
        topk = logits.topk(k, dim=-1).indices
        top3_correct += int((topk == labels.unsqueeze(-1)).any(dim=-1).sum())

        phase_ids = batch["phase_ids"].tolist()
        task_ids = batch["task_ids"].tolist()
        nps = batch["n_players"].tolist()
        ok_cpu = ok.cpu().tolist()
        expert_types = batch["expert_types"]
        actions = batch["actions"]
        n_atypes = len(ACTION_TYPES)
        pred_types: list[str] = []
        for b_idx in range(actions.shape[0]):
            ai = int(preds[b_idx])
            t_idx = int(actions[b_idx, ai, :n_atypes].argmax().item())
            pred_types.append(ACTION_TYPES[t_idx])

        for j, c in enumerate(ok_cpu):
            phase = PHASES[phase_ids[j]]
            task = TASKS[task_ids[j]]
            per_phase[phase][0] += int(c); per_phase[phase][1] += 1
            per_task[task][0] += int(c); per_task[task][1] += 1
            per_np[nps[j]][0] += int(c); per_np[nps[j]][1] += 1
            if task == "response_choice":
                er = 1 if expert_types[j] == "GUO" else 0
                pr = 1 if pred_types[j] == "GUO" else 0
                act_guo_conf[er, pr] += 1

    return {
        "acc": correct / max(total, 1),
        "acc_top3": top3_correct / max(total, 1),
        "acc_weighted": correct_w / max(weight_sum, 1e-6),
        "total": total,
        "per_phase": {k: v[0] / max(v[1], 1) for k, v in per_phase.items()},
        "per_phase_count": {k: v[1] for k, v in per_phase.items()},
        "per_task": {k: v[0] / max(v[1], 1) for k, v in per_task.items()},
        "per_task_count": {k: v[1] for k, v in per_task.items()},
        "per_np": {int(k): v[0] / max(v[1], 1) for k, v in per_np.items()},
        "act_guo_confusion": act_guo_conf.tolist(),
    }


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


def maybe_render(out_dir: Path):
    try:
        from visualize import render
        path = render(out_dir)
        print(f"[viz ] -> {path}")
    except Exception as e:  # noqa: BLE001
        print(f"[viz ] skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out_dir", type=str, default=str(HERE / "runs" / "bc"))
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--conv_blocks", type=int, default=3)
    parser.add_argument("--hist_layers", type=int, default=4)
    parser.add_argument("--hist_heads", type=int, default=8)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="每个 split 只读前 N 行（debug）")
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_logs", action="store_true",
                        help="不清空 train_log.jsonl / metrics.jsonl（默认开训前清空）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()

    train_log = out_dir / "train_log.jsonl"
    metrics_log = out_dir / "metrics.jsonl"
    config_path = out_dir / "config.json"
    if not args.keep_logs:
        for p in (train_log, metrics_log):
            if p.exists():
                p.unlink()
    config_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[cfg ] data_dir = {data_dir}")
    print(f"[cfg ] out_dir  = {out_dir}")
    print(f"[cfg ] device   = {args.device}   amp = {not args.no_amp}")
    print(f"[cfg ] hidden={args.hidden} conv={args.conv_blocks} hist={args.hist_layers}x{args.hist_heads} "
          f"dropout={args.dropout} ls={args.label_smoothing}")
    print(f"[cfg ] bs={args.batch_size} lr={args.lr} epochs={args.epochs} warmup={args.warmup_steps}")

    train_ds, train_loader = build_loader("train", args.batch_size, args.workers, True, data_dir, args.limit)
    valid_ds, valid_loader = build_loader("valid", args.eval_batch_size, args.workers, False, data_dir, args.limit)
    print(f"[data] train={len(train_ds)}  valid={len(valid_ds)}")

    device = torch.device(args.device)
    model = PointerPolicy(
        hidden=args.hidden, dropout=args.dropout,
        conv_blocks=args.conv_blocks, hist_layers=args.hist_layers, hist_heads=args.hist_heads,
    ).to(device)
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

    train_log_fh = train_log.open("a", encoding="utf-8")
    metrics_log_fh = metrics_log.open("a", encoding="utf-8")

    best_acc = -1.0
    step = 0
    t0 = time.time()

    try:
        for epoch in range(args.epochs):
            model.train()
            ep_loss = 0.0
            ep_n = 0
            ep_correct = 0
            for batch in train_loader:
                batch = move_to(batch, device)
                optim.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(batch)
                    loss = masked_label_smoothed_nll(
                        logits, batch["labels"], batch["action_mask"], batch["weights"],
                        smoothing=args.label_smoothing,
                    )

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
                    ep_correct += int((preds == batch["labels"]).sum())
                    ep_loss += float(loss.detach()) * bsz
                    ep_n += bsz

                step += 1
                if step % args.log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    avg_loss = ep_loss / max(ep_n, 1)
                    acc = ep_correct / max(ep_n, 1)
                    elapsed = time.time() - t0
                    print(f"[train] ep={epoch} step={step}/{total_steps} "
                          f"loss={avg_loss:.4f} acc={acc:.4f} lr={lr_now:.2e} "
                          f"elapsed={elapsed:.0f}s")
                    train_log_fh.write(json.dumps({
                        "step": step, "epoch": epoch,
                        "loss": avg_loss, "acc": acc, "lr": lr_now,
                        "elapsed": elapsed,
                    }) + "\n")
                    train_log_fh.flush()

            val_metrics = evaluate(model, valid_loader, device, max_batches=args.max_eval_batches)
            print(f"[eval ] ep={epoch} acc={val_metrics['acc']:.4f} "
                  f"top3={val_metrics['acc_top3']:.4f} "
                  f"acc_w={val_metrics['acc_weighted']:.4f}")
            print(f"        per_task    = {val_metrics['per_task']}")
            print(f"        per_phase   = {val_metrics['per_phase']}")
            print(f"        per_np      = {val_metrics['per_np']}")
            print(f"        act_guo_cm  = {val_metrics['act_guo_confusion']}  (rows=expert[ACT,GUO], cols=pred)")

            metrics_log_fh.write(json.dumps({"epoch": epoch, "val": val_metrics}, ensure_ascii=False) + "\n")
            metrics_log_fh.flush()

            torch.save({
                "model": model.state_dict(), "args": vars(args),
                "epoch": epoch, "val_metrics": val_metrics,
            }, out_dir / "last.pt")
            if val_metrics["acc"] > best_acc:
                best_acc = val_metrics["acc"]
                torch.save({
                    "model": model.state_dict(), "args": vars(args),
                    "epoch": epoch, "val_metrics": val_metrics,
                }, out_dir / "best.pt")
                print(f"        [best] acc={best_acc:.4f} -> {out_dir / 'best.pt'}")

            maybe_render(out_dir)
    finally:
        train_log_fh.close()
        metrics_log_fh.close()

    print(f"[done] best valid acc = {best_acc:.4f}")


if __name__ == "__main__":
    main()
