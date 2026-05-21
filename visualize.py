"""可视化训练曲线：读取 runs/bc/{train_log.jsonl, metrics.jsonl} 渲染一张 PNG。

用法：
    python visualize.py                                 # runs/bc/curves.png
    python visualize.py --run_dir runs/bc --out my.png  # 自定义输出
    python visualize.py --watch                         # 每 --interval 秒重新渲染一次

train.py 在每个 epoch 末会自动调用本模块的 render()。也可以在另一个终端开 --watch
来盯曲线。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return items


def render(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path is not None else run_dir / "curves.png"

    train_log = read_jsonl(run_dir / "train_log.jsonl")
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    cfg_path = run_dir / "config.json"
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    title = f"Daer BC Training  ·  {run_dir}"
    if cfg:
        title += (f"\nhidden={cfg.get('hidden')}  bs={cfg.get('batch_size')}  "
                  f"lr={cfg.get('lr')}  epochs={cfg.get('epochs')}  "
                  f"ls={cfg.get('label_smoothing')}  dropout={cfg.get('dropout')}")
    fig.suptitle(title, fontsize=12)

    # 1. Train loss
    ax = axes[0, 0]
    if train_log:
        xs = [r["step"] for r in train_log]
        ys = [r["loss"] for r in train_log]
        ax.plot(xs, ys, color="C0", linewidth=1.2)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(f"Train loss   (last={ys[-1]:.4f})")
    else:
        ax.set_title("Train loss   (no data)")
    ax.grid(alpha=0.3)

    # 2. Train acc + LR (twin axis)
    ax = axes[0, 1]
    if train_log:
        xs = [r["step"] for r in train_log]
        accs = [r["acc"] for r in train_log]
        lrs = [r["lr"] for r in train_log]
        ax.plot(xs, accs, color="C1", linewidth=1.2, label="train acc")
        ax.set_xlabel("step")
        ax.set_ylabel("train acc")
        ax.set_ylim(0, 1)
        ax2 = ax.twinx()
        ax2.plot(xs, lrs, color="C7", linewidth=1.0, linestyle="--", label="lr")
        ax2.set_ylabel("lr")
        ax.set_title(f"Train acc / LR   (acc={accs[-1]:.4f}, lr={lrs[-1]:.2e})")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax.set_title("Train acc / LR   (no data)")
    ax.grid(alpha=0.3)

    # 3. Valid acc by phase
    ax = axes[1, 0]
    if metrics:
        eps = [r["epoch"] for r in metrics]
        overall = [r["val"].get("acc", 0) for r in metrics]
        top3 = [r["val"].get("acc_top3", 0) for r in metrics]
        chu = [r["val"].get("per_phase", {}).get("CHU", 0) for r in metrics]
        chi = [r["val"].get("per_phase", {}).get("CHI", 0) for r in metrics]
        ax.plot(eps, overall, marker="o", linewidth=2.0, color="C0", label="overall (top-1)")
        ax.plot(eps, top3, marker="x", linewidth=1.5, color="C0", linestyle=":", alpha=0.6, label="overall (top-3)")
        ax.plot(eps, chu, marker="s", linewidth=1.5, color="C2", label="CHU (discard)")
        ax.plot(eps, chi, marker="^", linewidth=1.5, color="C3", label="CHI (chow/pass)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("valid acc")
        ax.set_ylim(0, 1)
        ax.set_title(f"Valid acc by phase   (best overall={max(overall):.4f})")
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.set_title("Valid acc by phase   (no data)")
    ax.grid(alpha=0.3)

    # 4. Valid acc by source
    ax = axes[1, 1]
    if metrics:
        eps = [r["epoch"] for r in metrics]
        sources: set[str] = set()
        for r in metrics:
            sources.update(r["val"].get("per_source", {}).keys())
        for i, s in enumerate(sorted(sources)):
            vals = [r["val"].get("per_source", {}).get(s, 0) for r in metrics]
            ax.plot(eps, vals, marker="o", linewidth=1.5, label=s)
        ax.set_xlabel("epoch")
        ax.set_ylabel("valid acc")
        ax.set_ylim(0, 1)
        ax.set_title("Valid acc by source")
        ax.legend(fontsize=7, loc="lower right")
    else:
        ax.set_title("Valid acc by source   (no data)")
    ax.grid(alpha=0.3)

    # 5. CHI-vs-GUO breakdown
    ax = axes[2, 0]
    if metrics:
        eps = [r["epoch"] for r in metrics]
        chi_recall: list[float] = []
        guo_recall: list[float] = []
        chi_vs_guo_acc: list[float] = []
        for r in metrics:
            cm = r["val"].get("chi_guo_confusion", [[0, 0], [0, 0]])
            tp_chi, fn = cm[0]
            fp, tp_guo = cm[1]
            chi_recall.append(tp_chi / max(tp_chi + fn, 1))
            guo_recall.append(tp_guo / max(fp + tp_guo, 1))
            total = tp_chi + fn + fp + tp_guo
            chi_vs_guo_acc.append((tp_chi + tp_guo) / max(total, 1))
        ax.plot(eps, chi_vs_guo_acc, marker="o", linewidth=2.0, color="C0", label="CHI-vs-GUO acc")
        ax.plot(eps, chi_recall, marker="s", linewidth=1.5, color="C2", label="CHI recall")
        ax.plot(eps, guo_recall, marker="^", linewidth=1.5, color="C3", label="GUO recall")
        ax.set_xlabel("epoch")
        ax.set_ylabel("rate")
        ax.set_ylim(0, 1)
        ax.set_title("CHI vs GUO (binary decision)")
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.set_title("CHI vs GUO   (no data)")
    ax.grid(alpha=0.3)

    # 6. Aux win BCE + acc_weighted
    ax = axes[2, 1]
    if metrics:
        eps = [r["epoch"] for r in metrics]
        aux = [r["val"].get("aux_win_bce", 0) for r in metrics]
        accw = [r["val"].get("acc_weighted", 0) for r in metrics]
        ax.plot(eps, aux, marker="o", color="C3", linewidth=1.5, label="aux win BCE")
        ax.set_xlabel("epoch")
        ax.set_ylabel("BCE")
        ax.set_title(f"Aux win BCE   (last={aux[-1]:.4f})")
        ax2 = ax.twinx()
        ax2.plot(eps, accw, marker="s", color="C0", linewidth=1.5, label="weighted acc")
        ax2.set_ylabel("weighted acc")
        ax2.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="lower right", fontsize=8)
    else:
        ax.set_title("Aux / weighted acc   (no data)")
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="runs/bc")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--watch", action="store_true", help="持续刷新")
    parser.add_argument("--interval", type=int, default=30, help="--watch 时刷新间隔（秒）")
    args = parser.parse_args()

    if args.watch:
        print(f"[viz] watching {args.run_dir}, refresh every {args.interval}s. Ctrl+C to stop.")
        while True:
            try:
                p = render(args.run_dir, args.out)
                print(f"[viz] {time.strftime('%H:%M:%S')} -> {p}")
            except Exception as e:
                print(f"[viz] error: {e}")
            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n[viz] stopped")
                break
    else:
        p = render(args.run_dir, args.out)
        print(f"[viz] -> {p}")


if __name__ == "__main__":
    main()
