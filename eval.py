"""加载 checkpoint，在任意 split 上跑评估。

用法：
    python eval.py --ckpt runs/bc/best.pt --split test
    python eval.py --ckpt runs/bc/best.pt --split valid --limit 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import DEFAULT_DATA_DIR, JsonlDataset, collate_fn, split_path
from model import PointerPolicy
from train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None,
                        help="若指定则覆盖 ckpt 中保存的 hidden 维度")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_args = ckpt.get("args", {})
    hidden = args.hidden if args.hidden is not None else ck_args.get("hidden", 256)
    print(f"[ckpt] {args.ckpt}  epoch={ckpt.get('epoch')}  hidden={hidden}")

    device = torch.device(args.device)
    model = PointerPolicy(hidden=hidden).to(device)
    model.load_state_dict(ckpt["model"])

    data_dir = Path(args.data_dir).resolve()
    ds = JsonlDataset(split_path(args.split, data_dir), limit=args.limit)
    print(f"[data] {args.split} = {len(ds)}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    metrics = evaluate(model, loader, device, max_batches=args.max_batches)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
