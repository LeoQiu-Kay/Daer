"""把 frontend_exports 全量样本展平、去重并按 game_id 划分成 train/valid/test.jsonl.

数据源：`/root/autodl-tmp/luzhoudaer/frontend_exports/{YYYYMMDD}/{game_id}_r{round}.json`
每文件是 List[Sample]。本脚本：
  1. glob 所有文件
  2. 按 game_id 用稳定哈希分到 train/valid/test（同一局不跨集，避免泄漏）
  3. 逐文件展平，按 decision_id 去重（同一局可能跨日期重复导出）
  4. 每条样本写一行到对应 split 的 jsonl

用法（在远端 AutoDL 上跑一次）：
    cd /root/autodl-tmp/luzhoudaer
    python frontend_exports_train/prepare_dataset.py \
        --input_root /root/autodl-tmp/luzhoudaer/frontend_exports \
        --out_dir   /root/autodl-tmp/luzhoudaer/frontend_exports_splits

    # 只取部分日期 / 限量做 smoke：
    python frontend_exports_train/prepare_dataset.py --glob "2026060*/*.json" --max_files 2000
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from data import DATA_ROOT, DEFAULT_DATA_DIR, PHASES, TASKS


SPLITS = ("train", "valid", "test")


def assign_split(game_id: str, val_ratio: float, test_ratio: float) -> str:
    """按 game_id 的稳定哈希映射到 [0,1)，再按比例切分。同一 game_id 永远落同一 split。"""
    h = hashlib.md5(str(game_id).encode("utf-8")).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    if frac < test_ratio:
        return "test"
    if frac < test_ratio + val_ratio:
        return "valid"
    return "train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default=str(DATA_ROOT),
                        help="frontend_exports 根目录")
    parser.add_argument("--glob", type=str, default="*/*.json",
                        help="相对 input_root 的 glob（如 '2026060*/*.json'）")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--max_files", type=int, default=None, help="只处理前 N 个文件（debug）")
    parser.add_argument("--keep_tasks", type=str, nargs="*", default=None,
                        help="只保留这些 feedback_task（默认全保留）")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(input_root / args.glob)))
    if args.max_files is not None:
        files = files[: args.max_files]
    print(f"[prep] input_root = {input_root}")
    print(f"[prep] matched {len(files)} files via glob '{args.glob}'")
    if not files:
        raise SystemExit("没有匹配到任何文件，请检查 --input_root / --glob")

    fhs = {s: (out_dir / f"{s}.jsonl").open("w", encoding="utf-8") for s in SPLITS}
    seen_decision: set[str] = set()
    counts = {s: 0 for s in SPLITS}
    games = {s: set() for s in SPLITS}
    phase_counter: Counter[str] = Counter()
    task_counter: Counter[str] = Counter()
    n_dup = 0
    n_bad = 0
    t0 = time.time()

    try:
        for fi, fp in enumerate(files):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception as e:  # noqa: BLE001 — 损坏文件跳过，不中断全量
                n_bad += 1
                if n_bad <= 10:
                    print(f"[warn] 跳过损坏文件 {fp}: {e}")
                continue

            for s in records:
                gid = s.get("game_id")
                did = s.get("decision_id") or s.get("sample_id")
                if gid is None or did is None:
                    n_bad += 1
                    continue
                if did in seen_decision:
                    n_dup += 1
                    continue
                seen_decision.add(did)

                task = s.get("feedback_task")
                if args.keep_tasks and task not in args.keep_tasks:
                    continue

                split = assign_split(gid, args.val_ratio, args.test_ratio)
                fhs[split].write(json.dumps(s, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[split] += 1
                games[split].add(gid)
                phase_counter[s.get("observation", {}).get("phase", "?")] += 1
                task_counter[task or "?"] += 1

            if (fi + 1) % 5000 == 0:
                done = sum(counts.values())
                print(f"[prep] {fi+1}/{len(files)} files  samples={done}  "
                      f"elapsed={time.time()-t0:.0f}s")
    finally:
        for fh in fhs.values():
            fh.close()

    stats = {
        "input_root": str(input_root),
        "glob": args.glob,
        "n_files": len(files),
        "n_files_bad": n_bad,
        "n_duplicates_skipped": n_dup,
        "counts": counts,
        "games": {s: len(g) for s, g in games.items()},
        "per_phase": {p: phase_counter.get(p, 0) for p in PHASES},
        "per_task": {t: task_counter.get(t, 0) for t in TASKS},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "prepare_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[prep] done")
    print(f"  samples : {counts}  (total {sum(counts.values())})")
    print(f"  games   : {stats['games']}")
    print(f"  phases  : {stats['per_phase']}")
    print(f"  tasks   : {stats['per_task']}")
    print(f"  dup skip: {n_dup}   bad: {n_bad}")
    print(f"  -> {out_dir}/{{train,valid,test}}.jsonl   (+ prepare_stats.json)")


if __name__ == "__main__":
    main()
