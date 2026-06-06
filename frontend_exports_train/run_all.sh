#!/usr/bin/env bash
# 一键跑通全流程：prepare -> train -> eval（frontend_exports 人类反馈 BC）。
#
# 用法：
#   bash run_all.sh                 # 全量：准备数据 + 训练 + 测试集评估
#   SMOKE=1 bash run_all.sh         # 冒烟：只取 2000 个文件 / 20000 样本 / 2 epoch
#   SKIP_PREPARE=1 bash run_all.sh  # 跳过数据准备（splits 已存在时复用）
#
# 常用可调环境变量（均有默认值）：
#   INPUT_ROOT   原始数据根       (default: /root/autodl-tmp/luzhoudaer/frontend_exports)
#   SPLIT_DIR    划分输出目录     (default: /root/autodl-tmp/luzhoudaer/frontend_exports_splits)
#   OUT_DIR      训练产物目录     (default: <脚本目录>/runs/bc)
#   EPOCHS BATCH_SIZE HIDDEN LR WORKERS   训练超参
set -euo pipefail

# 脚本所在目录（无论从哪里调用都能定位代码）
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 让 data.py / model.py 等可被 import
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
PY="${PYTHON:-python}"

# ---- 配置（环境变量可覆盖） ----
INPUT_ROOT="${INPUT_ROOT:-/root/autodl-tmp/luzhoudaer/frontend_exports}"
SPLIT_DIR="${SPLIT_DIR:-/root/autodl-tmp/luzhoudaer/frontend_exports_splits}"
OUT_DIR="${OUT_DIR:-$HERE/runs/bc}"

EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
HIDDEN="${HIDDEN:-384}"
LR="${LR:-5e-4}"
WORKERS="${WORKERS:-4}"

# 冒烟模式：小规模快速验证全链路
PREP_EXTRA=""
TRAIN_EXTRA=""
EVAL_EXTRA=""
if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "[run_all] SMOKE 模式：少量文件 / 限量样本 / 2 epoch"
  PREP_EXTRA="--max_files 2000"
  TRAIN_EXTRA="--limit 20000 --epochs 2 --batch_size 256 --workers 0"
  EVAL_EXTRA="--limit 20000 --workers 0"
fi

echo "============================================================"
echo "[run_all] HERE       = $HERE"
echo "[run_all] INPUT_ROOT = $INPUT_ROOT"
echo "[run_all] SPLIT_DIR  = $SPLIT_DIR"
echo "[run_all] OUT_DIR    = $OUT_DIR"
echo "[run_all] epochs=$EPOCHS bs=$BATCH_SIZE hidden=$HIDDEN lr=$LR workers=$WORKERS"
echo "============================================================"

# ---- 1) 准备数据（按 game_id 划分 + 去重 + 展平成 jsonl） ----
if [[ "${SKIP_PREPARE:-0}" == "1" ]]; then
  echo "[run_all] (1/3) 跳过 prepare（SKIP_PREPARE=1）"
elif [[ -f "$SPLIT_DIR/train.jsonl" && "${SMOKE:-0}" != "1" ]]; then
  echo "[run_all] (1/3) 检测到 $SPLIT_DIR/train.jsonl 已存在，跳过 prepare（删除该目录可重建）"
else
  echo "[run_all] (1/3) prepare_dataset.py ..."
  $PY prepare_dataset.py \
    --input_root "$INPUT_ROOT" \
    --out_dir "$SPLIT_DIR" \
    $PREP_EXTRA
fi

# ---- 2) 训练 ----
echo "[run_all] (2/3) train.py ..."
$PY train.py \
  --data_dir "$SPLIT_DIR" \
  --out_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --hidden "$HIDDEN" \
  --lr "$LR" \
  --workers "$WORKERS" \
  $TRAIN_EXTRA

# ---- 3) 测试集评估 ----
echo "[run_all] (3/3) eval.py (test split) ..."
$PY eval.py \
  --ckpt "$OUT_DIR/best.pt" \
  --data_dir "$SPLIT_DIR" \
  --split test \
  --out_json "$OUT_DIR/test_metrics.json" \
  $EVAL_EXTRA

echo "============================================================"
echo "[run_all] 全流程完成 ✓"
echo "  best ckpt   : $OUT_DIR/best.pt"
echo "  曲线图      : $OUT_DIR/curves.png"
echo "  测试集指标  : $OUT_DIR/test_metrics.json"
echo "============================================================"
