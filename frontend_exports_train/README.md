# 泸州大贰 · 人类反馈行为克隆（frontend_exports 版）

Pointer-style 策略网络，监督学习人类反馈数据中的两类决策：

- **`play_rank`**（出牌排序，phase=`CHU`）：从多张合法出牌里选一张
- **`response_choice`**（响应选择，phase ∈ `CHI/GUO/HU/BAO`）：在 吃/碰/招/胡/过/爆 中选动作

两类任务**混合训练**，共用一套变长合法动作的 pointer 结构。

数据格式见 `../data/README.md`，单文件示例见 `../data/2831330058_r9.json`。

## 目录

```
frontend_exports_train/
├── prepare_dataset.py   # frontend_exports/*/*.json → 按 game_id 划分 → {train,valid,test}.jsonl
├── data.py              # JsonlDataset (byte-offset 惰性加载) + 特征化 + collate
├── model.py             # 残差 Conv + 4 层 Transformer + dropout 的 PointerPolicy（无胜负头）
├── train.py             # 训练入口；step/epoch 双层日志，每轮自动渲染 curves.png
├── eval.py              # 单独评估
├── demo.py              # 单条样本推理可视化（控制台打印）
├── visualize.py         # 训练曲线可视化（PNG，支持 --watch 持续刷新）
├── requirements.txt
└── runs/bc/             # 训练产物：best.pt / last.pt / config.json /
                         #          train_log.jsonl / metrics.jsonl / curves.png
```

路径默认值（AutoDL 部署，本机不可见）：

- 原始数据根：`/root/autodl-tmp/luzhoudaer/frontend_exports`
- 展平/划分后：`/root/autodl-tmp/luzhoudaer/frontend_exports_splits`

## 安装

```powershell
cd D:\Data\Card\code\frontend_exports_train
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CUDA / PyTorch 版本按本机 GPU 配置即可（CPU 也能跑，慢得多）。

## 0. 准备数据（在远端跑一次）

原始数据是 `~18.7 万` 个 `List[Sample]` JSON，需要先展平 + 按 `game_id` 划分（同一局
不跨 split，避免泄漏），并按 `decision_id` 去重（同一局可能跨日期重复导出）：

```bash
cd /root/autodl-tmp/luzhoudaer
export PYTHONPATH="$PWD/frontend_exports_train:${PYTHONPATH:-}"
python frontend_exports_train/prepare_dataset.py \
    --input_root /root/autodl-tmp/luzhoudaer/frontend_exports \
    --out_dir   /root/autodl-tmp/luzhoudaer/frontend_exports_splits

# 只取部分日期 / 限量做 smoke：
python frontend_exports_train/prepare_dataset.py --glob "2026060*/*.json" --max_files 2000
```

输出：`{train,valid,test}.jsonl`（每行一条样本）+ `prepare_stats.json`（样本/牌局/阶段/任务统计）。
默认划分比例 train/valid/test = 0.90/0.05/0.05，可用 `--val_ratio --test_ratio` 调整。

## 1. 训练

```bash
# 默认配置：hidden=384, bs=1024, lr=5e-4, 25 epoch, label_smoothing=0.05, AMP 自动开
python train.py

# 加大容量
python train.py --hidden 512 --batch_size 2048 --epochs 30

# Smoke / 调试（每个 split 只取前 20000 条，2 epoch，单进程）
python train.py --limit 20000 --epochs 2 --workers 0 --batch_size 256
```

训练过程：

- 每 `--log_every` 步追加一条到 `runs/bc/train_log.jsonl`
- 每 epoch 末跑 valid，追加到 `runs/bc/metrics.jsonl`
- 按 valid acc 保留 `best.pt`，每轮覆盖写 `last.pt`，启动时把配置 dump 到 `config.json`
- **每轮自动调用 `visualize.render`，写出 `runs/bc/curves.png`**
- 默认开训前清空 `train_log.jsonl / metrics.jsonl`，要保留旧记录加 `--keep_logs`

## 2. 可视化训练曲线

`visualize.py` 把 `train_log.jsonl + metrics.jsonl` 渲染成一张 6 子图 PNG：
train loss / train acc + lr / valid acc by task / valid acc by phase /
ACT-vs-GUO 分解 / weighted acc。

```bash
python visualize.py                                  # -> runs/bc/curves.png
python visualize.py --run_dir runs/bc --out my.png
python visualize.py --watch --interval 30            # 另开终端实时盯曲线
```

## 3. 评估

```powershell
python eval.py --ckpt runs/bc/best.pt --split test
python eval.py --ckpt runs/bc/best.pt --split valid --limit 20000
```

输出含 `acc` / `acc_top3` / `acc_weighted`、`per_task`（play_rank / response_choice）、
`per_phase`（CHU/CHI/GUO/HU/BAO）、`per_np`（按人数）、`act_guo_confusion`
（response_choice 中“出手 vs 放弃(GUO)”的 2×2 混淆矩阵）。

## 4. 推理 / Demo

```bash
python demo.py                                  # 从 test.jsonl 随机抽 5 条
python demo.py --ckpt runs/bc/best.pt --n 8
python demo.py --task response_choice           # 只看响应选择
python demo.py --phase CHI                       # 只看 CHI 阶段
python demo.py --sample_index 1234
```

逐条打印：手牌、各家桌面牌组、弃牌、待决策牌、所有合法动作的模型概率（降序，
`★expert` 标专家、`←pred` 标模型选择），以及最终是否一致。红字牌（2/7/10/12/17/20）后加 `*`。

## 设计要点

1. **变长合法动作（pointer）**：不固定动作空间。每个 `legal_actions[i]` 独立特征化为
   `[动作类型 one-hot ⊕ 涉及牌 multi-hot ⊕ bi_cards multi-hot]`，与状态向量拼接过 MLP
   输出标量 logit，行内 mask + softmax。损失只对 `label_index` 求 CE，天然兼容
   `play_rank`（多候选）与 `response_choice`（少候选）。
2. **状态编码**：
   - **Card 通道** `(B, 28, 20)`：手牌 / 各家桌面牌组(PENG/CHI/ZHAO/LONG) / 各家弃牌 /
     各家「过」掉的牌(`pass_cards`) / 全局 `dead_cards` / `pending` / `last_discard`，
     过 3 个残差 ConvBlock + GroupNorm，mean + max 双池化。
   - **Scalar**：相对位、phase(5)、人数、庄家位、牌堆剩余、**任务 one-hot(2)**、
     `game_rules.special_way` multi-hot。
   - **History**：`recent_history` 近 16 步过 4 层 / 8 头 Transformer，masked mean pool
     （当前数据多为空，对全 padding 安全；为后续全量轨迹数据预留）。
   - 所有 MLP 加 `Dropout(0.1)`。
3. **玩家位归一化**：用 `player_order` 把所有玩家映射为 `self=0 / +1 / +2 / +3` 相对位，
   2/3 人局共用槽位（多余填 0）。
4. **加权 CE + label smoothing**：损失 `(nll * sample_weight).sum() / weight.sum()`，
   `label_smoothing=0.05` 在合法动作上做 mask-safe 平滑（不泄漏到非法动作）。

> 与 `../`（qc_report 版）的差异：observation 字段结构不同（list 化的 discards、
> PENG/CHI/ZHAO/LONG 牌组、`dead_cards`、`special_way`），且**无 `final_result`**，
> 故去掉了胜负辅助头；新增了任务 one-hot 输入与 per-task 指标。两套代码相互独立。

## 常见问题

- **OOM**：调小 `--batch_size`（512 / 256）或 `--hidden 256`。
- **DataLoader 卡住（Windows）**：`--workers 0`。
- **bf16 不支持**：自动回退 fp16；想完全关闭混合精度加 `--no_amp`。
- **`matplotlib` CJK 报错**：曲线图用英文标签，无 CJK 字体依赖。

## 下一步

- 用更细的特征（红字/胡牌型）或局级轨迹拼接增强 `recent_history`
- 在 BC 之上做 offline RL（需补 reward / final_result）
- 蒸馏成更小的推理模型上线
