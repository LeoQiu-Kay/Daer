# 泸州大贰 · 行为克隆训练（方案 A）

Pointer-style 小型策略网络，监督学习专家的 CHU（出牌）/ CHI（吃/过）决策。

## 目录

```
code/
├── data.py            # JsonlDataset (byte-offset 惰性加载) + 特征化 + collate
├── model.py           # CardEncoder + HistoryEncoder + StateEncoder + PointerPolicy
├── train.py           # 训练入口
├── eval.py            # 单独评估
├── requirements.txt
└── runs/bc/           # 训练产物：last.pt, best.pt, metrics.jsonl
```

数据路径相对本目录：`../qc_report_with_guo_balanced_12/splits/{train,valid,test}.jsonl`
（默认值即此路径，不需要传 `--data_dir`）。

## 安装

### 方案 1：conda 环境（推荐）

```powershell
# 1) 创建并激活环境（Python 3.11，与 PyTorch 兼容性最稳）
conda create -n daer python=3.11 -y
conda activate daer

# 2) 安装 PyTorch
#    GPU（CUDA 12.1）：
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
#    或 CPU-only：
# conda install pytorch cpuonly -c pytorch -y

# 3) 安装其他依赖
cd D:\Data\Card\code
pip install -r requirements.txt
```

验证：
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

后续每次使用前 `conda activate daer` 即可。

### 方案 2：纯 pip

```powershell
cd D:\Data\Card\code
pip install -r requirements.txt
```

CUDA / cuDNN 与 PyTorch 版本按本机 GPU 配置即可（CPU 也能跑，慢得多）。

## 训练

```powershell
# 默认配置：bs=512, 10 epoch, hidden=256, AMP 自动开启
python train.py

# 自定义
python train.py --batch_size 1024 --epochs 15 --lr 3e-4 --workers 8

# Smoke / 调试（每个 split 只取前 5000 条）
python train.py --limit 5000 --epochs 2 --workers 0 --batch_size 64
```

训练过程中每 `--log_every` 步打印一次 train loss / acc；每个 epoch 末跑一次 valid，
按 valid acc 保留 `best.pt`，同时每轮覆盖写 `last.pt`，所有指标追加到 `metrics.jsonl`。

## 评估

```powershell
python eval.py --ckpt runs/bc/best.pt --split test
python eval.py --ckpt runs/bc/best.pt --split valid --limit 20000
```

输出包含：
- `acc` / `acc_weighted`（按 `sample_weight` 加权）
- `per_phase`：CHU / CHI 各自 Top-1
- `per_source`：`prompt` / `synthetic_chi_from_3016` / `synthetic_guo_from_3014`
- `per_np`：2/3/4 人局
- `chi_guo_confusion`：CHI 阶段 expert × pred 的 2×2 混淆矩阵
- `aux_win_bce`：辅助胜负预测损失

## 设计要点

1. **变长合法动作**：不固定动作空间。每个 `legal_actions[i]` 独立特征化，
   `state ⊕ action_i` 过 MLP 输出标量 logit，行内 mask + softmax。
   损失只对 `label_index` 求 CE，自然兼容 CHU(8–15 个候选) 与 CHI(2–6 个候选)。
2. **状态编码**：
   - **Card 通道**：hand / 各家桌面牌组(PENG/CHI/ZHAO/LONG) / 各家弃牌 / pending / last_disc
     拼成 `(B, 23, 20)` 张量，过 Conv1D，取 mean + max 双池化。
   - **History**：近 16 步动作（类型 + 玩家相对位 + 涉及的牌 multi-hot）过小 Transformer，masked mean pool。
   - **Scalar**：相对位、phase、人数、庄家位、牌堆剩余、`game_rules.specialOptions` multi-hot 等。
3. **玩家位归一化**：用 `player_order` 把所有玩家映射为 `self=0 / +1 / +2 / +3` 的相对位，2/3/4 人局共用一套槽位（多余的槽位填 0）。
4. **加权 CE**：训练损失 `(nll * sample_weight).sum() / weight.sum()`。
5. **辅助任务**：从 `final_result` 取 `is_win` 做 BCE，权重 `--aux_weight 0.1`，提升数据效率。

## 常见问题

- **OOM**：把 `--batch_size` 调小，或 `--hidden 192 --workers 0`。
- **DataLoader 卡住（Windows）**：把 `--workers 0` 即可，单进程读取也不会成为瓶颈，因为 IO 本身较轻。
- **bf16 不支持**：自动回退到 fp16；想完全关闭混合精度加 `--no_amp`。
- **`is_valid_label=False` 的样本**：实测全集中均为 True，未做过滤。如确实碰到，建议预扫一遍把这些行排除。

## 下一步（如果继续往上做）

- 用 `final_result.score` 做 reward，在 BC 之上跑 offline RL（CQL / AWAC）
- 接规则引擎做 self-play AlphaZero
- 把当前模型蒸馏成更小的 INT8 / INT4 推理模型上线
