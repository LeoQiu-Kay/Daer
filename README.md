# 泸州大贰 · 行为克隆训练（方案 A）

Pointer-style 小型策略网络，监督学习专家的 CHU（出牌）/ CHI（吃/过）决策。

## 目录

```
code/
├── data.py            # JsonlDataset (byte-offset 惰性加载) + 特征化 + collate
├── model.py           # V2: 残差 Conv + 4 层 Transformer + dropout，默认 hidden=384
├── train.py           # 训练入口；step/epoch 双层日志，每轮自动渲染 curves.png
├── eval.py            # 单独评估
├── demo.py            # 单条样本推理可视化（控制台打印）
├── visualize.py       # 训练曲线可视化（PNG，支持 --watch 持续刷新）
├── requirements.txt
└── runs/bc/           # 训练产物：best.pt / last.pt / config.json /
                       #          train_log.jsonl / metrics.jsonl / curves.png
```

数据路径默认值（AutoDL 部署）：
`/root/autodl-tmp/luzhoudaer/qc_report_with_guo_balanced_12/splits/{train,valid,test}.jsonl`

如本机数据在别处，用 `--data_dir` 覆盖：
```powershell
python train.py --data_dir D:\Data\Card\qc_report_with_guo_balanced_12\splits
python eval.py  --ckpt runs/bc/best.pt --data_dir /path/to/splits --split test
```

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

```bash
# 默认配置（V2）：hidden=384, bs=1024, lr=5e-4, 25 epoch, label_smoothing=0.05, AMP 自动开
python train.py

# 进一步加大容量
python train.py --hidden 512 --batch_size 2048 --epochs 30

# Smoke / 调试（每个 split 只取前 20000 条，2 epoch）
python train.py --limit 20000 --epochs 2 --workers 0 --batch_size 256
```

训练过程：
- 每 `--log_every` 步追加一条到 `runs/bc/train_log.jsonl`
- 每 epoch 末跑 valid，追加到 `runs/bc/metrics.jsonl`
- 按 valid acc 保留 `best.pt`，每轮覆盖写 `last.pt`，启动时把配置 dump 到 `config.json`
- **每轮自动调用 `visualize.render`，写出 `runs/bc/curves.png`**
- 默认开训前会清空 `train_log.jsonl / metrics.jsonl`，要保留旧记录加 `--keep_logs`

## 可视化训练曲线

`visualize.py` 把 `train_log.jsonl + metrics.jsonl` 渲染成一张 6 子图 PNG：
train loss / train acc + lr / valid acc by phase / valid acc by source /
CHI-vs-GUO 分解 / aux win BCE。

```bash
# 一次性渲染
python visualize.py                                  # -> runs/bc/curves.png
python visualize.py --run_dir runs/bc --out my.png

# 持续刷新（在另一个终端运行，配合训练实时看曲线）
python visualize.py --watch --interval 30
```

> 训练 `train.py` 本身在每个 epoch 末已经自动调用 `render`，所以最简单的看图方法
> 就是用任意图片浏览器打开 `runs/bc/curves.png`，每个 epoch 末它会自动被覆盖更新。

## 评估

```powershell
python eval.py --ckpt runs/bc/best.pt --split test
python eval.py --ckpt runs/bc/best.pt --split valid --limit 20000
```

## 推理 / Demo

加载 `best.pt`，在 test.jsonl 中抽样本逐条打印状态、合法动作概率与专家动作对比：

```bash
# 默认随机抽 5 条
python demo.py

# 指定 ckpt / 阶段 / 数量
python demo.py --ckpt runs/bc/best.pt --phase CHI --n 8

# 看指定下标的一条
python demo.py --sample_index 1234
```

输出包含：手牌、各家桌面牌组、弃牌历史、最近 8 步动作、所有合法动作的模型概率（按降序排列，
`★expert` 标专家选择、`←pred` 标模型选择）、最终是否一致。红字牌后加 `*` 标记。

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
2. **状态编码 V2**：
   - **Card 通道**：hand / 各家桌面牌组(PENG/CHI/ZHAO/LONG) / 各家弃牌 / pending / last_disc
     拼成 `(B, 23, 20)` 张量，过 **3 个残差 ConvBlock + GroupNorm**，取 mean + max 双池化。
   - **History**：近 16 步动作（类型 + 玩家相对位 + 涉及的牌 multi-hot）过 **4 层 / 8 头 Transformer**，masked mean pool。
   - **Scalar**：相对位、phase、人数、庄家位、牌堆剩余、`game_rules.specialOptions` multi-hot 等。
   - 所有 MLP 加 `Dropout(0.1)`。
3. **玩家位归一化**：用 `player_order` 把所有玩家映射为 `self=0 / +1 / +2 / +3` 的相对位，2/3/4 人局共用一套槽位（多余的槽位填 0）。
4. **加权 CE + label smoothing**：训练损失 `(nll * sample_weight).sum() / weight.sum()`，
   `label_smoothing=0.05` 在合法动作上做 mask-safe 平滑（不会泄漏到非法动作）。
5. **辅助任务**：从 `final_result` 取 `is_win` 做 BCE，权重 `--aux_weight 0.1`，提升数据效率。

## 当前性能

V1 (`hidden=256, 10 epoch`, ~1.77M params)：
- valid acc 56.5%（CHU 49.0% / CHI 79.5%；CHI-vs-GUO 二分类 98.3%）

V2 改动相对 V1：
- 模型容量 ~1.77M → ~7–8M（hidden=384, 残差 conv, 4 层 history transformer）
- batch_size 512 → 1024，lr 3e-4 → 5e-4，epochs 10 → 25，warmup 1000 → 2000
- 新增 label smoothing 0.05、各 MLP dropout 0.1
- 新增 Top-3 acc 指标、step 级日志、训练曲线自动渲染

预期 CHU 49% → **60% 以上**。

## 常见问题

- **OOM**：把 `--batch_size` 调小（512 / 256），或 `--hidden 256`。
- **DataLoader 卡住（Windows）**：把 `--workers 0` 即可，单进程读取也不会成为瓶颈。
- **bf16 不支持**：自动回退到 fp16；想完全关闭混合精度加 `--no_amp`。
- **旧 ckpt 无法加载**：V2 架构与 V1 不兼容，需要重新训练（旧 `best.pt` 仅 demo.py 兼容用作展示）。
- **`matplotlib` 渲染 Chinese 报错**：本仓库的曲线图用英文标签，避免 CJK 字体依赖。

## 下一步（如果继续往上做）

- 用 `final_result.score` 做 reward，在 BC 之上跑 offline RL（CQL / AWAC）
- 接规则引擎做 self-play AlphaZero
- 把当前模型蒸馏成更小的 INT8 / INT4 推理模型上线
