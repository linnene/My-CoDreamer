# CoDreamer 复刻计划

> 本文档是复现工作的**唯一工作台账**。结构性决策、待定项、改动登记都写在这里。

## 1. 目标与成功标准

复刻对象：**CoDreamer: Communication-Based Decentralised World Models**（arXiv:2406.13600v1，
CoCoMARL workshop @ RLC 2024，Edan Toledo / Amanda Prorok）。**无官方代码仓库。**

后续目标：复刻 **DMAWM**（arXiv 未收录，ICML 2026，南京大学）。目前未找到官方仓库。

### 成功标准分级

论文实际规模：5 算法 × 4 种子 × 7 任务 = **140 次训练 × 每次约 9 小时（TPU v2/v3）**。

| 级别 | 标准 | 我们是否追求 |
|---|---|---|
| **L1** | 代码跑通，损失下降，agent 能学到东西 | ✅ 必须 |
| **L2** | 定性顺序正确：CoDreamer > IDreamer > IPPO；且 WM Comm / AC Comm 消融方向与论文一致 | ✅ **主目标** |
| **L3** | 归一化分数数值对齐论文（VMAS IQM 0.84、Estimate Game 0.97） | ⚠️ 仅在最便宜的环境上尝试 |

**不追求全量复刻**（140 次训练 × 9 小时不可能）。

### 方法谱系

| 方法 | 世界模型 | 通信 |
|---|---|---|
| IPPO | 无 | 无 |
| IDreamer | DreamerV3，N agent 参数共享 | 无 |
| **CoDreamer** | IDreamer + GNN | **两层**：RSSM/reward/cont 头 + actor/critic torso |
| WM Comm（消融） | 有 | 仅世界模型层 |
| AC Comm（消融） | 无 | 仅 actor-critic 层 |

## 2. 关键发现：超参已对上的部分

把论文 Table 2 / Table 5 与 `third_party/dreamerv3-torch/configs.yaml` 的 `defaults` 逐条对照：

| 论文项 | 论文值 | 上游默认 | |
|---|---|---|---|
| GRU recurrent units | 512 | `dyn_deter: 512` | ✅ |
| Dense hidden units | 512 | `dyn_hidden: 512` / `units: 512` | ✅ |
| CNN multiplier | 32 | `encoder.cnn_depth: 32` | ✅ |
| Number of latents | 32 | `dyn_stoch: 32` | ✅ |
| Classes per latent | 32 | `dyn_discrete: 32` | ✅ |
| MLP layers | 2 | `actor.layers` / `critic.layers` = 2 | ✅ |
| Replay capacity | 1e6 | `dataset_size: 1000000` | ✅ |
| Batch size × length | 16 × 64 | `batch_size: 16` / `batch_length: 64` | ✅ |
| β_dyn / β_rep | 0.5 / 0.1 | `dyn_scale: 0.5` / `rep_scale: 0.1` | ✅ |
| Reconstruction scale | 1.0 | （decoder 隐式 1.0） | ✅ |
| WM lr / adam eps / grad clip | 1e-4 / 1e-8 / 1000 | `model_lr` / `opt_eps` / `grad_clip` | ✅ |
| AC lr / adam eps / grad clip | 3e-5 / 1e-5 / 100 | `actor/critic.lr/eps/grad_clip` | ✅ |
| Imagination horizon H | 15 | `imag_horizon: 15` | ✅ |
| Discount γ | 0.997 | `discount: 0.997` | ✅ |
| Return λ | 0.95 | `discount_lambda: 0.95` | ✅ |
| Target critic Polyak | 0.02 | `critic.slow_target_fraction: 0.02` | ✅ |
| Actor entropy scale | 3e-4 | `actor.entropy: 3e-4` | ✅ |
| 激活 / 归一化 | LayerNorm + SiLU | `norm: True` / `act: 'SiLU'` | ✅ |
| **预训练步数** | **500** | `pretrain: 100` | ❌ **唯一差异** |

**结论：论文作者基本是拿 DreamerV3 官方默认超参跑的，只把 `pretrain` 从 100 改成 500。**

推断出的额外信息：`train_ratio: 512`（论文未提）。其含义为
`Every(batch_size * batch_length / train_ratio) = Every(1024/512) = Every(2)`
—— **每 2 个环境步做 1 次梯度更新**。

## 3. 待拍板的结构性问题

论文全文唯一读不出来的地方：**RSSM 里那层 GNN 插在哪。**

论文只给了三句无法互相闭合的表述：

1. "the addition of $k$ GNN layers **within the RSSM** and prediction heads"
2. "only the reward and terminal state prediction heads use communication, while **the decoder head remains independent**"
3. "graphs of observations $G_o^t$ … graphs of stochastic posterior $G_z^t$ and prior $\hat G_z^t$ … recurrent states are also represented as graphs $G_h^t$"

表 2 只给了一个约束：`RSSM GNN Layers = 1`。

### 已有的推理

- $e$（编码观测）**不可能**是唯一插入点：imagination 阶段没有任何观测，插在 $e$ 上的 GNN 会失效
- 功能上这层 GNN 必须**影响 prior**，否则通信进不了想象轨迹（这排除了"只改 $z$ 再喂预测头"的方案）
- 因此候选只剩：**$h$ 出口**、**$z$ 出口**、或**两者**

### 对上游代码的关键观察

`RSSM.obs_step` 的实现是：

```python
prior = self.img_step(prev_state, prev_action)      # prior 走这里
x = torch.cat([prior["deter"], embed], -1)          # posterior 用 prior 的 deter
post = {"stoch": stoch, "deter": prior["deter"], ...}  # post 直接复用
```

即 **`img_step` 里 GRU 之后是唯一同时通向 prior 与 posterior 的咽喉点**。
在那里插一层 GNN，prior / posterior / `get_feat`（=`cat([deter, stoch])`，被 decoder、
actor、critic 共用）会**全部自动带上通信信息**。

**待定**：decoder 那句 "independent" 按哪种读法实现 —— 是"decoder 吃通信后的 $h$"
（宽松读法），还是"decoder 保留一份未经通信的原始 $h$"（严格读法）。
这决定了要不要在 RSSM 里额外维护一路局部状态。

### 其他必须自己定的数值（论文未给）

| # | 缺口 | 影响 |
|---|---|---|
| 1 | GATv2 的 hidden dim / head 数 | 参数量 |
| 2 | 通信范围 $C$（VMAS 三个任务各自的值） | 图密度 |
| 3 | 边特征如何注入 GATv2（GATv2 原生不支持边特征） | 架构选择 |
| 4 | CoDreamer 是否也给观测拼 one-hot agent index（论文只说 IDreamer 加了） | 对比公平性 |
| 5 | VMAS 里位置 $p_i$ 从哪来（观测内？环境特权信息？执行时是否可得） | **可行性** |

第 5 条最容易被忽略：GNN 建图需要 $p_i$，若它不在观测里、只有仿真器知道，
那"去中心化执行"就打了折扣。

## 4. 分阶段计划

**约束：本地（Mac M4 / 16GB）只写代码，不跑实验。所有执行都在远端
（Windows + RTX 4060 8GB / 16GB RAM）。**

```
步骤 0  远端跑通原版 dreamerv3-torch            ← 当前阶段
        目的：隔离验证工具链（Python / torch+CUDA / 依赖 / 显存）
        不改任何上游文件，纯命令行覆盖

步骤 1  多智能体化 = IDreamer
        在自建环境上跑通，产出 IDreamer 基线

步骤 2  加 GNN = CoDreamer
        与步骤 1 对比，看是否复现论文的定性顺序

（之后）DMAWM 复刻
```

### 环境路线

| 环境 | 用途 | 依赖 |
|---|---|---|
| **Sequential Estimate Game** | 首选。论文自造，Table 1 给了完整规格 | **纯 Python，零依赖** |
| **VMAS** | 真实 benchmark，向量观测，4060 可跑 | `vmas` 包 |
| Melting Pot | 像素 + 强 inter-agent 依赖 | ❌ 本地/单机不现实，需 Linux/Docker |

Sequential Estimate Game 优先的理由：它是论文**唯一能干净验证"表达力"主张**的环境，
且其 reward / transition **显式依赖邻居** —— 正是我们最关心的性质。

### 硬件成本现实

`train_ratio: 512` → 每 2 环境步 1 次梯度更新 → 500k 环境步 = **25 万次梯度更新**，
且加入 agent 维度后有效 batch 还要 ×N。4060 8GB 上这是硬瓶颈。

因此复现阶段**必须缩减规模**：先 100k 环境步而非 500k，随机种子数也要减少。
这会让结果达不到 L3，但 L2（定性顺序）仍然可判。

## 5. 改动清单

对 `third_party/dreamerv3-torch/` 的每一处修改都登记在此。

| 文件 | 改动 | 阶段 | 标注 |
|---|---|---|---|
| （暂无） | | | |

## 6. 相关文档

- `assets/`（**不入库**）：论文 PDF、`brainstorm.md`（想法与锚点记录）
- `third_party/THIRD_PARTY.md`：第三方代码溯源与修改约定
