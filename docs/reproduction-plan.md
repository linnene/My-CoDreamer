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

推断出的额外信息：`train_ratio: 512`（论文未提）。

### 训练强度公式（易错，已更正一次）

```
每次梯度更新所需环境步 = (batch_size × batch_length / train_ratio) × action_repeat
```

因为 `_should_train = Every(batch_steps / train_ratio)` 作用在
`self._step = logger.step // action_repeat` 上，而不是直接作用在环境步上。

代入实际配置：`Every(16×64/512) = Every(2)`，再乘 `action_repeat`：

| 平台 | `action_repeat` | 更新频率 | 500k 环境步的更新次数 |
|---|---|---|---|
| DMC（冒烟用的） | 2 | 每 **4** 个环境步 1 次 | ~12.5 万 |
| **VMAS**（待确认） | 1（推测） | 每 **2** 个环境步 1 次 | ~25 万 |

两个后果，都会影响算力预算：

1. 我最初写的「每 2 个环境步 1 次」是错的，正确值是每 4 个环境步 1 次（DMC）
2. **`action_repeat` 会直接放大训练强度**。换到 VMAS 时如果它用 `action_repeat=1`，
   训练强度会**自动翻倍** —— 这是 `action_repeat` 与 `train_ratio` 之间一个不显眼的耦合，
   接 VMAS 前必须先确认它的取值

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

按上面的公式，DMC 下 500k 环境步 ≈ **12.5 万次梯度更新**（换到 VMAS 可能翻倍），
且加入 agent 维度后有效 batch 还要 ×N。4060 8GB 上这是硬瓶颈。

因此复现阶段**必须缩减规模**：先 100k 环境步而非 500k，随机种子数也要减少。
这会让结果达不到 L3，但 L2（定性顺序）仍然可判。

## 5. 改动清单

对 `third_party/dreamerv3-torch/` 的每一处修改都登记在此。

| 文件 | 改动 | 阶段 | 标注 |
|---|---|---|---|
| `dreamer.py` | 第 7 行 `os.environ["MUJOCO_GL"] = "osmesa"` → `setdefault(...)`，允许外部覆盖 | 步骤 0 | `# [CoDreamer]` |
| `dreamer.py` | `make_env` 增加 `seg` 分支（队伍级环境，直接返回不接 gym wrapper 链） | 步骤 1 | `# [CoDreamer]` |
| `dreamer.py` | `main()`：读 `env.num_agents`；多智能体时跳过 `Damy`/`Parallel`；预填随机策略的 batch 改为 `envs*num_agents`；三处收集调用改为 `collect`（= `tools_marl.simulate_marl`）；**train_ratio 自动补偿** | 步骤 1 | `# [CoDreamer]` |
| `configs.yaml` | 新增 `codreamer` 配置块（`seg` 任务、队伍数、超参） | 步骤 1 | `# [CoDreamer]` |
| `models.py` | `preprocess` 里 `obs["image"] /= 255` 改为条件执行（向量观测环境没有 image key） | 步骤 1 | `# [CoDreamer]` |
| `dreamer.py` | `_train` 增加 `update_behavior` 分支：预训练阶段**只更新世界模型**，跳过 actor-critic 与探索策略。上游无条件两者都训，与论文 D.3 的 "before any behaviour is learnt" 不符 | 步骤 1 | `# [CoDreamer]` |

新增文件（不在上游清单里，因此不影响"清单不一致 ⟺ 真改了"这个不变式）：

| 文件 | 作用 |
|---|---|
| `envs/seg.py` | Sequential Estimate Game 本体（纯 numpy） |
| `envs/seg_gym.py` | 队伍级 gym 包装 |
| `tools_marl.py` | 多智能体收集循环（N 条独立 episode 流） |

### 步骤 0 远端实测记录（2026-10-12, Windows + RTX 4060 Laptop 8GB）

环境：`.venv`（Python 3.10.8）+ `torch 2.4.1+cu121`。注意 PyPI 上 Windows 的 torch
默认 wheel 是 **CPU 版**（装完是 `2.4.1+cpu`），必须走 `--index-url https://download.pytorch.org/whl/cu121`。
其余按 `requirements.txt` 对齐（numpy 1.23.5 / gym 0.22.0 / mujoco 2.3.5 / dm_control 1.0.9）。

跑通原版 dreamerv3-torch 时踩到的平台问题（均已在脚手架侧处理）：

| # | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | `run_smoke.ps1` 解析报错，中文乱码 | 文件是 UTF-8 **无 BOM**，PS 5.1 按 GBK 读；且 Windows 检查脚本用 `python - @"..."@` 把 here-string 当成了命令行**参数**而非 stdin，python 转进 REPL | 改 `@"..."@ \| python -`；文件转 UTF-8 with BOM |
| 2 | 完整性校验 24/24 全部 FAILED | Windows `core.autocrlf=true` 在检出时把 LF 转了 CRLF，而清单是在 Mac 上按 LF 生成的；另 `rel.lstrip("./")` 会吃掉 `.gitignore` 的首点 | 新增 `.gitattributes`（`third_party/dreamerv3-torch/** -text`）并重新检出，使「清单不一致 == 真的改了」成立；`lstrip` → `rel[2:]` |
| 3 | 训练进程一起就退出 | PS 5.1 在 `$ErrorActionPreference="Stop"` 下，`2>&1 \|` 会把原生程序 stderr（Gym 弃用警告）升级成终止性错误 | 训练调用处临时置 `Continue`，改用 `$LASTEXITCODE` 判成败 |
| 4 | `ImportError: Unable to load OpenGL library ... OSMesa` | `dreamer.py` 写死 `MUJOCO_GL=osmesa`，Windows 上没有 OSMesa；而 `envs/dmc.py` 每步都 `render()`，必须有真 GL 后端 | 上游改为 `setdefault`（见上表），Windows 脚本设 `MUJOCO_GL=glfw` |
| 5 | 冒烟测试远超「几分钟」 | `Every(batch_size*batch_length / train_ratio)`：冒烟把 batch 砍到 `2×8=16`，`train_ratio` 仍是 512 → `Every(0.03125)` = **每环境步 32 次梯度更新**（真实配置是每 4 个环境步 1 次） | `.sh` / `.ps1` 均已覆盖 `--train_ratio 8` → `Every(16/8)=Every(2)`，与真实配置同频 |

### 步骤 0 结论：冒烟通过

`powershell -ExecutionPolicy Bypass -File scripts\run_smoke.ps1` 退出码 0，全程 **2 分 18 秒**
（dmc_proprio / dmc_walker_walk；循环内 1500 训练环境步 + 3 次评估，另加 prefill 200）。

可判据（全部出自 `logdir/smoke/`）：

- `model_loss` 28.7 → 18.5 → 16.8 → 14.3 单调下降 → **L1（代码跑通、损失下降）达成**。
- `update_count` 严格对账：末次标量日志 `update_count 725`，而此时 `_step = 2900/2 = 1450`，
  即 `100 (pretrain) + (1450-200)/2 = 725` —— 与 `Every(2)` 严丝合缝。
  （修 `train_ratio` 前，这个数会是同一区间的 ~32 倍。）
- `eval_return` 36.4 / 43.2 / 33.3。冒烟**只验链路不验效果**，3 条 episode 的波动无意义。
- 退出条件自洽：`while agent._step < steps//action_repeat + eval_every//action_repeat`
  = `2000//2 + 1000//2 = 1500`，对上日志末行 `logger.step 3000`（=2×1500）与 `dataset_size 1500`。

未验证：CoDreamer、多智能体、VMAS、任何学习效果。

### 回传证据的入库策略

**只有小体积文本证据入库。二进制一律不入库。**

为什么要立这条规矩：步骤 0 首次回传时把整个 `logdir/smoke/`（9 个文件、8.5 MB）提了进来，
但其中只有 `run.log`（8 KB）和 `metrics.jsonl`（5 KB）是真正的证据；
剩下的 `tfevents`（2.8 MB）和 6 个 episode npz（6.7 MB）都是 byproduct ——
可重新生成，且会让仓库体积随实验次数单调增长。

当前规则（见 `.gitignore`）：

```
logdir/*
!logdir/smoke/
logdir/smoke/*
!logdir/smoke/*.log
!logdir/smoke/*.jsonl
```

即白名单是**显式的**：要新增一种入库的文本证据，必须在 `.gitignore` 里显式开口子。
这个设计是故意的 —— 比「忽略二进制后缀」更难绕过。

| 内容 | 入库？ |
|---|---|
| `run.log`（stdout） | ✅ |
| `metrics.jsonl`（标量指标） | ✅ |
| `events.out.tfevents.*` | ❌ 可重生 |
| `train_eps/*.npz`、`eval_eps/*.npz` | ❌ 可重生，冒烟阶段无价值 |
| `latest.pt` 等权重 | ❌ 219 MB，超 GitHub 单文件上限 |
| 录像 / mp4 | ❌ |

已知代价：步骤 0 的 8.5 MB 二进制仍留在 git 历史里（`a442bfb`）。
仓库还年轻，未来需要时可以重写历史抹掉；目前不值得为此 force push。

`run.log` 原本被 PowerShell 5.1 的 `Tee-Object` 写成 UTF-16LE，
**已无损转成 UTF-8**（16534 → 8278 字节，往返校验一致）以便 diff / grep。

## 6. 相关文档

- `assets/`（**不入库**）：论文 PDF、`brainstorm.md`（想法与锚点记录）
- `third_party/THIRD_PARTY.md`：第三方代码溯源与修改约定
