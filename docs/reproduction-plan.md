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

表 2 与表 3 给出了完整的 GNN 配置（**已查清**）：

```
Table 2 World Model Size          Table 3 Actor-Critic Model Size
  GRU recurrent units     512       Dense hidden units   512
  Dense hidden units      512       MLP layers           2
  MLP layers              2         GNN Layers           1     ← k=1
  RSSM GNN Layers         1  ← k=1  Actor Parameters     1.6M
  Reward & Cont GNN Layers 1 ← k=1  Critic Parameters    1.7M
  Vector Environment Params 16M
```

**三处各 1 层，k=1。** 且 Table 2 的 16M 已经是**含两层 GNN 之后**的数字。

### 参数量的矛盾（待校准，尚未解决）

| | 我们 | 论文 | 差 |
|---|---|---|---|
| 世界模型 | 16,429,829（**不含 GNN**） | 16M（**含 GNN**） | 我们偏大 |
| actor | 1,052,676 | 1.6M | 0.55M |
| critic | 1,181,439 | 1.7M | 0.52M |

而我们自己实现的 1 层 GATv2(dim=512) = **1,312,768**，是那个缺口的 **2.4 倍**。

**结论：论文那层 GNN 不是标准 GATv2 的规模。** 可能它作用在更小的维度上，
或得分函数与值变换共享了权重，或少了 skip 投影。**不靠猜去凑数字** ——
先按标准 GATv2 实现，如实记录偏差，等 actor/critic 接完线用实测缺口倒推。
（一个有意思的巧合：`2d² = 2×512² = 524,288` 恰好等于 critic 的缺口 0.52M。
但这不足以当结论。）

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

| # | 缺口 | 影响 | 当前处理 |
|---|---|---|---|
| 1 | GATv2 的 head 数 | 参数量 | 取 1（保守；heads 只改注意力粒度数） |
| 2 | 通信范围 $C$（VMAS 三个任务各自的值） | 图密度 | 待定（VMAS 阶段） |
| 3 | 边特征如何注入 GATv2（原版 GATv2 没有边特征） | 架构选择 | **本实现把 $W_e E_{ij}$ 加进得分**：$e_{ij}=a^\top\mathrm{LeakyReLU}(W_l h_i + W_r h_j + W_e E_{ij})$。论文只说 "edge features ... are used with observations in the GAT V2 to calculate attention coefficients"，没给公式 |
| 4 | CoDreamer 是否也给观测拼 one-hot agent index（论文只说 IDreamer 加了） | 对比公平性 | **是**（两边都拼）—— 否则 CoDreamer 与 IDreamer 的差异不止 GNN |
| 5 | VMAS 里位置 $p_i$ 从哪来（观测内？环境特权信息？执行时是否可得） | **可行性** | 待定（VMAS 阶段） |

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

> **进展**：Sequential Estimate Game 已跑完并判定为**饱和**（六配置扫描 + 三 seed
> 对照全部收敛到同一回报），不再用于通信对照。已转入 **VMAS**。
> `envs/vmas_env.py` 完成并通过自检（三任务 × 图性质 / 形状 / 越界 / 可重现）。


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
| `models.py` | `preprocess` 里 `obs["image"] /= 255` 改为条件执行（向量观测环境没有 image key） | 步骤 1 | `# [CoDreamer]` || `dreamer.py` | `_train` 增加 `update_behavior` 分支：预训练阶段**只更新世界模型**，跳过 actor-critic 与探索策略。上游无条件两者都训，与论文 D.3 的 "before any behaviour is learnt" 不符 | 步骤 1 | `# [CoDreamer]` |
| `dreamer.py` | `make_env` 的 `seg` 分支改为用 `config.seg_*` 构造 `Spec`，并传 `obs_mode` / `graph_obs`；默认值与论文 Table 1 一致 | 跨步对比 + 步骤 2 | `# [CoDreamer]` |
| `configs.yaml` | `codreamer` 块增加 `seg_*` 环境规格键（`num_agents` / `horizon` / `num_actions` / `edge_density` / `local_state_ratio` / `target_transform` / `obs_mode` / `graph_obs`）。每个键都会自动变成命令行参数 | 跨步对比 + 步骤 2 | `# [CoDreamer]` |
| `configs.yaml` | `encoder`/`decoder` 的 `mlp_keys` 由 `.*` 改为 `^(?!adjacency$)(?!edge_features$).*`，把图拓扑（邻接矩阵行、边特征行）挡在 encoder/decoder 之外 —— 它们只给 GNN 用，不是要编码/重建的特征。默认（`seg_graph_obs: false`）下行为与 `.*` 完全一致 | 步骤 2 | `# [CoDreamer]` |
| `configs.yaml` | `batch_size` 改为 8：图批次下它的单位是**队伍数**，论文 B=16 在 12GB 上会 OOM（实测 12 队可行、16 队不行） | 步骤 2 | `# [CoDreamer]` |
| `tools.py` | `SymlogDist.log_prob` 的归约由 `sum(range(len(shape))[2:])` 改为 `sum(-1)`：前者把"第 2 维之后全部"归约掉，多智能体下会把 agent 轴也加掉（loss 从 (B,T,N) 变成 (B,T)，与 kl_loss 对不上）。本类只用于向量观测，event 恒为 1 维，故归约末维与上游等价 | 步骤 2 | `# [CoDreamer]` |
| `networks.py` | `RSSM.__init__` 新增 `gnn_layers` / `gnn_edge_dim` / `gnn_heads`，建一个 `GraphAggregator`；`img_step` 在 **GRU 之后**做邻域聚合（唯一同时通向 prior 与 posterior 的咽喉点）；`observe` / `obs_step` / `imagine_with_action` 多一路 `graph` 参数 | 步骤 2 | `# [CoDreamer]` |
| `networks.py` | `RSSM.initial` 接受 int 或形状元组（上游只取 `len(is_first)`，会丢掉 agent 轴）；`obs_step` 的 `is_first[:, None]` 改为 `unsqueeze(-1)`（前者在位置 1 插轴，`(B,N)` 下得到 `(B,1,N)` 无法广播）；"全部重置"的判断由 `len(is_first)` 改为 `numel()` | 步骤 2 | `# [CoDreamer]` |
| `models.py` | `_train` 不再折平 agent 轴（整条链路都只对末维操作，折平反而让 agent 彼此不可见）；新增 `graph_from_data`；两处 `assert shape == embed.shape[:2]` 改为按自身维数对齐 | 步骤 2 | `# [CoDreamer]` |
| `models.py` | `WorldModel.__init__` 把 `gnn_layers` / 边特征维度（取自 `edge_features` 的末维）/ `gnn_heads` 传给 RSSM | 步骤 2 | `# [CoDreamer]` |
| `models.py` | `ImagBehavior._imagine` **临时**把 agent 轴也折进 batch（upstream 的 `lambda_return` 末尾写死 `reshape([shape[0], shape[1], 1])`，带 agent 轴改动面很大）。**步骤 5 要取消这一步**，同时给 AC 加 GNN 并把冻结的起点图喂给想象轨迹 | 步骤 2（临时） | `# [CoDreamer]` |
| `configs.yaml` | 新增 `gnn_layers`（默认 0 = IDreamer）与 `gnn_heads` | 步骤 2 | `# [CoDreamer]` |
| `networks.py` | `MLP` 新增 `gnn_layers` / `gnn_edge_dim` / `gnn_heads`，在 **self.layers 的隐层输出**上做邻域聚合（不是头的 1536 维输入 —— GATv2 是 5·d²，那里是 1180 万参数）；`forward` 多一路 `graph` 参数 | 步骤 2 | `# [CoDreamer]` |
| `models.py` | reward / cont 两个头接 GNN；新增 `_comm_heads`。**decoder 不接** —— 论文："only the reward and terminal state prediction heads use communication, while the decoder head remains independent" | 步骤 2 | `# [CoDreamer]` |
| `configs.yaml` | 新增 `gnn_head_layers`（默认 0；论文 Table 2 = 1） | 步骤 2 | `# [CoDreamer]` |
| `models.py` | 新增 `flatten_graph`（(B,T,N,N)->(B*T,N,N)，与 `_imagine` 对隐状态的合并方式一致）；`WorldModel` 暴露 `edge_dim` 供 AC 建自己的 GNN | 步骤 2 | `# [CoDreamer]` |
| `models.py` | `ImagBehavior` 的 actor / critic 各加一层 GNN（论文 Table 3）；`_train` / `_imagine` / `_compute_target` / `_compute_actor_loss` 全部多一路 `graph` | 步骤 2 | `# [CoDreamer]` |
| `models.py` | **取消** `_imagine` 里对 agent 轴的临时折平（步骤 3 留下的）。`:imagine` 现在把**冻结的起点图**传给 actor 与 `img_step`，整段轨迹复用同一张图（论文明确要求不重新预测邻接矩阵） | 步骤 2 | `# [CoDreamer]` |
| `dreamer.py` | `_train` 取出图、用 `flatten_graph` 扁平化，同一份图既给 reward 头也给 `ImagBehavior` | 步骤 2 | `# [CoDreamer]` |
| `configs.yaml` | 新增 `gnn_behavior_layers`（默认 0；论文 Table 3 = 1） | 步骤 2 | `# [CoDreamer]` |
| `tools.py` / `models.py` | `lambda_return` 改为返回张量 `(time, *batch, 1)`（不再返回元组），两处 `torch.stack(target, dim=1)` 随之删除；两处 `[:, :, None]` 改为 `[..., None]`；`axis!=0` 明确报错 | 步骤 2 | `# [CoDreamer]` |
| `dreamer.py` | `__call__` 新增 `collect_policy`；`_train` 新增 `update_world_model`；新增 `set_world_model_frozen`；循环后补一次评测；切断时存 `cutoff.pt`；checkpoint 里附带 `config` | 步骤 2 | `# [CoDreamer]` |
| `tools_marl.py` | 新增 `simulate_imagination`（Phase 2：环境切断后只推进训练节奏）与 `count_dataset_size` | 步骤 2 | `# [CoDreamer]` |
| `dreamer.py` | `main()` 多智能体时改用 `tools_marl.make_graph_dataset` 构造训练集；`video_pred_log` 与之互斥（未适配会直接报错） | 步骤 2 | `# [CoDreamer]` |
| `dreamer.py` | 按 `ckpt_every` 另存带步号的 `ckpt_NNNNNNNNN.pt`。上游只留 `latest.pt` 且每轮覆盖，事后只剩最终点——"某个量随训练怎么变"这条曲线因此永远拿不到。2026-10-15 的诊断正是卡在这里：只有最终 checkpoint，无法判断模型利用是随训练收紧还是保持不动。默认 0 = 不存，行为与上游逐位一致 | 诊断（2026-10-15） | `# [CoDreamer]` |
| `configs.yaml` | 新增 `ckpt_every`（默认 0） | 诊断（2026-10-15） | `# [CoDreamer]` |
| `models.py` | `_compute_target` 的 λ-return 自举项可用 `value_bootstrap: 'slow'` 换成慢目标。**上游把在线 critic 同时用作自举与中间混合项**，而慢目标只出现在损失正则里 —— 目标因此没有外部锚。实测症状：`target_mean ≡ value_mean`、V 与真实回报相关 0.03、价值膨胀到每步奖励的 ~1000 倍且不收敛、actor advantage 归零。默认 `'online'` = 与上游逐位一致 | 诊断（2026-10-15） | `# [CoDreamer]` |
| `configs.yaml` | 新增 `value_bootstrap`（默认 `'online'`） | 诊断（2026-10-15） | `# [CoDreamer]` |
| `dreamer.py` | `make_env` 增加 `overcooked` 分支（队伍级环境，直接返回，不接 gym wrapper 链） | 演示环境（2026-10-16） | `# [CoDreamer]` |
| `configs.yaml` | `codreamer` 块新增 `overcooked_*` 10 个键（num_agents / layout / max_steps / cook_time / recipe / sight / 通信半径 / 图开关 …） | 演示环境（2026-10-16） | `# [CoDreamer]` |
| `scripts/eval_policy.py` | `config_key` 决定"要不要重建模型"。原先写死 `seg_*` 键，对别的 suite 取到的是 yaml 里与本 run 无关的默认值。**改为用 checkpoint 里 `agent_state_dict` 的权重形状表**（`_arch_key`）—— 枚举 config 键这条路过两代都会漏（写死 `seg_*` 漏；按前缀枚举则漏掉任何没被命名成 `<suite>_*` 的结构性旋钮，`seg_obs_mode` 就是差点漏掉的一个），而形状表机械完备：它直接回答"这份权重装不装得进当前模型"。`_env_spec` 保留但**只用于打印** | 演示环境（2026-10-16） | `# [CoDreamer]` |

> **⚠️ `value_bootstrap: 'slow'` 已被量测否定，不要指望它修好任何东西。**
> 实测慢目标与在线 critic 只差 **0.19%（IDreamer）/ 0.56%（CoDreamer）**，
> 且与真实回报的相关性和在线 critic 一样烂（+0.037 / +0.118）。
> 它只是延迟约 50 次更新的副本（`slow_target_fraction: 0.02`，而全程约 32000 次更新），
> **不是锚**。开关保留只为复现这次否定，不是候选修法。
>
> 真正的原因用比值表述最清楚：价值目标里自举项 `γ·V ≈ 12.0`，奖励项 `r̄ = 0.012`，
> **比值约 997:1** —— 目标里 99.9% 是 critic 自己的输出，0.1% 来自环境。

新增文件（不在上游清单里，因此不影响"清单不一致 ⟺ 真改了"这个不变式）：

| 文件 | 作用 |
|---|---|
| `envs/seg.py` | Sequential Estimate Game 本体（纯 numpy） |
| `envs/seg_gym.py` | 队伍级 gym 包装 |
| `tools_marl.py` | 多智能体收集循环（N 条独立 episode 流）；队伍归组与**图采样器** `sample_episode_graphs` |
| `networks_gnn.py` | GATv2 层 + 防过平滑跳连；`GraphAggregator(layers=0)` 是严格恒等 |
| `scripts/test_graph_pipeline.py` | 数据管线 `(B,T,N,·)` 的**数值恒等**验收（阈值 0） |
| `envs/overcooked_env.py` | Overcooked 的队伍级包装（含自检） |
| `scripts/test_team_env_wiring.py` | 队伍级环境接入的**集成点**测试（8 节，纯 CPU） |
| `scripts/render_overcooked.py` | Overcooked 回放到 GIF/MP4（matplotlib Agg，不依赖显示服务器） |
| `scripts/probe_comm_reach.py` | **通信探针**：信息从 j 传到 i 吗、走几跳。免训练、无 seed 方差 |

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

---

## 外部审查后的修复与重跑（Codex review）

一次独立审查指出训练/评测代码里存在**会直接改变实验含义**的错误。逐条核对后
全部成立，其中两条是致命的。以下按修复顺序登记。

### 致命 1：预训练什么都没训

```python
# 修改前（dreamer.py）
update_behavior=not pretraining,
update_world_model=not (pretraining or self._world_model_frozen)   # ← 两个都是 False
```

注释写的是「预训练阶段只更新世界模型」，表达式却把 `pretraining` 也并进了 `or`，
于是那 500 步是 **500 次不更新任何参数的前向**，日志照旧打印「预训练完成」。
引入于 `ca1a68d`（Phase 2 的提交）—— 是我自己的「修复」弄坏的，早先的
`43b71b1` 本来是对的。冻结世界模型属于 Phase 2，与预训练无关，不该共用表达式。

### 致命 2：执行路径从不通信

`Dreamer._policy()` 的四个调用点（`obs_step` ×1、`actor` ×3）全都不传图。
后果：**收集数据用的动作来自一个看不见图的 actor**，部署评测评的也是看不见图的
策略。CoDreamer 在真实执行中从未使用通信；此前产出的每一个 CoDreamer 回报数字
都是无通信策略的数字。

这个失效模式在 ⑤ 里已经被**诊断并修复过一次**（`_imagine`），但真实执行的四个
调用点被放过了。修复过程中又暴露出同一个问题的另外两层：

1. 图形状：环境每个 agent 给的是整张 `(N, N)`，`simulate_marl` 铺平后是
   `(batch, N)`（一堆行），GNN 要 `(batch, N, N)`。
2. agent 轴：训练数据是 `(B, T, N, ·)`，执行时被铺平成 `(n_teams*N, ·)`，
   **agent 轴消失**，GNN 把 batch 维当成了邻居维。

两者同源：执行路径缺的是 agent 轴。`_policy` 现在把整步算在 `(n_teams, N, ·)`
布局下，算完铺平回去；`n_agents == 1` 完全不动形状，保证单智能体路径与上游逐位一致。

### 其他修复

| 问题 | 修复 |
|---|---|
| README 称 `gnn_layers` 是唯一开关 | 三处 GNN 改由单一 `comm_variant` 驱动，`-1` 表示跟随 |
| 部署加载的 strict 校验恒为空 | `[k for k in missing if not k.startswith("_")]` —— 本项目每个真参数的键都以 `_` 开头，过滤后永远为空。改为任何 missing/unexpected 都失败 |
| 不确定性按 agent 聚合 | 同队 agent 共享状态、邻接与奖励，非独立样本。改为先按队求均值再估计区间，并断言各 checkpoint 的 seed 相同 |
| 图损失被当作「想象能力」的证据 | 那是 replay 上的**训练**损失，且用的是**看过当步观测的 posterior**。新增 `scripts/eval_world_model.py`：held-out 上只喂真实动作、不喂观测的多步开环误差 |

### 元问题：为什么 566 行测试一条都没抓到

`scripts/test_graph_pipeline.py` 全绿，因为它测的全是**模块**（GATv2 形状、数据
折平、RSSM 前向、参数量守恒）。两个致命错误都长在**集成点**上。

新增 `scripts/test_training_loop.py` 补这一层，只断言从外面可观测的行为：

```
[1] 预训练后世界模型权重必须变，actor 必须逐位不变
[2] 对照：预训练结束后 actor 必须开始变（否则 [1] 会因为「什么都不学」假通过）
[3] 冻结世界模型：世界模型不变，actor 变
[4] codreamer：改邻接矩阵必须改变动作 / 对数概率
[5] idreamer：改邻接矩阵必须逐位不变（对照，证明 [4] 的敏感性来自 GNN）
[6] comm_variant 同时驱动三处 GNN
```

### 重跑结果（3 seed，2500 步，`batch_size 4`）

**策略性能：没有差别，环境已饱和。**

| seed | IDreamer（step 700 → 3100） | CoDreamer |
|---|---|---|
| 0 | −0.171 → −0.133 | −0.171 → −0.159 |
| 1 | −0.121 → −0.052 | −0.121 → −0.073 |
| 2 | −0.174 → −0.092 | −0.174 → −0.103 |

从 step 700 起两条曲线在 seed 0/1 上**逐位相同** —— 两者都收敛到同一个饱和行为，
环境分不出差别。这与 `scripts/sweep_seg.sh` 的六配置扫描结论一致。

**注意与修复前的对比**：修复前 IDreamer 在 step 700 是 **−2.36**，修复后是 **−0.17**。
此前观察到的「IDreamer 策略塌陷」是预训练 bug 造成的，不是算法现象。

**世界模型质量：CoDreamer 明确且一致地更好（held-out 开环）。**

h=16 时（3 个 seed）：

| | IDreamer | CoDreamer |
|---|---|---|
| 隐状态漂移 KL | 5.59 / 3.98 / 9.26（均值 6.28） | 1.18 / 1.12 / 0.92（均值 **1.07**） |
| continuation MAE | 0.867 / 0.867 / 0.867 | 0.740 / 0.526 / 0.190（均值 **0.485**） |

KL 相差 **5.9 倍**，三个 seed **完全不重叠**。CoDreamer 的开环漂移**不随 horizon
增长**（始终 ~1.0），IDreamer 的则发散。IDreamer 的 continuation MAE 在三个 seed
上逐位相同（0.867）—— 它退化成了常数预测。

这正是论文 Figure 12 的主张，用正确的口径测出来了。

### 当前结论

1. **核心命题（世界模型能扛住策略的持续改进）**：在 IDreamer 上成立。
2. **CoDreamer 的世界模型确实更好** —— held-out 开环误差上一致、大幅胜出。
3. **但这个环境翻译不出策略差异** —— 饱和在 ≈ −0.09。
   所以「CoDreamer 与 IDreamer 的回报谁高」在本环境上**没有信息量**，
   需要 VMAS。
4. **Phase 2 降级为诊断**：基础对照与通信消融成立之前，它不应该当主结果。

---

## 转入 VMAS（2026-10-15）

### 为什么换：论文自己的消融给出了环境排序

附录 C 的三段消融合起来是一张判别力的排序表，**与"论文正文主实验=最能区分"的
直觉相反**：

| 环境 | 论文原话 | 判别力 |
|---|---|---|
| Melting Pot | "probability > 0.5 of improving upon IDreamer **with statistical significance**"；WM Comm 贡献最大 | **最强** |
| Estimate Game | "this overlap of information is **highly evident by the construction of the environment**" | 中 |
| VMAS | "**not highly inter-agent dependent**"、"IDreamer and CoDreamer are **similar**... due to **stochasticity**" | **最弱** |

VMAS 仍是必做项：它是**正文主实验**（Figure 4），且向量观测 / 4~5 agent / 500 步
在本机跑得动。但它同时解释了为什么"回报分不出差别"不能当作复现失败 ——
论文自己就承认在 VMAS 上是在看随机噪声。

**因此 VMAS 上的关键指标不是回报，而是我们在 Estimate Game 上验证过的那一个：
held-out 开环隐状态漂移**（那里它给出了 5.9 倍、三 seed 不重叠的差异，而回报
完全分不出）。把同一指标搬到 VMAS 上，检验那个发现能否推广。

### 已完成的落地

`envs/vmas_env.py` —— 队伍级包装，接口与 `envs/seg_gym.py` 逐条对齐
（单 agent 视角的 space 声明、队伍级 reset/step、逐 agent 奖励、同样的图键），
所以 `tools_marl.simulate_marl` 一个字都不用改。自检：
`python -m envs.vmas_env`。

### 三个必须自己定、而不是查得到的东西

| # | 缺口 | 处理 |
|---|---|---|
| 1 | **通信半径 C** | 论文只说"边当且仅当 ‖p_i−p_j‖ ≤ C，C 由环境设定"，但 VMAS 没有这个概念，论文也没给三个任务的值。**实测互距分布后再选**：flocking N=4 中位 1.13 → C=1.5（平均度 2.45/3）；discovery N=5 中位 1.11 → C=1.5；buzz_wire 互距**恒 0.49**（刚性连杆）→ C=0.5 恒连通。可用 `--vmas_comm_radius` 覆盖 |
| 2 | **版本漂移** | vmas 1.5.2 的离散动作是 `nvec=[3,3]`（9 个组合、对外一个索引），论文写 Discrete (5)；观测维 Discovery 19（论文 21）、Buzz Wire 6（论文 8），只有 Flocking 的 18 与论文一致。**需要严格对齐时要把 vmas 钉回论文时期的版本** |
| 3 | **buzz_wire 的图不变化** | 两 agent 由刚性连杆相连，互距几乎恒定 → 拓扑要么恒连通要么恒断开。不影响 CoDreamer 的语义（通信=聚合邻居隐状态），但**该任务不能用于图稀疏度消融** |

### 下一步

1. `dreamer.py` 的 `make_env` 加 `vmas` 分支（读 `config.vmas_task` 等）
2. `configs.yaml` 加 VMAS 配置块（`batch_size`、`action_repeat` 需重新标定 ——
   注意 `action_repeat` 与 `train_ratio` 的耦合会直接放大训练强度）
3. 冒烟：跑通 IDreamer / CoDreamer 各一小段
4. **把开环漂移指标接进训练日志**（不只是事后脚本），使它随训练过程出曲线
5. 三 seed 对照，主看开环漂移，回报作为次要

### VMAS 接线完成（冒烟通过）

`make_env` 的 `vmas` 分支 + `configs.yaml` 的 `vmas_*` 键。冒烟两变体均跑通：

```
Multi-agent: num_agents=4, teams=1
train_ratio 512 -> 128 (compensating x4)      ← 多智能体补偿生效
Action Space Discrete(9)
Encoder MLP shapes: {'state': (22,)}          ← 18 观测 + 4 one-hot

idreamer   model 16,467,222 / actor 1,055,241 / value 1,181,439
codreamer  model 20,405,526 / actor 2,368,009 / value 2,494,207
```

### ⚠️ 两项真实实验前必须处理的实操约束

1. **prefill 必须 ≥ 一局长度。**
   VMAS 单局 500 步（论文 Table 8），而 replay 只在**整局结束**时才写盘。
   所以 `--prefill 500` 以下根本攒不出一个完整队伍，`make_graph_dataset` 会直接
   报 "no complete team in the replay buffer"。
   冒烟为了快用了 `--vmas_max_steps 20`，**这不是论文设定**。

2. **`batch_size` 的含义在 VMAS 下要重新标定。**
   它仍然是"队伍数"，但每队 4~5 个 agent、batch 里还要乘 N。
   flocking（N=4）下 `batch_size=4` 就是 16 条序列。
   参考：SEG（batch_length 较小）上 12 队可行、16 队 OOM，VMAS 要实测。

### 下一步

- **把 held-out 开环漂移接进训练日志**（目前只有事后脚本
  `scripts/eval_world_model.py`）。它需要在训练中定期跑一次 open-loop，
  代价是每隔 `eval_every` 多一批前向，可接受。
- 三 seed × 两变体对照。**主指标是开环漂移，回报作次要** ——
  论文自己说 VMAS 上的回报差异可能是随机性。

### 开环指标已进训练日志（冒烟验证通过）

`models.openloop_error` 是**唯一实现**，训练循环与事后脚本 `scripts/eval_world_model.py`
共用它（脚本原先自己有一份拷贝，那样两个口径会各自演化）。

数据来自 `evaldir`：评测局现在会落盘，它们由当前策略现跑、从不参与梯度更新，
天生是 held-out。`simulate_marl` 原先在 `is_eval` 分支里不写盘。

冒烟实测（VMAS flocking，200 步，只为验证管线）：

```
[60] openloop_h1_reward_mae 0.0 / openloop_h16_kl 7.1 / openloop_h16_state_mse 5.6
     ... reward_loss 1.0 / state_loss 0.5 / ...
```

**同一时刻 `reward_loss=1.0` 而 held-out 开环 `reward_mae=0.022`，差 45 倍。**
这就是为什么拿 replay 训练损失判断"世界模型想得准不准"没有意义 ——
它衡量的是重建（而且用的是看过当步观测的 posterior），不是预测。

### ⚠️ 远端协作的坑：push 和 fetch 都会静默失败

wyb 到 GitHub 的 443 时通时断，`git` 在连不上时的行为**极易骗人**，已经白追了一轮：

1. `git push` 报成功，远端其实还是旧 commit → 训练跑的是没修复的代码；
2. **更隐蔽**：`git fetch` 也静默失败，此时本地的 `origin/main` 是**旧**引用，
   于是 `git reset --hard origin/main` 会**成功地**把代码重置回错误的 commit，
   全程无报错。

固化在 `scripts/sync_remote.sh`：fetch → 比对 hash → reset → **再验一次 HEAD**，
任一步不匹配就重试。远端运行脚本里**不再写 git 命令**（它们不知道自己有没有成功）。

### VMAS 下一步

冒烟数字（200 步）是噪声，不是结果。正式对照需要：

- `prefill >= 一局长度`（论文设定 500 步），否则攒不出完整队伍
- `batch_size` 重新标定（仍是"队伍数"，但每队 4~5 agent）
- 三 seed × 两变体，**主指标开环漂移，回报次要**（论文自己说 VMAS 上回报差异可能是随机性）

---

## 吞吐真相（纠正之前所有相关结论）

### 我之前的错误

我一度判断"有约 3.9 秒/迭代的开销不知去向"，并据此说"算力差两个数量级"。**两个结论都不成立。**

根因是一个口径错误：**`tools.Every.__call__` 返回的是次数，不是布尔**。

```python
class Every:
    def __call__(self, step):
        ...
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count          # ← 次数
```

于是 `batch_steps = batch_size × batch_length`，而

```
每 _step 单位的更新次数 = 1 / (batch_steps / train_ratio) = train_ratio / batch_steps
```

**缩小 `batch_size` / `batch_length` 只会让单次更新更便宜、次数按比例变多 ——
每个 agent 步的总训练量由 `train_ratio` 锁死，完全不变。**
我把"单次更新 0.057 秒"直接和端到端 fps 比，**漏乘了每次调用里实际的更新次数**
（实测 32 次），那 3.9 秒根本不存在。

复算：32 × 0.057 = 1.82 s / 次调用，每次调用推进 4 agent 步 → 0.456 s/agent 步
→ 上限 2.2 agent 步/秒；实测 1.6，**差距 0.17 s/agent 步，完全解释得下**。

### 顺带纠正的另外两个错误（旧 profiler 脚本）

| 错误 | 后果 |
|---|---|
| 主循环是 `while agent._step < steps + eval_every`，而我为了"不评测"把 `eval_every` 设成 1e6 | 要跑 **100 万步**，剖析永远不会结束 |
| `train_ratio` 方向搞反（越大 → 间隔越小 → 更新**越频繁**） | 我那个"关闭训练"的对照组 `train_ratio=1e8`，实际是**最激进**的更新组 |

### 正确的模块级计时（`scripts/prof_vmas.py`）

固定迭代次数、每段 GPU 前后 `torch.cuda.synchronize()`、热身、逐模块：

```
next(dataset) 采样一个图批次        0.49 ms   cpu
世界模型 fwd+bwd+opt              51.68 ms   gpu    ← 85%
actor-critic 想象训练             33.73 ms   gpu
_policy 前向（执行路径）            2.61 ms   gpu
VMAS env.step                    4.79 ms   cpu
save_episodes（写盘）              0.65 ms   io
──────────────────────────────────────────────
一次完整训练更新                   88.0 ms
每个 agent 步（×2 次更新）         176.0 ms
→ 上限吞吐                       5.68 agent 步/秒
```

**没有悬空开销**：训练更新占 97%，环境只占 5%。

### 可行性账（`train_ratio=512`，即论文设定）

| 步数 | 单次运行 | 6 次（3 seed × 2 变体） |
|---|---|---|
| 5e5（论文） | 24.4 h | **6.1 天** |
| 1e5 | 4.9 h | 1.2 天 |
| 2e4 | 1.0 h | 0.2 天 |

**唯一能真正提速的旋钮是 `train_ratio` 本身**（会偏离论文设定；但两个变体用同一
设定时，对比仍然公平）。批次大小不是旋钮 —— 它被 `train_ratio` 反噬掉了。

### 方法论修正：KL 不能单独作为"世界模型更准"的证据

之前用 `h=16 时 5.9 倍 latent KL` 作为 CoDreamer 世界模型更好的主证据。这个说法
**过强**：两个模型学的是**各自的 latent space**，KL 的绝对值不可直接横向比较。

更可靠的替代口径（按可信度排序）：

1. **解码后的多步 state MSE**（`openloop_h*_state_mse`，已在日志里）
2. **reward MAE / continuation 校准**（已在日志里）
3. 最终任务表现

KL 仍可作为**同一模型内部**随 horizon 漂移的指标，但不该单独用来说明"跨模型更准"。

### Estimate Game 的重新定位

不是"完全放弃"。它不适合比策略回报，但适合做：

- 数据管线回归测试
- GNN 是否生效的单元环境
- 世界模型开环诊断
- IDreamer / CoDreamer 架构消融的**廉价**测试床

---

## VMAS 变体对照：一次有结论的调查（2026-10-15）

### 起因

VMAS 冒烟里 CoDreamer 的开环误差比 IDreamer 差 4~5 倍（两轮独立采样一致），
看起来是个明确的负面结果。追了一天之后，**其中大部分是评测口径的伪影**。

### 已确认的结论

**1. 那次 4~5 倍主要是评测混淆，不是模型差异。**

两臂各自 on-policy 采集时，两个世界模型根本不在同一批数据上比。让它们评在
**同一批** held-out 轨迹上（新增 `--rollout random` / `--rollout-dir`）：

| h | state_mse IDreamer | CoDreamer | 倍数 |
|---|---|---|---|
| 1 | 0.0621 | 0.0803 | 1.29× |
| 8 | 0.0810 | 0.1342 | 1.66× |
| 14 | 0.0926 | 0.1568 | 1.69× |

4~5 倍 → **1.3~1.7 倍**，且 `reward_mae` 反号（共享数据上 CoDreamer 每个 horizon 都更低）。
分解：数据因子 ≈ 3.8×，模型因子 ≈ 1.4×，`3.8 × 1.4 ≈ 5.3` 对上原来的 5.4。

**2. "CoDreamer 在 VMAS 上更差"在 n=3 下不成立，也不被否定。**

```
                 step 250   12250   24250(终)
IDreamer  s0      −4.37     −1.12    +0.78
IDreamer  s1      −5.04     −2.43    −2.50
IDreamer  s2      −2.51     −0.73    +0.23
CoDreamer s0      −0.69     −2.07    +0.11
CoDreamer s1      −3.20     −1.94    −6.18
CoDreamer s2      −1.34    −16.80    −5.79
```

终点均值 IDreamer −0.50（std 1.68）、CoDreamer −3.95（std 3.20），
差 3.45、合并 SE 2.09 → **1.65σ**。已按下面的样本量处方补跑到 8 seed/臂。

**3. 方差分解 —— 决定该拧哪个旋钮。**

- σ_seed ≈ **2.4**（同变体不同 seed 的真实差异）
- σ_eval ≈ **0.5**（单次评测的采样误差，n=4 队）
- `Var(臂均值) ≈ (σ_seed² + σ_eval²)/k`

**seed 方差比评测误差大一个量级，所以加评测局数几乎无用**（纯前向、看着便宜，
但压在不起作用的那一项上）。要分辨 d≈1 需要 **k ≈ 25 seed/臂**；
分辨当前这个 d=3.45 需要 8。**先做方差分解，再选旋钮。**

**4. critic 是弱向导 —— 跨两个环境成立。**

固定 held-out 轨迹上比 `V(s_t)` 与真实折扣回报 `G_t`（新增
`scripts/check_critic_calibration.py`）：

| | VMAS | SEG |
|---|---|---|
| `corr(V, G)` | +0.035 / +0.125 | +0.243 / +0.209 |
| V 与回报的**尺度比** | **~1000×** | ~0.4× |

换成与训练同视界的 10 步回报，相关系数几乎不动（0.035→0.036）——
"你比的视界不对"这个反驳排除。

**5. VMAS 的价值膨胀是环境特有的。**

SGE 的 V（−0.05）与它自己的回报（−0.12）同量级；VMAS 的 V（12）是其回报（~0）的
1000 倍。`target_mean` 从 0.038 涨到 11.92（**314×**）而进入目标的唯一数据项
`imag_reward_mean` 纹丝不动（0.023 → 0.012）。

**最干净的表述是比值，不是数值**：价值目标里自举项 `γ·V ≈ 12.0`、奖励项 `r̄ = 0.012`，
**约 997:1** —— 目标里 99.9% 是 critic 自己的输出。

**6. 模型对策略自身的动作系统性高估**（新增 `scripts/check_model_exploitation.py`）。

| | IDreamer | CoDreamer |
|---|---|---|
| 模型承诺（每步） | +0.0113 | +0.0011 |
| **环境实际给** | −0.0013 | **−0.0136** |
| 承诺 vs 实际的相关 | +0.2675 | +0.3851 |

判据是"**模型对策略所选动作的承诺 vs 环境对此的实际交付**"，与 `openloop reward_mae`
问的不是同一件事（后者喂真实动作，测的是准不准；前者测的是策略能不能找到模型的错）。
同一个模型可以同时"预测得准"（MAE 0.02）和"对策略说谎"。

### 已否定（都是本次调查中自己提出又推翻的）

| 假设 | 如何被推翻 |
|---|---|
| 想象塌缩到常数 | `check_imagination.py`：想象奖励散度 k=1→15 只从 2.4e-2 变到 1.9e-2，**没有趋零**。日志里的 `imag_reward_std 0.0` 只是**按 1 位小数打印**，0.019 显示成 "0.0" |
| `value_bootstrap='slow'` 能修好价值 | 慢目标与在线 critic 只差 **0.19% / 0.56%**，与真实回报的相关性和在线 critic 一样烂（+0.037 / +0.118）。它是延迟约 50 次更新的副本（fraction 0.02，全程约 32000 次更新），**不是锚** |

### 新增的工具与开关（都在 `exp/comm-ablation` 分支）

| 文件 | 作用 |
|---|---|
| `scripts/check_critic_calibration.py` | V(s_t) vs 真实折扣回报，同一批固定轨迹 |
| `scripts/check_imagination.py` | prior 前滚时各头输出是否塌缩 |
| `scripts/check_model_exploitation.py` | 模型对策略自身动作的承诺 vs 环境实际交付 |
| `scripts/ablation_comm.sh` | 四臂通信消融（含参数量恒等式机械检查） |
| `scripts/eval_world_model.py` | 新增 `--rollout {self,random}` 与 `--rollout-dir`，可让多个 checkpoint 评在同一批轨迹上 |
| `configs.yaml` | `ckpt_every`（默认 0）：按步号另存 checkpoint。上游只留 `latest.pt` 且每轮覆盖，"某个量随训练怎么变"这类问题**在原理上无从提问** |
| `configs.yaml` | `value_bootstrap`（默认 `'online'`）：见上面的否定，保留只为复现 |

### ★ 核心机制：有效折扣由 `cont` 决定，而 `cont` 由「想象视界 vs 局长」决定

λ-return 的折扣是 `config.discount * cont_head`，不是 `config.discount`。逐 k 实测：

```
SEG  IDreamer   cont:  k=1: 0.3071  k=2: 0.3044  k=3: 0.2118  k=4: 0.0619  k=5: 0.0000
VMAS IDreamer   cont:  k=1..5 恒为 1.0000
```

| | `cont`（起点） | 隐含累积视界 `1/(1−γ·cont)` | 实测 `V/r̄` |
|---|---|---|---|
| VMAS IDreamer | **1.0000** | **333 步** | **955** |
| SEG IDreamer s0 | **0.307** | **≈ 1.4 步** | **0.4** |
| SEG CoDreamer s0 | 0.202 (k=5) | ≈ 1.3 步 | **1.2** |

**同一个 `γ=0.997`，有效累积视界相差约 240 倍。** 两个工作区的 `V/r̄` 与隐含视界
同向、同量级（差 ≤ 10 倍，不是精确吻合）。

**注意 `cont` 本身是对的，不是坏掉：** SEG 的局真的只有 6 步，VMAS 的局有 201 步且
只在最后一步标终止。继续概率头在两个环境上都如实反映了这一点。**出问题的不是这个头，
是 `imag_horizon` 与局长这个从未被一起设置过的关系：**

- **想象视界相对局长太长**（SEG：15 vs 6）→ 想象轨迹一滚就穿过终止点
  → `cont → 0` → **critic 退化成只看一步**，优势信号没有未来。
  SEG 上"回报饱和、分不出变体"很可能源于此，而不是环境太简单。
- **局长远长于想象视界**（VMAS：201 vs 10）→ 永远够不到截断点 → `cont ≡ 1`
  → **critic 累积 333 步**，把世界模型每步 0.013 的高估放大成约 4.3 的虚假价值
  → actor 去追模型高估的地方 → 策略在环境里拿到负奖励（实测 −0.0136/步）。

⚠️ 记录一次差点犯的错：初次测量只取了 `k=5`，读到 `cont = 0.0000`，差点报成
"继续概率头塌缩了"。**而 k=5 在 SEG 上本来就是终止步。** 只看单点而不看逐 k 曲线
会得到一个完全错误的机制。凡是"某个量在末步很小"，先问末步本身有没有特殊性。

**因果链（每一环都有实测）：**

```
模型对策略所选动作高估 +0.013/步      （check_model_exploitation.py）
      ↓ × 1/(1−γ·cont) = 333
约 4.3 的虚假价值（V 实测 12，真实回报 ≈ 0）
      ↓
actor 最大化这个优势 → 去找模型高估的区域
      ↓
策略在环境里平均拿到负奖励 → 回报不升反降
```

**这条同时解释了论文为什么在 VMAS 上分不出两个变体**：不是通信没用，是无论哪个架构，
优势信号都被放大后的模型偏差淹没。与结构无关，与 `γ × 局长` 有关。

**可检验的推论（尚未做）**：把 VMAS 局长缩到 SEG 量级（或让局真实终止），`V/r̄` 应按
比例回到 O(1)。这是这条机制的直接否证点。

### ★★ 最终结论：VMAS 上 CoDreamer 更差，8 seed 下显著

补齐到 8 seed/臂后的终点回报（step 24250）：

```
IDreamer   n=8  mean −0.548  std 1.317  中位数   0.00
CoDreamer  n=8  mean −4.199  std 2.995  中位数  −4.70
差值 +3.651   合并 SE 1.157   t = 3.16   Welch df 9.61   → p ≈ 0.011
```

**不是平移，是重尾**：最低四名全是 CoDreamer（−8.7 / −6.2 / −5.9 / −5.8）；
IDreamer 最差 −2.6。CoDreamer 有四次低于 −5.5，IDreamer 零次。

**完整因果链（每一环都有实测）：**

```
① CoDreamer 的世界模型对策略所选动作高估 +0.0147/步（IDreamer +0.0127）—— 只差 15%
② 有效累积视界 333 步（cont ≡ 1），15% 被放大
③ CoDreamer 策略在环境里拿 −0.0136/步（IDreamer −0.0013）—— 差 10 倍
④ 整局累计 → 回报差 3.65；偶发深度利用 → 重尾
```

**关键是否定了直觉的一半：这不是"GNN 让世界模型变坏了"。** 共享轨迹上模型差距只有
1.3~1.7 倍，而且 CoDreamer 的**奖励预测反而更好**（`reward_mae` 在每个 horizon 都更低）。
差的是**策略对自己模型的利用程度**，而 333 步视界把它放大成策略差异。

**必须写明的边界**：该结论成立于本配置（15k 步、4 队、200 步截断局，论文的 3%），
**不是**对论文 CoDreamer 的判定。而且这个配置恰落在放大最强的工作区——它测的是
"在此工作区内 A 与 B 的差异"，不是方法的普遍优劣。

**这也解释了论文自己那句话**（"IDreamer 与 CoDreamer 在 VMAS 上相似，差异可归因于
随机性"）：在论文的 500k 步预算下，世界模型拟合得更好，每步高估更小，经 333 倍放大后
仍落在噪声里。**我们之所以能看到差异，恰恰因为训练不足放大了高估**。

### 由此收口：SEG 为什么分不出变体（推翻台账此前的归因）

本文件前面把 SEG 的饱和记为"环境已饱和、不适合比策略回报"，语气上归因于**环境简单**。
**这个归因是错的**，正确的归因是**有效视界**：

- SEG 的有效累积视界 ≈ **1.4 步**（`cont` 起点 0.307 × γ=0.997）
- 而 SEG 的奖励只取决于**当前动作 vs 当前目标** —— 贪心即最优

所以**世界模型对 SEG 的策略根本没有用武之地，不是它不够好**。这与实测自洽：
两个变体的开环世界模型质量**确实有差别**（1.3~1.7 倍，共享轨迹上测的），
但那份差别传不到回报上——策略只需要看一步。

**"环境简单"和"有效视界只有 1.4 步"是两个不同的诊断，修法也不同。** 前者无解，
后者是一个可以调的参数关系。

### 方法论（本次付出代价换来的）

- **先做 seed 复制，再找原因。** 第一天花了几轮解释 0.78 vs 0.105，而它在同变体的
  另一个 seed 上是 0.78 vs −2.50。
- **判据是比值，不是数值。** SEG 的 V=−0.05、VMAS 的 V=12，单看数值分不出谁对；
  各自与自己回报的比值（0.4 vs 1000）立刻分开。同理：`target` 涨 314 倍而它唯一的
  数据输入不动 —— 不需要不动点公式就能断定驱动力来自自身。
- **"组件存在"不等于"组件起作用"。** 慢目标确实在、确实被 Polyak 更新，但它和在线
  critic 是同一个函数的两个时刻。
- **便宜 ≠ 有效。** 加评测局数比加 seed 便宜得多，但 σ_seed 主导时它基本无用。
- **带 n=1 的臂间比较不要下结论。** 变体差异 0.68，而 seed 波动 3.3。

---

## 性能测量的协议（2026-10-16）

> 来源：wyb 上一次独立的性能调查。以下六条**不是经验好习惯，每一条都有实测支撑**。
> 之所以写进台账而不是只留在那份交付里：这个坑在任何未来的性能测量中都会重现，
> 包括顺手测个数的时候。

### 为什么需要它

同一个配置、同一个进程、同一段代码，**位置**就能让墙钟差 11.2%：

```
第 1 次重建      77.30 ms
后续 5 次均值    85.93 ms
```

四个实验交叉对照后，来源被锁死在一个变量上：

| 实验 | 测量前跑过 profiler | 测量前丢弃一次重建 | 位置效应 |
|---|---|---|---|
| 位置探针 | 有 | 无 | **+11.2%** |
| 稳态重测 | 有 | **有** | +1% |
| 空档探针 | **无** | 无 | +0.8% |
| 时钟探针 | **无** | — | 1.1% |

**唯一对得上的是 `torch.profiler`**：它注册 CUPTI 回调、分配缓冲区，测完不释放；
"测量前丢弃一次重建"正好把它清掉。GC 与 caching allocator 都已被对照实验排除
（GC 开关无差异；`num_alloc_retries` 恒 0、peak 恒定 740.6 MiB）。

**它最危险的地方是自掩**：那次调查里每个 arm 的计数是在计时**之后**跑 profiler 取的，
于是每测一个 arm 就给下一个埋了 11% 的坑。**没有协议的话，任何人都能在这台机器上
"测出"任何方向的 10% 收益。**

### 六条规格

任何性能声明必须满足：

1. **每个 arm 前丢弃一次重建**（把进程内状态带到稳态）。
2. **同配置连测两次落在同一档**（差异 < 2%）—— 这是对第 1 条本身的自证，
   否则"丢弃一次"可能只是把快档挪给了下一个 arm。
3. **给出同批次的本底**，不能引用别次的。
4. **小于本底的收益不得声明。**
5. **同时报结构性计数（kernel 数 / GPU 时间 / 图数）。** 它们的职责不是佐证收益，
   而是**证伪测量条件**：实测出现过 wall 没变而 kernel 少 1100 个的情况 ——
   那说明两次测的不是同一个东西。
6. **跨进程不得比较绝对耗时。** 同配置跨进程 70.89 vs 79.77（12%），同批次只差 2%。

   ⚠️ **这条只管耗时，不管结果。** 回报类的跨进程对比是合法的 —— 被测量不是时长，
   而且噪声尺度差两个数量级（跨进程耗时漂移 12%，而 seed 间回报 σ=2.4 相对效应量 3.65）。
   **同一条规则在两类量上不适用，因为分母差两个数量级。**

### 与 kernel 数的关系

那次调查里"减少 kernel 数"**四次都没有转化成墙钟**（kernel 降幅 −18% / −22% / −29% / −36%，
对应墙钟变化 0~5%），第五次甚至在同一档内出现"kernel 少 1100 个、wall 不变"。

**墙钟不由工作量决定，而由空档决定** —— 而 kernel 数与 CPU 提交时长都不含空档。
这正是第 5 条里结构计数的定位是"证伪条件"而不是"预测收益"的原因。

**同一批测量的实际收益**：把编译真正接到 `observe`/`img_step`（上游的
`torch.compile` 只拦 `forward`，而 RSSM 里那两段计算在独立方法中，从未被编译到）
约 3%（n=2，区间 2%~5%，本底 1%~2%，刚过本底）。**上界 1.27×，兑现 1.03×。**
对"分辨 d≈1 需要 25 seed/臂"的项目来说，优先投 seed 而不是继续推这条路径。

---

## 演示环境的选型与通信探针（2026-10-16）

> 目标：找一个**简单的多人协作游戏**，把算法接上去，让"通信在起作用"这件事
> 能直接看出来，而不是靠一条噪声很大的回报曲线。分支 `feat/foraging-demo`。

### 选型：Overcooked（胡闹厨房）

**第一步选的是 Level-Based Foraging，被否掉了 —— 理由是用户不认识它。**
原判据是"不通信在信息论上是否做不到"（VMAS flocking 的 lidar 对邻居完全盲、
奖励是成对距离的 potential shaping，每个 agent 看自己那份就够，通信因此冗余）。
LBF 在这条判据上合格，但它是个网格觅食任务，**看不出是个"游戏"**。
改用 **Overcooked**：洋葱、锅、盘子、上菜口不需要解释，谁在配合、谁在原地打转，
看回放就分得出 —— 而"直观"本来就是这个任务的第一需求。

代价是 Overcooked **默认全可观**（官方 featurize 给出整个厨房，含所有玩家的
位置/朝向/手持），正是要避开的那件事。所以包装**不使用**官方编码，自己写限视野
特征。这是选型与实现的交换：**用一个家喻户晓的游戏，换一层自己写的观测。**

**只依赖 numpy。** `overcooked_ai_py.mdp.overcooked_env` 顶层 `import gym, tqdm`，
而本项目装的是 `gym 0.22.0`（vmas 的依赖）；`overcooked_mdp` 只依赖 numpy。
所以直接驱动 `OvercookedGridworld`，顺带避开官方 `lossless_state_encoding` 里
`num_players == 2` 的假设 —— 玩家数由**布局里的玩家数字**决定，我自写了三人厨房。

**两个必须先压下去的默认值**（否则短局长在语义上是空的，与 Melting Pot 被否掉的
原因是同一个）：

| 默认 | 问题 | 处理 |
|---|---|---|
| `cook_time = 20` | 25 步的局长里连一锅汤都煮不完 | 默认取 **5** |
| 配方 3 个洋葱 | 光"去分发器→回锅"×3 就要 30+ 步 | 默认改为 **1 个洋葱** |

压完之后，自检里的完整流程（取洋葱→下锅→开火→煮熟）**第 9 步**就走完。

**限视野的判据不是"窗口占多少面积"。** 厨房大部分是台面，占面积大不等于看得见锅。
正确的判据是：站在一个典型位置，**任务相关站点**（锅/洋葱/盘子/上菜口）看得见几个。
实测：

```
7x5  sight=2   平均看得见 80% 的站点； 80% 的位置至少漏一个   ← 几乎没造出信息差
9x6  sight=2   平均看得见 49% 的站点；**100%** 的位置至少漏一个   ← 默认
9x7  sight=2   平均看得见 39% 的站点；100% 的位置至少漏一个
```

所以三人厨房定为 **9x6** —— 它是"每一处都漏点什么"的最小尺寸，同时不至于把
交付流程推到 25 步装不下。

**C 也是实测选的**（9x6，2000 局随机起局）：

```
C=1  平均度 0.43/2   全连通  1.9%   全断开 46.4%   ← 太稀
C=2  平均度 1.00/2   全连通 18.4%   全断开  9.1%   ← 默认（= sight）
C=3  平均度 1.50/2   全连通 48.4%   全断开  0.0%   ← 太密
```

**局长 35，不是 25 —— 这条也是实测逼出来的。** 用脚本策略追一局：

    第 2 步取洋葱 → 第 5 步下锅 → 第 6 步开火 → 第 10 步煮好
    → 第 13 步取盘子 → 第 19 步端起汤 → ……

**第 25 步被截断，差两三步没送到。** 9x6 厨房里锅到上菜口有 5 格远，25 步装不下
一次完整交付（trace 里那一局奖励 0）。提到 35 后同一个策略拿到 **20 分**。
这也说明"局长够不够"不能靠推，要看一条完整交付链实际用掉多少步。

**一个不显然的机制**：下锅与开火是**两次交互** —— 只有玩家**空手**对锅交互时
才会 `begin_cooking()`（`overcooked_mdp.py:1168-1172`）。少了第二步
`cooking_tick` 永远是 -1，汤永远煮不好。自检断言 tick 从 -1 变为非负，
并按 tick 判"煮好了"而不是与 0 相等（`begin_cooking` 置 0 之后，**同一步**的
`cook()` 就把它推到 1）。

**观测的构造原则**：窗口之外的格子**根本不进向量**，而不是填 0 —— 填 0 等于
告诉 agent"那里是空的"，那也是一种信息。窗外队友只有"不可见"一个比特，其相对
位置被置零；自检里有反证断言"可见时相对位置必须非零"，否则那条会因为该字段恒为 0
而假通过。

⚠️ **LBF 已随这次改选整体移除**（`envs/foraging_env.py`、`foraging_*` 配置键、
`make_env` 分支、接线测试的对应内容）。它测过的东西（观测局部性、协作强制性、
C 的选法）都已迁到 Overcooked 上重做，探针的实测表也是重新测的 —— **没有留下
任何一个在已删除环境上测出来的数字**。

### 通信探针：比"变体间比回报"更合适的仪器

回报对比在这里是坏仪器：σ_seed 3.3 而效应量 0.68~3.65，8 个 seed 才勉强显著，
而且测的是混合物。探针把问题拆到最小 —— **通信到底传不传得过去，传几跳**。
它是架构属性，所以**不需要训练、没有 seed 方差**（同一份权重、同一份观测、
共同随机数），而且**可证伪**：reach 必须等于各段 GNN 层数。

做法：`Dreamer._policy` 是 (obs, state) 的函数，于是同一 incoming state 下跑两次、
只把 agent j 的**观测内容**换成 agent k 的（保留 j 自己的 one-hot 索引、图不动），
两次之差就是 j 对 i 的影响。

**实测（overcooked_demo，3 agent，C=2，60 条轨迹 × 6 步，随机初始化）：**

| 跳数 | deter(t+1) 相对变化 | 动作(t) 翻转 | 动作(t+1) 翻转 | logprob(t) 相对变化 | 样本数 |
|---|---|---|---|---|---|
| IDreamer（对照） | **0** | **0** | **0** | **0** | 702 / 94 |
| CoDreamer 1 跳 | 5.557% | 6.13% | 6.15% | 3.07e-02 | 702 |
| CoDreamer 2 跳 | 0.114% | **0.0000** | 0.0000 | **恰好 0** | 94 |

读法：**actor 那层 GNN 的 reach = 1**（动作(t) 与 logprob 在 2 跳恰好为 0）；
**deter(t+1) 的 reach = 2**（见下），3 跳归零。IDreamer 逐格严格为零，证明测到的
差异确实来自通信。同一份探针在 seg_seq 与 vmas_flocking 上给出同一结构。

⚠️ **探针免训练，所以上表只是"路径存在"的证据，不是"哪个通道更强"的证据** ——
它比的是两个**随机初始化**的模型，训练后的幅度不可由此外推。

### ★ `deter(t+1)` 为什么是 2 跳：两段 GNN 分处两个时刻 ★

**决定性证据是消融**（SEG，静态图，同一批 108 个 2 跳样本）：

    rssm=1 head=1 behavior=1    2 跳 deter = 0.000281    非零
    rssm=1 head=0 behavior=0    2 跳 deter = 恰好 0
    rssm=0 head=0 behavior=1    2 跳 deter = 恰好 0

两段缺一不可，所以路径必然是复合的：

    扰动 obs_t[j] → stoch_t[j] 变
      → 【图 t 上的 actor GNN】把 j 的信息传给 action_t[m]，m ∈ N_t(j)   ← 第 1 跳
      → action_t 是 t+1 时刻 GRU 的**输入**（`x = cat([prev_stoch, prev_action])`）
      → 【图 t+1 上的 RSSM GNN】再聚合一跳                               ← 第 2 跳
      → deter_{t+1}

**这条路径与图是否随时间变化无关**：SEG 的邻接矩阵只在 `reset()` 里采样一次
（`envs/seg.py:243`），`step()` 从不重采样，局内静态，而这个 2 跳照样出现。

⚠️ **但"2 跳那一行非零"有两个不同的来源，必须分开**（我在这一点上先说错过一次——
写下"与图变化无关"就收工了，而它只对上面那条路径成立）：

| 来源 | 需要什么 | 静态图上出现吗 |
|---|---|---|
| **跨时刻复合**：actor@t → `action_t` → RSSM@t+1 | rssm=1 **且** behavior=1 | **出现**（SEG 实测 0.000281） |
| **分类口径**：图 t 与图 t+1 不同，一对 agent 在图 t 上只隔 1 跳、却被图 t+1 归成 2 跳 | 任何一条单跳通道 | **不出现**（SEG 实测恰好 0） |

判别证据是同一个配置在**静态图**与**动态图**上的对比：

    --gnn 0 0 1   SEG（静态图，局内邻接矩阵不变）    2 跳 deter = 恰好 0（108 样本）
    --gnn 0 0 1   Overcooked（动态图，agent 一直移动）2 跳 deter = 0.002161（94 样本）

`rssm=0` 时不存在复合通道（没有 RSSM 的 GNN 可接第二棒），所以动态图上那
0.002161 **只能**是分类口径。而 SEG 静态，分类口径不可能产生非零 —— 所以它
那个 0.000281（`gnn 1 1 1`）**只能**是跨时刻复合。两个环境各自排除了一个来源，
**合起来才是决定性的 —— 单个环境分不开这两支。**

**对探针的用法含义**：`deter(t+1)` 那一列在**动态图**上混了两种东西，只有
**静态图**（SEG）上的 2 跳才能干净地读成"有没有第二条通路"。

⚠️ **一处被实测否掉的解释，记在这里免得有人重新发明**：曾写成"i 的 `deter_t` 里
已经有 j 的信息，GRU 把它带到了 t+1"。**这是错的** —— `deter_t` 可证不受 `obs_t`
影响（它由 `state_{t-1}` 算出）。载体是 `action_t`，走的是 GRU 的**动作输入**，
不是循环状态。（这个错是 Maindev 从"分类与解释自相矛盾"里挑出来的：我一边说那些
agent 在 t 时刻只隔 1 跳，一边又把它们归为 2 跳。**分类与解释对不上时，先怀疑解释。**）

**推论一**：端到端的 reach **不等于单段层数**，两段分处相邻时刻的 GNN 会串成一条
跨时刻的链。所以测 reach 时不能由 `(rssm, behavior)` 层数**推算**，必须以实测为准
—— 这正是这个探针存在的理由。

**推论二（★ 直接影响消融的解读，包括正在跑的 SEG 四臂）★**：
既然 `action_t` 是下一步 GRU 的输入，那么在**想象**里 actor 的输出同样进入世界模型
的下一步动力学（`_imagine`: `action = policy(feat)` → `img_step(state, action)`）。
于是**四个消融臂不是四个独立的通信开关**：

    ac_comm（rssm=0 head=0 behavior=1）  世界模型那一侧的通信**不等于零** ——
                                        RSSM 的 GNN 确实是恒等，但邻居信息仍然
                                        经由动作通道流进了 deter_{t+1}

实测证据就在上面那次消融里，`--gnn 0 0 1` 那一行：

    1 跳 deter(t+1) = 0.004082     ← rssm 的 GNN 关着，隐状态照样变了

对照 `--gnn 1 0 0`（只有 RSSM）的 1 跳 deter = 0.009934 —— 两条路线**量级相同**，
只是入口不同。所以"哪个臂的通信在起作用"不能按 config 里的三个开关去读，
`idreamer / wm_comm / ac_comm / codreamer` 这组命名会误导。

**所以 reach 的判断必须以实测为准——包括消融臂的命名。**

### 探针已并入回路测试（`scripts/test_training_loop.py` 第 [7] 节）

不留在 `scripts/` 当独立工具，理由是台账自己记着的那条：**566 行模块级断言全绿
却漏掉两个致命 bug，因为它们长在集成点上** —— 而通信接线正是集成点（执行路径的
三次形状 bug 全出在那里）。独立脚本要靠人记得跑，这一层不行。

它与既有两节的分工是本轮最值得写下来的一点：

| | 测什么 | 抓不到的错 |
|---|---|---|
| `[4]`/`[5]` | **敏感性**：改邻接矩阵，动作变不变 | GNN **接错位置但仍对邻接敏感**（例如插在 encoder 之后、或只接了一半 agent） |
| `[7]` | **结构**：哪几跳有、哪几跳无 | —— |

而且 [7] 的判据是**二值**的（1 跳有 / 2 跳恰好没有），因此跨环境可比、不随 seed 漂、
不需要噪声本底。这正是本项目在"变体间比回报"上缺的三样性质。

节内也堵了空断言：先断言样本里确实存在 1 跳与 2 跳的 (i,j) 对，否则那两条
"恰好为 0"会在没有对应样本时假通过。用 **5 agent + 小 C** 是为了让 2 跳真的存在。

**"恰好为 0"那两条断言落在 `logprob` 上，不落在翻转率上** —— 这是本节最后定下来的
形态，理由见下面「依据能承载它的量级」。翻转率是阈值化的量，用它会引出样本量论证，
而 (i,j) 对在同一局里共享状态、并不独立（有效样本量上限 = 局数 × 可用 t 数，
设计效应可达数倍）。`logprob` 是连续量，"恰好没变"于是成了 float 精确断言 ——
实测 60 个 2 跳样本上 `0.000e+00`，不需要任何样本量论证。

⚠️ 那条 2 跳断言失败时，**第一问是「是不是有人把层数改成 2 了」**，不是
「是不是掩码泄漏了」—— 它本质是一句关于 GNN 层数的断言。而且前提是那个 0 来自
**硬掩码**（结构性无通路）：若将来换成软门控，它会失败 —— **那时失败是正确行为**。

### ★ 依据能承载它的量级，把主张放对地方 ★

这一轮最值钱的一条，比这个测试本身大。它同时收编了本文件前面几条分散的教训：

| 量 | 能承载什么 | 本项目里的实例 |
|---|---|---|
| 学习曲线 / 回报 | 只能承载**大**效应 | σ_seed 3.3 而效应量 0.68~3.65，要 25 seed/臂才分辨 d≈1 |
| 训练损失 | **什么都不能承载**（estimand 错） | `reward_loss=1.0` 而 held-out 开环 `reward_mae=0.022`，差 45 倍 |
| **阈值化**的量（翻转率、是否成功） | 只能抓**大**错（层数错、接线错），抓不住边缘效应 | 翻转率 4~6%，弱影响会被 argmax 吃掉 |
| **float 精确**的结构量 | 可以承载"有/无通路"这类**二值**主张，**不需要样本量论证** | logprob 在 2 跳恰好 0.000e+00 |

推论：**先问"这个量能承载什么"，再问"它现在是多少"。**
具体到选指标——把一个统计问题**消掉**，优于把它**修好**。

这一条是被一次接近的错逼出来的。我最初把"actor 在 2 跳无通路"这条主张放在**翻转率**
上，而翻转率是阈值化的：测到 0 次既可能是真没通路、也可能只是样本不够，于是必须
做样本量论证。而那个论证又有第二个坑 —— **(i,j) 对在同一局里共享状态、并不独立**，
有效样本量上限是 `局数 × 可用 t 数`，设计效应可达数倍，`n ≳ 83` 的阈值可能只剩
`n_eff ≈ 14`。把它换成连续的 `logprob`（贪心动作未翻转时，两次比的是**同一个动作**
在两个分布下的对数概率），"恰好没变"就成了一条 float 精确断言：真有通路的话第一个
样本就会给出非零值。独立性、设计效应、簇相关统统不进入。

**边界（必须写明）**：这个论证成立的前提是那个 0 来自**硬掩码**（结构性无通路），
而不是算术相消。若将来有人把硬掩码换成软门控，测试会失败 —— **那时失败是正确行为**
（结构变了），不是误报。

### 探针开发中犯的三个错（都靠对照抓出来）

1. **第一版测在 `state=None` 的第一步**，`deter` 对观测恒为常数 —— 因为
   `obs_step` 里 `deter_t = img_step(prev_state_{t-1}, ...)` **不含 obs_t**，
   当前观测只进 posterior 的 `stoch`。**RSSM 的通信相对观测晚一步。**
   症状是两个变体都报 0.000e+00，看起来像"通信没接线"。
2. **拿扰动调用的 `D_t` 去比 baseline 的 `D_{t+1}`**，测到的是隐状态自己的
   时间漂移而非扰动 —— 完全没有通信的 IDreamer 因此显示出 0.64 的"巨大变化"。
   正确做法是扰动后**再走一步**再看 deter。
3. **所有量按同一时刻的图分组**，而 agent 会移动：t 时刻隔 2 跳、t+1 只隔 1 跳的
   样本混进了"2 跳"那一行，于是那一行永远不可能是 0。每个量必须按**它自己那一
   时刻**的图分组。

三条都写成了断言（确定性自证、扰动生效自证、**层数全 0 时跨 agent 影响必须恰好为 0**
的对照）。第三条对照是关键：它把第 2 类错误从"一份看起来很像结论的表"变成当场报错。

### 已知坑

- `networks.MLP.__init__` 的 `device` 默认值是**字面量 `"cuda"`**（networks.py:706），
  而 `config.encoder` 里没有这个键 —— 所以整个代码库**在不暴露 CUDA 时连构造都过不去**
  （`CUDA_VISIBLE_DEVICES=""` 会炸在 `torch.tensor(..., device="cuda")`）。
- `cfg.comm_variant` 的 yaml 默认是 `'idreamer'`。不显式设它，`codreamer` 的 run 会
  静默建成 IDreamer，而探针照样打得出一张完整的表 —— 两行一模一样。必须调
  `dreamer.resolve_comm_variant()`（唯一实现），并把参数量当机械证据打出来。
- **Overcooked 里下锅与开火是两次交互**：只有玩家**空手**对锅交互时才
  `begin_cooking()`（`overcooked_mdp.py:1168-1172`）。少了第二步 `cooking_tick`
  永远是 -1，汤永远煮不好。而且 `begin_cooking` 置 0 之后**同一步**的 `cook()`
  就把它推到 1 —— 判"刚开火"要判"不再 idle"，不是判等于 0。
- **`get_state_transition` 返回 `(new_state, infos)`**，奖励在
  `infos["sparse_reward_by_agent"]` 里，不是直接返回奖励。只有上菜才给稀疏奖励。
- **布局的四条边界必须全非空、每行等长、玩家数字恰为 1..n**（`_assert_valid_grid`），
  否则 `from_grid` 直接抛断言。第一版自写的三人厨房两处都踩了。
- **`from_grid` 要的是行列表**（`GRID.split("
")`），传整串会被当成逐字符迭代。
- `overcooked_env.py` 的 `adjacency` / `edge_features` 是 property，**reset 之前
  访问会炸**（`self._state` 还是 None），于是 `hasattr` 静默返回 False ——
  接线测试因此必须在 reset 之后再查接口面。
- 渲染：官方有 `state_string`（文本）与基于 **pygame** 的可视化（要显示后端）。
  `scripts/render_overcooked.py` 用 matplotlib 的 Agg 后端纯离屏出帧，
  **不碰屏幕也不需要额外系统包**；画面上带**每个 agent 的视野窗口**，
  "谁看得见什么"因此直接可见 —— 那正是通信要解决的问题。
- `SoupState.cook_time` 在汤**还没开火**时会抛
  `ValueError: Recipe is not determined until soup begins cooking` ——
  未开火时配方没定下来。渲染进度条这类地方要先判 `is_idle` 再读它。
