# CoDreamer

**Communication-Based Decentralised World Models**
(CoCoMARL workshop @ RLC 2024 · [arXiv:2406.13600](https://arxiv.org/abs/2406.13600))

基于 DreamerV3 的多智能体世界模型复现。论文**没有官方代码**，本仓库从零实现。

```
基座      NM512/dreamerv3-torch  (MIT)   vendored @ 6ef8646
环境      Sequential Estimate Game       纯 numpy，按论文 Table 1 实现
堆栈      Python 3.10 · PyTorch 2.4.1 · CUDA 12.1
硬件      Ubuntu 22.04 + RTX 3060 12GB（单机）
```

---

## 这是什么

CoDreamer 把 DreamerV3 扩展成多智能体算法，核心主张是：**让世界模型在 agent
之间通信**，而不只是让策略通信。

典型的基于世界模型的多智能体方法（论文称之为 IDreamer）里，每个 agent 各自
建模环境，把其他 agent 视作环境的一部分。CoDreamer 的改动是：把 agent 表示成
图上的节点，在**世界模型内部**做邻域聚合，让每个 agent 的隐状态包含邻居的信息。

本仓库提供该算法的可运行实现，包括环境中立的数据管线、图神经网络通信层，
以及一套用于验证正确性的数值测试。

---

## 特性

- **Sequential Estimate Game** —— 论文附录 B 的协作博弈环境，纯 numpy 实现，
  参数完全按论文 Table 1。它是一个**可证明需要非局部信息**的压力测试场：
  每个 agent 要预测的目标值有 70% 取决于邻居状态。
- **多智能体数据管线** —— agent 轴是样本内部的一维 `(B, T, N, ·)` 而非摊平进
  batch，因为 GNN 必须能在同一时刻对 N 个 agent 做邻域聚合。
- **GATv2 通信层** —— 支持边特征、带原特征跳连（防过平滑）、孤立节点安全。
- **RSSM 内通信** —— 在 GRU 咽喉点插入 GNN，通信信息自动流到 prior、posterior
  及所有下游预测头。
- **IDreamer / CoDreamer 单一开关** —— `gnn_layers=0` 时通信层是严格恒等
  （返回输入对象本身），整条链路退化为 IDreamer。
- **配置驱动** —— 环境规格、GNN 层数、观测模式等全部可通过命令行覆盖。

实现状态与复现细节见 [docs/reproduction-plan.md](docs/reproduction-plan.md)。

---

## CoDreamer 与 IDreamer 的唯一区别

两者**共用一套世界模型权重**（论文明确使用 parameter sharing），每个 agent
拥有自己的隐状态 $(h_i, z_i)$。差别只在一件事：**IDreamer 的 agent 彼此不可见，
CoDreamer 的 agent 会在同一时刻沿邻居做一次邻域聚合。**

```
              t-1                t                 t+1
          ┌────┬────┐        ┌────┬────┐        ┌────┬────┐
   隐状态  │ h1 │ h2 │        │ h1 │ h2 │        │ h1 │ h2 │
          └────┴────┘        └────┴────┘        └────┴────┘
              ╲  ╱               ╲  ╱               ╲  ╱
               ╳  ← GNN 在这里交换  ╳                 ╳
              ╱  ╲               ╱  ╲               ╱  ╲
              │  │               │  │               │  │
          ┌───┴──┴───┐       ┌───┴──┴───┐       ┌───┴──┴───┐
          │ h1'│ h2' │  ──►  │ h1'│ h2' │  ──►  │ h1'│ h2' │
          └───┬──┬───┘       └───┬──┬───┘       └───┬──┬───┘
              └──┘               └──┘               └──┘
            RNN 沿时间推进（有方向）    GNN 在同一时刻内交换（无方向）
```

两个轴是正交的：**时间是递归的、有方向的；通信是同时的、对称的。**
一层 GNN 里没有"谁先谁后"—— 所有节点都只读彼此**更新前**的值，一次算完。
$k$ 层即在同一时刻内传播 $k$ 跳。

### 图由环境提供

论文明确规定通信图来自环境（欧氏距离 ≤ 通信半径 $C$），CoDreamer 不预测它。
唯一例外是想象阶段：轨迹起点的图被冻结，全程复用。

### 通信插在 RSSM 的哪一处

`RSSM.img_step` 里 GRU **之后**是唯一同时通向 prior 与 posterior 的咽喉点 ——
`obs_step` 把它既当 prior、又拿它拼 embed 算 posterior，而 `get_feat` 又用
`deter` 拼出 decoder / actor / critic 共用的特征。在这一处插一层 GNN，
通信信息自动流到全部下游，无需任何额外接线。

---

## 安装

```bash
conda create -n codreamer python=3.10
conda activate codreamer

# PyPI 上 Windows 默认装的是 CPU 版 torch，必须显式指定 CUDA 索引
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r third_party/dreamerv3-torch/requirements.txt
```

安装后先跑自检，它会在第一屏报告 Python / torch / CUDA / 显存 / 依赖 /
上游文件完整性：

```bash
python scripts/check_env.py
```

---

## 使用

### 冒烟测试

```bash
bash scripts/run_smoke.sh                                        # Linux / macOS / WSL
powershell -ExecutionPolicy Bypass -File scripts\run_smoke.ps1   # Windows
```

### 训练

```bash
cd third_party/dreamerv3-torch

# IDreamer —— agent 之间不通信
python dreamer.py --configs codreamer --comm_variant idreamer --seg_graph_obs True \
    --logdir ~/runs/idreamer

# CoDreamer —— RSSM / 预测头 / actor-critic 各 1 层 GNN（论文 Table 2/3）
python dreamer.py --configs codreamer --comm_variant codreamer --seg_graph_obs True \
    --logdir ~/runs/codreamer
```

**`comm_variant` 是两个变体之间唯一的日常开关。** 它同时驱动三处 GNN，避免
出现"半个 CoDreamer"：只打开 RSSM 而关着头与 actor-critic，得到的既不是论文的
CoDreamer，也不是 IDreamer。

想跑单点消融时，`--gnn_layers` / `--gnn_head_layers` / `--gnn_behavior_layers`
可以逐个覆盖 `comm_variant`（三处不一致时会打印警告）。

### 监控与可视化

```bash
python scripts/monitor.py --log ~/runs/codreamer/main.log
python scripts/plot_metrics.py --logdir ~/runs/codreamer --out figures/curve.png
python scripts/compare_runs.py --run "IDreamer=~/runs/idreamer" \
                               --run "CoDreamer=~/runs/codreamer" --out figures/ab.png
```

---

## 测试

本项目的测试不依赖"跑两遍对比学习曲线"。曲线受随机性影响，而且**结构错误往往
不会让曲线变难看，只会让它安静地学错东西**。因此关键路径使用数值断言：

```bash
cd third_party/dreamerv3-torch
python -m envs.seg              # 环境本体：奖励语义、邻接性质、可复现性
python -m envs.seg_gym          # gym 包装：空间声明、agent 轴、图结构
python -m tools_marl            # 多智能体收集、队伍归组、图采样
python -m networks_gnn          # GATv2：孤立节点、防过平滑、梯度回流

cd ../.. && python scripts/test_graph_pipeline.py   # 数据管线等价性
```

`scripts/test_graph_pipeline.py` 断言的是**数值恒等式**：

> 同一份数据、同一套权重、同一份优化器配置下，
> 图批次路径与折平批次路径的损失必须一致。
> 关闭通信层时这是恒等变换，相对误差应处于 float32 机器精度（~1e-7）量级。

任何对齐错误（切错 agent、轴顺序颠倒、拼接错位）都会以**好几个数量级**的偏差
暴露，而不是 1e-7。

---

## 目录结构

```
CoDreamer/
├── docs/
│   └── reproduction-plan.md        ← 复现台账：假设、偏离、未定项
├── scripts/
│   ├── check_env.py                ← 环境自检
│   ├── test_graph_pipeline.py      ← 数据管线数值恒等测试
│   ├── run_smoke.{sh,ps1}          ← 冒烟测试
│   ├── run_seg.sh / sweep_seg.sh   ← 训练 / 环境配置扫描
│   ├── monitor.py                  ← 训练监控
│   ├── plot_metrics.py             ← 学习曲线
│   ├── compare_runs.py             ← 多 run 对比
│   ├── remote_metrics.py           ← 远端指标拉取
│   └── viz_seg.py                  ← 环境可视化
├── third_party/
│   ├── THIRD_PARTY.md              ← 第三方溯源与修改约定
│   └── dreamerv3-torch/            ← 基座（MIT）
│       ├── networks_gnn.py         ← ★ GATv2 通信层
│       ├── envs/seg.py             ← ★ Sequential Estimate Game
│       ├── envs/seg_gym.py         ← ★ 队伍级 gym 包装
│       ├── tools_marl.py           ← ★ 多智能体收集与图采样
│       └── UPSTREAM_MANIFEST.sha256 ← 上游文件指纹清单
├── logdir/smoke/                   ← 冒烟证据（小体积文本）
├── assets/                         ← 论文 PDF 与笔记（不入库）
└── README.md
```

★ 标记的文件不在上游指纹清单里，因此**"清单不一致 ⟺ 上游被修改过"**
这一不变式成立 —— 任何对基座的改动都能被机械地检出。

---

## 复现说明

论文有几处表述无法互相闭合，实现时必须自行判定。所有推断、偏离与未定项都
显式记录在 [docs/reproduction-plan.md](docs/reproduction-plan.md)，包括：

- 奖励目标公式与动作区间范围不自洽时的取舍
- 论文未给出的 GNN 内部维度与边特征注入方式
- 单机显存受限时对 batch 规模的调整

---

## 第三方与许可

`third_party/dreamerv3-torch/` 为 **MIT** 许可，版权归 NM512 (2023)，
原许可文件保留在该目录下。溯源说明见
[third_party/THIRD_PARTY.md](third_party/THIRD_PARTY.md)。

本仓库其余部分为复现工作。

---

## 引用

```bibtex
@inproceedings{codreamer2024,
  title         = {CoDreamer: Communication-Based Decentralised World Models},
  booktitle     = {CoCoMARL Workshop at RLC},
  year          = {2024},
  eprint        = {2406.13600},
  archivePrefix = {arXiv}
}
```
