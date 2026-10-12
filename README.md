# CoDreamer 复现

复现 **CoDreamer: Communication-Based Decentralised World Models**
（arXiv:[2406.13600v1](https://arxiv.org/abs/2406.13600)，CoCoMARL workshop @ RLC 2024）。
该论文**没有官方代码仓库**。

在此之后，计划复现 **DMAWM**（*Learning Disentangled Multi-Agent World Model
for Decentralized Control*，ICML 2026，南京大学）。

> 完整的工作台账（目标、结构性决策、待定项、改动登记）见
> [`docs/reproduction-plan.md`](docs/reproduction-plan.md)。

## 当前状态

**步骤 0** —— 让远端跑通原版 `dreamerv3-torch`，隔离验证工具链。
尚未对上游做任何修改。

## 仓库结构

```
CoDreamer/
├── docs/
│   └── reproduction-plan.md      ← 工作台账（先读这个）
├── scripts/
│   ├── check_env.py              ← 环境自检（跑任何东西之前先跑它）
│   ├── run_smoke.sh              ← 步骤 0 冒烟测试（bash）
│   └── run_smoke.ps1             ← 步骤 0 冒烟测试（Windows PowerShell）
├── third_party/
│   ├── THIRD_PARTY.md            ← 第三方代码溯源与修改约定
│   └── dreamerv3-torch/          ← PyTorch 版 DreamerV3（MIT），复现基座
├── assets/                       ← 本地留存，**不入库**（论文 PDF、头脑风暴笔记）
└── README.md
```

## 为什么用 PyTorch 而不是官方 JAX 实现

实验机是 Windows + RTX 4060。**JAX 在 Windows 上没有官方 GPU 支持**（需绕道 WSL），
因此基座选 PyTorch。

选用 [`NM512/dreamerv3-torch`](https://github.com/NM512/dreamerv3-torch) 的关键原因：
**它的默认超参与 CoDreamer 论文 Table 2 / Table 5 逐条吻合**，唯一差异是
`pretrain`（论文 500，上游 100）。也就是说复现**不需要猜任何超参**。
对照表见 `docs/reproduction-plan.md`。

## 工作流

**本地（Mac）只写代码，不跑实验。** 所有执行都在远端：

```
本地写代码 → push → 远端 agent 拉取并运行 → 回传日志
```

因此代码必须自带可验证性：`scripts/check_env.py` 负责在第一屏报告
Python / torch / CUDA / 显存 / 依赖 / 上游完整性，避免"训到一半才发现链路不通"。

## 步骤 0：冒烟测试

**先跑自检：**

```bash
python scripts/check_env.py
```

**再跑冒烟测试**（约几分钟，只验链路，不验效果）：

```bash
# Linux / macOS / WSL
bash scripts/run_smoke.sh

# Windows PowerShell
powershell -ExecutionPolicy Bypass -File scripts\run_smoke.ps1
```

用的是 `dmc_proprio` 配置 —— 向量观测 + MLP 编解码器，**不需要 CNN**，
结构上与后续要接入的 VMAS 最接近。所有参数覆盖都走命令行，
**上游 `configs.yaml` 保持原样**。

日志输出到 `logdir/smoke/`（已 gitignore），跑完请回传整个目录。

## 已确认的行事约定

| 约定 | 说明 |
|---|---|
| 本地不跑实验 | 见 `docs/reproduction-plan.md` |
| 不改上游文件除非必要 | 若必须改，用 `# [CoDreamer]` 标注并在台账登记 |
| 上游文件有 sha256 清单 | `third_party/dreamerv3-torch/UPSTREAM_MANIFEST.sha256`，用于识别我们改了什么 |
| `assets/` 不入库 | 论文 PDF、头脑风暴笔记、锚点记录都留本地 |

## 许可

`third_party/dreamerv3-torch/` 为 MIT 许可，版权归 NM512 (2023)，
原许可文件保留在该目录下。本仓库其余部分为复现工作。
