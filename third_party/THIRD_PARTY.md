# 第三方代码

## dreamerv3-torch

| 项 | 值 |
|---|---|
| 上游仓库 | https://github.com/NM512/dreamerv3-torch |
| 上游 commit | `6ef8646d807cd10ce0c88e10a7e943211e7fc44c` |
| commit 日期 | 2026-03-08 |
| 许可 | MIT（见 `dreamerv3-torch/LICENSE`，版权归 NM512, 2023） |
| 落地路径 | `third_party/dreamerv3-torch/` |
| 落地方式 | 原样拷贝，剥离 `.git/`、`imgs/`、`__pycache__/` |

### 为什么用它

CoDreamer 的基座是 DreamerV3（官方实现为 JAX）。JAX 在 Windows 上没有官方 GPU 支持，
而本项目的实验机是 Windows + RTX 4060，因此改用 PyTorch 实现作为基座。

### 为什么是这一份

`dreamerv3-torch` 的超参默认值与 CoDreamer 论文 Table 2 / Table 5 **逐条吻合**
（唯一差异：论文的 `pretrain` 为 500，上游默认为 100）。
详见 `docs/reproduction-plan.md` 的对照表。

这意味着复现时**不需要猜任何超参**，只需覆盖 `pretrain=500`。

### 完整性校验

`dreamerv3-torch/UPSTREAM_MANIFEST.sha256` 记录了落地时全部 24 个文件的 sha256。

用途：后续我们会在 `networks.py` / `models.py` 等处做修改。校验命令**预期会在被我们改过的文件上报 FAIL** —— 这正是它的作用，它把「哪些文件偏离了上游」变成一个可枚举的事实，而不是靠回忆。

校验方式（在 `third_party/dreamerv3-torch/` 下）：

```bash
shasum -a 256 -c UPSTREAM_MANIFEST.sha256
```

输出里 `OK` 的行 = 与上游一致；`FAILED` 的行 = 被我们改过。据此可以：

1. 生成干净的 diff（把那几个文件与上游逐行对比）
2. 上游更新时评估合并成本

### 修改约定

对 `third_party/dreamerv3-torch/` 内文件的任何改动，必须：

1. 以注释标注，格式：`# [CoDreamer] <说明>`
2. 在多行改动块的首尾都标注
3. 在 `docs/reproduction-plan.md` 的「改动清单」一节登记

目的是让"我们改了什么"这件事永远可查，而不是淹没在 4000 行上游代码里。
