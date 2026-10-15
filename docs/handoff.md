# 交接提示词 —— CoDreamer 复现

> 把下面整段直接交给接手的人 / Claude Code。它假设读者**没有任何上下文**。

---

## 0. 你要做的事，一句话

复现 **CoDreamer**（arXiv:2406.13600，RLC 2024 CoCoMARL workshop，**没有官方代码**）：
一个把 GNN 插进 DreamerV3 的世界模型与 actor-critic、让多智能体通过"通信"共享隐状态的方法。
基线是 IDreamer（同一套代码，关掉通信）。

**当前状态：基础设施与关键 bug 都已处理完，卡在"哪个环境能真正区分两个变体"和算力上。**

---

## 1. 环境与仓库

| 项 | 值 |
|---|---|
| 本地（只写代码，**不跑实验**） | `/Users/a0000/dev/CoDreamer`，Mac，**没装 numpy/torch** |
| 远端（所有实验） | `ssh wyb`，`wyb-ThinkStation-K-C2490`，Ubuntu 22.04，RTX 3060 12GB，15 GiB RAM，i7-12700 |
| 远端仓库 | `/home/wyb/CoDreamer` |
| conda 环境 | `codreamer`（Py3.10.22 + torch 2.4.1+cu121）。**不要动其它 5 个环境** |
| PATH | `export PATH=/home/wyb/anaconda3/envs/codreamer/bin:$PATH` |
| 基座 | `NM512/dreamerv3-torch`，上游 commit `6ef8646d...`，放在 `third_party/dreamerv3-torch/` |
| 无免密 sudo | 装系统包要走 conda，不能 apt |

**唯一的真相来源是 `docs/reproduction-plan.md`（641 行）。**
里面有：论文锚点、每一步的设计决策与理由、上游改动清单、所有复现缺口。
**动手前先读它。** 本文档只是索引 + 踩坑清单。

---

## 2. ⚠️ 运维铁律（每条都是踩出来的）

### 2.1 长命令必须走 tmux + 全 stdio 重定向

```bash
ssh wyb "tmux new -d -s NAME 'bash script.sh > out 2>&1' </dev/null >/dev/null 2>&1; sleep 3; tmux ls"
```

**不加 `</dev/null >/dev/null 2>&1`，ssh 会挂起。**
我也犯过"把一个含 20 次重试的同步脚本放前台跑"的错 —— 直接卡死。

### 2.2 push **和** fetch 都会静默失败，必须比对 hash

wyb 到 GitHub 的 443 时通时断，git 的表现会骗人：

- `git push` 报成功，远端其实还是旧 commit；
- **更隐蔽**：`git fetch` 静默失败时，本地 `origin/main` 是**旧引用**，于是
  `git reset --hard origin/main` 会**成功地**把代码重置回错的 commit，**全程零报错**。

**用 `scripts/sync_remote.sh <sha>`**：fetch → 比对 → reset → **再验一次 HEAD**，不匹配就重试。
远端任务脚本里**不要写任何 git 命令**（它们不知道自己有没有成功）。

推送后也要验：

```bash
LOCAL=$(git rev-parse HEAD)
for i in $(seq 1 10); do
  git push origin main >/dev/null 2>&1
  R=$(git ls-remote origin main | awk '{print $1}')
  [ "$LOCAL" = "$R" ] && break
  sleep 8
done
```

### 2.3 `tmux kill-session` 不杀子进程

python 会变孤儿并**占着显存**（我遇到过一个孤儿占 5.8 GB，导致后续 OOM 且报错信息指向错误的地方）。

```bash
tmux kill-session -t NAME; pkill -f dreamer.py; sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # 确认
```

### 2.4 清出来的临时目录

`/tmp/runs/` 是各种实验的输出。远端可写，根分区剩 ~500G。

---

## 3. 代码地图

### 3.1 我们写的（不在上游 manifest 里）

| 文件 | 作用 |
|---|---|
| `third_party/dreamerv3-torch/envs/seg.py` | Sequential Estimate Game 本体（论文附录 B） |
| `third_party/dreamerv3-torch/envs/seg_gym.py` | SEG 的队伍级 gym 包装 |
| `third_party/dreamerv3-torch/envs/vmas_env.py` | **VMAS 队伍级包装（当前主线）** |
| `third_party/dreamerv3-torch/networks_gnn.py` | GATv2 + GraphAggregator（0 层严格恒等） |
| `third_party/dreamerv3-torch/tools_marl.py` | 多智能体 rollout、episode 存储、图批次、Phase 2 |
| `scripts/test_training_loop.py` | **回路级自检（最重要）** |
| `scripts/test_graph_pipeline.py` | 模块级自检 |
| `scripts/prof_vmas.py` | 模块级耗时剖析 |
| `scripts/eval_world_model.py` | held-out 多步开环误差（事后） |
| `scripts/eval_policy.py` | 部署评测（多 checkpoint、同一批固定局） |
| `scripts/sync_remote.sh` | 验证式同步远端 |
| `scripts/vmas_burnin.sh` | VMAS 可行性试跑 |

### 3.2 改过的上游文件

`dreamer.py` / `models.py` / `networks.py` / `tools.py` / `configs.yaml`。
**所有改动都有 `# [CoDreamer]` 标记**，并登记在 `docs/reproduction-plan.md` 的「改动清单」。
`grep -rn "\[CoDreamer\]" third_party/dreamerv3-torch/` 可以看全部。

---

## 4. 已验证 / 未验证

### ✅ 已验证（有测试或实测证据）

- **回路级**（`python scripts/test_training_loop.py` 全绿）：
  - 预训练确实只更新世界模型、actor 逐位不变（含反证对照组）
  - 冻结世界模型时只训策略
  - **执行路径确实带图**（codreamer 改邻接矩阵 → 动作分布变）；**关掉时逐位不变**（无泄漏）
  - `comm_variant` 单一开关同时驱动三处 GNN
- **VMAS 包装**（`python -m envs.vmas_env` 全绿）：三任务 × 图性质/形状/越界/可重现
- **开环指标**：训练日志里确实产出（`openloop_h*_reward_mae/cont_mae/kl/state_mse`）
- **吞吐**：`scripts/prof_vmas.py` —— 88 ms/更新，训练更新占 97%，**无悬空开销**

### ❌ 未验证 / 已知不成立

- **CoDreamer 是否真的更好** —— 没有任何可信证据
- **VMAS 上能否区分两个变体** —— 论文自己的消融说不能（见 §6）
- Melting Pot 完全没碰

---

## 5. 当前正在跑的东西（接手第一件事：检查它）

```bash
tmux ls                      # 会话名 burn
grep -vE 'Warning|warn' /home/wyb/burn.out
```

**⚠️ 这个 run 有问题，应该重跑：**
它用 `batch_length 16` + `openloop_horizons "1 4 16"` 启动，而约束是
`batch_length >= max(horizon) + 2`。于是主指标被跳过：

```
[CoDreamer] 跳过开环误差：batch_length=16 装不下 horizon=16（需要 >= 18）
```

**它只产出回报** —— 而回报恰恰是论文说分不出变体的那个口径。

修正已经在 `scripts/vmas_burnin.sh`（`BATCH_LENGTH=24`，并把两个值钉在一起防止再漂移）。
**重跑方式：**

```bash
cd /Users/a0000/dev/CoDreamer
git push origin main && bash scripts/sync_remote.sh
ssh wyb "tmux kill-session -t burn 2>/dev/null; pkill -f dreamer.py; sleep 2; \
  tmux new -d -s burn 'bash /home/wyb/vmas_burnin.sh > /home/wyb/burn.out 2>&1' </dev/null >/dev/null 2>&1"
```

（约 40 分钟/变体，共 ~1.5 小时。单 seed、短预算 —— **是可行性试跑，不是结论**。）

---

## 6. 科学上的真实困境（务必理解，否则会做无用功）

### 6.1 论文自己的消融给出了一张判别力排序表

| 环境 | 论文原话 | 判别力 |
|---|---|---|
| **Melting Pot** | *"probability > 0.5 of improving upon IDreamer **with statistical significance**"* | **最强** |
| **Estimate Game** | *"this overlap of information is **highly evident by the construction of the environment**"* | 中（靠环境特殊性） |
| **VMAS** | *"**not highly inter-agent dependent**"*、*"IDreamer and CoDreamer are **similar**... due to **stochasticity**"* | **最弱** |

**而 VMAS 是正文主实验。** 这是复现这件事最尴尬的地方：
必须做 VMAS，但论文自己说 VMAS 上回报分不出这两个变体。

### 6.2 SEG 已判定饱和，但不要叫它"失败"

六配置扫描 + 三 seed 对照，全部收敛到同一回报。**它不适合比策略回报**，
但适合做：数据管线回归测试、GNN 是否生效的单元环境、世界模型开环诊断、架构消融的廉价测试床。

### 6.3 算力

| 步数 | 单次运行 | 6 次（3 seed × 2 变体） |
|---|---|---|
| 5e5（论文） | 24.4 h | **6.1 天** |
| 1e5 | 4.9 h | 1.2 天 |
| 2e4 | 1.0 h | 0.2 天 |

**`train_ratio` 锁死每 agent 步的更新次数**（`train_ratio / (batch_size × batch_length)`）。
缩小 batch 只会让单次更新更便宜、次数按比例变多，**总训练量不变**。
**唯一能真正提速的旋钮是 `train_ratio` 本身**（会偏离论文设定；只要两个变体用同一设定，
对比仍然公平）。

### 6.4 指标的口径（血泪教训）

**不要用 `metrics.jsonl` 里的 `reward_loss` / `state_loss` 判断"世界模型想得准不准"。**
它们是 replay 上的**训练损失**，且用的是**看过当步观测的 posterior**（`feat = get_feat(post)`）
—— 衡量的是"给定当前帧重建当前帧"，不是预测未来。

**实测对照：同一时刻 `reward_loss = 1.0`，而 held-out 开环 `reward_mae = 0.022`，差 45 倍。**

正确的口径是 `models.openloop_error`（训练日志里的 `openloop_h*_*`）：
posterior 吃完序列 → 取某时刻状态 → 之后**只喂真实动作、不再看观测**，让 prior 往前走。

**注意**：`evaldir` 里的局由**当前策略**现跑、从不参与梯度更新，是天然的 held-out
（为此 `simulate_marl` 的 `is_eval` 分支也落盘了）。

**另外**：不要用 latent KL 的**绝对值**跨模型比较 —— 两个模型学的是各自的 latent space。
KL 只适合作为同一模型内部的漂移指标。跨模型要看**解码后的 state MSE** 和 **reward MAE**。

---

## 7. 已经修掉的严重 bug（别再犯）

| # | bug | 教训 |
|---|---|---|
| 1 | **预训练两个开关同时为 False**（`not (pretraining or frozen)`），500 步不更新任何参数，日志照打"预训练完成" | 布尔表达式把两个无关语义混在一起 |
| 2 | **`_policy()` 从不传图** → 收集数据用的是看不见图的 actor，部署评测也是 | 同一个失效模式我在 `_imagine` 修过一次、放过了真实执行的 4 个调用点 |
| 3 | **执行路径缺 agent 轴**：训练是 `(B,T,N,·)`，执行被铺平成 `(n_teams*N,·)`，GNN 把 batch 维当邻居维 | 图相关的形状错了三次，每次都是 GNN 内部报错而不是调用点 |
| 4 | **部署加载校验恒为空**：`[k for k in missing if not k.startswith("_")]`，而本项目所有真参数的键都以 `_` 开头 | 护栏本身是假的 |
| 5 | **收尾评测缺失**：多智能体下 `_step += batch × eval_every`，最后一整段训练没评测 | 静默 |
| 6 | **`dist.mean` 有时是属性、有时是方法**（reward 头 vs cont 头） | 踩了三次 |
| 7 | **预测头 `(...,1)` 与标签 `(...,)` 广播从右边对齐** → 把 agent 轴对齐到了时间轴 | 小 batch 时侥幸不报错，换 batch 才炸 |
| 8 | `Every.__call__` **返回次数不是布尔** | 见 §6.3 |

**元教训**：566 行模块级测试全绿，却漏掉了 #1 和 #2 —— 因为**它们长在集成点上**。
`scripts/test_training_loop.py` 就是为补这一层而生，它**只断言从外面可观测的行为**
（权重变没变、动作分布随不随图变），不检查内部张量形状。**新加功能时请在这里加断言。**

---

## 8. 下一步（按优先级）

1. **重跑 VMAS 试跑**（见 §5 的命令），确认：
   - 两个变体都跑完、开环指标真的产出
   - 出现**稳定学习信号**（`value_mean` 是否单调上升是个好观察点）
2. **只有第 1 步确认后才追加 seed 和预算。** 单 seed 短预算不能得出任何论文结论。
3. 如果 VMAS 有信号 → 加 seed、加步数；如果没信号 → 回到 §6.1 重新选环境。
4. **不要现在转向 Melting Pot**（像素 + 1000~1500 步 episode，比 VMAS 重一个量级，且无 sudo）；
   **也不要现在自建环境**（那更适合作为之后"不可靠通信"研究的受控实验平台，
   不能拿来当复现证据）。

---

## 9. 行为约定

- **不要为了好看而扩大结论。** 这个项目已经因为过度解读曲线翻过一次车
  （"IDreamer 策略塌陷"其实是预训练 bug 的假象）。
- 图表标签**一律英文** —— wyb 没有 CJK 字体，中文渲染成方块。
- commit message 一律英文。
- `assets/` 不入库。
- **改动必须标注 `# [CoDreamer]` 并登记进 `docs/reproduction-plan.md` 的改动清单。**
- **README 只写项目本身，不写我们的实验进度。**
- 回答要简单直接，不随意扩展。

---

## 10. 现在就可以执行的检查

```bash
# 1) 远端状态
ssh wyb "date; tmux ls; nvidia-smi --query-gpu=memory.used --format=csv,noheader"

# 2) 代码是否同步（必须一致）
cd /Users/a0000/dev/CoDreamer && git rev-parse HEAD
ssh wyb "cd /home/wyb/CoDreamer && git rev-parse HEAD"

# 3) 自检是否还绿（约 5 分钟）
ssh wyb "export PATH=/home/wyb/anaconda3/envs/codreamer/bin:\$PATH; cd /home/wyb/CoDreamer \
  && python scripts/test_training_loop.py 2>&1 | tail -20"

# 4) VMAS 环境自检（约 1 分钟）
ssh wyb "export PATH=/home/wyb/anaconda3/envs/codreamer/bin:\$PATH; cd /home/wyb/CoDreamer/third_party/dreamerv3-torch \
  && python -m envs.vmas_env 2>&1 | tail -8"
```
