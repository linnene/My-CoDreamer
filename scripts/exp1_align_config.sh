#!/usr/bin/env bash
# 实验 1：**对齐论文配置，预算不动** —— 把"配置偏离"和"预算偏离"分开。
#
# 为什么是这个实验
# ----------------
# 现有 7×2 个 run 的三点轨迹显示差值随训练**变大**（−0.69 → +3.48 → +4.07），
# 所以"预算不够"这个假设反而被削弱。剩下的头号嫌疑是配置偏离：
#
#     单局长度   200 → 500       （论文 Table 8）
#     imag_horizon 10 → 15       （论文 D.3）
#     batch_length 16 → 64       （论文 D.3）
#     batch_size   4 队 → 16 队  （论文 B=16）
#     envs         4 → 8         （论文 "8 independent workers"）
#
# 这些直接改变**有效折扣视界**：imag_horizon 10 相对单局 200 太短，
# `cont` 头学不到"会终止"，于是 cont≡1、有效视界 ≈ 1/(1−γ) ≈ 333 步，
# 把每步 15% 的模型高估放大成 10 倍的策略差距（见台账的因果链）。
# **对齐后这条放大链会变，所以这是最便宜的判据。**
#
# 预算刻意**不提高**：两臂都只跑到约 25k agent 步，与现有 run 同量级，
# 这样"配置"是唯一变动的量。
#
# 检查点：eval_every 1250 × num_agents 4 = 每次迭代 5000 agent 步，
# ckpt_every 5000 → **每次迭代正好一个检查点**（这个耦合见 dreamer.py 的启动告警）。
export PATH=/home/wyb/anaconda3/envs/codreamer/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/wyb/CoDreamer/third_party/dreamerv3-torch

# 等 GPU 空出来（另一个会话的 Overcooked 训练）。我自己还没起进程，所以这个判据安全。
while pgrep -f "dreamer.py" > /dev/null; do
  echo "  等待 GPU… $(date +%T)"
  sleep 60
done

for VAR in idreamer codreamer; do
  D=/tmp/runs/e1_align_${VAR}_s1
  rm -rf "$D"
  echo "===== exp1 $VAR (论文配置)  $(date +%T) ====="
  python -u dreamer.py --configs codreamer --task vmas_flocking --comm_variant $VAR \
    --vmas_max_steps 500 --vmas_graph_obs True --vmas_add_agent_index True \
    --prefill 500 --envs 8 --batch_size 16 --batch_length 64 --imag_horizon 15 \
    --train_ratio 512 --steps 20000 --eval_every 1250 --eval_episode_num 4 \
    --ckpt_every 5000 --log_every 1250 --action_repeat 1 --seed 1 \
    --openloop_horizons "" \
    --logdir "$D" --device cuda:0 > "$D.log" 2>&1
  echo "  退出码=$?  墙钟 $(date +%T)"
  echo "  检查点: $(ls "$D"/ckpt_*.pt 2>/dev/null | wc -l) 个  占用: $(du -sh "$D" 2>/dev/null | cut -f1)"
  grep -a "eval_return" "$D.log" | tail -4 | sed 's/^/  /'
done
echo "===== exp1 完成 $(date +%T) ====="
