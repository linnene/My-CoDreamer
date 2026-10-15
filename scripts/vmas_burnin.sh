#!/usr/bin/env bash
# VMAS 可行性试跑 —— 单 seed、双变体、短预算。
#
# 定位（照 Codex 的措辞）：这不是论文结论，是**可行性试跑**。
# 目的只有两个：
#   1. 确认实现能扩展到正式的多智能体 benchmark；
#   2. 确认可观测空间里的开环预测是否改善（我们在 SEG 上测到的那个效应）。
# 论文自己说 VMAS 上 IDreamer 与 CoDreamer 的**回报**不易区分，所以不期待回报给出结论。
#
# 与论文设定的偏离（必须记录）：
#   vmas_max_steps 200（论文 500）—— 为了让短预算里能有足够多完整局
#   batch_length 16（论文 64）、batch_size 4 队（论文 16）、imag_horizon 10（论文 15）
#   train_ratio 512 保持论文设定 —— 它锁死每 agent 步的更新次数，是"训练强度"本身
# 两个变体用**完全相同**的设定，所以对比是公平的。
export PATH=/home/wyb/anaconda3/envs/codreamer/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/wyb/CoDreamer/third_party/dreamerv3-torch

for VAR in idreamer codreamer; do
  D=/tmp/runs/vmas_burn_$VAR
  rm -rf "$D"
  echo "===== $VAR  $(date +%T) ====="
  python -u dreamer.py --configs codreamer --task vmas_flocking --comm_variant $VAR \
    --vmas_max_steps 200 --steps 15000 --prefill 250 \
    --envs 4 --batch_size 4 --batch_length 16 --imag_horizon 10 \
    --train_ratio 512 --eval_every 3000 --eval_episode_num 4 --log_every 1000 \
    --action_repeat 1 --openloop_horizons "1 4 16" \
    --logdir "$D" --device cuda:0 > "$D.log" 2>&1
  echo "  退出码=$?  墙钟 $(date +%T)"
  echo "  参数量: $(grep -a 'Optimizer model_opt' "$D.log" | sed 's/.*has //;s/ variables.*//')"
  grep -a 'held-out 开环误差' "$D.log" | tail -2 | sed 's/^/  /'
done
echo "===== 完成 $(date +%T) ====="
