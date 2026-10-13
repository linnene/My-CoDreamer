#!/usr/bin/env bash
# 在【实验机】上跑一次 IDreamer（多智能体 DreamerV3）训练。
#
# 用法：
#     bash scripts/run_seg.sh                 # 默认 5000 步
#     STEPS=20000 bash scripts/run_seg.sh
#     LOGDIR=~/runs/seg_v2 bash scripts/run_seg.sh
#
# 远端建议挂 tmux，避免 SSH 断开把训练带走：
#     tmux new -d -s codreamer 'bash scripts/run_seg.sh'
#     tmux attach -t codreamer

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D3T="$REPO/third_party/dreamerv3-torch"

# --- conda 环境（本机上 torch + CUDA 都在这里）
CONDA_ENV="${CONDA_ENV:-/home/wyb/anaconda3/envs/codreamer}"
export PATH="$CONDA_ENV/bin:$PATH"

# --- 规模。默认值刻意比论文小很多：
#     论文 500k 环境步 × 4 种子，我们一次先跑得完一轮为准。
STEPS="${STEPS:-5000}"
PREFILL="${PREFILL:-1000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_EPISODES="${EVAL_EPISODES:-40}"   # 按 agent episode 计：40 = 10 局（N=4）
LOG_EVERY="${LOG_EVERY:-250}"
DEVICE="${DEVICE:-cuda:0}"
LOGDIR="${LOGDIR:-$HOME/runs/seg_base}"
SEED="${SEED:-0}"

mkdir -p "$LOGDIR"
LOG="$LOGDIR/main.log"

cd "$D3T"

echo "=============================================================="
echo " IDreamer @ Sequential Estimate Game"
echo " 仓库    : $REPO"
echo " 日志    : $LOG"
echo " 规模    : steps=$STEPS prefill=$PREFILL eval_every=$EVAL_EVERY"
echo " 设备    : $DEVICE   seed=$SEED"
echo " 开始    : $(date '+%F %T')"
echo "=============================================================="

# batch_size=16 / batch_length=64 直接用 defaults，不覆盖 —— 与论文一致。
python -u dreamer.py \
  --configs codreamer \
  --steps "$STEPS" \
  --prefill "$PREFILL" \
  --eval_every "$EVAL_EVERY" \
  --eval_episode_num "$EVAL_EPISODES" \
  --log_every "$LOG_EVERY" \
  --seed "$SEED" \
  --logdir "$LOGDIR" \
  --device "$DEVICE" \
  2>&1 | tee -a "$LOG"

echo "=============================================================="
echo " 结束    : $(date '+%F %T')"
echo "=============================================================="
