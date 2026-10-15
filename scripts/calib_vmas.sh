#!/usr/bin/env bash
# VMAS 吞吐标定 —— 在 wyb 上跑。
#
# 为什么要先标定：冒烟实测 fps≈4.7（agent 步/秒），按论文的 5e5 步要 30 小时/次，
# 完全不可行。必须先知道"每秒能走多少个 agent 步"，才能定出可完成的步数预算，
# 并把"我们只跑了论文的百分之几"这件事说清楚。
#
# 用法（在 wyb 上）：bash calib_vmas.sh
export PATH=/home/wyb/anaconda3/envs/codreamer/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/wyb/CoDreamer/third_party/dreamerv3-torch

for CFG in "4 4" "8 8" "16 8"; do
  set -- $CFG
  ENVS=$1
  BS=$2
  echo "===== envs=$ENVS batch_size=$BS  $(date +%T) ====="
  rm -rf /tmp/runs/calib
  timeout 900 python -u dreamer.py --configs codreamer --task vmas_flocking \
    --comm_variant codreamer --vmas_max_steps 30 --steps 1600 --prefill 80 \
    --envs "$ENVS" --batch_size "$BS" --batch_length 64 --eval_every 1000000 \
    --log_every 200 --action_repeat 1 --openloop_horizons "" \
    --logdir /tmp/runs/calib --device cuda:0 > /tmp/runs/calib.log 2>&1
  echo "  退出码=$?"
  grep -ao "fps [0-9.]*" /tmp/runs/calib.log | tail -3 | tr '\n' ' ' | sed 's/^/  fps: /'
  echo
  if grep -qaE "CUDA out of memory|OutOfMemory" /tmp/runs/calib.log; then
    echo "  ⚠️ OOM"
  fi
done
echo "===== 完成 $(date +%T) ====="
