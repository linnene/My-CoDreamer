#!/usr/bin/env bash
# 实验 3：另外两个 VMAS 任务 —— flocking 是不是特例？
#
# 论文的 VMAS 结论是**三个任务聚合**出来的（Figure 4 的 Median/IQM/Mean 跨
# flocking / discovery / buzz_wire）。我们此前只跑了 flocking，而它恰好是三个里
# **唯一观测维与论文完全一致**的那个（18；discovery 19 vs 21，buzz_wire 6 vs 8）。
#
# 配置与现有 8-seed 那批**逐字相同**（vmas_max_steps 200、T=16、H=10、4 队），
# 这样新任务上的结果与已有 flocking 结果可直接比。
#
# discovery  N=5（论文 Table 8），agent 步数按 N 放大，墙钟约 1.3 h/臂
# buzz_wire  N=2，墙钟约 0.6 h/臂
export PATH=/home/wyb/anaconda3/envs/codreamer/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/wyb/CoDreamer/third_party/dreamerv3-torch

for TASK in vmas_discovery vmas_buzz_wire; do
  for VAR in idreamer codreamer; do
    D=/tmp/runs/e3_${TASK#vmas_}_${VAR}_s1
    rm -rf "$D"
    echo "===== exp3 $TASK $VAR  $(date +%T) ====="
    python -u dreamer.py --configs codreamer --task "$TASK" --comm_variant $VAR \
      --vmas_max_steps 200 --prefill 250 \
      --envs 4 --batch_size 4 --batch_length 16 --imag_horizon 10 \
      --train_ratio 512 --steps 15000 --eval_every 3000 --eval_episode_num 16 \
      --log_every 1000 --action_repeat 1 --seed 1 --openloop_horizons "" \
      --logdir "$D" --device cuda:0 > "$D.log" 2>&1
    echo "  退出码=$?  墙钟 $(date +%T)"
    grep -a "eval_return" "$D.log" | tail -3 | sed 's/^/  /'
  done
done
echo "===== exp3 完成 $(date +%T) ====="
