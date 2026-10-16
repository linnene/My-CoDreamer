#!/usr/bin/env bash
# 补上"另一半协议"：interval checkpoint + best 选择规则。
#
# 为什么只改一个键
# ----------------
# 台账 §8 的结论是两处协议修正**方向相反**：修"取最终点"会把差值从 3.647 压到
# 1.419，修"局不配对"会把它抬到 4.402，而两者不能叠加 —— 因为统一重评用的
# `latest.pt` 是最终权重，手里根本没有中间检查点。
#
# 所以这个实验**唯一要加的东西是 `--ckpt_every`**。其余每一个参数都与
# `seed_rep.sh` 逐字相同（同样 15000 步、4 队、batch_length 16、imag_horizon 10、
# train_ratio 512、单局 200），这样"选择规则"是唯一变动的量。
#
# eval_every 也**故意不动**：训练时的评测口径与 8-seed 那批保持一致，可比。
# best 选择是**离线**在固定局上做的（`eval_unified.py --run-dir`），
# 用的是所有 checkpoint 共用的同一批 32 局 —— 顺带把配对缺陷也一起修了。
#
# 产出：/tmp/runs/iv_<variant>_s1/ckpt_NNNNNNNNN.pt（约 20 个，每个 0.2~0.3 GB）
export PATH=/home/wyb/anaconda3/envs/codreamer/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/wyb/CoDreamer/third_party/dreamerv3-torch

# 等 GPU 空出来（sampled 重评还在跑）
while pgrep -f "eval_unified.py" > /dev/null; do
  echo "  等待 eval_unified 结束… $(date +%T)"
  sleep 30
done

for VAR in idreamer codreamer; do
  D=/tmp/runs/iv_${VAR}_s1
  rm -rf "$D"
  echo "===== $VAR (interval ckpt)  $(date +%T) ====="
  python -u dreamer.py --configs codreamer --task vmas_flocking --comm_variant $VAR \
    --vmas_max_steps 200 --steps 15000 --prefill 250 \
    --envs 4 --batch_size 4 --batch_length 16 --imag_horizon 10 \
    --train_ratio 512 --eval_every 3000 --eval_episode_num 16 --log_every 1000 \
    --action_repeat 1 --seed 1 --openloop_horizons "" \
    --ckpt_every 1200 \
    --logdir "$D" --device cuda:0 > "$D.log" 2>&1
  echo "  退出码=$?  墙钟 $(date +%T)"
  echo "  检查点数量: $(ls "$D"/ckpt_*.pt 2>/dev/null | wc -l)  占用: $(du -sh "$D" | cut -f1)"
done
echo "===== 完成 $(date +%T) ====="
