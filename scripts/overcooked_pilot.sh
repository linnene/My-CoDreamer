#!/usr/bin/env bash
#
# ⚠️⚠️  这个脚本的产物**只能回答一个问题：有没有学习信号**  ⚠️⚠️
#
# 它**不能**用来读"IDreamer 与 CoDreamer 哪个更好"，两个理由：
#
#   1. seed 数远不够。台账里的方差分解是 σ_seed ≈ 2.4、σ_eval ≈ 0.5；
#      n=2 时每臂 SE ≈ 1.7，**只能分辨约 5 的差异**。两臂接近**不构成"无差异"**。
#   2. 预算远低于论文（见下），而本项目唯一统计显著的结论本身也是"3% 预算下的性质"。
#
# 这个限制写在这里、写在产物路径里、也写在台账里 —— 三处，因为三个月后
# 最可能误引它的就是我自己。
#
# 目的（只有这一个）：确认两个变体都能跑起来、而且**回报确实在动**。
# 若没有信号，就不该加 seed 和步数 —— 那是本项目已经付过代价的教训。
#
# 与论文设定的偏离（必须记录）：
#   overcooked_max_steps 35      —— 实测：脚本策略跑完一条完整交付链要 ~25 步，
#                                   9x6 厨房里锅到上菜口 5 格，25 步会在差两三步处截断
#   cook_time 5、配方 1 个洋葱   —— 默认 20 tick + 3 洋葱在短局长里一锅都出不了
#   batch_length 20、batch_size 4 队、imag_horizon 10（论文 64 / 16 / 15）
#   train_ratio 512 保持论文设定 —— 它锁死每 agent 步的更新次数，是"训练强度"本身
# 两个变体用**完全相同**的设定，所以对比在"同预算"这个意义上是公平的。
#
# ⚠️ prefill 必须 >= 一局长度（35）：replay 只在整局结束时才写盘，低于它
#    make_graph_dataset 会直接报 "no complete team in the replay buffer"。
#    这里取 40。
set -u

export PATH=/home/wyb/anaconda3/envs/codreamer/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/wyb/CoDreamer_lbf/third_party/dreamerv3-torch

# ---- 预算：由 scripts/prof_vmas.py 的实测吞吐反推后填入 ----
# 先跑吞吐测量，把 ms/更新 换成"这个预算要多少墙钟"，再回来定。
STEPS=${STEPS:-10000}
SEEDS=${SEEDS:-"0 1"}
RUN_ROOT=${RUN_ROOT:-/tmp/runs/oc_pilot}

BATCH_LENGTH=20
HORIZONS="1 4 16"
EP_LEN=35          # = --overcooked_max_steps
PREFILL=40         # 必须 >= EP_LEN

MAX_H=$(echo "$HORIZONS" | tr ' ' '\n' | sort -n | tail -1)
if [ $((MAX_H + 2)) -gt "$BATCH_LENGTH" ]; then
  echo "配置不自洽：horizon=$MAX_H 需要 batch_length >= $((MAX_H + 2))，当前 $BATCH_LENGTH" >&2
  exit 1
fi
# prefill < 一局长度时 replay 里攒不出完整队伍，make_graph_dataset 会直接报错。
# 这里挡在启动前，因为那两个数分散在不同行、容易改一个忘另一个。
if [ "$PREFILL" -lt "$EP_LEN" ]; then
  echo "配置不自洽：prefill=$PREFILL 必须 >= 一局长度 $EP_LEN" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
SHA=$(git -C /home/wyb/CoDreamer_lbf rev-parse HEAD 2>/dev/null || echo "unknown")
# 这个副本没有 .git（rsync 覆盖式同步），所以 SHA 多半取不到 —— 那是**已知且已标注**的：
# 每个 run 的 log 里 dreamer.py 自己会打印 [CoDreamer] git_sha=...
cat > "$RUN_ROOT/README.txt" <<EOF
本目录是 Overcooked 上的**信号检查试跑**，不是变体对照。
产物**不可**用于判断 IDreamer 与 CoDreamer 孰优：seed 数与预算都远不够
（σ_seed ≈ 2.4，n=2 时每臂 SE ≈ 1.7，只能分辨约 5 的差异）。
唯一可读的是：两个变体是否都跑起来了、回报是否在动。
steps=$STEPS  seeds="$SEEDS"  batch_length=$BATCH_LENGTH  horizons="$HORIZONS"
外层取到的 git SHA: ${SHA}（rsync 副本无 .git，以各 run log 里的 git_sha 为准）
启动时间: $(date -Iseconds)
EOF

echo "===== Overcooked pilot  steps=$STEPS  seeds=$SEEDS  $(date +%T) ====="
for SEED in $SEEDS; do
  for VAR in idreamer codreamer; do
    D="$RUN_ROOT/${VAR}_s${SEED}"
    rm -rf "$D"
    echo "----- $VAR seed=$SEED  $(date +%T) -----"
    python -u dreamer.py --configs codreamer --task overcooked_demo \
      --comm_variant "$VAR" \
      --overcooked_max_steps 35 --overcooked_layout three_player \
      --overcooked_cook_time 5 --overcooked_recipe onion --overcooked_sight 2 \
      --steps "$STEPS" --prefill "$PREFILL" \
      --envs 4 --batch_size 4 --batch_length "$BATCH_LENGTH" --imag_horizon 10 \
      --train_ratio 512 --eval_every 2000 --eval_episode_num 4 --log_every 500 \
      --action_repeat 1 --openloop_horizons "$HORIZONS" --seed "$SEED" \
      --logdir "$D" --device cuda:0 > "$D.log" 2>&1
    echo "  退出码=$?  墙钟 $(date +%T)"
    grep -a 'git_sha=' "$D.log" | head -1 | sed 's/^/  /'
    echo "  参数量: $(grep -a 'Optimizer model_opt' "$D.log" | sed 's/.*has //;s/ variables.*//')"
    grep -a 'held-out 开环误差' "$D.log" | tail -1 | sed 's/^/  /'
    # 回报必须真的在动，否则"有没有信号"这个问题没有答案 —— 直接打出来
    grep -a 'train_return' "$D.log" | awk 'NR%4==1' | tail -4 | sed 's/^/  /'
  done
done
echo "===== 完成 $(date +%T) ====="
echo "⚠️ 产物只回答『有没有学习信号』，不可用于变体对照。见 $RUN_ROOT/README.txt"
