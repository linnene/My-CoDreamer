#!/usr/bin/env bash
#
# Overcooked 接入冒烟 —— 回答一个问题：**这个游戏真的接进去了吗？**
#
# 单步能构造、能 step 只算"接上了线"，不算"接进去了"。真正要验的是训练循环
# 在这套观测上端到端跑得通：
#
#   1. 世界模型 / actor-critic 两个优化器都建起来（参数量打印出来）
#   2. 图批次是真的（`make_graph_dataset` 在 Overcooked 的 obs 上不报错）
#   3. **开环误差指标真的产出** —— 这是最容易静默消失的一个（见下）
#   4. 评测跑得动、有回报
#   5. 没有 OOM
#
# 只跑一个变体：这一步验的是**接入**，不是变体对比（那是另一个对话的活）。
#
# ⚠️ 两个会静默失效的配置关系，挡在启动前：
#   prefill >= 一局长度：replay 只在整局结束时才写盘，低了会报
#     "no complete team in the replay buffer"
#   batch_length >= max(openloop horizon) + 2：低了 dreamer.py 会**整体跳过**
#     开环计算，只打一行"跳过"然后照常跑完 —— 45 分钟跑完、一个开环数字都没有。
#     这个坑本项目踩过一次，所以这里硬挡。
set -uo pipefail

export PATH=/home/wyb/anaconda3/envs/codreamer/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=${REPO:-/home/wyb/CoDreamer_lbf}
D3T="$REPO/third_party/dreamerv3-torch"
LOGDIR=${LOGDIR:-/tmp/runs/oc_smoke}
STEPS=${STEPS:-2000}
EP_LEN=35
PREFILL=40
BATCH_LENGTH=20
HORIZONS="1 4 16"

MAX_H=$(echo "$HORIZONS" | tr ' ' '\n' | sort -n | tail -1)
if [ $((MAX_H + 2)) -gt "$BATCH_LENGTH" ]; then
  echo "配置不自洽：horizon=$MAX_H 需要 batch_length >= $((MAX_H + 2))，当前 $BATCH_LENGTH" >&2
  exit 1
fi
if [ "$PREFILL" -lt "$EP_LEN" ]; then
  echo "配置不自洽：prefill=$PREFILL 必须 >= 一局长度 $EP_LEN" >&2
  exit 1
fi

echo "=============================================================="
echo " Overcooked 接入冒烟   steps=$STEPS  $(date +%T)"
echo "=============================================================="
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
cd "$D3T"

python -u dreamer.py --configs codreamer --task overcooked_demo \
  --comm_variant codreamer \
  --overcooked_max_steps "$EP_LEN" --overcooked_layout three_player \
  --overcooked_cook_time 5 --overcooked_recipe onion --overcooked_sight 2 \
  --steps "$STEPS" --prefill "$PREFILL" \
  --envs 2 --batch_size 2 --batch_length "$BATCH_LENGTH" --imag_horizon 10 \
  --train_ratio 512 --eval_every 1000 --eval_episode_num 4 --log_every 500 \
  --action_repeat 1 --openloop_horizons "$HORIZONS" \
  --logdir "$LOGDIR" --device cuda:0 > "$LOGDIR/run.log" 2>&1
RC=$?
echo "训练退出码=$RC  墙钟 $(date +%T)"
echo

echo "=============== 接入自检：关键日志行是否真的产出 ==============="
FAIL=0
chk() {  # chk <说明> <grep 模式>
  if grep -qaE "$2" "$LOGDIR/run.log"; then
    echo "  [ok  ] $1"
    grep -aE "$2" "$LOGDIR/run.log" | tail -1 | cut -c1-150 | sed 's/^/         /'
  else
    echo "  [FAIL] $1"
    FAIL=$((FAIL+1))
  fi
}

chk "环境被识别为多智能体（3 agent）"      "Multi-agent: num_agents=3"
chk "世界模型优化器建起来了"                "Optimizer model_opt has"
chk "actor / value 优化器建起来了"          "Optimizer (actor|value)_opt has"
chk "GitHub 无关：git_sha 被记录（None 也算）" "git_sha="
chk "**开环误差真的产出**（最容易静默消失）"  "openloop_h"
chk "评测跑出回报"                          "eval_return"
chk "训练回报在动"                          "train_return"

# OOM 是"不该出现"而不是"应该出现"，所以单独查，不走 chk
if grep -qa "out of memory\|CUDA out of memory" "$LOGDIR/run.log"; then
  echo "  [FAIL] 出现 OOM"; FAIL=$((FAIL+1))
else
  echo "  [ok  ] 没有 OOM"
fi

if grep -qa "没有完整队伍\|no complete team" "$LOGDIR/run.log"; then
  echo "  [FAIL] replay 里攒不出完整队伍（prefill 太小）"; FAIL=$((FAIL+1))
else
  echo "  [ok  ] replay 攒得出完整队伍"
fi

echo
if [ "$FAIL" -gt 0 ]; then
  echo "接入自检 FAILED（$FAIL 项）—— 日志：$LOGDIR/run.log"
  exit 1
fi
if [ "$RC" -ne 0 ]; then
  echo "接入自检通过，但训练退出码=$RC，看一眼日志"
  exit 1
fi
echo "接入自检全部通过。日志：$LOGDIR/run.log"
