#!/usr/bin/env bash
# 两级通信的四臂消融 —— 论文 C.1（Estimate Game）的核心结果。
#
# ============================================================================
# 为什么需要它
# ============================================================================
# 我们此前只跑过 comm_variant 的两个端点（三处全 0 / 三处全 1）。而论文的结论是
# **两级分解**，且两级的作用随环境反转：
#
#   Estimate Game  WM Comm **显著优于** IDreamer；IDreamer 优于 AC Comm
#   VMAS           WM Comm **显著变差**；CoDreamer 与 IDreamer 在置信区间内相近
#
# 两个端点看不出这件事 —— 它们只是这个四点实验的一个投影。
#
# ============================================================================
# 臂的定义
# ============================================================================
# 三个键逐个显式给出，不依赖 comm_variant 的展开（comm_variant 只在三个键为 -1
# 时生效；这里全部显式，它退化为一个标签）。三处不一致时 dreamer.py 会打印一行
# 警告 —— **wm_comm / ac_comm 两臂出现那行警告是预期的**，它不是错误。
#
#   idreamer   0 0 0
#   wm_comm    1 1 0   ← 只在世界模型里通信（论文的 WM Comm）
#   ac_comm    0 0 1   ← 只在 actor-critic 里通信（论文的 AC Comm）
#   codreamer  1 1 1   ← 两级都通信（论文的 CoDreamer）
#
# ============================================================================
# 跑完会机械核对参数量恒等式
# ============================================================================
# 打开哪一处的 GNN，就只能改变那一处的参数量：
#
#   世界模型：idreamer == ac_comm      且  wm_comm == codreamer
#   actor/critic：idreamer == wm_comm  且  ac_comm == codreamer
#
# 不成立的话这四个臂就不是干净的对照，结论全部作废 —— 所以它必须是**机械的**，
# 而不是靠读日志确认。
#
# 用法（在 wyb 上）：bash scripts/ablation_comm.sh
set -uo pipefail

export PATH=/home/wyb/anaconda3/envs/codreamer/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/wyb/CoDreamer/third_party/dreamerv3-torch

# 臂名|comm_variant 标签|rssm|head|behavior
# comm_variant 只影响日志可读性（三处均显式给出，展开不会生效）：
# 让它跟随多数，grep 出来的一行才读得通。
ARMS=(
  "idreamer|idreamer|0|0|0"
  "wm_comm|codreamer|1|1|0"
  "ac_comm|codreamer|0|0|1"
  "codreamer|codreamer|1|1|1"
)

# 与 /home/wyb/base.sh 的 SEG 基线**完全一致**，否则新旧结果不可比。
STEPS=2500
PREFILL=100
EVAL_EVERY=150
EVAL_EPISODES=40
LOG_EVERY=150
BATCH_SIZE=4
SEEDS="0 1 2"

echo "=============================================================="
echo " 四臂通信消融 @ Sequential Estimate Game"
echo " 臂   : idreamer / wm_comm / ac_comm / codreamer"
echo " seed : $SEEDS   步数: $STEPS   batch_size: $BATCH_SIZE（队伍数）"
echo " 开始 : $(date '+%F %T')"
echo "=============================================================="

declare -A NWORLD NACTOR NVALUE   # 每个臂取 seed 0 的参数量，其余 seed 与之比对

for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm cvar R H B <<<"$spec"
  for seed in $SEEDS; do
    D=/tmp/runs/abl_${arm}_s${seed}
    rm -rf "$D"
    echo "----- $arm seed=$seed  gnn=($R,$H,$B)  $(date +%T) -----"
    python -u dreamer.py --configs codreamer --comm_variant "$cvar" \
      --gnn_layers "$R" --gnn_head_layers "$H" --gnn_behavior_layers "$B" \
      --seg_graph_obs True --seed "$seed" \
      --steps "$STEPS" --prefill "$PREFILL" --eval_every "$EVAL_EVERY" \
      --eval_episode_num "$EVAL_EPISODES" --log_every "$LOG_EVERY" \
      --batch_size "$BATCH_SIZE" --logdir "$D" --device cuda:0 > "$D.log" 2>&1
    rc=$?
    errs=$(grep -acE 'Traceback|CUDA out of memory' "$D.log")
    echo "  退出码=$rc  异常行=$errs"

    # 展开后的三处层数必须与本次指定的完全一致 —— 防的是"覆盖没生效"
    got=$(grep -aoE 'gnn: rssm=[0-9]+ head=[0-9]+ behavior=[0-9]+' "$D.log" | head -1)
    [ "$got" = "gnn: rssm=$R head=$H behavior=$B" ] \
      || echo "  ⚠️ 展开不符：期望 rssm=$R head=$H behavior=$B，实际 $got"

    m=$(grep -aoE 'Optimizer model_opt has [0-9]+' "$D.log" | head -1 | grep -oE '[0-9]+$')
    a=$(grep -aoE 'Optimizer actor_opt has [0-9]+' "$D.log" | head -1 | grep -oE '[0-9]+$')
    v=$(grep -aoE 'Optimizer value_opt has [0-9]+' "$D.log" | head -1 | grep -oE '[0-9]+$')
    echo "  params: model=${m:-?} actor=${a:-?} value=${v:-?}"

    if [ "$seed" = "0" ]; then
      NWORLD[$arm]=${m:-0}; NACTOR[$arm]=${a:-0}; NVALUE[$arm]=${v:-0}
    else
      [ "${m:-0}" = "${NWORLD[$arm]}" ] || echo "  ⚠️ model_opt 跨 seed 不一致"
      [ "${a:-0}" = "${NACTOR[$arm]}" ] || echo "  ⚠️ actor_opt 跨 seed 不一致"
      [ "${v:-0}" = "${NVALUE[$arm]}" ] || echo "  ⚠️ value_opt 跨 seed 不一致"
    fi
  done
done

echo
echo "=============================================================="
echo " 参数量恒等式（打开哪一处 GNN，只改那一处）"
echo "=============================================================="
printf '  %-11s %-10s %12s %12s %12s\n' 臂 gnn world actor value
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm _ R H B <<<"$spec"
  printf '  %-11s (%s,%s,%s)%7s %12s %12s %12s\n' \
    "$arm" "$R" "$H" "$B" "" "${NWORLD[$arm]}" "${NACTOR[$arm]}" "${NVALUE[$arm]}"
done

fail=0
expect_eq() {  # 名字 左 右
  if [ "$2" = "$3" ]; then
    echo "  [ok  ] $1"
  else
    echo "  [FAIL] $1  ($2 vs $3)"; fail=1
  fi
}
expect_eq "世界模型: idreamer == ac_comm"    "${NWORLD[idreamer]}" "${NWORLD[ac_comm]}"
expect_eq "世界模型: wm_comm  == codreamer"  "${NWORLD[wm_comm]}"  "${NWORLD[codreamer]}"
expect_eq "actor   : idreamer == wm_comm"    "${NACTOR[idreamer]}" "${NACTOR[wm_comm]}"
expect_eq "actor   : ac_comm  == codreamer"  "${NACTOR[ac_comm]}"  "${NACTOR[codreamer]}"
expect_eq "value   : idreamer == wm_comm"    "${NVALUE[idreamer]}" "${NVALUE[wm_comm]}"
expect_eq "value   : ac_comm  == codreamer"  "${NVALUE[ac_comm]}"  "${NVALUE[codreamer]}"

# 存在性对照：少了这条，上面六条会因为"处处相等"而假通过（比如 GNN 根本没接上）
if [ "${NWORLD[codreamer]}" -gt "${NWORLD[idreamer]}" ] \
   && [ "${NACTOR[codreamer]}" -gt "${NACTOR[idreamer]}" ]; then
  echo "  [ok  ] 对照：打开 GNN 确实增加了参数量"
else
  echo "  [FAIL] 对照：GNN 好像没有接上（参数量没变）"; fail=1
fi

echo
echo "=============================================================="
echo " 结果"
echo "=============================================================="
python3 - <<'PY'
import json, os

arms = ["idreamer", "wm_comm", "ac_comm", "codreamer"]
rows = {}
for arm in arms:
    vals = []
    for s in (0, 1, 2):
        p = f"/tmp/runs/abl_{arm}_s{s}/metrics.jsonl"
        if not os.path.exists(p):
            vals.append(None); continue
        r = [json.loads(l) for l in open(p) if l.strip()]
        ev = [x["eval_return"] for x in r if "eval_return" in x]
        vals.append(round(ev[-1], 4) if ev else None)
    rows[arm] = vals

print(f"  {'arm':<11} {'seed0':>9} {'seed1':>9} {'seed2':>9} {'mean':>9}")
for arm in arms:
    v = rows[arm]
    ok = [x for x in v if x is not None]
    mean = f"{sum(ok)/len(ok):.4f}" if ok else "-"
    cells = " ".join(f"{x:>9}" if x is not None else f"{'(缺失)':>9}" for x in v)
    print(f"  {arm:<11} {cells} {mean:>9}")

print()
print("  论文在 Estimate Game 上的三条**有方向**的预测：")
print("    1) wm_comm   > idreamer   （WM 通信显著更好）")
print("    2) idreamer  > ac_comm    （只给 AC 通信反而更差）")
print("    3) codreamer 最好，且 idreamer / ac_comm 的方差明显更大")
print("  三条同时成立才算复现。单看均值不够 —— 论文的原始结论建立在置信区间上。")
PY

echo
echo " 图（每臂 3 条曲线叠加）:"
echo "   cd /home/wyb/CoDreamer && python scripts/compare_runs.py \\"
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm _ _ _ _ <<<"$spec"
  echo "     --run '$arm=/tmp/runs/abl_${arm}_s0' \\"
done
echo "     --out /tmp/runs/abl_comm.png"

echo
echo "=============================================================="
[ "$fail" = "0" ] && echo " 参数量恒等式全部成立" || echo " ⚠️ 参数量恒等式**有失败** —— 结论作废"
echo " 结束 : $(date '+%F %T')"
echo "=============================================================="
exit "$fail"
