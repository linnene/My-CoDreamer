#!/usr/bin/env bash
# 横向对比实验：找出"通信真正可能有用"的环境配置。
#
# 背景：论文 Table 1 的默认配置下，IDreamer（不通信）已经逼近上界
# （我们测到归一化 0.971，而论文 CoDreamer 也只有 0.97）。既然不通信就
# 接近满分，再加 GNN 也无处可涨。所以先扫一遍环境配置，找出
# 「IDreamer 离上界还有明显距离」的区间 —— 那才是步骤 2 该用的擂台。
#
# 用法：
#     bash scripts/sweep_seg.sh                 # 默认 1 个种子，全部 6 个配置
#     SEEDS=3 bash scripts/sweep_seg.sh
#     ONLY=A_baseline,D_actions8 bash scripts/sweep_seg.sh
#
# 结果落在 $ROOT/<名字>_s<种子>/。

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D3T="$REPO/third_party/dreamerv3-torch"
export PATH="${CONDA_ENV:-/home/wyb/anaconda3/envs/codreamer}/bin:$PATH"

STEPS="${STEPS:-5000}"
SEEDS="${SEEDS:-1}"
ROOT="${ROOT:-$HOME/runs/sweep}"
ONLY="${ONLY:-}"
TEAM_EPISODES="${TEAM_EPISODES:-10}"   # 每次评测跑几个【团队】局

# 配置表： 名字 | agent 数 N | 额外命令行参数
EXPERIMENTS=(
  "A_baseline|4|"
  "B_oracle|4|--seg_obs_mode oracle"
  "C_stretch|4|--seg_target_transform stretch"
  "D_actions8|4|--seg_num_actions 8"
  "E_eta01|4|--seg_local_state_ratio 0.1"
  "F_agents8|8|--seg_num_agents 8"
)

selected() {
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

mkdir -p "$ROOT"
cd "$D3T"

echo "=============================================================="
echo " 横向对比实验"
echo " 根目录 : $ROOT"
echo " 规模   : steps=$STEPS  seeds=$SEEDS"
echo " 开始   : $(date '+%F %T')"
echo "=============================================================="

for entry in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name n overrides <<< "$entry"
  selected "$name" || continue
  eval_eps=$((TEAM_EPISODES * n))

  for seed in $(seq 0 $((SEEDS - 1))); do
    logdir="$ROOT/${name}_s${seed}"
    mkdir -p "$logdir"

    echo
    echo "── $name  seed=$seed  N=$n  eval_episodes=$eval_eps  ${overrides:-（无额外参数）}"

    START=$(date +%s)
    # shellcheck disable=SC2086
    python -u dreamer.py \
      --configs codreamer \
      --steps "$STEPS" \
      --prefill 1000 \
      --eval_every 500 \
      --eval_episode_num "$eval_eps" \
      --log_every 250 \
      --seed "$seed" \
      --logdir "$logdir" \
      --device "${DEVICE:-cuda:0}" \
      $overrides \
      > "$logdir/main.log" 2>&1
    RC=$?
    ELAPSED=$(( $(date +%s) - START ))

    if [ $RC -ne 0 ]; then
      echo "   ❌ 失败 rc=$RC  日志尾部："
      tail -12 "$logdir/main.log" | sed 's/^/      /'
    else
      python - "$logdir/metrics.jsonl" "$ELAPSED" <<'PY'
import json, pathlib, sys
rows = [json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.strip()]
ev = [r["eval_return"] for r in rows if "eval_return" in r]
tr = [r["train_return"] for r in rows if "train_return" in r]
s = f"   ✅ {sys.argv[2]}s   "
s += f"eval末值 {ev[-1]:+.3f}" if ev else "eval 无数据"
if tr:
    s += f"   最后50局均 {sum(tr[-50:])/len(tr[-50:]):+.3f}"
print(s + f"   (eval {len(ev)} 次, 局数 {len(tr)})")
PY
    fi
  done
done

echo
echo "=============================================================="
echo " 结束   : $(date '+%F %T')"
echo
echo " 汇总命令："
printf '   cd %s && python %s/scripts/compare_runs.py \\\n' "$D3T" "$REPO"
for entry in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name n overrides <<< "$entry"
  selected "$name" || continue
  printf '     --run "%s=%s/%s_s0" \\\n' "$name" "$ROOT" "$name"
done
printf '     --out %s/figures/sweep.png\n' "$REPO"
echo "=============================================================="
