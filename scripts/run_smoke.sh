#!/usr/bin/env bash
# 步骤 0 冒烟测试 —— 原版 dreamerv3-torch，不改任何上游文件。
#
# 目的：只验证工具链（Python / torch+CUDA / 依赖 / 显存 / 训练循环能转起来）。
# 不验证效果，也不验证 CoDreamer。
#
# 用法：
#     bash scripts/run_smoke.sh                # 默认 cuda:0
#     bash scripts/run_smoke.sh cpu            # 无 GPU 时
#     DEVICE=mps bash scripts/run_smoke.sh     # macOS
#
# 日志落在 logdir/smoke/ 下（已 gitignore），跑完把整个目录回传。

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D3T="$REPO/third_party/dreamerv3-torch"
DEVICE="${1:-${DEVICE:-cuda:0}}"
LOGDIR="$REPO/logdir/smoke"

echo "=============================================================="
echo " 步骤 0 冒烟测试：原版 dreamerv3-torch"
echo " 仓库   : $REPO"
echo " 设备   : $DEVICE"
echo " 日志   : $LOGDIR"
echo "=============================================================="
echo

echo "--- 1/3 环境自检 ---"
python "$REPO/scripts/check_env.py"
echo

echo "--- 2/3 上游完整性 ---"
python - <<'PY'
import hashlib, pathlib
d3t = pathlib.Path("third_party/dreamerv3-torch")
man = d3t / "UPSTREAM_MANIFEST.sha256"
bad = []
for ln in man.read_text().splitlines():
    want, rel = ln.split("  ", 1)
    p = d3t / rel.lstrip("./")
    got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"
    if got != want:
        bad.append(rel)
print(f"  与上游不一致的文件数: {len(bad)}")
for b in bad:
    print(f"    ! {b}")
if not bad:
    print("  （步骤 0 应恒为 0。若不为 0，说明上游文件被改过）")
PY
echo

echo "--- 3/3 启动训练 ---"
mkdir -p "$LOGDIR"
cd "$D3T"

# 所有覆盖都走命令行 —— 上游 configs.yaml 保持原样。
# dmc_proprio：向量观测 + MLP 编解码器，结构与 VMAS 最接近，无需 CNN。
python dreamer.py \
  --configs dmc_proprio \
  --task dmc_walker_walk \
  --logdir "$LOGDIR" \
  --device "$DEVICE" \
  --steps 2000 \
  --prefill 200 \
  --batch_size 2 \
  --batch_length 8 \
  --envs 1 \
  --eval_every 1000 \
  --eval_episode_num 1 \
  --log_every 500 \
  --video_pred_log False \
  --compile False \
  2>&1 | tee "$LOGDIR/run.log"

echo
echo "=============================================================="
echo " 完成。请回传整个目录：$LOGDIR"
echo "=============================================================="
