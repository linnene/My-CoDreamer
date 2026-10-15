#!/usr/bin/env bash
# 把远端仓库同步到指定的 commit —— 用**验证**代替信任。
#
# ============================================================================
# 为什么需要这个脚本（两次踩坑换来的）
# ============================================================================
# wyb 到 GitHub 的 443 时通时断，而 git 在连不上时的表现**非常容易骗人**：
#
#   1. `git push` 报了成功，远端其实还是旧 commit。
#      → 训练跑的是没修复的代码，报的是已经修过的错。
#
#   2. 更隐蔽：`git fetch` 也会静默失败。此时本地那个 `origin/main` 引用是
#      **旧的**，于是 `git reset --hard origin/main` 会**成功地把代码重置回
#      错误的 commit** —— 没有任何报错。我在这上面白追了一轮。
#
# 所以：**永远比对 hash，不要相信命令的退出码。** push 和 fetch 都是。
#
# 用法：
#   bash scripts/sync_remote.sh              # 同步到本地当前的 HEAD
#   bash scripts/sync_remote.sh <sha>        # 同步到指定 commit
#   HOST=wyb REPO=/home/wyb/CoDreamer bash scripts/sync_remote.sh
#
# 远端运行脚本里**不要**再写 git 命令 —— 那些脚本不知道自己有没有同步成功。
# 先跑本脚本，再启动任务。
set -uo pipefail

HOST="${HOST:-wyb}"
REPO="${REPO:-/home/wyb/CoDreamer}"
SHA="${1:-$(git rev-parse HEAD)}"
ATTEMPTS="${ATTEMPTS:-20}"

echo "同步 $HOST:$REPO -> $SHA"

remote_sha() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" \
    "cd $REPO && git rev-parse origin/main 2>/dev/null" 2>/dev/null
}

for i in $(seq 1 "$ATTEMPTS"); do
  # fetch 可能静默失败，所以每一步都不看退出码，只看 hash
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" \
    "cd $REPO && git fetch -q origin main >/dev/null 2>&1 || true" 2>/dev/null

  cur="$(remote_sha)"
  if [ "$cur" = "$SHA" ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" \
      "cd $REPO && git reset -q --hard $SHA" 2>/dev/null
    # reset 之后**再验一次** —— reset 也可能因为别的原因没生效
    got="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" \
      "cd $REPO && git rev-parse HEAD" 2>/dev/null)"
    if [ "$got" = "$SHA" ]; then
      echo "已同步并确认: $SHA"
      exit 0
    fi
    echo "  尝试 $i: reset 后 HEAD=${got:0:7} ≠ ${SHA:0:7}"
  else
    echo "  尝试 $i: origin/main=${cur:0:7} ≠ ${SHA:0:7}"
  fi
  sleep 6
done

echo "同步失败：$ATTEMPTS 次都没拿到 $SHA" >&2
exit 1
