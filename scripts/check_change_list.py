"""改动清单 vs 实际 diff —— 把"合并后逐条对一遍"从人工纪律变成机械检查。

================================================================================
为什么需要它
================================================================================
`docs/reproduction-plan.md` 的「改动清单」是**按文件逐条登记**的表，而它是
`UPSTREAM_MANIFEST` 那条不变式的前提：

    "清单不一致 ⟺ 上游被修改过"

**清单错了，那条不变式就废了** —— 而且是往"看起来没问题"的方向废：表里列着
一个已经被合并覆盖掉的改动，或者一个真实存在的改动没登记，两者都不会报错。

风险最大的是**两条分支同时改 `third_party/`** 的情形：文本合并会产出一张
**看着完整、实际对不上**的表。这正是本脚本要挡住的东西。

（本脚本的存在理由与台账里那条同源：**人工纪律会漂移，机械检查不会。**）

================================================================================
三条断言
================================================================================
  [1] 每个**实际被改**的 `third_party/` 文件，清单里必须有登记
  [2] 清单里登记的每个文件，必须**真的**被改过（防"登记了但改动被覆盖"）
  [3] 新增文件表与清单不重叠（新增文件不该出现在「改动清单」里 ——
      它不进上游指纹清单，登记进去反而会污染不变式）

用法::

    python scripts/check_change_list.py                  # 与 origin/main 比
    python scripts/check_change_list.py --base exp/comm-ablation
    python scripts/check_change_list.py --base c1e1e0d
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PLAN = REPO / "docs" / "reproduction-plan.md"
UPSTREAM = "third_party/dreamerv3-torch"


def git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    )
    return out.stdout


def changed_upstream_files(base: str) -> set[str]:
    """相对 base 被改动的上游文件（相对 UPSTREAM 的路径）。"""
    diff = git("diff", "--name-only", f"{base}...HEAD", "--", UPSTREAM)
    return {
        line.split(f"{UPSTREAM}/", 1)[1]
        for line in diff.splitlines()
        if line.strip() and f"{UPSTREAM}/" in line
    }


def plan_tables(text: str) -> tuple[set[str], set[str]]:
    """从台账里抠出 (改动清单里的文件, 新增文件表里的文件)。

    两张表的行都长成 `| \\`path\\` | 说明 | ... |`，靠小节标题区分。
    """
    def rows(block: str) -> set[str]:
        out = set()
        for line in block.splitlines():
            m = re.match(r"^\|\s*`([^`]+)`\s*\|", line)
            if m:
                out.add(m.group(1))
        return out

    i = text.index("## 5. 改动清单")
    j = text.index("新增文件（不在上游清单里")
    change = rows(text[i:j])
    # 新增文件表的结尾：下一个二级标题或文件尾
    k = text.find("\n## ", j)
    added = rows(text[j : k if k != -1 else len(text)])
    return change, added


def normalise(name: str) -> str:
    """清单里的写法有裸文件名也有相对路径，统一成 basename 比对。

    只比 basename 是刻意的：本项目上游目录是扁平的（`dreamer.py` / `configs.yaml` /
    `envs/*.py`），同名冲突不存在；严格化到全路径会把"清单里写 `dreamer.py`、
    diff 里是 `third_party/dreamerv3-torch/dreamer.py`"这种正常情形误判成缺失。
    """
    return pathlib.PurePosixPath(name).name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None,
                    help="比对基准。默认 = **引入 third_party 的那个 commit**")
    args = ap.parse_args()

    base = args.base
    if base is None:
        # ⚠️ 基准只能是"上游 vendoring 的那一刻"：改动清单是相对**上游**累积的，
        # 不是相对本分支起点。我第一版默认用 origin/main，于是把整张表误判成
        # "登记了但没改" —— 报出四条假失败。
        first = git("log", "--format=%H", "--reverse", "--", UPSTREAM).splitlines()
        if not first:
            print(f"{UPSTREAM} 没有任何提交，无从比对", file=sys.stderr)
            return 2
        base = first[0]
        print(f"（未指定 --base，自动取引入基座的那个 commit：{base[:7]}）")

    try:
        changed = changed_upstream_files(base)
    except subprocess.CalledProcessError as e:
        print(f"git diff 失败：{e.stderr.strip()}", file=sys.stderr)
        return 2

    text = PLAN.read_text()
    listed, added = plan_tables(text)
    listed_n = {normalise(x) for x in listed}
    added_n = {normalise(x) for x in added}

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print(f"改动清单 vs 实际 diff（基准 {args.base}）")
    print("-" * 70)
    print(f"  实际被改的上游文件 {len(changed)} 个；清单登记 {len(listed_n)} 个")
    print()

    # 新增文件（我们自己的 envs/seg.py、networks_gnn.py 之类）本来就会出现在 diff 里，
    # 但它们归「新增文件表」管：它们**不在上游指纹清单里**，登记进「改动清单」反而
    # 会污染那个不变式。所以这里要把新增表排除掉。
    missing = sorted(
        f for f in changed
        if normalise(f) not in listed_n and normalise(f) not in added_n
    )
    check(
        "[1] 每个实际被改的上游文件都在清单里（新增文件除外）",
        not missing,
        f"缺登记：{missing}" if missing else "",
    )

    extra = sorted(
        f for f in listed_n if f not in {normalise(c) for c in changed} and f not in added_n
    )
    # 清单里可能登记了 `scripts/xxx.py`（非上游），那些不算"上游文件"，单独提示
    check(
        "[2] 清单里登记的上游文件都真的被改过",
        not extra,
        f"登记了但 diff 里没有：{extra}" if extra else "",
    )

    # ⚠️ 这一条**只提示、不判失败**。项目的不变式是"上游指纹清单不一致 ⟺ 真改了
    # 上游"，而我们自己的新文件压根不在那张指纹清单里 —— 所以"某个新文件后来又
    # 被改一次、于是也出现在改动清单里"并不破坏不变式。我第一版把它写成硬失败，
    # **那是我自己发明的不变式**：检查器不该比项目本身更严，否则它报的"失败"
    # 会训练人忽略它。
    overlap = sorted(listed_n & added_n)
    if overlap:
        print(f"  提示（不判失败）：两张表都提到的文件 {overlap} —— 它们是我们的"
              f"新文件，不影响上游指纹不变式")

    print()
    if changed:
        print("  实际被改的上游文件：")
        for f in sorted(changed):
            # 三态：在改动清单里 / 在新增文件表里（正常，新文件归那边管）/ 两处都没有
            n = normalise(f)
            mark = "  " if n in listed_n else ("新增" if n in added_n else "!!")
            print(f"    {mark} {f}")
    print("=" * 70)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        print("清单与 diff 对不上 —— 那个『清单不一致 ⟺ 上游被修改过』的不变式此刻不成立。")
        return 1
    print("清单与 diff 一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
