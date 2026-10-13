#!/usr/bin/env python3
"""在【本地】实时监控远端主机的 CPU / GPU / 内存 / 温度 / 进程 / 训练进度。

它通过 ssh 调用远端上的 `scripts/remote_metrics.py` 取一次数据，然后在本机渲染。
每次刷新只建立一个 ssh 连接。

用法：
    python scripts/monitor.py                      # 默认监控 wyb，每 3 秒刷新
    python scripts/monitor.py --interval 2
    python scripts/monitor.py --log /tmp/runs/seg/main.log --steps 20000
    python scripts/monitor.py --once               # 只采一次，便于重定向到文件

退出：Ctrl-C
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time

# ------------------------------------------------------------------ 小工具

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"


def color_for(pct: float, warn: float = 70.0, bad: float = 90.0) -> str:
    if pct >= bad:
        return RED
    if pct >= warn:
        return YELLOW
    return GREEN


def bar(pct: float, width: int = 28, warn: float = 70.0, bad: float = 90.0) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    return f"{color_for(pct, warn, bad)}{'█' * filled}{DIM}{'·' * (width - filled)}{RESET}"


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def section(title: str) -> str:
    return f"{BOLD}{CYAN}── {title} {'─' * max(0, 56 - len(title))}{RESET}"


def pick_temp(temps: dict) -> tuple[str, float] | None:
    """挑一个最能代表 CPU 的温度。优先 coretemp / x86_pkg_temp。"""
    order = ["coretemp", "x86_pkg_temp", "TCPU"]
    for key in order:
        for name, val in temps.items():
            if key.lower() in name.lower():
                return name, val
    if temps:
        k = max(temps, key=lambda k: temps[k])
        return k, temps[k]
    return None


# ------------------------------------------------------------------ 渲染

def render(d: dict, args) -> str:
    out: list[str] = []
    w = shutil.get_terminal_size((100, 40)).columns

    header = (f"{BOLD}{d['host']}{RESET}  "
              f"{DIM}uptime {human_time(d['uptime'])}{RESET}")
    out.append(header)
    out.append("")

    # --- CPU
    cpu = d["cpu"]
    out.append(section("CPU"))
    out.append(f"  总体   {bar(cpu['overall'])} {cpu['overall']:5.1f}%"
               f"   load {cpu['load1']:.1f} {cpu['load5']:.1f} {cpu['load15']:.1f}")
    cores = cpu["cores"]
    per_row = max(1, (w - 8) // 7)
    for i in range(0, len(cores), per_row):
        chunk = cores[i:i + per_row]
        cells = " ".join(f"{v:5.1f}" for v in chunk)
        out.append(f"  {'核':<4} {cells}")

    # --- 温度
    t = pick_temp(d["temps"])
    if t:
        name, val = t
        tcolor = GREEN if val < 70 else (YELLOW if val < 85 else RED)
        out.append(f"  温度   {tcolor}{val:5.1f}°C{RESET}  {DIM}{name}{RESET}")

    # --- 内存
    mem = d["mem"]
    out.append(section("内存"))
    out.append(f"  RAM    {bar(mem['percent'], warn=80, bad=95)} "
               f"{mem['used_gib']:.1f} / {mem['total_gib']:.1f} GiB ({mem['percent']:.0f}%)")
    if mem["swap_total_gib"]:
        used = mem["swap_total_gib"] - mem["swap_free_gib"]
        pct = 100.0 * used / mem["swap_total_gib"]
        out.append(f"  Swap   {bar(pct, warn=50, bad=80)} "
                   f"{used:.1f} / {mem['swap_total_gib']:.1f} GiB")

    # --- GPU
    g = d["gpu"]
    out.append(section("GPU"))
    if not g.get("available"):
        out.append(f"  {RED}不可用{RESET} {g.get('error', '')}")
    else:
        memp = 100.0 * g["mem_used"] / g["mem_total"] if g["mem_total"] else 0.0
        tcolor = GREEN if g["temp"] < 70 else (YELLOW if g["temp"] < 83 else RED)
        out.append(f"  {g['name']}")
        out.append(f"  利用率 {bar(g['util'])} {g['util']:5.1f}%")
        out.append(f"  显存   {bar(memp, warn=80, bad=92)} "
                   f"{g['mem_used']:.0f} / {g['mem_total']:.0f} MiB")
        out.append(f"  温度   {tcolor}{g['temp']:5.1f}°C{RESET}   "
                   f"功耗 {g['power']:5.1f} / {g['power_limit']:.0f} W   "
                   f"SM {g['clock_sm']:.0f} MHz")
        for p in g.get("procs", []):
            out.append(f"    {DIM}pid {p['pid']:<7} {p['mem_mib']:>7.0f} MiB  "
                       f"{p['name'][:48]}{RESET}")

    # --- 进程
    out.append(section("占用最高的进程"))
    out.append(f"  {'PID':>7} {'CPU%':>6} {'MEM%':>6} {'运行时长':>9} {'线程':>5}  命令")
    for p in d["procs"]:
        c = color_for(p["cpu"], 50, 90)
        out.append(f"  {p['pid']:>7} {c}{p['cpu']:>6.1f}{RESET} {p['mem']:>6.1f} "
                   f"{human_time(p['etimes']):>9} {p['threads']:>5}  "
                   f"{p['cmd'][:40]}")

    # --- 训练
    log = d.get("log", {})
    out.append(section("训练"))
    if not log.get("path"):
        out.append(f"  {DIM}（未指定 --log）{RESET}")
    elif not log.get("exists"):
        out.append(f"  {DIM}{log['path']} 尚不存在{RESET}")
    else:
        age = log.get("mtime_age", 0)
        acolor = GREEN if age < 30 else (YELLOW if age < 300 else RED)
        out.append(f"  {DIM}{log['path']}{RESET}   "
                   f"最后写入 {acolor}{age:.0f}s 前{RESET}")
        latest = None
        for ln in log.get("lines", []):
            if "update_count" in ln or "eval_return" in ln or "train_return" in ln:
                latest = ln
        if latest:
            out.append(f"  {BOLD}{latest.strip()[:w - 4]}{RESET}")
        else:
            for ln in log.get("lines", [])[-3:]:
                out.append(f"  {DIM}{ln.strip()[:w - 4]}{RESET}")

    return "\n".join(out)


# ------------------------------------------------------------------ 采集

def collect(args) -> dict:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.host,
           f"python3 {args.remote_script} --top {args.top} --log-lines 8"]
    if args.log:
        cmd[-1] += f" --log {args.log}"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"ssh 超时（>{args.timeout}s）"}
    if p.returncode != 0:
        return {"error": (p.stderr or p.stdout).strip()[:300]}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": f"无法解析远端输出: {p.stdout[:200]!r}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="wyb")
    ap.add_argument("--remote-script",
                    default="$HOME/CoDreamer/scripts/remote_metrics.py")
    ap.add_argument("--log", default=None, help="远端训练日志的绝对路径")
    ap.add_argument("--steps", type=float, default=None, help="训练总步数（保留）")
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    try:
        while True:
            t0 = time.time()
            d = collect(args)
            if "error" in d:
                body = f"{RED}采集失败:{RESET} {d['error']}"
            else:
                body = render(d, args)
            # 重绘：清屏 + 回到左上角
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(body + "\n")
            sys.stdout.write(f"\n{DIM}刷新间隔 {args.interval:.0f}s   "
                             f"{time.strftime('%H:%M:%S')}   Ctrl-C 退出{RESET}\n")
            sys.stdout.flush()
            if args.once:
                return 0
            time.sleep(max(0.5, args.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        sys.stdout.write(RESET + "\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
