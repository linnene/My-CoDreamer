#!/usr/bin/env python3
"""在【远端主机】上采集一次系统状态，输出一行 JSON 到 stdout。

配套 `scripts/monitor.py`（在本地跑）使用：本地通过 ssh 调它，解析 JSON 后渲染。

为什么单独写这个而不是用 nvtop/htop：
    远端这台机器没装 nvtop / htop / gpustat，而且装它们要 sudo。
    但所需数据本就能直接读 —— /proc/stat、/sys/class/thermal、/sys/class/hwmon、
    nvidia-smi —— 不需要任何额外依赖。

用法（在远端）：
    python3 scripts/remote_metrics.py
    python3 scripts/remote_metrics.py --log /tmp/runs/seg/main.log --top 5
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time


# --------------------------------------------------------------------- CPU

def cpu_snapshot() -> tuple[int, int, list[tuple[int, int]]]:
    """读一次 /proc/stat，返回 (总忙, 总闲, [(每核忙, 每核闲)])。"""
    busy_total = idle_total = 0
    per_core: list[tuple[int, int]] = []
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            parts = line.split()
            name = parts[0]
            if name == "cpu":
                vals = [int(x) for x in parts[1:11]]
                idle = vals[3] + vals[4]  # idle + iowait
                busy_total = sum(vals) - idle
                idle_total = idle
            else:
                vals = [int(x) for x in parts[1:11]]
                idle = vals[3] + vals[4]
                per_core.append((sum(vals) - idle, idle))
    return busy_total, idle_total, per_core


def cpu_usage(sample_seconds: float = 0.35) -> dict:
    """采两次样算真实占用率（瞬时读数没有意义）。"""
    b1, i1, c1 = cpu_snapshot()
    time.sleep(sample_seconds)
    b2, i2, c2 = cpu_snapshot()
    db, di = b2 - b1, i2 - i1
    total = db + di
    overall = 100.0 * db / total if total else 0.0
    cores = []
    for (pb, pi), (b, i) in zip(c1, c2):
        d = (b - pb) + (i - pi)
        cores.append(100.0 * (b - pb) / d if d else 0.0)
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0, 0, 0)
    return {
        "overall": overall,
        "cores": cores,
        "load1": load[0],
        "load5": load[1],
        "load15": load[2],
    }


# ----------------------------------------------------------------- 内存 / 温度

def memory() -> dict:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                info[k] = int(v.split()[0]) / 1024 / 1024  # GiB
    total = info.get("MemTotal", 0.0)
    avail = info.get("MemAvailable", 0.0)
    return {
        "total_gib": total,
        "used_gib": total - avail,
        "avail_gib": avail,
        "percent": 100.0 * (total - avail) / total if total else 0.0,
        "swap_total_gib": info.get("SwapTotal", 0.0),
        "swap_free_gib": info.get("SwapFree", 0.0),
    }


def temperatures() -> dict:
    """读 sysfs。这台机器没有 lm-sensors，但内核已经把温度暴露在 sysfs 里。"""
    out: dict[str, float] = {}

    # 1) /sys/class/hwmon/*/temp*_input  —— 带名字，最可读
    for hw in sorted(pathlib.Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            name = (hw / "name").read_text().strip()
        except OSError:
            continue
        for t in sorted(hw.glob("temp*_input")):
            try:
                raw = int(t.read_text().strip())
            except (OSError, ValueError):
                continue
            label_file = t.with_name(t.name.replace("_input", "_label"))
            label = label_file.read_text().strip() if label_file.exists() else t.stem
            out[f"{name}/{label}"] = raw / 1000.0

    # 2) thermal_zone —— 兜底
    for z in sorted(pathlib.Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            ztype = (z / "type").read_text().strip()
            raw = int((z / "temp").read_text().strip())
        except (OSError, ValueError):
            continue
        # 只保留有意义的（acpitz 经常是假的 27.8，但留着无妨）
        if ztype not in out:
            out[f"zone/{ztype}"] = raw / 1000.0
    return out


# ----------------------------------------------------------------------- GPU

GPU_FIELDS = [
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
]


def gpu() -> dict:
    try:
        q = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"available": False}
    if q.returncode != 0:
        return {"available": False, "error": q.stderr.strip()[:200]}

    vals = [v.strip() for v in q.stdout.strip().splitlines()[0].split(",")]
    d = dict(zip(GPU_FIELDS, vals))

    def num(key):
        try:
            return float(d.get(key, "nan"))
        except ValueError:
            return float("nan")

    procs = []
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        for row in p.stdout.strip().splitlines():
            if not row.strip():
                continue
            pid, pname, mem = [x.strip() for x in row.split(",", 2)]
            procs.append({"pid": int(pid), "name": pname, "mem_mib": float(mem)})
    except Exception:
        pass

    return {
        "available": True,
        "name": d.get("name", "?"),
        "util": num("utilization.gpu"),
        "mem_used": num("memory.used"),
        "mem_total": num("memory.total"),
        "temp": num("temperature.gpu"),
        "power": num("power.draw"),
        "power_limit": num("power.limit"),
        "clock_sm": num("clocks.sm"),
        "procs": procs,
    }


# ------------------------------------------------------------------ 进程

def processes(top: int) -> list[dict]:
    q = subprocess.run(
        ["ps", "-eo", "pid,ppid,user,pcpu,pmem,etimes,nlwp,comm",
         "--sort=-pcpu", "--no-headers"],
        capture_output=True, text=True, timeout=10,
    )
    out = []
    for line in q.stdout.splitlines()[:top]:
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        pid, ppid, user, pcpu, pmem, etimes, nlwp, comm = parts
        try:
            out.append({
                "pid": int(pid), "ppid": int(ppid), "user": user,
                "cpu": float(pcpu), "mem": float(pmem),
                "etimes": int(etimes), "threads": int(nlwp), "cmd": comm,
            })
        except ValueError:
            continue
    return out


# ------------------------------------------------------------------ 日志

def log_tail(path: str | None, lines: int = 6) -> dict:
    """读训练日志尾部，抽最新一行的关键指标。"""
    if not path:
        return {"path": None}
    p = pathlib.Path(path)
    if not p.is_file():
        return {"path": path, "exists": False}
    try:
        # 只读尾部，避免大文件卡住
        size = p.stat().st_size
        with p.open("rb") as f:
            f.seek(max(0, size - 65536))
            chunk = f.read().decode("utf-8", "replace")
        tail = chunk.splitlines()[-lines:]
    except OSError as e:
        return {"path": path, "error": str(e)}
    return {"path": path, "exists": True, "lines": tail,
            "mtime_age": time.time() - p.stat().st_mtime}


# ------------------------------------------------------------------ 主

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--log-lines", type=int, default=6)
    args = ap.parse_args()

    data = {
        "ts": time.time(),
        "host": os.uname().nodename,
        "uptime": float(open("/proc/uptime").read().split()[0]),
        "cpu": cpu_usage(),
        "mem": memory(),
        "temps": temperatures(),
        "gpu": gpu(),
        "procs": processes(args.top),
        "log": log_tail(args.log, args.log_lines),
    }
    print(json.dumps(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
