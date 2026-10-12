#!/usr/bin/env python3
"""环境自检 —— 在远端开跑之前先执行这个。

设计原则：**永不因缺少可选依赖而崩溃**。它只负责报告事实，
让"链路到底通不通"这件事在第一屏就可判，而不是训到一半才发现。

用法：
    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D3T = REPO / "third_party" / "dreamerv3-torch"


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def line(label: str, value: object) -> None:
    print(f"  {label:<28} {value}")


def check_python() -> None:
    section("Python / 系统")
    line("python", sys.version.split()[0])
    line("executable", sys.executable)
    line("platform", platform.platform())
    line("machine", platform.machine())
    line("cwd", os.getcwd())
    line("repo root", REPO)
    try:
        import multiprocessing

        line("cpu count", multiprocessing.cpu_count())
    except Exception as exc:  # pragma: no cover
        line("cpu count", f"<error: {exc}>")


def check_torch() -> bool:
    section("PyTorch / GPU")
    try:
        import torch
    except ImportError as exc:
        line("torch", f"缺失 ({exc})")
        print("\n  >>> 这是阻塞项：先装 torch 再继续。")
        return False

    line("torch", torch.__version__)
    line("torch 编译的 CUDA", getattr(torch.version, "cuda", None))
    line("cudnn", getattr(torch.backends.cudnn, "version", lambda: None)())
    cuda_ok = torch.cuda.is_available()
    line("cuda available", cuda_ok)
    mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    line("mps available", bool(mps_ok))

    if cuda_ok:
        n = torch.cuda.device_count()
        line("cuda device count", n)
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / 1024**3
            line(f"  gpu[{i}]", f"{props.name}  {vram:.1f} GB  cc{props.major}.{props.minor}")
        # 一次真实的显存分配，确认不是"看得见用不了"
        try:
            t = torch.zeros(1024, 1024, device="cuda")
            del t
            line("显存分配测试", "PASS")
        except Exception as exc:
            line("显存分配测试", f"FAIL: {exc}")
    else:
        print("\n  >>> 注意：CUDA 不可用。若这是远端实验机，请先排查驱动/torch 版本。")
    return True


def check_deps() -> None:
    section("依赖")
    required = ["numpy", "yaml", "ruamel.yaml", "einops", "matplotlib", "tensorboard"]
    optional = ["gym", "gymnasium", "dm_control", "mujoco", "crafter", "vmas"]

    for name in required:
        _report(name, required=True)
    for name in optional:
        _report(name, required=False)


def _report(name: str, required: bool) -> None:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        line(name, f"OK  {ver}")
    except Exception as exc:
        tag = "缺失" if required else "缺失(可选)"
        line(name, f"{tag}  {type(exc).__name__}")


def check_repo() -> None:
    section("仓库")
    line("third_party/dreamerv3-torch", "存在" if D3T.is_dir() else "**缺失**")
    manifest = D3T / "UPSTREAM_MANIFEST.sha256"
    if not manifest.is_file():
        line("UPSTREAM_MANIFEST", "缺失")
        return
    if shutil.which("shasum") is None:
        line("shasum", "不可用，跳过完整性校验")
        return
    proc = subprocess.run(
        ["shasum", "-a", "256", "-c", "UPSTREAM_MANIFEST.sha256"],
        cwd=D3T,
        capture_output=True,
        text=True,
    )
    changed = [ln for ln in proc.stdout.splitlines() if "FAILED" in ln]
    total = sum(1 for _ in manifest.read_text().splitlines())
    line("上游文件总数", total)
    line("与上游不一致的文件", len(changed))
    for ln in changed:
        print(f"      ! {ln.split(':', 1)[0]}")


def main() -> int:
    check_python()
    torch_ok = check_torch()
    check_deps()
    check_repo()

    section("结论")
    if not torch_ok:
        print("  BLOCKED：torch 不可用，无法继续。")
        return 1
    print("  自检完成。以上信息请连同运行日志一并回传。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
