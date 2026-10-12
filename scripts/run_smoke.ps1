<#
 步骤 0 冒烟测试 —— 原版 dreamerv3-torch，不改任何上游文件。（Windows / PowerShell）

 目的：只验证工具链（Python / torch+CUDA / 依赖 / 显存 / 训练循环能转起来）。
 不验证效果，也不验证 CoDreamer。

 用法：
     powershell -ExecutionPolicy Bypass -File scripts\run_smoke.ps1
     powershell -ExecutionPolicy Bypass -File scripts\run_smoke.ps1 -Device cpu

 日志落在 logdir\smoke\ 下（已 gitignore），跑完把整个目录回传。
#>

param(
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"

# Windows 上没有 OSMesa，而 envs/dmc.py 每步都 render（obs["image"] = self.render()），
# 所以必须给 dm_control 一个真的 GL 后端。上游 dreamer.py 里是 setdefault("osmesa")，这里给出真值。
if (-not $env:MUJOCO_GL) { $env:MUJOCO_GL = "glfw" }

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$D3T  = Join-Path $Repo "third_party\dreamerv3-torch"
$LogDir = Join-Path $Repo "logdir\smoke"

Write-Host "=============================================================="
Write-Host " 步骤 0 冒烟测试：原版 dreamerv3-torch"
Write-Host " 仓库   : $Repo"
Write-Host " 设备   : $Device"
Write-Host " 日志   : $LogDir"
Write-Host "=============================================================="
Write-Host ""

Write-Host "--- 1/3 环境自检 ---"
python (Join-Path $Repo "scripts\check_env.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "自检未通过，中止。" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host ""

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

Write-Host "--- 2/3 上游完整性 ---"
# 注意：here-string 必须通过管道喂给 `python -`（走 stdin）。
# 若写成 `python - @"..."@`，PowerShell 会把它当成命令行参数，python 转而进入 REPL。
@"
import hashlib, pathlib
d3t = pathlib.Path(r"$D3T")
man = d3t / "UPSTREAM_MANIFEST.sha256"
bad = []
for ln in man.read_text(encoding="utf-8").splitlines():
    want, rel = ln.split("  ", 1)
    p = d3t / rel[2:]  # 清单条目形如 "./x"；不能用 lstrip("./")（会吃掉 .gitignore 的首点）
    got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"
    if got != want:
        bad.append(rel)
print(f"  与上游不一致的文件数: {len(bad)}")
for b in bad:
    print(f"    ! {b}")
if not bad:
    print("  （基线应恒为 0。不为 0 时逐条对照 docs/reproduction-plan.md 的改动清单）")
"@ | python -
Write-Host ""

Write-Host "--- 3/3 启动训练 ---"
Set-Location $D3T
$LogFile = Join-Path $LogDir "run.log"

# 所有覆盖都走命令行 —— 上游 configs.yaml 保持原样。
# dmc_proprio：向量观测 + MLP 编解码器，结构与 VMAS 最接近，无需 CNN。
# 注意：bool 参数必须写成 "True" / "False"（tools.args_type 只认这两个字面量）。
# 注意：train_ratio 必须跟着 batch 一起缩。上游 _should_train = Every(batch_size*batch_length/train_ratio)：
#   真实配置 Every(16*64/512) = Every(2) → 每 2 环境步 1 次梯度更新；
#   冒烟把 batch 砍成 2*8 后若仍用 512，会变成 Every(0.03125) → 每环境步 32 次更新（多 64 倍），
#   冒烟会从几分钟变成几十分钟。取 train_ratio 8 → Every(2*8/8) = Every(2)，与真实配置同频。
# 注意：PS 5.1 在 $ErrorActionPreference = "Stop" 下，`2>&1 |` 会把原生程序的 stderr
# 变成终止性错误（Gym 的弃用警告就能把脚本直接打断，训练根本没启动）。
# 这里临时降级为 Continue，改用 $LASTEXITCODE 判定成败。
$ErrorActionPreference = "Continue"
python -u dreamer.py `
  --configs dmc_proprio `
  --task dmc_walker_walk `
  --logdir "$LogDir" `
  --device "$Device" `
  --steps 2000 `
  --prefill 200 `
  --batch_size 2 `
  --batch_length 8 `
  --train_ratio 8 `
  --envs 1 `
  --eval_every 1000 `
  --eval_episode_num 1 `
  --log_every 500 `
  --video_pred_log False `
  --compile False `
  2>&1 | Tee-Object -FilePath $LogFile
$TrainExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"

Write-Host ""
if ($TrainExit -ne 0) {
    Write-Host "训练进程退出码 $TrainExit，日志：$LogFile" -ForegroundColor Red
    exit $TrainExit
}
Write-Host "=============================================================="
Write-Host " 完成。请回传整个目录：$LogDir"
Write-Host "=============================================================="
