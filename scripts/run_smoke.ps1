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
python - @"
import hashlib, pathlib
d3t = pathlib.Path(r"$D3T")
man = d3t / "UPSTREAM_MANIFEST.sha256"
bad = []
for ln in man.read_text(encoding="utf-8").splitlines():
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
"@
Write-Host ""

Write-Host "--- 3/3 启动训练 ---"
Set-Location $D3T
$LogFile = Join-Path $LogDir "run.log"

# 所有覆盖都走命令行 —— 上游 configs.yaml 保持原样。
# dmc_proprio：向量观测 + MLP 编解码器，结构与 VMAS 最接近，无需 CNN。
# 注意：bool 参数必须写成 "True" / "False"（tools.args_type 只认这两个字面量）。
python dreamer.py `
  --configs dmc_proprio `
  --task dmc_walker_walk `
  --logdir "$LogDir" `
  --device "$Device" `
  --steps 2000 `
  --prefill 200 `
  --batch_size 2 `
  --batch_length 8 `
  --envs 1 `
  --eval_every 1000 `
  --eval_episode_num 1 `
  --log_every 500 `
  --video_pred_log False `
  --compile False `
  2>&1 | Tee-Object -FilePath $LogFile

Write-Host ""
Write-Host "=============================================================="
Write-Host " 完成。请回传整个目录：$LogDir"
Write-Host "=============================================================="
