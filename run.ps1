# LlamaForge one-click runner.
# Reads config.json, starts the llama.cpp router + the LlamaForge backend,
# then opens the dashboard in your browser. Safe to run repeatedly.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# config.json is per-machine and deliberately not in the repo. Without this the
# first run died on a raw "Get-Content: path does not exist" exception that said
# nothing about config.example.json sitting right next to it.
$cfgPath = Join-Path $here "config.json"
if (-not (Test-Path $cfgPath)) {
  $example = Join-Path $here "config.example.json"
  if (-not (Test-Path $example)) {
    Write-Host "config.json is missing and config.example.json was not found in $here." -ForegroundColor Red
    Write-Host "Re-clone the repo, or create config.json by hand." -ForegroundColor Red
    exit 1
  }
  Copy-Item $example $cfgPath
  Write-Host "config.json not found - created one from config.example.json." -ForegroundColor Yellow
  Write-Host "Set your llama.cpp paths and model folders in the dashboard's Setup tab." -ForegroundColor Yellow
}
$cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json

function Listening($port){ [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) }

$logDir = Join-Path $here "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Resolve the active binary/registry and sanitize with the shared Python path.
# A failed schema probe reports a warning while the dashboard still opens.
if (-not (Listening $cfg.router_port)) {
  & python (Join-Path $here "backend\runtime_registry.py")
} else {
  $ownerPid = Get-NetTCPConnection -LocalPort $cfg.router_port -State Listen -ErrorAction SilentlyContinue |
              Select-Object -First 1 -ExpandProperty OwningProcess
  $ownerName = if ($ownerPid) { (Get-Process -Id $ownerPid -ErrorAction SilentlyContinue).ProcessName }
  if ($ownerName -and $ownerName -notmatch "llama") {
    Write-Host "port $($cfg.router_port) is already in use by '$ownerName' (PID $ownerPid)." -ForegroundColor Yellow
    Write-Host "The router was not started. Stop that process, or change router_port in Setup." -ForegroundColor Yellow
  }
}

# 2. LlamaForge backend (dashboard)
if (-not (Listening $cfg.panel_port)) {
  Start-Process -FilePath "python" -ArgumentList (Join-Path $here "backend\server.py") `
                -WorkingDirectory (Join-Path $here "backend") -WindowStyle Hidden
  Write-Host "started LlamaForge dashboard on port $($cfg.panel_port)"
}

# 3. open the dashboard
Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:$($cfg.panel_port)/"
