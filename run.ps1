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

# 1. llama.cpp / ik_llama router (only if not already up)
if (-not (Listening $cfg.router_port)) {
  # Choose binary based on active_engine setting
  $engine = if ($cfg.active_engine) { $cfg.active_engine } else { "llamacpp" }
  if ($engine -eq "ikllama") {
    $serverBin = $cfg.ik_llama_server_bin
    $engineLabel = "ik_llama"
    # Mirror config.ini_path(): derive a sibling of models_ini, splitting on the
    # extension. .Replace(".ini", ...) is a global replace and would also rewrite
    # any parent directory whose name contains ".ini".
    $modelsIni = if ($cfg.ik_llama_models_ini) { $cfg.ik_llama_models_ini } else {
      Join-Path ([IO.Path]::GetDirectoryName($cfg.models_ini)) `
                ([IO.Path]::GetFileNameWithoutExtension($cfg.models_ini) + "-ikllama" +
                 [IO.Path]::GetExtension($cfg.models_ini))
    }
  } else {
    $serverBin = $cfg.server_bin
    $engineLabel = "llama.cpp"
    $modelsIni = $cfg.models_ini
  }
  # Mirror config._abs(): the router is started without -WorkingDirectory, and
  # config.example.json ships "./models.ini", so a relative value resolved
  # against whatever directory the user ran this from - the router then read an
  # empty registry and loaded 0 models.
  if ($modelsIni -and -not [IO.Path]::IsPathRooted($modelsIni)) {
    $modelsIni = [IO.Path]::GetFullPath((Join-Path $here $modelsIni))
  }
  # Mirror config.ensure_models_ini(): llama-server refuses to start without
  # this file, and the router is launched here, before the backend can make one.
  if ($modelsIni -and -not (Test-Path $modelsIni)) {
    Set-Content -Path $modelsIni -Encoding utf8 -Value @(
      "; LlamaForge model registry - read by llama-server's router.",
      "; Sections are model ids; keys are llama-server flags.",
      "version = 1",
      "",
      "[*]",
      "ctx-size = 150000")
    Write-Host "created $modelsIni"
  }
  if (Test-Path $serverBin) {
    $routerHost = if ($cfg.router_host) { $cfg.router_host } else { "127.0.0.1" }
    # llama.cpp's router evicts by loaded-model COUNT, so at --models-max 1
    # loading a second model always unloaded the first. 0 means unlimited and
    # hands the decision to LlamaForge, so test for presence here rather than
    # truthiness - a configured 0 must survive.
    $modelsMax = if ($null -ne $cfg.router_models_max) { $cfg.router_models_max } else { 1 }
    $args = @("--models-preset", $modelsIni, "--models-max", "$modelsMax", "--offline",
              "--host", $routerHost, "--port", "$($cfg.router_port)", "--metrics")
    if ($cfg.router_api_key) { $args += @("--api-key", $cfg.router_api_key) }
    Start-Process -FilePath $serverBin -ArgumentList $args -WindowStyle Hidden `
                  -RedirectStandardOutput (Join-Path $logDir "router.out.log") `
                  -RedirectStandardError  (Join-Path $logDir "router.err.log")
    Write-Host "started $engineLabel router on $($routerHost):$($cfg.router_port)"
  } else {
    Write-Host "server_bin not found ($serverBin) - open the dashboard Build tab to build $engineLabel first." -ForegroundColor Yellow
  }
} else {
  # Something already holds the router port. If it isn't a llama-server, the
  # dashboard would come up with every model "offline" and no stated reason -
  # port 8080 collides with XAMPP/Apache and plenty of other dev servers.
  $ownerPid = Get-NetTCPConnection -LocalPort $cfg.router_port -State Listen -ErrorAction SilentlyContinue |
              Select-Object -First 1 -ExpandProperty OwningProcess
  $ownerName = if ($ownerPid) { (Get-Process -Id $ownerPid -ErrorAction SilentlyContinue).ProcessName }
  if ($ownerName -and $ownerName -notmatch "llama") {
    Write-Host "port $($cfg.router_port) is already in use by '$ownerName' (PID $ownerPid)." -ForegroundColor Yellow
    Write-Host "The router was not started. Stop that process, or change router_port in the Setup tab." -ForegroundColor Yellow
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
