@echo off
setlocal EnableExtensions
title ComfyUI [GUI Launcher]

set "COMFY_ROOT=N:\ComfyUI"
set "VENV_DIR=N:\ComfyUI\venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\Activate.ps1"
set "PORT=8188"
set "REQ_LOCK=D:\COMFY_CACHE\venv_requirements\requirements_lock.txt"
set "REQ_TXT=D:\COMFY_CACHE\venv_requirements\requirements.txt"
set "INSTALLER_BAT=D:\COMFY_CACHE\new_install\launch.bat"

if not exist %COMFY_ROOT% (
  echo [ERROR] Missing ComfyUI root: %COMFY_ROOT%
  echo         Update 'Comfy Root' in GUI or install ComfyUI
  exit /b 1
)
if not exist %COMFY_ROOT%\main.py (
  echo [ERROR] Missing ComfyUI main.py in %COMFY_ROOT%
  echo         Ensure ComfyUI is cloned and up to date
  exit /b 1
)

cd /d %COMFY_ROOT%

set "TEMP=D:\COMFY_CACHE\temp"
set "TMP=D:\COMFY_CACHE\temp"
set "TORCH_HOME=D:\COMFY_CACHE\torch"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="
set "TRANSFORMERS_CACHE=D:\COMFY_CACHE\transformers_cache"
set "XFORMERS_DISABLE=1"

%VENV_PY% -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)"
if errorlevel 1 (
  echo [ERROR] Python interpreter is not 3.10: %VENV_PY%
  echo         Rebuild venv via D:\COMFY_CACHE\new_install\launch.bat or point CUSTOM_VENV_PATH to 3.10
  exit /b 1
)

if not exist %VENV_PY% (
  echo [ERROR] Missing venv python: %VENV_PY%
  echo         Rebuild it via D:\COMFY_CACHE\new_install\launch.bat
  exit /b 1
)

set "NEED_REPAIR=0"
%VENV_PY% -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('torch') else 1)"
if errorlevel 1 set NEED_REPAIR=1
%VENV_PY% -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('aiohttp') else 1)"
if errorlevel 1 set NEED_REPAIR=1
if "%NEED_REPAIR%" NEQ "0" (
  echo [Repair] Missing torch/aiohttp detected. Rehydrating virtual environment...
  if exist "%REQ_LOCK%" (
    echo [Repair] Installing pinned stack from %REQ_LOCK%
    %VENV_PY% -m pip install --no-input --upgrade -r "%REQ_LOCK%"
  ) else if exist "%REQ_TXT%" (
    echo [Repair] Installing stack from %REQ_TXT%
    %VENV_PY% -m pip install --no-input --upgrade -r "%REQ_TXT%"
  ) else (
    echo [ERROR] requirements files missing; run %INSTALLER_BAT% manually.
    exit /b 1
  )
  if errorlevel 1 (
    echo [ERROR] Automatic dependency repair failed. Run %INSTALLER_BAT% manually.
    exit /b 1
  )
)

if not exist "D:\SparkAI\Comfy_Cache\logs" mkdir "D:\SparkAI\Comfy_Cache\logs"

start "ComfyUI" /b powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "& { $ErrorActionPreference='Stop'; Set-Location $env:COMFY_ROOT; $venvPy=$env:VENV_PY; $activate=$env:VENV_ACTIVATE; if (Test-Path $activate) { . $activate }; & $venvPy -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)'; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; $ErrorActionPreference='Continue'; & $venvPy main.py --disable-xformers 2>&1 | Tee-Object -File 'D:\SparkAI\Comfy_Cache\logs\comfyui_server.log'; $last=$LASTEXITCODE; exit $last }"

setlocal EnableDelayedExpansion
set /a __WAITED=0
set /a __TIMEOUT=120
:waitloop
timeout /t 2 >nul
set /a __WAITED+=2
powershell.exe -NoLogo -NoProfile -Command "try { $null = Invoke-WebRequest http://127.0.0.1:%PORT% -UseBasicParsing; exit 0 } catch { exit 1 }"
if errorlevel 1 if !__WAITED! LSS !__TIMEOUT! goto waitloop
if errorlevel 1 (
  echo [ERROR] ComfyUI did not start within !__TIMEOUT! seconds.
  echo         See server log: D:\SparkAI\Comfy_Cache\logs\comfyui_server.log
  powershell.exe -NoLogo -NoProfile -Command "if (Test-Path 'D:\SparkAI\Comfy_Cache\logs\comfyui_server.log') { Get-Content -Tail 80 -Path 'D:\SparkAI\Comfy_Cache\logs\comfyui_server.log' } else { Write-Host 'No server log found.' }"
  exit /b 1
)

start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --user-data-dir="D:\COMFY_CACHE\utils\ChromeProfile_ComfyUI" --app=http://127.0.0.1:8188 --start-fullscreen --kiosk

exit /b 0
