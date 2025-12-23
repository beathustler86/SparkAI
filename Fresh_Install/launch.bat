@echo off
setlocal EnableExtensions EnableDelayedExpansion

title ComfyUI Venv Clean Rebuild

REM Set NONINTERACTIVE=1 to disable pauses (useful for automated runs)
set "PAUSE_ON_EXIT=1"
if defined NONINTERACTIVE set "PAUSE_ON_EXIT=0"

set "COMFY_ROOT=N:\ComfyUI"
set "VENV_DIR=%COMFY_ROOT%\venv"
if defined CUSTOM_VENV_PATH set "VENV_DIR=%CUSTOM_VENV_PATH%"
set "LOCK_DIR=D:\COMFY_CACHE\venv_requirements"
set "LOG_DIR=D:\COMFY_CACHE\new_install"

set "RUN_LOG=%LOG_DIR%\install_run.log"

REM Self-log everything to RUN_LOG unless disabled
if not defined NO_SELF_LOG if not defined __SELF_LOGGING (
  set "__SELF_LOGGING=1"
  call "%~f0" %* > "%RUN_LOG%" 2>&1
  exit /b %ERRORLEVEL%
)

echo [INFO] Starting ComfyUI venv rebuild
echo [INFO] NONINTERACTIVE=%NONINTERACTIVE%
echo [INFO] COMFY_ROOT=%COMFY_ROOT%
echo [INFO] VENV_DIR=%VENV_DIR%
echo [INFO] LOCK_DIR=%LOCK_DIR%
echo [INFO] LOG_DIR=%LOG_DIR%
echo [INFO] RUN_LOG=%RUN_LOG%

if defined TRACE echo [TRACE] Entered script main body

REM Isolation: avoid leaking user-site packages or external python env vars
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="

REM Force pip usage to be inside a virtual environment
set "PIP_REQUIRE_VIRTUALENV=1"

REM Ensure CUDA-enabled PyTorch wheels are discoverable (torch==*+cu128 etc)
REM This avoids relying on any global pip config.
set "PIP_INDEX_URL=https://pypi.org/simple"
set "PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128"

set "REQ=%LOCK_DIR%\requirements.txt"
set "LOCK=%LOCK_DIR%\requirements_lock.txt"
set "FREEZE=%LOCK_DIR%\pip_freeze.txt"

if defined TRACE (
  echo [TRACE] REQ=%REQ%
  echo [TRACE] LOCK=%LOCK%
  echo [TRACE] FREEZE=%FREEZE%
)

if not exist "%COMFY_ROOT%\main.py" (
  echo [WARN] ComfyUI main.py not visible at: "%COMFY_ROOT%\main.py"
  echo        Continuing venv rebuild anyway ^(this check is non-fatal^).
  echo        If this is unexpected, verify the drive mapping / permissions for N:.
  if defined TRACE echo [TRACE] Missing: %COMFY_ROOT%\main.py
) else (
  if defined TRACE echo [TRACE] Found ComfyUI main.py
)

if not exist "%REQ%" (
  echo [ERROR] Missing locked requirements: "%REQ%"
  echo         Create/verify D:\COMFY_CACHE\venv_requirements\requirements.txt
  if defined TRACE echo [TRACE] Missing: %REQ%
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

if defined TRACE echo [TRACE] Found requirements.txt

echo ======================================================
echo ComfyUI venv clean rebuild
echo - COMFY_ROOT: %COMFY_ROOT%
echo - VENV_DIR  : %VENV_DIR%
echo - LOCK_DIR  : %LOCK_DIR%
echo ======================================================

REM Pick a python to create the venv (Python 3.10 is required for this stack)
set "PY_CREATE="
where py >nul 2>nul && set "PY_CREATE=py -3.10"
if not defined PY_CREATE (
  echo [ERROR] Python 3.10 is required but the Windows Python Launcher ^(py.exe^) was not found.
  echo         Install Python 3.10 and ensure the Python Launcher is available ^("py -3.10"^).
  if defined TRACE echo [TRACE] where py failed / PY_CREATE empty
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

if defined TRACE echo [TRACE] PY_CREATE=%PY_CREATE%

echo [1/6] Removing existing venv (if present)...
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if exist "%VENV_DIR%\Scripts\python.exe" (
  echo     Found existing venv: "%VENV_DIR%"
  echo     Attempting to stop any running processes using: "%VENV_PY%"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -and ($_.Path -ieq $env:VENV_PY) } | Stop-Process -Force -ErrorAction SilentlyContinue } catch { }" >nul 2>nul
)
if exist "%VENV_DIR%" (
  set "_RM_TRIES=0"
  :__rm_try
  set /a _RM_TRIES+=1
  rmdir /s /q "%VENV_DIR%" 2>nul
  if exist "%VENV_DIR%" (
    if !_RM_TRIES! GEQ 5 (
      echo [ERROR] Failed to remove existing venv after !_RM_TRIES! attempts.
      echo         Close any running ComfyUI/python processes and ensure you have permission to delete:
      echo           "%VENV_DIR%"
      if "%PAUSE_ON_EXIT%"=="1" pause
      exit /b 1
    )
    timeout /t 2 /nobreak >nul
    goto __rm_try
  )
)

if defined TRACE echo [TRACE] Venv directory removed (if existed)

echo [2/6] Creating venv...
%PY_CREATE% -m venv "%VENV_DIR%"
if errorlevel 1 (
  echo [ERROR] venv creation failed.
  if defined TRACE echo [TRACE] venv creation errorlevel=%ERRORLEVEL%
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

if defined TRACE echo [TRACE] Venv created

echo Verifying venv Python version is 3.10...
"%VENV_DIR%\Scripts\python.exe" -c "import sys; v=f'{sys.version_info.major}.{sys.version_info.minor}'; print(v); raise SystemExit(0 if v=='3.10' else 1)"
if errorlevel 1 (
  echo [ERROR] Venv python is not 3.10.
  echo         This stack is pinned for Python 3.10 for compatibility.
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

echo [3/6] Upgrading pip tooling...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [ERROR] pip tooling upgrade failed.
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

echo [4/6] Installing locked requirements...
if exist "%FREEZE%" (
  echo     Using exact freeze: "%FREEZE%"
  "%VENV_DIR%\Scripts\python.exe" -m pip install --only-binary :none: -r "%FREEZE%"
) else (
  echo     Freeze not found; using requirements.txt
  if exist "%LOCK%" (
    echo     Applying constraints from requirements_lock.txt
    "%VENV_DIR%\Scripts\python.exe" -m pip install --only-binary :none: -r "%REQ%" -c "%LOCK%"
  ) else (
    "%VENV_DIR%\Scripts\python.exe" -m pip install --only-binary :none: -r "%REQ%"
  )
)
if errorlevel 1 (
  echo [ERROR] pip install failed.
  echo         See output above for the first failing dependency.
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b 1
)

echo [5/6] Capturing exact installed versions...
"%VENV_DIR%\Scripts\python.exe" -m pip freeze > "%LOCK_DIR%\pip_freeze.txt"

echo [6/6] Verifying environment integrity (pip check)...
"%VENV_DIR%\Scripts\python.exe" -m pip check > "%LOG_DIR%\pip_check.txt"
set "PIP_CHECK_RC=%ERRORLEVEL%"
if not "%PIP_CHECK_RC%"=="0" (
  echo [WARN] pip check reported issues.
  echo        Review: "%LOG_DIR%\pip_check.txt"
  if "%PAUSE_ON_EXIT%"=="1" pause
  exit /b %PIP_CHECK_RC%
)

echo [OK] Venv rebuilt and dependencies verified.
echo      - Freeze: "%LOCK_DIR%\pip_freeze.txt"
echo      - Check : "%LOG_DIR%\pip_check.txt"

if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0
