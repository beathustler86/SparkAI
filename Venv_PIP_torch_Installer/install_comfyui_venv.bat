@echo off
setlocal ENABLEEXTENSIONS

REM ==========================================================
REM ComfyUI VENV INSTALLER (EXACT REPRODUCTION)
REM
REM VENV:
REM   D:\SparkAI\Comfy_Cache\venv
REM
REM REQUIREMENTS:
REM   D:\SparkAI\Comfy_Cache\venv_requirements\pip_freeze.txt
REM   D:\SparkAI\Comfy_Cache\venv_requirements\requirements_lock.txt
REM ==========================================================

set VENV_PYTHON=D:\SparkAI\Comfy_Cache\venv\Scripts\python.exe
set REQ_DIR=D:\SparkAI\Comfy_Cache\venv_requirements
set FREEZE_FILE=%REQ_DIR%\pip_freeze.txt
set LOCK_FILE=%REQ_DIR%\requirements_lock.txt

echo.
echo =========================================
echo   ComfyUI VENV INSTALL (DOUBLE-CLICK)
echo =========================================
echo.

REM ----------------------------------------------------------
REM 1. Sanity checks
REM ----------------------------------------------------------
if not exist "%VENV_PYTHON%" (
    echo ERROR: Venv Python not found:
    echo %VENV_PYTHON%
    pause
    exit /b 1
)

if not exist "%REQ_DIR%" (
    echo ERROR: Requirements directory not found:
    echo %REQ_DIR%
    pause
    exit /b 1
)

REM ----------------------------------------------------------
REM 2. Upgrade pip
REM ----------------------------------------------------------
echo === Upgrading pip ===
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto error

REM ----------------------------------------------------------
REM 3. Install dependencies
REM ----------------------------------------------------------
if exist "%FREEZE_FILE%" (
    echo.
    echo === EXACT REPRODUCTION MODE (pip_freeze.txt) ===
    echo Using: %FREEZE_FILE%
   "%VENV_PYTHON%" -m pip install -r "%FREEZE_FILE%" --no-deps --extra-index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 goto error

) else if exist "%LOCK_FILE%" (
    echo.
    echo === PINNED LOCK MODE (requirements_lock.txt) ===
    echo Using: %LOCK_FILE%
    "%VENV_PYTHON%" -m pip install -r "%LOCK_FILE%" --extra-index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 goto error

) else (
    echo ERROR: No requirements file found.
    echo Expected:
    echo   %FREEZE_FILE%
    echo   %LOCK_FILE%
    pause
    exit /b 1
)

REM ----------------------------------------------------------
REM 4. Basic health check
REM ----------------------------------------------------------
echo.
echo === BASIC HEALTH CHECK ===
"%VENV_PYTHON%" -c "import sys, pip; print('Python:', sys.version.split()[0]); print('Executable:', sys.executable); print('pip:', pip.__version__)"

echo.
echo =========================================
echo   INSTALL COMPLETE - SUCCESS
echo =========================================
pause
exit /b 0

:error
echo.
echo =========================================
echo   INSTALL FAILED - SEE ERROR ABOVE
echo =========================================
pause
exit /b 1

"%VENV_PYTHON%" -m pip install antlr4-python3-runtime==4.13.2

& "D:\SparkAI\Comfy_Cache\venv\Scripts\python.exe" -c "import torch, torchvision, torchaudio; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('torchaudio:', torchaudio.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"


