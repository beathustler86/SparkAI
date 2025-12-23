@echo off
setlocal EnableExtensions EnableDelayedExpansion

title ComfyUI GUI Launcher

REM ==========================================================
REM Paths
REM ==========================================================
set "ROOT=%~dp0"
set "GUI=%ROOT%GUI_Launcher\comfyui_launcher_gui.py"

REM ==========================================================
REM HARD LOCK: Working Python 3.10 VENV
REM ==========================================================
set "CUSTOM_VENV_PATH=D:\SparkAI\Comfy_Cache\venv"
set "PYTHON_EXE=%CUSTOM_VENV_PATH%\Scripts\python.exe"

REM ==========================================================
REM Sanity checks
REM ==========================================================
if not exist "%GUI%" (
  echo [ERROR] Missing GUI script:
  echo "%GUI%"
  pause
  exit /b 1
)

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python executable not found:
  echo "%PYTHON_EXE%"
  pause
  exit /b 1
)

REM ==========================================================
REM Launch GUI (env var overrides internal logic)
REM ==========================================================
echo [INFO] Launching GUI with:
echo %PYTHON_EXE%
echo.

"%PYTHON_EXE%" "%GUI%"
set "code=%ERRORLEVEL%"

if %code% NEQ 0 (
  echo.
  echo [ERROR] GUI exited with code %code%
  pause
  exit /b %code%
)

exit /b 0
