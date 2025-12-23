import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
LOG_DIR = ROOT_DIR / "logs"
TELEMETRY_DIR = ROOT_DIR / "telemetry"
GPU_BENCHMARK_LOG = TELEMETRY_DIR / "comfyui_gpu_benchmark_log.csv"
LEGACY_CHROME_PROFILE = ROOT_DIR / "Conda_ComfyUI_Chromium_Launchers" / "ChromeProfile_ComfyUI"
NEW_CHROME_PROFILE = ROOT_DIR / "utils" / "ChromeProfile_ComfyUI"
CONFIG_PATH = APP_DIR / "launcher_config.json"
GENERATED_DIR = APP_DIR / "generated"
GENERATED_BAT = GENERATED_DIR / "launch_last.bat"

DEFAULTS = {
    "comfy_root": r"N:\ComfyUI",
    "port": 8188,
    "chrome_exe": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "chrome_profile_dir": str(NEW_CHROME_PROFILE),
    "cache_temp": r"D:\COMFY_CACHE\temp",
    "cache_torch_home": r"D:\COMFY_CACHE\torch",
    "lock_dir": r"D:\COMFY_CACHE\venv_requirements",
    "installer_bat": r"D:\COMFY_CACHE\new_install\launch.bat",
    "mode": "standard",
    "env": {},
    "custom_venv_locked": False,
}

GITHUB_DARK = {
    # Darkened base for better contrast; panel is darker to avoid "white" header feel
    "bg": "#0a0e14",
    "panel": "#0f1520",
    "text": "#e1e6ee",
    "muted": "#9ba6b5",
    "accent": "#2f81f7",
    "green": "#3fb950",
    "red": "#f85149",
    "border": "#253040",
}

GITHUB_LIGHT = {
    "bg": "#f7f7fb",
    "panel": "#edeef4",
    "text": "#1c1f26",
    "muted": "#5b6370",
    "accent": "#2f81f7",
    "green": "#1e9e4a",
    "red": "#d43f3a",
    "border": "#c8cdd7",
}


@dataclass
class EnvField:
    key: str
    category: str
    kind: str  # "bool" | "text" | "int" | "choice"
    label: str
    help_text: str
    default: Any = None
    choices: Optional[List[str]] = None


ENV_FIELDS: List[EnvField] = [
    # 1) Performance Settings
    EnvField("CUDA_VISIBLE_DEVICES", "Performance", "text", "CUDA_VISIBLE_DEVICES", "Select GPU ids to use (e.g., 0 or 0,1)", ""),
    EnvField("PYTORCH_CUDA_ALLOC_CONF", "Performance", "text", "PYTORCH_CUDA_ALLOC_CONF", "CUDA alloc config (e.g., max_split_size_mb:256)", ""),
    EnvField("NUM_WORKERS", "Performance", "int", "NUM_WORKERS", "Workers for data loading", ""),
    EnvField("TORCH_CUDNN_BENCHMARK", "Performance", "bool", "TORCH_CUDNN_BENCHMARK", "Enable cuDNN benchmarking", False),
    EnvField("TORCH_DEVICE", "Performance", "choice", "TORCH_DEVICE", "Force CUDA or CPU", "", ["cuda", "cpu"]),
    EnvField("USE_MIXED_PRECISION", "Performance", "bool", "USE_MIXED_PRECISION", "Enable mixed precision", False),
    EnvField("ENABLE_OPTIMIZER_STATE_SHARING", "Performance", "bool", "ENABLE_OPTIMIZER_STATE_SHARING", "Share optimizer state", False),
    EnvField("USE_FUSED_OPTIMIZER", "Performance", "bool", "USE_FUSED_OPTIMIZER", "Use fused optimizers", False),
    EnvField("USE_BFLOAT16", "Performance", "bool", "USE_BFLOAT16", "Enable bfloat16", False),
    EnvField("USE_FP16", "Performance", "bool", "USE_FP16", "Enable FP16", False),

    # 2) Telemetry / Health & Monitoring
    EnvField("DO_NOT_TRACK", "Telemetry", "bool", "DO_NOT_TRACK", "Disable telemetry for HF models/datasets", False),
    EnvField("PIP_CHECK", "Telemetry", "bool", "PIP_CHECK", "Run pip check before launch", True),
    EnvField("DEBUG_MODE", "Telemetry", "bool", "DEBUG_MODE", "Enable debug logging", False),
    EnvField("ENABLE_PROFILER", "Telemetry", "bool", "ENABLE_PROFILER", "Enable profiler", False),
    EnvField("PROFILE_INTERVAL", "Telemetry", "int", "PROFILE_INTERVAL", "Profiler interval", ""),
    EnvField("LOG_LEVEL", "Telemetry", "choice", "LOG_LEVEL", "Verbosity", "", ["", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    EnvField("SYNCHRONOUS_GPU", "Telemetry", "bool", "SYNCHRONOUS_GPU", "Synchronous GPU mode", False),

    # 3) Models & Training
    EnvField("TRANSFORMERS_CACHE", "Models", "text", "TRANSFORMERS_CACHE", "Cache path for HuggingFace transformers", r"D:\COMFY_CACHE\transformers_cache"),
    EnvField("HF_HOME", "Models", "text", "HF_HOME", "HuggingFace home directory", ""),
    EnvField("MODEL_TYPE", "Models", "text", "MODEL_TYPE", "Model type (e.g., GPT, Diffuser)", ""),
    EnvField("TRAINING_BATCH_SIZE", "Models", "int", "TRAINING_BATCH_SIZE", "Training batch size", ""),
    EnvField("MAX_TRAINING_STEPS", "Models", "int", "MAX_TRAINING_STEPS", "Max training steps", ""),
    EnvField("ENABLE_GRADIENT_CHECKPOINTING", "Models", "bool", "ENABLE_GRADIENT_CHECKPOINTING", "Enable gradient checkpointing", False),
    EnvField("USE_XFORMERS", "Models", "bool", "USE_XFORMERS", "Enable xFormers attention", False),

    # 4) Customization & System
    EnvField("CUSTOM_VENV_PATH", "Customization", "text", "CUSTOM_VENV_PATH", "Custom venv path (optional)", ""),
    EnvField("TORCH_HOME", "Customization", "text", "TORCH_HOME", "PyTorch cache directory", ""),
    EnvField("USE_TRITON", "Customization", "bool", "USE_TRITON", "Enable Triton kernels", False),

    # 5) CUDA Tuning
    EnvField("CUDA_MEMORY_FORMAT", "CUDA Tuning", "choice", "CUDA_MEMORY_FORMAT", "CUDA tensor memory format", "", ["", "channels_last", "contiguous"]),
    EnvField("CUDA_LAUNCH_BLOCKING", "CUDA Tuning", "bool", "CUDA_LAUNCH_BLOCKING", "Synchronous CUDA execution", False),
    EnvField("CUDA_DEVICE_MAX_CONNECTIONS", "CUDA Tuning", "int", "CUDA_DEVICE_MAX_CONNECTIONS", "Max concurrent CUDA queues", ""),
    EnvField("PYTORCH_NO_CUDA_MEMORY_CACHING", "CUDA Tuning", "bool", "PYTORCH_NO_CUDA_MEMORY_CACHING", "Disable CUDA memory caching", False),
    EnvField("PYTORCH_NVML_BASED_CUDA_CHECK", "CUDA Tuning", "bool", "PYTORCH_NVML_BASED_CUDA_CHECK", "Use NVML for CUDA check", False),

    # 6) Caches & Misc
    EnvField("CACHE_MODEL_PARAMS", "Caches", "bool", "CACHE_MODEL_PARAMS", "Cache model params", False),
    EnvField("CUSTOM_PIPELINE", "Caches", "text", "CUSTOM_PIPELINE", "Custom pipeline name", ""),
    EnvField("HF_HUB_CACHE", "Caches", "text", "HF_HUB_CACHE", "HF hub repo cache", ""),
    EnvField("HF_XET_CACHE", "Caches", "text", "HF_XET_CACHE", "HF Xet cache", ""),
    EnvField("HF_ASSETS_CACHE", "Caches", "text", "HF_ASSETS_CACHE", "HF downstream assets cache", ""),
    EnvField("HF_DATASETS_CACHE", "Caches", "text", "HF_DATASETS_CACHE", "HF datasets cache", ""),
    EnvField("CUDA_CACHE_DISABLE", "Caches", "bool", "CUDA_CACHE_DISABLE", "Disable PTX JIT cache", False),
    EnvField("CUDA_CACHE_PATH", "Caches", "text", "CUDA_CACHE_PATH", "PTX JIT cache path", ""),
    EnvField("CUDA_CACHE_MAXSIZE", "Caches", "int", "CUDA_CACHE_MAXSIZE", "PTX JIT cache max size (bytes)", ""),
    EnvField("CUDA_FORCE_PTX_JIT", "Caches", "bool", "CUDA_FORCE_PTX_JIT", "Force PTX JIT compilation", False),
    EnvField("CUDA_DISABLE_PTX_JIT", "Caches", "bool", "CUDA_DISABLE_PTX_JIT", "Disable PTX JIT (use embedded)", False),
    EnvField("DIFFUSERS_DISABLE_FP16", "Caches", "bool", "DIFFUSERS_DISABLE_FP16", "Disable FP16 in diffusers", False),
    EnvField("NO_NORMALIZE_INPUTS", "Caches", "bool", "NO_NORMALIZE_INPUTS", "Disable input normalization", False),
    EnvField("USE_ACCURATE_ATTENTION", "Caches", "bool", "USE_ACCURATE_ATTENTION", "Use accurate attention", False),
    EnvField("USE_COMPRESSION", "Caches", "bool", "USE_COMPRESSION", "Enable compression", False),
    EnvField("GENERATE_TRAINING_DATA", "Caches", "bool", "GENERATE_TRAINING_DATA", "Auto-generate training data", False),
    EnvField("TEXT_EMBEDDING_MODE", "Caches", "text", "TEXT_EMBEDDING_MODE", "Embedding mode (if used)", ""),
    EnvField("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "Caches", "bool", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "Allow TF32 in cuBLAS", False),
    EnvField("TORCH_DTYPE", "Caches", "choice", "TORCH_DTYPE", "Force dtype", "", ["", "float32", "float16", "bfloat16"]),
]


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            merged = {**DEFAULTS, **data}
            merged.setdefault("env", {})
            # Migrate legacy Chrome profile path to new utils location
            legacy_profile = str(LEGACY_CHROME_PROFILE)
            new_profile = str(NEW_CHROME_PROFILE)
            if str(merged.get("chrome_profile_dir", "")).strip() == legacy_profile:
                merged["chrome_profile_dir"] = new_profile
            return merged
        except Exception:
            return dict(DEFAULTS)
    return dict(DEFAULTS)


def save_config(cfg: Dict[str, Any]) -> None:
    cfg_to_save = dict(cfg)
    CONFIG_PATH.write_text(json.dumps(cfg_to_save, indent=2), encoding="utf-8")


def _bat_quote(value: str) -> str:
    value = value.replace('"', '""')
    return f'"{value}"'


def build_launch_bat(cfg: Dict[str, Any]) -> str:
    comfy_root = cfg["comfy_root"]
    port = int(cfg.get("port", 8188))
    chrome_exe = cfg["chrome_exe"]
    chrome_profile_dir = cfg["chrome_profile_dir"]
    cache_temp = cfg["cache_temp"]
    cache_torch_home = cfg["cache_torch_home"]
    lock_dir_value = cfg.get("lock_dir") or DEFAULTS["lock_dir"]
    lock_dir = str(Path(lock_dir_value))
    requirements_lock = str(Path(lock_dir) / "requirements_lock.txt")
    requirements_txt = str(Path(lock_dir) / "requirements.txt")
    installer_bat = cfg.get("installer_bat", DEFAULTS["installer_bat"])

    mode = cfg.get("mode", "xformers")

    custom_locked = bool(cfg.get("custom_venv_locked", DEFAULTS["custom_venv_locked"]))
    custom_venv_raw = cfg.get("env", {}).get("CUSTOM_VENV_PATH")
    custom_venv = str(custom_venv_raw or "").strip().strip('"').strip("'")

    if custom_locked:
        if not custom_venv:
            raise ValueError("CUSTOM_VENV_PATH is locked but empty")
        if any(ch in custom_venv for ch in ["&", "|", "<", ">"]):
            raise ValueError("CUSTOM_VENV_PATH contains unsupported characters (& | < >)")

    venv_dir = custom_venv if (custom_locked and custom_venv) else str(Path(comfy_root) / "venv")

    comfy_args: List[str] = []
    # ComfyUI no longer accepts --xformers; absence of --disable-xformers lets xformers load if available.
    if mode == "standard":
        comfy_args = []
    elif mode == "safe":
        comfy_args = ["--disable-xformers"]
    elif mode == "benchmark":
        comfy_args = ["--disable-xformers", "--benchmark"]
    elif mode == "high_performance":
        # Stub for future tuning
        comfy_args = []
    else:
        comfy_args = []

    env_vars: Dict[str, str] = {}

    # Always set cache + isolation (matches existing venv launchers)
    env_vars["TEMP"] = cache_temp
    env_vars["TMP"] = cache_temp
    env_vars["TORCH_HOME"] = cache_torch_home
    env_vars["PYTHONNOUSERSITE"] = "1"
    env_vars["PYTHONPATH"] = ""
    env_vars["PYTHONHOME"] = ""

    # Apply user-selected env vars (whitelisted by UI)
    for k, v in cfg.get("env", {}).items():
        if k == "CUSTOM_VENV_PATH":
            continue
        if k == "ENABLE_BACKDOOR_LLM":
            continue
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                env_vars[k] = "1"
            continue
        v_str = str(v).strip()
        if v_str == "":
            continue
        env_vars[k] = v_str

    # Map USE_XFORMERS to legacy disable flag expected by ComfyUI
    use_xf = bool(cfg.get("env", {}).get("USE_XFORMERS", False))
    if not use_xf:
        env_vars["XFORMERS_DISABLE"] = "1"
    else:
        env_vars.pop("XFORMERS_DISABLE", None)

    # chrome app mode: no address bar
    chrome_args = [
        f"--user-data-dir={_bat_quote(chrome_profile_dir)}",
        f"--app=http://127.0.0.1:{port}",
        "--start-fullscreen",
        "--kiosk",
    ]

    lines: List[str] = []
    lines.append("@echo off")
    lines.append("setlocal EnableExtensions")
    lines.append("title ComfyUI [GUI Launcher]")
    lines.append("")
    lines.append(f"set \"COMFY_ROOT={comfy_root}\"")
    if custom_locked and custom_venv:
        lines.append(f"set \"CUSTOM_VENV_PATH={custom_venv}\"")
    lines.append(f"set \"VENV_DIR={venv_dir}\"")
    lines.append("set \"VENV_PY=%VENV_DIR%\\Scripts\\python.exe\"")
    lines.append("set \"VENV_ACTIVATE=%VENV_DIR%\\Scripts\\Activate.ps1\"")
    lines.append(f"set \"PORT={port}\"")
    lines.append(f"set \"REQ_LOCK={requirements_lock}\"")
    lines.append(f"set \"REQ_TXT={requirements_txt}\"")
    lines.append(f"set \"INSTALLER_BAT={installer_bat}\"")
    lines.append("")

    # Pre-flight checks for Comfy root and main.py
    lines.append("if not exist %COMFY_ROOT% (")
    lines.append("  echo [ERROR] Missing ComfyUI root: %COMFY_ROOT%")
    lines.append("  echo         Update 'Comfy Root' in GUI or install ComfyUI")
    lines.append("  exit /b 1")
    lines.append(")")
    lines.append("if not exist %COMFY_ROOT%\\main.py (")
    lines.append("  echo [ERROR] Missing ComfyUI main.py in %COMFY_ROOT%")
    lines.append("  echo         Ensure ComfyUI is cloned and up to date")
    lines.append("  exit /b 1")
    lines.append(")")
    lines.append("")
    lines.append("cd /d %COMFY_ROOT%")
    lines.append("")

    for k, v in env_vars.items():
        lines.append(f"set \"{k}={v}\"")

    lines.append("")

    # Verify interpreter version before background start (fail fast)
    lines.append("%VENV_PY% -c \"import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)\"")
    lines.append("if errorlevel 1 (")
    lines.append("  echo [ERROR] Python interpreter is not 3.10: %VENV_PY%")
    lines.append("  echo         Rebuild venv via D:\\COMFY_CACHE\\new_install\\launch.bat or point CUSTOM_VENV_PATH to 3.10")
    lines.append("  exit /b 1")
    lines.append(")")
    lines.append("")
    lines.append("if not exist %VENV_PY% (")
    lines.append("  echo [ERROR] Missing venv python: %VENV_PY%")
    lines.append("  echo         Rebuild it via D:\\COMFY_CACHE\\new_install\\launch.bat")
    lines.append("  exit /b 1")
    lines.append(")")
    lines.append("")

    lines.append("set \"NEED_REPAIR=0\"")
    lines.append("%VENV_PY% -c \"import importlib.util as u, sys; sys.exit(0 if u.find_spec('torch') else 1)\"")
    lines.append("if errorlevel 1 set NEED_REPAIR=1")
    lines.append("%VENV_PY% -c \"import importlib.util as u, sys; sys.exit(0 if u.find_spec('aiohttp') else 1)\"")
    lines.append("if errorlevel 1 set NEED_REPAIR=1")
    lines.append("if \"%NEED_REPAIR%\" NEQ \"0\" (")
    lines.append("  echo [Repair] Missing torch/aiohttp detected. Rehydrating virtual environment...")
    lines.append("  if exist \"%REQ_LOCK%\" (")
    lines.append("    echo [Repair] Installing pinned stack from %REQ_LOCK%")
    lines.append("    %VENV_PY% -m pip install --no-input --upgrade -r \"%REQ_LOCK%\"")
    lines.append("  ) else if exist \"%REQ_TXT%\" (")
    lines.append("    echo [Repair] Installing stack from %REQ_TXT%")
    lines.append("    %VENV_PY% -m pip install --no-input --upgrade -r \"%REQ_TXT%\"")
    lines.append("  ) else (")
    lines.append("    echo [ERROR] requirements files missing; run %INSTALLER_BAT% manually.")
    lines.append("    exit /b 1")
    lines.append("  )")
    lines.append("  if errorlevel 1 (")
    lines.append("    echo [ERROR] Automatic dependency repair failed. Run %INSTALLER_BAT% manually.")
    lines.append("    exit /b 1")
    lines.append("  )")
    lines.append(")")
    lines.append("")

    # Optional: pip check before launch
    pip_check_enabled = bool(cfg.get("env", {}).get("PIP_CHECK", True))
    if pip_check_enabled:
        lines.append("echo [Health] Running pip check...")
        lines.append("%VENV_PY% -m pip check")
        lines.append("if errorlevel 1 (")
        lines.append("  echo [WARN] pip check reported issues.")
        lines.append("  echo        Fix stack or rebuild via D:\\COMFY_CACHE\\new_install\\launch.bat")
        lines.append("  pause")
        lines.append("  exit /b 1")
        lines.append(")")
        lines.append("")

    # Ensure telemetry directory exists (used by benchmark mode)
    try:
        TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Start ComfyUI
    if mode == "benchmark":
        lines.append(
            "start \"NVIDIA Monitor\" cmd /c \"nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,temperature.gpu,memory.total,memory.used,memory.free --format=csv -l 1 > "
            + _bat_quote(str(GPU_BENCHMARK_LOG))
            + "\""
        )
        lines.append("")

    comfy_args_str = " ".join(comfy_args)
    # Run in background via PowerShell; validate interpreter is 3.10.
    # Use $env: variables to avoid cmd.exe percent-expansion inside the -Command string.
    py_ver_check = (
        "import sys; "
        "raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)"
    )
    # Ensure log directory exists before launch so Tee-Object can write
    lines.append(f"if not exist {_bat_quote(str(LOG_DIR))} mkdir {_bat_quote(str(LOG_DIR))}")
    lines.append("")

    # Capture server output to log for debugging
    server_log = str(LOG_DIR / "comfyui_server.log")
    ps_cmd = (
        "& { "
        "$ErrorActionPreference='Stop'; "
        "Set-Location $env:COMFY_ROOT; "
        "$venvPy=$env:VENV_PY; $activate=$env:VENV_ACTIVATE; "
        "if (Test-Path $activate) { . $activate }; "
        f"& $venvPy -c '{py_ver_check}'; "
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; "
        "$ErrorActionPreference='Continue'; "
        f"& $venvPy main.py {comfy_args_str} 2>&1 | Tee-Object -File '{server_log}'; "
        "$last=$LASTEXITCODE; "
        "exit $last "
        "}"
    )
    lines.append(
        "start \"ComfyUI\" /b powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "
        + _bat_quote(ps_cmd)
    )
    lines.append("")

    # Wait loop with timeout to avoid hanging forever
    lines.append("setlocal EnableDelayedExpansion")
    lines.append("set /a __WAITED=0")
    lines.append("set /a __TIMEOUT=120")
    lines.append(":waitloop")
    lines.append("timeout /t 2 >nul")
    lines.append("set /a __WAITED+=2")
    lines.append(
        "powershell.exe -NoLogo -NoProfile -Command "
        + _bat_quote(
            "try { $null = Invoke-WebRequest http://127.0.0.1:%PORT% -UseBasicParsing; exit 0 } catch { exit 1 }"
        )
    )
    lines.append("if errorlevel 1 if !__WAITED! LSS !__TIMEOUT! goto waitloop")
    lines.append("if errorlevel 1 (")
    lines.append("  echo [ERROR] ComfyUI did not start within !__TIMEOUT! seconds.")
    lines.append(f"  echo         See server log: {server_log}")
    lines.append(
        "  powershell.exe -NoLogo -NoProfile -Command "
        + _bat_quote(f"if (Test-Path '{server_log}') {{ Get-Content -Tail 80 -Path '{server_log}' }} else {{ Write-Host 'No server log found.' }}")
    )
    lines.append("  exit /b 1")
    lines.append(")")
    lines.append("")

    # Launch Chrome
    chrome_line = "start \"\" " + _bat_quote(chrome_exe) + " " + " ".join(chrome_args)
    lines.append(chrome_line)
    lines.append("")
    lines.append("exit /b 0")
    lines.append("")

    return "\r\n".join(lines)


class ScrollFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.canvas = tk.Canvas(
            self,
            highlightthickness=0,
            bd=0,
            bg=GITHUB_DARK["bg"],
        )
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        # Use tk.Frame for reliable background coloring inside a Canvas.
        self.inner = tk.Frame(self.canvas, bg=GITHUB_DARK["bg"], bd=0)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self._window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # Stretch inner frame to canvas width for cleaner resizing
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self._window_id, width=e.width),
        )
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event: tk.Event):
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass


class LauncherGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ComfyUI Launcher")
        # Larger default viewport so bottom controls remain visible
        self.geometry("1920x1080")
        self.minsize(1280, 880)

        self.cfg = load_config()
        self._palette = GITHUB_DARK
        self._theme_var = tk.StringVar(value="dark")
        self._vars: Dict[str, tk.Variable] = {}
        self._status_var = tk.StringVar(value="Ready")
        # Used by Models tab to show Transformers cache stats
        self._transformers_cache_info_var = tk.StringVar(value="Transformers cache: (pending)")
        self._current_python = sys.executable
        self._actions_frame: Optional[ttk.Frame] = None
        self._relaunch_bar: Optional[ttk.Frame] = None
        self._relaunch_lbl: Optional[ttk.Label] = None
        self._rebuild_proc: Optional[subprocess.Popen] = None
        self._toggle_buttons: Dict[str, ttk.Button] = {}
        self._tooltips: Dict[str, tk.Toplevel] = {}

        self._setup_style()
        self._build_ui()
        self._load_cfg_into_vars()
        self._refresh_active_envs()

        # Prevent background threads from crashing the app if the window closes
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        # Best-effort: don't hard-kill an installer run, but avoid Tk calls after destroy.
        try:
            self.destroy()
        except Exception:
            pass

    def _safe_after(self, delay_ms: int, func, *args):
        try:
            return self.after(delay_ms, func, *args)
        except Exception:
            return None

    def _switch_theme(self):
        mode = self._theme_var.get().strip().lower()
        self._palette = GITHUB_LIGHT if mode == "light" else GITHUB_DARK
        self._setup_style()

    def _toggle_bool(self, key: str):
        var = self._vars.get(key)
        if var is None:
            return
        try:
            var.set(not bool(var.get()))
        except Exception:
            var.set(False)
        self._sync_toggle_button(key)
        self._on_any_change()

    def _sync_toggle_button(self, key: str):
        btn = self._toggle_buttons.get(key)
        var = self._vars.get(key)
        if btn is None or var is None:
            return
        val = bool(var.get())
        btn.configure(text=("On" if val else "Off"), style=("ToggleOn.TButton" if val else "ToggleOff.TButton"))

    def _make_info_icon(self, parent: tk.Widget, text: str):
        icon = tk.Label(parent, text="ⓘ", fg=self._palette["accent"], bg=self._palette["bg"], cursor="question_arrow")

        def show_tip(_event=None):
            if not text:
                return
            if icon in self._tooltips:
                return
            tip = tk.Toplevel(icon)
            tip.wm_overrideredirect(True)
            tip.configure(bg=self._palette["panel"], padx=8, pady=6)
            lbl = tk.Label(tip, text=text, justify="left", background=self._palette["panel"], foreground=self._palette["text"], wraplength=320)
            lbl.pack()
            x = icon.winfo_rootx() + 16
            y = icon.winfo_rooty() + 20
            tip.wm_geometry(f"+{x}+{y}")
            self._tooltips[icon] = tip

        def hide_tip(_event=None):
            tip = self._tooltips.pop(icon, None)
            if tip:
                tip.destroy()

        icon.bind("<Enter>", show_tip)
        icon.bind("<Leave>", hide_tip)
        return icon

    def _setup_style(self):
        palette = getattr(self, "_palette", GITHUB_DARK)
        self.configure(bg=palette["bg"])

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Softer, higher-contrast control palette
        entry_bg = "#1b2330" if palette is GITHUB_DARK else "#f6f6fb"
        entry_fg = "#dfe6f2" if palette is GITHUB_DARK else "#1c1f26"
        entry_sel = palette["accent"]
        cream = "#f2e9d8" if palette is GITHUB_DARK else "#ebe2cf"
        cream_active = "#e8dec7" if palette is GITHUB_DARK else "#e1d6bd"
        cream_border = "#d6ccb7" if palette is GITHUB_DARK else "#c4b89d"

        style.configure("TFrame", background=palette["bg"])
        style.configure("Panel.TFrame", background=palette["panel"])
        style.configure("TLabel", background=palette["bg"], foreground=palette["text"])
        style.configure("Muted.TLabel", background=palette["bg"], foreground=palette["muted"])
        style.configure(
            "TButton",
            padding=8,
            background=cream,
            foreground="#1c1c1c",
            bordercolor=cream_border,
            focusthickness=2,
        )
        style.map(
            "TButton",
            background=[("active", cream_active)],
            bordercolor=[("active", cream_border)],
        )
        style.configure(
            "Accent.TButton",
            padding=8,
            foreground="#1c1c1c",
            background=cream,
            bordercolor=cream_border,
            focusthickness=2,
        )
        style.map("Accent.TButton", background=[("active", cream_active)], bordercolor=[("active", cream_border)])
        style.configure(
            "Success.TButton",
            padding=8,
            foreground="#0f2310",
            background="#d5f0d1",
            bordercolor="#9acb92",
            focusthickness=2,
        )
        style.map("Success.TButton", background=[("active", "#c4e7be")], bordercolor=[("active", "#89bc82")])

        style.configure("TNotebook", background=palette["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=palette["panel"], foreground=palette["text"], padding=(14, 8))
        style.map(
            "TNotebook.Tab",
            background=[("selected", palette["bg"])]
        )

        style.configure(
            "TEntry",
            fieldbackground=entry_bg,
            foreground=entry_fg,
            insertcolor=entry_fg,
            selectbackground=entry_sel,
            bordercolor=palette["border"],
            lightcolor="#1f2836",
            darkcolor="#111723",
        )
        style.configure(
            "TCombobox",
            fieldbackground=entry_bg,
            foreground=entry_fg,
            insertcolor=entry_fg,
            selectbackground=entry_sel,
            bordercolor=palette["border"],
            lightcolor="#1f2836",
            darkcolor="#111723",
        )
        # Make text widgets and entries more legible with bold, high-contrast cursor/text
        self.option_add("*TEntry.font", ("Segoe UI", 10, "bold"))
        self.option_add("*TEntry.insertBackground", entry_fg)
        self.option_add("*TCombobox.font", ("Segoe UI", 10, "bold"))

        # Toggle styles for green/red buttons
        style.configure(
            "ToggleOn.TButton",
            padding=8,
            background=palette["green"],
            foreground="#0c140c",
            bordercolor=palette["green"],
            focusthickness=2,
        )
        style.map("ToggleOn.TButton", background=[("active", palette["green"])])
        style.configure(
            "ToggleOff.TButton",
            padding=8,
            background=palette["red"],
            foreground="#0c0c0c",
            bordercolor=palette["red"],
            focusthickness=2,
        )
        style.map("ToggleOff.TButton", background=[("active", palette["red"])])

    def report_callback_exception(self, exc, val, tb):
        # Called by Tkinter on exceptions raised in Tk callbacks.
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            crash_log = LOG_DIR / "gui_crash.log"
            crash_log.write_text(
                "".join(traceback.format_exception(exc, val, tb)),
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            pass
        try:
            messagebox.showerror("GUI Error", f"An unexpected error occurred.\n\nDetails were written to:\n{crash_log}")
        except Exception:
            pass

    def _build_ui(self):
        top = ttk.Frame(self, style="Panel.TFrame")
        top.pack(side="top", fill="x", padx=12, pady=(12, 8))

        title = ttk.Label(top, text="ComfyUI Launcher", font=("Segoe UI", 16, "bold"), background=GITHUB_DARK["panel"], foreground=GITHUB_DARK["text"])
        title.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 2))

        subtitle = ttk.Label(top, text="Configure env vars for the next run, then launch in Chrome app-mode.", style="Muted.TLabel")
        subtitle.configure(background=GITHUB_DARK["panel"])
        subtitle.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 10))

        # Theme selector
        theme_row = ttk.Frame(top, style="Panel.TFrame")
        theme_row.grid(row=0, column=1, sticky="e", padx=10, pady=(10, 2))
        ttk.Label(theme_row, text="Theme:").pack(side="left", padx=(0, 6))
        theme_combo = ttk.Combobox(theme_row, width=10, state="readonly", textvariable=self._theme_var, values=["dark", "light"])
        theme_combo.pack(side="left")
        theme_combo.bind("<<ComboboxSelected>>", lambda e: self._switch_theme())

        # Mode row
        mode_row = ttk.Frame(self)
        mode_row.pack(side="top", fill="x", padx=12, pady=(0, 8))

        ttk.Label(mode_row, text="Launch mode:").pack(side="left")
        self._vars["mode"] = tk.StringVar()
        mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self._vars["mode"],
            values=[
                "standard",
                "safe",
                "benchmark",
                "high_performance",
            ],
            state="readonly",
            width=18,
        )
        mode_combo.pack(side="left", padx=(8, 16))
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self._on_any_change())

        ttk.Label(mode_row, text="Port:").pack(side="left")
        self._vars["port"] = tk.StringVar()
        port_entry = ttk.Entry(mode_row, textvariable=self._vars["port"], width=8)
        port_entry.pack(side="left", padx=(8, 16))
        port_entry.bind("<KeyRelease>", lambda e: self._on_any_change())

        # Tabs
        notebook = ttk.Notebook(self)
        notebook.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 10))

        self._tab_frames: Dict[str, ScrollFrame] = {}
        for cat in ["Performance", "Telemetry", "Models", "Customization", "CUDA Tuning", "Caches"]:
            sf = ScrollFrame(notebook)
            self._tab_frames[cat] = sf
            notebook.add(sf, text=cat)

        self._build_fields()

        # Bottom active envs + actions
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="both", expand=True, padx=12, pady=(0, 12))

        active_label = ttk.Label(bottom, text="Active env vars for next launch:")
        active_label.pack(side="top", anchor="w")

        self.active_text = tk.Text(
            bottom,
            height=6,
            bg=GITHUB_DARK["panel"],
            fg=GITHUB_DARK["text"],
            insertbackground=GITHUB_DARK["text"],
            relief="solid",
            bd=1,
        )
        self.active_text.pack(side="top", fill="x", expand=False, pady=(6, 10))
        self.active_text.tag_configure("active", foreground=GITHUB_DARK["green"])
        self.active_text.tag_configure("muted", foreground=GITHUB_DARK["muted"])

        console_label = ttk.Label(bottom, text="Console (rebuild / checks):")
        console_label.pack(side="top", anchor="w")

        console_frame = ttk.Frame(bottom)
        console_frame.pack(side="top", fill="both", expand=True, pady=(6, 10))

        self.console_text = tk.Text(
            console_frame,
            height=8,
            bg=GITHUB_DARK["panel"],
            fg=GITHUB_DARK["text"],
            insertbackground=GITHUB_DARK["text"],
            relief="solid",
            bd=1,
            wrap="none",
        )
        self.console_text.pack(side="left", fill="both", expand=True)
        self.console_text.tag_configure("ok", foreground=GITHUB_DARK["green"])
        self.console_text.tag_configure("err", foreground=GITHUB_DARK["red"])
        self.console_text.tag_configure("muted", foreground=GITHUB_DARK["muted"])

        sb = ttk.Scrollbar(console_frame, orient="vertical", command=self.console_text.yview)
        sb.pack(side="right", fill="y")
        self.console_text.configure(yscrollcommand=sb.set)

        actions = ttk.Frame(bottom)
        actions.pack(side="bottom", fill="x")
        self._actions_frame = actions

        launch_btn = ttk.Button(actions, text="Launch", style="Accent.TButton", command=self._launch)
        launch_btn.pack(side="left")

        clear_btn = ttk.Button(actions, text="Clear Envs", command=self._clear_envs)
        clear_btn.pack(side="left", padx=(8, 0))

        save_btn = ttk.Button(actions, text="Save", command=self._save)
        save_btn.pack(side="left", padx=(8, 0))

        health_btn = ttk.Button(actions, text="Health / Stack Check", command=self._health_check)
        health_btn.pack(side="left", padx=(8, 0))

        rebuild_btn = ttk.Button(actions, text="Rebuild / Refresh Venv", command=self._rebuild_venv)
        rebuild_btn.pack(side="left", padx=(8, 0))

        open_locks_btn = ttk.Button(actions, text="Open Lock Folder", command=self._open_lock_folder)
        open_locks_btn.pack(side="left", padx=(8, 0))

        open_script_btn = ttk.Button(actions, text="Open Generated Script", command=self._open_generated)
        open_script_btn.pack(side="left", padx=(8, 0))

        clear_console_btn = ttk.Button(actions, text="Clear Console", command=self._clear_console)
        clear_console_btn.pack(side="left", padx=(8, 0))

        # Optional packages selector for test venv
        opt_frame = ttk.Frame(actions)
        opt_frame.pack(side="left", padx=(12, 0))

        ttk.Label(opt_frame, text="Optional pkg (Test venv):").pack(side="left")
        self._opt_pkg_var = tk.StringVar()
        self._opt_pkg_combo = ttk.Combobox(opt_frame, textvariable=self._opt_pkg_var, state="readonly", width=34)
        self._opt_pkg_combo.pack(side="left", padx=(6, 6))
        self._opt_pkg_combo.bind("<<ComboboxSelected>>", lambda e: self._on_optional_pkg_select())
        self._opt_pkg_combo.configure(postcommand=self._refresh_optional_packages_combo)

        self._opt_install_btn = ttk.Button(opt_frame, text="Install", style="Accent.TButton", command=self._install_selected_optional_package)
        self._opt_install_btn.pack(side="left")
        self._optional_pkg_map: Dict[str, str] = {}
        self._refresh_optional_packages_combo()

        status = ttk.Label(actions, textvariable=self._status_var, style="Muted.TLabel")
        status.pack(side="right")

        # Hidden warning bar: shows if GUI is running inside the target venv
        self._relaunch_bar = ttk.Frame(actions)
        self._relaunch_lbl = ttk.Label(self._relaunch_bar, text="", style="Muted.TLabel")
        self._relaunch_lbl.pack(side="left")
        ttk.Button(self._relaunch_bar, text="Relaunch GUI (py 3.10)", command=self._relaunch_with_system_python).pack(side="left", padx=(8, 0))

        self._buttons_to_disable: List[ttk.Button] = [
            launch_btn,
            clear_btn,
            save_btn,
            health_btn,
            rebuild_btn,
            open_locks_btn,
            open_script_btn,
            clear_console_btn,
            self._opt_install_btn,
        ]

    def _build_fields(self):
        # Layout per tab: rows with label + widget + info icon
        cats: Dict[str, List[EnvField]] = {}
        for f in ENV_FIELDS:
            cats.setdefault(f.category, []).append(f)

        for cat, fields in cats.items():
            frame = self._tab_frames[cat].inner
            for row, f in enumerate(fields):
                lbl = ttk.Label(frame, text=f.label)
                lbl.grid(row=row, column=0, sticky="w", padx=(12, 6), pady=6)

                widget: tk.Widget
                if f.kind == "bool":
                    var = tk.BooleanVar()
                    self._vars[f.key] = var
                    btn = ttk.Button(frame, text="Off", command=lambda k=f.key: self._toggle_bool(k), width=8)
                    btn.grid(row=row, column=1, sticky="w", padx=(0, 8))
                    self._toggle_buttons[f.key] = btn
                    self._sync_toggle_button(f.key)
                    widget = btn
                elif f.kind == "int":
                    var = tk.StringVar()
                    self._vars[f.key] = var
                    ent = ttk.Entry(frame, textvariable=var, width=24)
                    ent.grid(row=row, column=1, sticky="w", padx=(0, 8))
                    ent.bind("<KeyRelease>", lambda e: self._on_any_change())
                    widget = ent
                elif f.kind == "choice":
                    var = tk.StringVar()
                    self._vars[f.key] = var
                    combo = ttk.Combobox(frame, textvariable=var, values=f.choices or [], state="readonly", width=22)
                    combo.grid(row=row, column=1, sticky="w", padx=(0, 8))
                    combo.bind("<<ComboboxSelected>>", lambda e: self._on_any_change())
                    widget = combo
                else:
                    var = tk.StringVar()
                    self._vars[f.key] = var
                    if f.key == "CUSTOM_VENV_PATH":
                        container = ttk.Frame(frame)
                        container.grid(row=row, column=1, sticky="w", padx=(0, 8))

                        ent = ttk.Entry(container, textvariable=var, width=44)
                        ent.pack(side="left")
                        ent.bind("<KeyRelease>", lambda e: self._on_any_change())
                        self._custom_venv_path_entry = ent

                        browse_btn = ttk.Button(container, text="Browse...", command=self._browse_custom_venv)
                        browse_btn.pack(side="left", padx=(8, 0))
                        self._custom_venv_browse_btn = browse_btn

                        self._vars["custom_venv_locked"] = tk.BooleanVar(value=bool(self.cfg.get("custom_venv_locked", DEFAULTS["custom_venv_locked"])))
                        lock_btn = ttk.Button(
                            container,
                            text="Lock",
                            command=lambda: self._toggle_custom_lock(),
                            width=6,
                        )
                        lock_btn.pack(side="left", padx=(8, 0))
                        widget = container
                    else:
                        ent = ttk.Entry(frame, textvariable=var, width=44)
                        ent.grid(row=row, column=1, sticky="w", padx=(0, 8))
                        ent.bind("<KeyRelease>", lambda e: self._on_any_change())
                        widget = ent

                info = self._make_info_icon(frame, f.help_text)
                if info is not None:
                    info.grid(row=row, column=2, sticky="w", padx=(0, 6))

            frame.grid_columnconfigure(3, weight=1)

        # Models tab: transformers cache stats + quick open
        models_frame = self._tab_frames["Models"].inner
        stats_lbl = ttk.Label(models_frame, textvariable=self._transformers_cache_info_var, style="Muted.TLabel")
        stats_lbl.grid(row=len(cats.get("Models", [])) + 1, column=0, columnspan=3, sticky="w", padx=10, pady=(14, 4))

        open_btn = ttk.Button(models_frame, text="Open Transformers Cache Folder", command=self._open_transformers_cache)
        open_btn.grid(row=len(cats.get("Models", [])) + 2, column=0, sticky="w", padx=10, pady=(0, 8))

        # Caches tab: quick open/clear actions
        caches_frame = self._tab_frames.get("Caches")
        if caches_frame:
            base_row = len(cats.get("Caches", [])) + 2
            ttk.Button(caches_frame.inner, text="Open Torch Cache (TORCH_HOME)", command=self._open_torch_cache).grid(row=base_row, column=0, sticky="w", padx=10, pady=(10, 4))
            ttk.Button(caches_frame.inner, text="Clear Torch Cache", command=self._clear_torch_cache).grid(row=base_row, column=1, sticky="w", padx=10, pady=(10, 4))
            ttk.Button(caches_frame.inner, text="Open Transformers Cache", command=self._open_transformers_cache).grid(row=base_row + 1, column=0, sticky="w", padx=10, pady=(4, 4))
            ttk.Button(caches_frame.inner, text="Clear Transformers Cache", command=self._clear_transformers_cache).grid(row=base_row + 1, column=1, sticky="w", padx=10, pady=(4, 4))

    def _load_cfg_into_vars(self):
        self._vars["mode"].set(self.cfg.get("mode", DEFAULTS["mode"]))
        self._vars["port"].set(str(self.cfg.get("port", DEFAULTS["port"])))

        env = self.cfg.get("env", {})
        for f in ENV_FIELDS:
            if f.key not in self._vars:
                continue
            v = env.get(f.key, f.default)
            if f.key == "TRANSFORMERS_CACHE" and (v is None or str(v).strip() == ""):
                v = f.default
            if f.kind == "bool":
                self._vars[f.key].set(bool(v))
                self._sync_toggle_button(f.key)
            else:
                self._vars[f.key].set("" if v is None else str(v))

        if "custom_venv_locked" in self._vars:
            try:
                self._vars["custom_venv_locked"].set(bool(self.cfg.get("custom_venv_locked", DEFAULTS["custom_venv_locked"])))
            except Exception:
                pass
        self._apply_custom_venv_lock_state()
        self._refresh_transformers_cache_info_async()

    def _vars_to_cfg(self) -> Dict[str, Any]:
        cfg = dict(self.cfg)
        cfg["mode"] = self._vars["mode"].get().strip() or DEFAULTS["mode"]

        try:
            cfg["port"] = int(self._vars["port"].get().strip())
        except Exception:
            cfg["port"] = DEFAULTS["port"]

        env: Dict[str, Any] = {}
        for f in ENV_FIELDS:
            var = self._vars.get(f.key)
            if var is None:
                continue
            if f.kind == "bool":
                env[f.key] = bool(var.get())
            elif f.kind == "int":
                s = str(var.get()).strip()
                if s == "":
                    env[f.key] = ""
                else:
                    # keep as string to avoid breaking if user types non-int; BAT will receive it
                    env[f.key] = s
            else:
                env[f.key] = str(var.get()).strip()

        # Explicitly strip excluded key if it ever appears
        env.pop("ENABLE_BACKDOOR_LLM", None)

        cfg["env"] = env
        if "custom_venv_locked" in self._vars:
            try:
                cfg["custom_venv_locked"] = bool(self._vars["custom_venv_locked"].get())
            except Exception:
                cfg["custom_venv_locked"] = DEFAULTS["custom_venv_locked"]
        return cfg

    def _refresh_active_envs(self):
        cfg = self._vars_to_cfg()
        self.active_text.configure(state="normal")
        self.active_text.delete("1.0", "end")

        self.active_text.insert("end", f"Mode: {cfg['mode']}\n", ("muted",))
        self.active_text.insert("end", f"Port: {cfg['port']}\n\n", ("muted",))

        self.active_text.insert("end", f"Locks: {cfg.get('lock_dir', DEFAULTS['lock_dir'])}\n", ("muted",))
        self.active_text.insert("end", f"Installer: {cfg.get('installer_bat', DEFAULTS['installer_bat'])}\n\n", ("muted",))

        env = cfg.get("env", {})
        active_items: List[Tuple[str, str]] = []
        for k, v in env.items():
            if k == "CUSTOM_VENV_PATH":
                if str(v).strip():
                    active_items.append((k, str(v).strip()))
                continue
            if isinstance(v, bool):
                if v:
                    active_items.append((k, "1"))
                continue
            v_str = str(v).strip()
            if v_str:
                active_items.append((k, v_str))

        if not active_items:
            self.active_text.insert("end", "(No env vars selected)\n", ("muted",))
        else:
            for k, v in sorted(active_items):
                self.active_text.insert("end", f"{k}={v}\n", ("active",))

        self.active_text.configure(state="disabled")

    def _on_any_change(self):
        self._refresh_active_envs()
        self._update_relaunch_warning()
        self._refresh_transformers_cache_info_async()
        self._update_relaunch_warning()

    def _browse_custom_venv(self):
        chosen = filedialog.askdirectory(title="Select venv folder (will be created/replaced)")
        if not chosen:
            return
        self._vars["CUSTOM_VENV_PATH"].set(chosen)
        # Auto-lock after browsing (requested)
        if "custom_venv_locked" in self._vars:
            try:
                self._vars["custom_venv_locked"].set(True)
            except Exception:
                pass
        self._apply_custom_venv_lock_state()
        self._on_any_change()

    def _on_custom_venv_lock_change(self):
        self._apply_custom_venv_lock_state()
        self._on_any_change()

    def _toggle_custom_lock(self):
        var = self._vars.get("custom_venv_locked")
        if var is not None:
            try:
                var.set(not bool(var.get()))
            except Exception:
                var.set(False)
        self._on_custom_venv_lock_change()

    def _target_venv_dir(self) -> str:
        cfg = self._vars_to_cfg()
        custom_locked = bool(cfg.get("custom_venv_locked", DEFAULTS["custom_venv_locked"]))
        custom_venv = str(cfg.get("env", {}).get("CUSTOM_VENV_PATH") or "").strip().strip('"').strip("'")
        if os.name == "nt" and "/" in custom_venv:
            custom_venv = custom_venv.replace("/", "\\")
        if custom_locked and custom_venv:
            return custom_venv
        return str(Path(cfg["comfy_root"]) / "venv")

    def _update_relaunch_warning(self):
        try:
            if not self._actions_frame or not self._relaunch_bar:
                return
            venv_dir = self._target_venv_dir()
            venv_py = str(Path(venv_dir) / "Scripts" / "python.exe")
            cur = os.path.normcase(os.path.abspath(self._current_python))
            tgt = os.path.normcase(os.path.abspath(venv_py))
            should_warn = cur == tgt
            # Show/hide bar accordingly
            if should_warn:
                if self._relaunch_lbl:
                    self._relaunch_lbl.configure(text=f"Warning: GUI running from target venv ({venv_py}). Rebuild may close it.")
                # Only pack if not already visible
                if not self._relaunch_bar.winfo_ismapped():
                    self._relaunch_bar.pack(side="left", padx=(12, 0))
            else:
                if self._relaunch_bar.winfo_ismapped():
                    self._relaunch_bar.pack_forget()
        except Exception:
            pass

    def _relaunch_with_system_python(self):
        try:
            if shutil.which("py") is None:
                messagebox.showerror("Relaunch", "Python Launcher (py.exe) not found. Install Python 3.10.")
                return
            script = str(Path(__file__).resolve())
            creationflags = 0
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                creationflags = subprocess.CREATE_NO_WINDOW
            subprocess.Popen(["py", "-3.10", script], cwd=str(APP_DIR), creationflags=creationflags)
            # Close this instance shortly after spawning the new one
            self.after(200, self._on_close)
        except Exception as e:
            messagebox.showerror("Relaunch", f"Failed to relaunch via system Python:\n\n{e}")

    def _install_optional_packages_test_venv(self):
        # Install pip-based "optional" packages into the dedicated test venv only
        test_venv_dir = str((ROOT_DIR / "_test_custom_venv").resolve())
        venv_py = str(Path(test_venv_dir) / "Scripts" / "python.exe")
        pkgs_file = ROOT_DIR / "venv_bonus_packages" / "optional_packages.txt"

        if not Path(venv_py).exists():
            messagebox.showerror(
                "Optional Packages",
                f"Missing test venv python:\n{venv_py}\n\nCreate the test venv first (py -3.10 -m venv _test_custom_venv).",
            )
            return
        if not pkgs_file.exists():
            messagebox.showerror(
                "Optional Packages",
                f"Missing list file:\n{pkgs_file}",
            )
            return

        # Extract lines that start with "pip install" (case-insensitive)
        cmds: List[str] = []
        try:
            for raw in pkgs_file.read_text(encoding="utf-8", errors="replace").splitlines():
                s = raw.strip()
                if not s:
                    continue
                low = s.lower()
                if low.startswith("pip install "):
                    # Keep original casing/args for package spec
                    cmds.append(s[len("pip install "):].strip())
        except Exception as e:
            messagebox.showerror("Optional Packages", f"Failed parsing optional packages file:\n\n{e}")
            return

        if not cmds:
            messagebox.showinfo("Optional Packages", "No 'pip install' commands found to run.")
            return

        self._set_busy(True)
        self._status_var.set("Installing optional packages to test venv...")
        self._console_append("\n===== OPTIONAL PACKAGES (TEST VENV) =====\n", "muted")
        self._console_append(f"Test venv: {test_venv_dir}\n", "muted")

        def run():
            try:
                # Use CREATE_NO_WINDOW to avoid new console
                creationflags = 0
                if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                    creationflags = subprocess.CREATE_NO_WINDOW

                for spec in cmds:
                    cmd = [venv_py, "-m", "pip", "install"] + spec.split()
                    self._safe_after(0, self._console_append, f"[pip] installing: {spec}\n", "muted")
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(ROOT_DIR),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                        creationflags=creationflags,
                    )
                    if proc.stdout is None:
                        raise RuntimeError("subprocess stdout is None")
                    for line in proc.stdout:
                        self._safe_after(0, self._console_append, line)
                    rc = proc.wait()
                    if rc != 0:
                        self._safe_after(0, self._console_append, f"[ERROR] pip returned {rc} for: {spec}\n", "err")
                        # Continue to attempt remaining installs
                self._safe_after(0, self._console_append, "===== OPTIONAL PACKAGES DONE =====\n", "muted")
                self._safe_after(0, self._status_var.set, "Ready")
            except Exception as e:
                self._safe_after(0, self._status_var.set, "Optional packages failed")
                self._safe_after(0, messagebox.showerror, "Optional Packages", str(e))
            finally:
                self._safe_after(0, self._set_busy, False)

        threading.Thread(target=run, daemon=True).start()

    def _parse_optional_packages_file(self) -> List[Tuple[str, str]]:
        pkgs_file = ROOT_DIR / "venv_bonus_packages" / "optional_packages.txt"
        if not pkgs_file.exists():
            return []

        results: List[Tuple[str, str]] = []
        current_label: Optional[str] = None
        seen = set()

        try:
            for raw in pkgs_file.read_text(encoding="utf-8", errors="replace").splitlines():
                s = raw.strip()
                if not s:
                    continue
                low = s.lower()

                # Capture a friendly label if the line looks like a heading (no spaces/colon)
                if ":" not in s and " " not in s and s:
                    current_label = s
                    continue

                # Extract pip install commands in multiple formats
                spec: Optional[str] = None
                if low.startswith("pip install"):
                    spec = s[len("pip install") :].strip()
                elif "pip install" in low:
                    idx = low.index("pip install")
                    spec = s[idx + len("pip install") :].strip()

                if spec:
                    label = current_label or spec
                    display_label = label.strip() or spec
                    key = (display_label, spec)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append((display_label, spec))
        except Exception:
            return []

        return results

    def _refresh_optional_packages_combo(self):
        pkgs = self._parse_optional_packages_file()
        self._optional_pkg_map = {}
        display_values: List[str] = []
        for label, spec in pkgs:
            display = f"{label} — {spec}" if label != spec else spec
            self._optional_pkg_map[display] = spec
            display_values.append(display)

        # Preserve selection if still present
        current = self._opt_pkg_var.get()
        self._opt_pkg_combo["values"] = display_values
        if display_values:
            if current not in display_values:
                self._opt_pkg_var.set(display_values[0])
        else:
            self._opt_pkg_var.set("")

    def _on_optional_pkg_select(self):
        try:
            if self._opt_install_btn:
                self._opt_install_btn.configure(style="Accent.TButton", text="Install")
        except Exception:
            pass

    def _install_selected_optional_package(self):
        test_venv_dir = str((ROOT_DIR / "_test_custom_venv").resolve())
        venv_py = str(Path(test_venv_dir) / "Scripts" / "python.exe")
        pkgs_file = ROOT_DIR / "venv_bonus_packages" / "optional_packages.txt"

        if not Path(venv_py).exists():
            messagebox.showerror(
                "Optional Packages",
                f"Missing test venv python:\n{venv_py}\n\nCreate the test venv first (py -3.10 -m venv _test_custom_venv).",
            )
            return
        if not pkgs_file.exists():
            messagebox.showerror(
                "Optional Packages",
                f"Missing list file:\n{pkgs_file}",
            )
            return

        self._refresh_optional_packages_combo()
        selection = self._opt_pkg_var.get().strip()
        if not selection:
            messagebox.showinfo("Optional Packages", "Select a package to install.")
            return
        spec = self._optional_pkg_map.get(selection) or selection
        if not spec:
            messagebox.showinfo("Optional Packages", "No spec found for the selected package.")
            return

        try:
            install_args = shlex.split(spec)
        except ValueError as e:
            messagebox.showerror("Optional Packages", f"Invalid spec: {spec}\n\n{e}")
            return

        if not install_args:
            messagebox.showinfo("Optional Packages", "No install arguments parsed for selection.")
            return

        self._set_busy(True)
        self._status_var.set("Installing optional package...")
        self._console_append("\n===== OPTIONAL PACKAGE (TEST VENV) =====\n", "muted")
        self._console_append(f"Test venv: {test_venv_dir}\n", "muted")
        self._console_append(f"[pip] installing: {' '.join(install_args)}\n", "muted")

        def run():
            try:
                creationflags = 0
                if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                    creationflags = subprocess.CREATE_NO_WINDOW

                cmd = [venv_py, "-m", "pip", "install"] + install_args
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                )
                if proc.stdout is None:
                    raise RuntimeError("subprocess stdout is None")
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._safe_after(0, self._console_append, line)
                rc = proc.wait()
                if rc == 0:
                    self._safe_after(0, self._console_append, "[OK] optional package install complete\n", "ok")
                    self._safe_after(0, self._status_var.set, "Ready")
                    self._safe_after(0, self._flash_opt_install_success)
                else:
                    self._safe_after(0, self._console_append, f"[ERROR] pip returned {rc} for: {' '.join(install_args)}\n", "err")
                    self._safe_after(0, self._status_var.set, "Optional package failed")
            except Exception as e:
                self._safe_after(0, self._console_append, f"[ERROR] {e}\n", "err")
                self._safe_after(0, self._status_var.set, "Optional package failed")
            finally:
                self._safe_after(0, self._set_busy, False)

        threading.Thread(target=run, daemon=True).start()

    def _flash_opt_install_success(self):
        try:
            if self._opt_install_btn:
                self._opt_install_btn.configure(style="Success.TButton", text="Installed ✓")
                self._safe_after(1600, lambda: self._opt_install_btn.configure(style="Accent.TButton", text="Install"))
        except Exception:
            pass

    def _apply_custom_venv_lock_state(self):
        locked = False
        if "custom_venv_locked" in self._vars:
            try:
                locked = bool(self._vars["custom_venv_locked"].get())
            except Exception:
                locked = False
        try:
            if self._custom_venv_path_entry is not None:
                self._custom_venv_path_entry.configure(state=("disabled" if locked else "normal"))
            if self._custom_venv_browse_btn is not None:
                self._custom_venv_browse_btn.configure(state=("disabled" if locked else "normal"))
        except Exception:
            pass

    def _open_transformers_cache(self):
        cfg = self._vars_to_cfg()
        path = str(cfg.get("env", {}).get("TRANSFORMERS_CACHE") or r"D:\COMFY_CACHE\transformers_cache").strip()
        if not path:
            messagebox.showerror("Transformers cache", "TRANSFORMERS_CACHE is empty.")
            return
        p = Path(path)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if p.exists():
            os.startfile(str(p))
        else:
            messagebox.showerror("Transformers cache", f"Missing cache folder:\n{p}")

    def _clear_transformers_cache(self):
        cfg = self._vars_to_cfg()
        path = str(cfg.get("env", {}).get("TRANSFORMERS_CACHE") or r"D:\COMFY_CACHE\transformers_cache").strip()
        if not path:
            messagebox.showerror("Transformers cache", "TRANSFORMERS_CACHE is empty.")
            return
        p = Path(path)
        try:
            if p.exists():
                for child in p.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
            p.mkdir(parents=True, exist_ok=True)
            messagebox.showinfo("Transformers cache", f"Cleared cache at:\n{p}")
        except Exception as e:
            messagebox.showerror("Transformers cache", f"Failed to clear cache:\n{e}")

    def _open_torch_cache(self):
        cfg = self._vars_to_cfg()
        path = str(cfg.get("env", {}).get("TORCH_HOME") or cfg.get("cache_torch_home", DEFAULTS["cache_torch_home"])).strip()
        if not path:
            messagebox.showerror("Torch cache", "TORCH_HOME is empty.")
            return
        p = Path(path)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if p.exists():
            os.startfile(str(p))
        else:
            messagebox.showerror("Torch cache", f"Missing cache folder:\n{p}")

    def _clear_torch_cache(self):
        cfg = self._vars_to_cfg()
        path = str(cfg.get("env", {}).get("TORCH_HOME") or cfg.get("cache_torch_home", DEFAULTS["cache_torch_home"])).strip()
        if not path:
            messagebox.showerror("Torch cache", "TORCH_HOME is empty.")
            return
        p = Path(path)
        try:
            if p.exists():
                for child in p.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
            p.mkdir(parents=True, exist_ok=True)
            messagebox.showinfo("Torch cache", f"Cleared cache at:\n{p}")
        except Exception as e:
            messagebox.showerror("Torch cache", f"Failed to clear cache:\n{e}")

    def _refresh_transformers_cache_info_async(self):
        cfg = self._vars_to_cfg()
        cache_dir = str(cfg.get("env", {}).get("TRANSFORMERS_CACHE") or r"D:\COMFY_CACHE\transformers_cache").strip()
        if not cache_dir:
            self._transformers_cache_info_var.set("Transformers cache: (not set)")
            return

        def worker(path_str: str):
            p = Path(path_str)
            try:
                if not p.exists():
                    self._safe_after(0, self._transformers_cache_info_var.set, f"Transformers cache: {path_str} (missing)")
                    return
                total = 0
                files = 0
                for root, _, names in os.walk(p):
                    for name in names:
                        files += 1
                        try:
                            total += (Path(root) / name).stat().st_size
                        except Exception:
                            pass
                mb = total / 1024 / 1024
                self._safe_after(0, self._transformers_cache_info_var.set, f"Transformers cache: {path_str} | {files} files | {mb:.1f} MB")
            except Exception as e:
                self._safe_after(0, self._transformers_cache_info_var.set, f"Transformers cache: {path_str} (error: {e})")

        threading.Thread(target=worker, args=(cache_dir,), daemon=True).start()

    def _clear_envs(self):
        if not messagebox.askyesno("Clear Envs", "Clear all env selections?"):
            return
        for f in ENV_FIELDS:
            var = self._vars.get(f.key)
            if var is None:
                continue
            if f.kind == "bool":
                var.set(False)
            else:
                var.set("")
        self._refresh_active_envs()
        self._status_var.set("Cleared envs")

    def _save(self):
        self.cfg = self._vars_to_cfg()
        save_config(self.cfg)
        self._status_var.set("Saved")

    def _open_generated(self):
        if GENERATED_BAT.exists():
            os.startfile(str(GENERATED_BAT))
        else:
            messagebox.showinfo("Generated script", "No script generated yet.")

    def _clear_console(self):
        self.console_text.configure(state="normal")
        self.console_text.delete("1.0", "end")
        self.console_text.configure(state="disabled")

    def _console_append(self, text: str, tag: Optional[str] = None):
        try:
            safe_text = text.replace("\x00", "")
            self.console_text.configure(state="normal")
            use_tag = tag
            if use_tag is None:
                lower = safe_text.lower()
                if "[error]" in lower or "error:" in lower:
                    use_tag = "err"
                elif "[warn" in lower or "warning" in lower:
                    use_tag = "muted"
                elif "[ok]" in lower or "[info]" in lower:
                    use_tag = "ok"
            if use_tag:
                self.console_text.insert("end", safe_text, (use_tag,))
            else:
                self.console_text.insert("end", safe_text)
            self.console_text.see("end")
            self.console_text.configure(state="disabled")
        except Exception:
            # If the app is closing or Tk is not available, ignore.
            pass

    def _set_busy(self, busy: bool):
        for b in getattr(self, "_buttons_to_disable", []):
            try:
                b.configure(state=("disabled" if busy else "normal"))
            except Exception:
                pass

    def _launch(self):
        self.cfg = self._vars_to_cfg()
        save_config(self.cfg)

        # If custom venv is locked, ensure launch binds to it too.
        custom_locked = bool(self.cfg.get("custom_venv_locked", DEFAULTS["custom_venv_locked"]))
        custom_venv = str(self.cfg.get("env", {}).get("CUSTOM_VENV_PATH") or "").strip().strip('"').strip("'")
        if custom_locked and not custom_venv:
            messagebox.showerror(
                "Launch",
                "CUSTOM_VENV_PATH is locked but empty.\n\nUnlock it, browse to a folder, then lock again.",
            )
            return
        if custom_locked and any(ch in custom_venv for ch in ["&", "|", "<", ">"]):
            messagebox.showerror(
                "Launch",
                "CUSTOM_VENV_PATH contains unsupported characters (& | < >).\nPick a simpler folder path.",
            )
            return

        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        try:
            bat = build_launch_bat(self.cfg)
        except Exception as e:
            messagebox.showerror("Launch", f"Failed to build launch script:\n\n{e}")
            return
        GENERATED_BAT.write_text(bat, encoding="utf-8")

        self._status_var.set("Launching...")

        def run():
            try:
                subprocess.Popen(["cmd.exe", "/c", str(GENERATED_BAT)], cwd=str(ROOT_DIR))
                self._safe_after(0, self._status_var.set, "Launched")
            except Exception as e:
                self._safe_after(0, self._status_var.set, "Launch failed")
                self._safe_after(0, messagebox.showerror, "Launch failed", str(e))

        threading.Thread(target=run, daemon=True).start()

    def _health_check(self):
        cfg = self._vars_to_cfg()
        comfy_root = cfg["comfy_root"]
        lock_dir = cfg.get("lock_dir", DEFAULTS["lock_dir"])
        installer_bat = cfg.get("installer_bat", DEFAULTS["installer_bat"])
        custom_locked = bool(cfg.get("custom_venv_locked", DEFAULTS["custom_venv_locked"]))
        custom_venv = str(cfg.get("env", {}).get("CUSTOM_VENV_PATH") or "").strip().strip('"').strip("'")
        if os.name == "nt" and "/" in custom_venv:
            custom_venv = custom_venv.replace("/", "\\")
        if custom_locked and not custom_venv:
            messagebox.showerror(
                "Health check",
                "CUSTOM_VENV_PATH is locked but empty.\n\nUnlock it, browse to a folder, then lock again.",
            )
            return
        venv_dir = custom_venv if (custom_locked and custom_venv) else str(Path(comfy_root) / "venv")
        venv_py = str(Path(venv_dir) / "Scripts" / "python.exe")

        if not Path(venv_py).exists():
            messagebox.showerror("Health check", f"Missing venv python:\n{venv_py}\n\nRebuild via D:\\COMFY_CACHE\\new_install\\launch.bat")
            return

        self._status_var.set("Checking stack...")

        def run_check():
            try:
                out: List[str] = []
                out.append(f"venv: {venv_dir}")
                out.append(f"locks: {lock_dir}")
                out.append(f"installer: {installer_bat}")

                req = str(Path(lock_dir) / "requirements.txt")
                lock = str(Path(lock_dir) / "requirements_lock.txt")
                md = str(Path(lock_dir) / "STACK_LOCKDOWN.md")
                out.append(f"requirements.txt: {'OK' if Path(req).exists() else 'MISSING'}")
                out.append(f"requirements_lock.txt: {'OK' if Path(lock).exists() else 'MISSING'}")
                out.append(f"STACK_LOCKDOWN.md: {'OK' if Path(md).exists() else 'MISSING'}")

                ver = subprocess.check_output([venv_py, "-c", "import sys; print(sys.version)"]).decode("utf-8", errors="replace").strip()
                out.append(f"python: {ver}")

                pip_check = subprocess.run([venv_py, "-m", "pip", "check"], capture_output=True, text=True)
                if pip_check.returncode == 0:
                    out.append("pip check: OK")
                    if pip_check.stdout.strip():
                        out.append(pip_check.stdout.strip())
                else:
                    out.append("pip check: ISSUES")
                    out.append(pip_check.stdout.strip())
                    out.append(pip_check.stderr.strip())

                # Show a few key packages
                for pkg in ["torch", "xformers", "triton_windows", "diffusers", "transformers"]:
                    try:
                        pv = subprocess.check_output([venv_py, "-c", f"import importlib; import pkgutil; import sys;\nimport importlib.metadata as m;\nprint(m.version('{pkg}'))"], text=True).strip()
                        out.append(f"{pkg}: {pv}")
                    except Exception:
                        out.append(f"{pkg}: (not found)")

                # Installed package count
                try:
                    count = subprocess.check_output(
                        [
                            venv_py,
                            "-c",
                            "import importlib.metadata as m; print(len(list(m.distributions())))",
                        ],
                        text=True,
                    ).strip()
                    out.append(f"installed packages: {count}")
                except Exception:
                    pass

                # Full package listing
                try:
                    freeze = subprocess.check_output([venv_py, "-m", "pip", "freeze"], text=True)
                    out.append("")
                    out.append("pip freeze:")
                    out.append(freeze.strip())
                except Exception as e:
                    out.append("")
                    out.append(f"pip freeze: failed ({e})")

                body = "\n".join([line for line in out if line is not None and line != ""])
                self._safe_after(0, self._show_text_dialog, "Health / Stack", body)
                self._safe_after(0, self._status_var.set, "Ready")
            except Exception as e:
                self._safe_after(0, self._status_var.set, "Health check failed")
                self._safe_after(0, messagebox.showerror, "Health check failed", str(e))

        threading.Thread(target=run_check, daemon=True).start()

    def _show_text_dialog(self, title: str, body: str):
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("820x520")
        win.configure(bg=GITHUB_DARK["bg"])

        txt = tk.Text(win, bg=GITHUB_DARK["panel"], fg=GITHUB_DARK["text"], insertbackground=GITHUB_DARK["text"], relief="solid", bd=1)
        txt.pack(fill="both", expand=True, padx=12, pady=12)
        txt.insert("1.0", body)
        txt.configure(state="disabled")

    def _open_lock_folder(self):
        lock_dir = self._vars_to_cfg().get("lock_dir", DEFAULTS["lock_dir"])
        p = Path(lock_dir)
        if not p.exists():
            messagebox.showerror("Open Lock Folder", f"Missing lock folder:\n{lock_dir}")
            return
        os.startfile(str(p))

    def _rebuild_venv(self):
        cfg = self._vars_to_cfg()
        installer_bat = cfg.get("installer_bat", DEFAULTS["installer_bat"])
        if isinstance(installer_bat, str):
            installer_bat = installer_bat.strip().strip('"').strip("'")
        if not Path(installer_bat).exists():
            messagebox.showerror(
                "Rebuild venv",
                f"Missing installer script:\n{installer_bat}\n\nExpected: D:\\COMFY_CACHE\\new_install\\launch.bat",
            )
            return

        if self._rebuild_proc and self._rebuild_proc.poll() is None:
            messagebox.showinfo("Rebuild", "A rebuild is already running.")
            return

        custom_venv = str(cfg.get("env", {}).get("CUSTOM_VENV_PATH") or "").strip().strip('"').strip("'")
        custom_locked = bool(cfg.get("custom_venv_locked", DEFAULTS["custom_venv_locked"]))
        if custom_locked and not custom_venv:
            messagebox.showerror(
                "Rebuild / Refresh Venv",
                "CUSTOM_VENV_PATH is locked but empty.\n\nUnlock it, browse to a folder, then lock again.",
            )
            return
        if any(ch in custom_venv for ch in ["&", "|", "<", ">"]):
            messagebox.showerror(
                "Rebuild / Refresh Venv",
                "CUSTOM_VENV_PATH contains unsupported characters (& | < >).\nPick a simpler folder path.",
            )
            return

        target_venv = custom_venv if (custom_locked and custom_venv) else str(Path(cfg["comfy_root"]) / "venv")

        if not messagebox.askyesno(
            "Rebuild / Refresh Venv",
            f"This will delete and recreate:\n\n{target_venv}\n\n…and reinstall from your locked files.\n\nContinue?",
        ):
            return

        self._set_busy(True)
        self._status_var.set("Rebuilding venv...")
        self._console_append("\n===== REBUILD START =====\n", "muted")
        self._console_append(f"Installer: {installer_bat}\n", "muted")

        # custom_venv/custom_locked validated above; now used to pass into installer

        def run():
            try:
                # Avoid creating a new console window
                creationflags = 0
                if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                    creationflags = subprocess.CREATE_NO_WINDOW

                # Use shell=True so cmd receives the exact command string.
                # If passed as a list, Python will escape embedded quotes as \" which cmd treats literally.
                venv_part = ""
                if custom_venv:
                    venv_part = f' & set "CUSTOM_VENV_PATH={custom_venv}"'
                cmdline = f'chcp 65001 >nul & set NO_SELF_LOG=1 & set NONINTERACTIVE=1 & set TRACE=1{venv_part} & call "{installer_bat}"'
                self._safe_after(0, self._console_append, f"[cmd] {cmdline}\n", "muted")
                self._rebuild_proc = subprocess.Popen(
                    cmdline,
                    cwd=str(ROOT_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                    shell=True,
                )
                if self._rebuild_proc.stdout is None:
                    raise RuntimeError("subprocess stdout is None")

                assert self._rebuild_proc.stdout is not None
                for line in self._rebuild_proc.stdout:
                    self._safe_after(0, self._console_append, line)

                rc = self._rebuild_proc.wait()
                if rc == 0:
                    self._safe_after(0, self._console_append, "===== REBUILD OK =====\n", "ok")
                    self._safe_after(0, self._status_var.set, "Rebuild OK")
                else:
                    self._safe_after(0, self._console_append, f"===== REBUILD FAILED (exit {rc}) =====\n", "err")
                    self._safe_after(0, self._status_var.set, "Rebuild failed")
            except Exception as e:
                self._safe_after(0, self._console_append, f"[ERROR] {e}\n", "err")
                self._safe_after(0, self._status_var.set, "Rebuild failed")
            finally:
                self._safe_after(0, self._set_busy, False)

        threading.Thread(target=run, daemon=True).start()


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        app = LauncherGUI()
        app.mainloop()
    except Exception:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            (LOG_DIR / "gui_startup_crash.log").write_text(
                traceback.format_exc(),
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
