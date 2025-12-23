\# ComfyUI \+ Chromium (Kiosk/App) Launchers \+ One-Click Venv Rebuild

This folder contains the Windows launchers that:

- Start ComfyUI from the venv at `N:\ComfyUI\venv`
- Wait until ComfyUI is reachable on `http://127.0.0.1:8188`
- Launch Chrome in **app/kiosk** mode using a dedicated profile directory (no address bar, repeatable UI state)

It also integrates with the **one-click venv rebuild** system in `D:\COMFY_CACHE\new_install`.

---

## Quick start (most common)

1. Double-click `D:\COMFY_CACHE\Launch_ComfyUI_GUI.bat`
2. Use **Rebuild / Refresh Venv** if the venv is missing/broken
3. Choose a **Launch mode** (xformers / no-xformers / safe / benchmark)
4. Click **Launch**

---

## Folder map (where everything lives)

### Core paths

- **ComfyUI root**: `N:\ComfyUI`
- **Python venv**: `N:\ComfyUI\venv`

### Launcher + GUI tooling (this repo)

- **Workspace root**: `D:\COMFY_CACHE`
- **Chromium launchers**: `D:\COMFY_CACHE\Conda_ComfyUI_Chromium_Launchers\`
- **Dedicated Chrome profile** (important):
  - `D:\COMFY_CACHE\Conda_ComfyUI_Chromium_Launchers\ChromeProfile_ComfyUI\`
- **GUI launcher**:
  - Script: `D:\COMFY_CACHE\GUI_Launcher\comfyui_launcher_gui.py`
  - Config: `D:\COMFY_CACHE\GUI_Launcher\launcher_config.json`
  - Generated BAT (last launch): `D:\COMFY_CACHE\GUI_Launcher\generated\launch_last.bat`
  - GUI crash log (if a Tk callback dies): `D:\COMFY_CACHE\GUI_Launcher\gui_crash.log`

### Rebuild / locked stack

- **One-click rebuild script**: `D:\COMFY_CACHE\new_install\launch.bat`
- **Rebuild run log** (if logging enabled): `D:\COMFY_CACHE\new_install\install_run.log`
- **Dependency integrity output**: `D:\COMFY_CACHE\new_install\pip_check.txt`

- **Locked requirements folder**: `D:\COMFY_CACHE\venv_requirements\`
  - `requirements.txt` (pinned intent)
  - `requirements_lock.txt` (constraints)
  - `pip_freeze.txt` (exact installed snapshot)
  - `STACK_LOCKDOWN.md` (policy + notes)

### Cache folders (speed + disk control)

- **Temp**: `D:\COMFY_CACHE\temp`
- **Torch cache**: `D:\COMFY_CACHE\torch`

Note: There is also a separate, optional model-cache utility described below.

---

## Launcher modes (what they do)

All modes:

- Use `N:\ComfyUI\venv\Scripts\python.exe`
- Start ComfyUI
- Wait until `http://127.0.0.1:8188` responds
- Open Chrome in app/kiosk mode with the dedicated profile directory

Modes:

- **xformers**: starts ComfyUI with `--xformers`
- **no_xformers**: starts ComfyUI with `--disable-xformers`
- **safe**: starts ComfyUI with `--disable-xformers --no-localization --quiet`
- **benchmark**: starts ComfyUI with `--disable-xformers --benchmark` and also starts GPU logging (`nvidia-smi`)

---

## One-click rebuild (exact install procedure)

The rebuild script is designed to be repeatable and not depend on global machine state.

### What it does

`D:\COMFY_CACHE\new_install\launch.bat`:

1. Stops any running `python.exe` that is **using the venv interpreter** (prevents delete locks)
2. Deletes `N:\ComfyUI\venv` (with retries)
3. Creates a new venv with **Python 3.10 only** via `py -3.10 -m venv`
4. Upgrades pip tooling (`pip`, `setuptools`, `wheel`)
5. Installs dependencies from either:
   - `D:\COMFY_CACHE\venv_requirements\pip_freeze.txt` (preferred, exact), OR
   - `requirements.txt` (+ `requirements_lock.txt` as constraints)
6. Writes a new `pip_freeze.txt`
7. Runs `pip check` and writes `D:\COMFY_CACHE\new_install\pip_check.txt`

### Why the PyTorch/CUDA packages install reliably

The installer sets these env vars inside the rebuild process:

- `PIP_INDEX_URL=https://pypi.org/simple`
- `PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128`

This is required because packages like `torch==2.9.1+cu128` are hosted on the PyTorch wheel index, not standard PyPI.

### Running it manually

Non-interactive (recommended for automation / GUI):

```bat
cmd.exe /c "set NO_SELF_LOG=1& set NONINTERACTIVE=1& call D:\COMFY_CACHE\new_install\launch.bat"
```

Interactive (pauses on error):

```bat
D:\COMFY_CACHE\new_install\launch.bat
```

---

## GUI launcher (features)

`D:\COMFY_CACHE\Launch_ComfyUI_GUI.bat` opens the GUI.

GUI features:

- Choose launch mode (xformers / no-xformers / safe / benchmark)
- Configure whitelisted environment variables (categorized)
- Shows the active env vars (green) that will apply on next launch
- **Health / Stack Check**:
  - Confirms venv python exists
  - Shows python version
  - Runs `pip check`
  - Shows a `pip freeze` snapshot
- **Rebuild / Refresh Venv**:
  - Runs the one-click rebuild script and streams output into the GUI console

If the GUI ever errors in a Tk callback, it writes:

- `D:\COMFY_CACHE\GUI_Launcher\gui_crash.log`

---

## Env-var library (what they do / when to use)

These env vars are optional and only set if enabled/filled in the GUI.

### Performance

- `CUDA_VISIBLE_DEVICES`: restricts which GPU(s) are visible (e.g. `0`)
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA allocator tuning (common: `max_split_size_mb:256`)
- `OMP_NUM_THREADS`: CPU thread count for certain ops
- `CUDA_LAUNCH_BLOCKING=1`: forces sync CUDA calls; use only for debugging (slow)
- `XFORMERS_DISABLE=1`: disables xFormers attention even if installed
- `NUM_WORKERS`: dataloader worker count (only affects nodes that use it)
- `TORCH_USE_CUDNN`: `enabled`/`disabled` for cuDNN usage (advanced)
- `CUDA_MEMORY_FORMAT`: `channels_last` or `contiguous` (advanced)

### Models

- `TRANSFORMERS_CACHE`: HuggingFace transformers cache directory
- `HF_HOME`: HuggingFace base directory
- `DIFFUSERS_DISABLE_FP16=1`: disables FP16 paths in diffusers
- `USE_TRITON=1`: enables Triton kernels where supported
- `TORCH_DEVICE`: default device hint (e.g. `cuda`)
- `TRAINING_BATCH_SIZE`: only relevant for training pipelines
- `MAX_TRAINING_STEPS`: only relevant for training pipelines
- `MODEL_TYPE`: optional pipeline hint
- `TEXT_EMBEDDING_MODE`: optional pipeline hint
- `NO_NORMALIZE_INPUTS=1`: disables input normalization (pipeline-specific)
- `USE_ACCURATE_ATTENTION=1`: prefers accuracy over speed (pipeline-specific)
- `USE_COMPRESSION=1`: enables compression (pipeline-specific)
- `ENABLE_GRADIENT_CHECKPOINTING=1`: reduces VRAM at cost of speed (training)
- `GENERATE_TRAINING_DATA=1`: pipeline-specific automation

### Precision

- `USE_BF16=1`: enable bfloat16
- `USE_BFLOAT16=1`: force bfloat16 instead of fp16 (pipeline-specific)
- `USE_FP16=1`: enable float16
- `USE_FP32=1`: force float32
- `USE_MIXED_PRECISION=1`: mixed precision if supported
- `TORCH_DTYPE`: force dtype string (e.g. `float32`)
- `ENABLE_OPTIMIZER_STATE_SHARING=1`: training optimization (advanced)

### Debug

- `DEBUG_MODE=1`: enables debug logging (pipeline-specific)
- `ENABLE_PROFILER=1`: enables profiling (pipeline-specific)
- `PROFILE_INTERVAL`: profiler interval (pipeline-specific)
- `LOG_LEVEL`: verbosity (e.g. `DEBUG`)
- `CACHE_MODEL_PARAMS=1`: pipeline-specific
- `SYNCHRONOUS_GPU=1`: pipeline-specific
- `PRINT_TENSOR_SHAPES=1`: pipeline-specific

### Pipelines

- `CUSTOM_PIPELINE`: select a pipeline by name (pipeline-specific)
- `DISABLE_DATA_AUGMENTATION=1`: training pipeline option
- `ENABLE_BATCH_NORM=1`: training pipeline option

### Health

- `PIP_CHECK=1`: run `pip check` before launching ComfyUI (fails launch if broken)
- `CUSTOM_VENV_PATH`: override venv path (advanced; default is `N:\ComfyUI\venv`)

### Excluded (intentional)

- `ENABLE_BACKDOOR_LLM` is explicitly not supported in the GUI.

---

## Optional: Model cache manager

This folder also contains a utility script:

- `D:\COMFY_CACHE\Conda_ComfyUI_Chromium_Launchers\comfyui_cache_manager.py`

It supports copying model folders from a HDD location into an SSD cache folder and tracking usage.

Default paths used by the script (edit inside the file if needed):

- Cache root: `D:\COMFY_CACHE\model_cache`
- Model source: `H:\AI_MODELS`

Commands:

```bat
python comfyui_cache_manager.py --cache SDXL
python comfyui_cache_manager.py --list
python comfyui_cache_manager.py --clean
python comfyui_cache_manager.py --telemetry
```










