# Stack Lockdown

This folder stores backups of your working Python dependency stack for ComfyUI.

## Files
- `requirements.txt`: full pinned stack you provided
- `requirements_lock.txt`: smaller lock list (partially pinned)
- `pip_freeze.txt`: exact installed package versions from your current venv (generated)

## Recommended install workflows

### Exact reproduction (strongest lock)
1. Ensure venv exists: `N:\ComfyUI\venv`
2. Install exact frozen set:
   - `N:\ComfyUI\venv\Scripts\python.exe -m pip install --upgrade pip`
   - `N:\ComfyUI\venv\Scripts\python.exe -m pip install -r D:\COMFY_CACHE\venv_requirements\pip_freeze.txt`

### From your pinned requirements.txt (still locked, but not as strong as freeze)
- `N:\ComfyUI\venv\Scripts\python.exe -m pip install -r D:\COMFY_CACHE\venv_requirements\requirements.txt`

Notes:
- `pip_freeze.txt` is the best “no surprises” snapshot.
- If you later intentionally upgrade, regenerate `pip_freeze.txt`.
