# comfyui_cache_manager.py

import os
import shutil
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import torch

CACHE_ROOT = Path("D:/COMFY_CACHE/model_cache")
LOG_FILE = Path("D:/COMFY_CACHE/cache_log.json")
USAGE_FILE = Path("D:/COMFY_CACHE/cache_usage_history.json")
TELEMETRY_FILE = Path("D:/COMFY_CACHE/gpu_telemetry_log.json")
CACHE_RETENTION_DAYS = 7

MODEL_HDD_PATH = Path("H:/AI_MODELS")  # Change as needed

# Ensure log files exist
for log in [LOG_FILE, USAGE_FILE, TELEMETRY_FILE]:
    if not log.exists():
        log.write_text(json.dumps({}))

def log_model_usage(model_name):
    data = json.loads(USAGE_FILE.read_text())
    now = datetime.now().isoformat()
    data.setdefault(model_name, []).append(now)
    USAGE_FILE.write_text(json.dumps(data, indent=2))

def update_cache_log(model_name, path):
    data = json.loads(LOG_FILE.read_text())
    size_mb = sum(f.stat().st_size for f in Path(path).rglob('*')) / 1024 / 1024
    data[model_name] = {
        "path": str(path),
        "cached_at": datetime.now().isoformat(),
        "size_mb": round(size_mb, 2),
    }
    LOG_FILE.write_text(json.dumps(data, indent=2))

def is_cache_valid(model_name):
    data = json.loads(LOG_FILE.read_text())
    if model_name not in data:
        return False
    last_used = datetime.fromisoformat(data[model_name]["cached_at"])
    return (datetime.now() - last_used).days < CACHE_RETENTION_DAYS

def build_model_cache(model_name):
    print(f"[+] Building cache for: {model_name}")
    src = MODEL_HDD_PATH / model_name
    dst = CACHE_ROOT / model_name
    if not src.exists():
        print(f"[!] Model source not found: {src}")
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    update_cache_log(model_name, dst)
    log_model_usage(model_name)
    return True

def cleanup_old_cache():
    data = json.loads(LOG_FILE.read_text())
    for model_name, info in list(data.items()):
        last_used = datetime.fromisoformat(info["cached_at"])
        if (datetime.now() - last_used).days >= CACHE_RETENTION_DAYS:
            print(f"[-] Removing expired cache: {model_name}")
            shutil.rmtree(info["path"], ignore_errors=True)
            del data[model_name]
    LOG_FILE.write_text(json.dumps(data, indent=2))

def list_cache():
    data = json.loads(LOG_FILE.read_text())
    print("\n[Model Cache Summary]")
    for model, info in data.items():
        print(f"{model:25} | {info['size_mb']:8} MB | Cached at: {info['cached_at']}")

def auto_cache_model(model_name):
    if not is_cache_valid(model_name):
        build_model_cache(model_name)
    else:
        log_model_usage(model_name)

def log_gpu_telemetry():
    try:
        # Import NVML lazily to avoid hard dependency when telemetry isn't used
        try:
            import pynvml  # type: ignore
        except ImportError:
            telemetry = json.loads(TELEMETRY_FILE.read_text())
            timestamp = datetime.now().isoformat()
            telemetry[timestamp] = {
                "error": "pynvml_not_available",
                "message": "Install 'pynvml' to enable GPU telemetry",
            }
            TELEMETRY_FILE.write_text(json.dumps(telemetry, indent=2))
            print("[Telemetry] pynvml not installed; logged notice.")
            return

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        telemetry = json.loads(TELEMETRY_FILE.read_text())
        timestamp = datetime.now().isoformat()
        telemetry[timestamp] = {
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "mem_used_mb": round(mem_info.used / 1024 / 1024, 2),
            "mem_total_mb": round(mem_info.total / 1024 / 1024, 2),
            "temperature_C": temp
        }
        TELEMETRY_FILE.write_text(json.dumps(telemetry, indent=2))
    except Exception as e:
        print(f"[Telemetry Error] {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", help="Model name to cache")
    parser.add_argument("--list", action="store_true", help="List current cache")
    parser.add_argument("--clean", action="store_true", help="Clean expired cache")
    parser.add_argument("--telemetry", action="store_true", help="Log current GPU telemetry")
    args = parser.parse_args()

    if args.cache:
        auto_cache_model(args.cache)
    elif args.clean:
        cleanup_old_cache()
    elif args.list:
        list_cache()
    elif args.telemetry:
        log_gpu_telemetry()
    else:
        parser.print_help()
