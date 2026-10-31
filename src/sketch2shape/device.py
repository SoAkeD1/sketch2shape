"""Device detection and the hardware report. Run: python -m sketch2shape.device"""
import argparse
import gc
import json
import platform
import shutil
import subprocess
import sys

import psutil

from .config import REPO_ROOT, load_config
from .errors import DeviceError

WINDOWS_11_FIRST_BUILD = 22000


def _try_import_torch():
    try:
        import torch
        return torch
    except (ImportError, OSError):
        return None


def _is_rocm_build(torch):
    return getattr(torch.version, "hip", None) is not None or "rocm" in torch.__version__


def _gpu_runs(torch, index):
    try:
        return (torch.ones(1, device=f"cuda:{index}") + 1).item() == 2
    except Exception:
        return False


def best_cuda_index(torch, gpu_index="auto"):
    """Return the index of the usable GPU with the most memory, or None.

    Index 0 cannot be assumed. On a ROCm PC with integrated graphics, index 0 can
    be the iGPU, which reports as available and then fails on the first kernel.
    """
    if not torch.cuda.is_available():
        return None
    if gpu_index != "auto":
        return gpu_index if _gpu_runs(torch, gpu_index) else None
    indices = range(torch.cuda.device_count())
    usable = [i for i in indices if _gpu_runs(torch, i)]
    return max(usable, key=lambda i: torch.cuda.get_device_properties(i).total_memory, default=None)


def _available(name, torch, gpu_index="auto"):
    """Run a real tensor op: a backend counts only if it can compute."""
    if torch is None and name != "cpu":
        return False
    try:
        if name == "cuda":
            return best_cuda_index(torch, gpu_index) is not None
        if name == "mps":
            return torch.backends.mps.is_available()
        if name == "directml":
            import torch_directml
            return (torch.ones(1, device=torch_directml.device()) + 1).item() == 2
    except Exception:
        return False
    return name == "cpu"


def pick_device(cfg):
    """Return (device_name, dtype_name) from config, verifying the choice works."""
    torch = _try_import_torch()
    runtime = cfg["runtime"]
    requested = runtime["device"]
    gpu_index = runtime["gpu_index"]
    if requested == "auto":
        name = next((d for d in runtime["device_priority"] if _available(d, torch, gpu_index)), None)
        if name is None:
            raise DeviceError("no entry in runtime.device_priority is usable")
    elif _available(requested, torch, gpu_index):
        name = requested
    else:
        raise DeviceError(f"requested device {requested!r} is not usable", hint="set runtime.device=auto")
    dtype = runtime["dtype"]
    if dtype == "auto":
        dtype = "float16" if name in runtime["dtype_fp16_devices"] else "float32"
    if name == "cuda":
        index = best_cuda_index(torch, gpu_index)
        # Third-party code that says plain "cuda" must land on the same GPU
        torch.cuda.set_device(index)
        name = f"cuda:{index}"
    return name, dtype


def torch_device(name):
    """Turn a device name into the object torch expects."""
    if name == "directml":
        import torch_directml
        return torch_directml.device()
    return name


def free_gpu_memory():
    gc.collect()
    torch = _try_import_torch()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_peak_vram():
    torch = _try_import_torch()
    if torch and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_bytes():
    torch = _try_import_torch()
    return torch.cuda.max_memory_allocated() if torch and torch.cuda.is_available() else 0


def peak_ram_bytes():
    """Peak RAM of this process so far. Windows records a true peak; elsewhere this is current use."""
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss)


def _windows_adapters(bytes_per_gb):
    """Windows only. WMI caps AdapterRAM at 4 GB, so read the 64-bit registry value."""
    import winreg
    key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    adapters = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as root:
        for index in range(winreg.QueryInfoKey(root)[0]):
            try:
                with winreg.OpenKey(root, winreg.EnumKey(root, index)) as sub:
                    name = winreg.QueryValueEx(sub, "DriverDesc")[0]
                    memory = winreg.QueryValueEx(sub, "HardwareInformation.qwMemorySize")[0]
                    adapters.append({"name": name, "vram_gb": round(memory / bytes_per_gb, 1)})
            except OSError:
                continue
    return adapters


def _linux_nvidia_adapters():
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    adapters = []
    for line in output.strip().splitlines():
        name, memory = line.rsplit(",", 1)
        adapters.append({"name": name.strip(), "vram_gb": round(float(memory) / 1024, 1)})
    return adapters


def _adapters(bytes_per_gb):
    try:
        return _windows_adapters(bytes_per_gb) if sys.platform == "win32" else _linux_nvidia_adapters()
    except OSError:
        return []


def hardware_report(cfg):
    bytes_per_gb = cfg["runtime"]["bytes_per_gb"]
    torch = _try_import_torch()
    report = {
        # platform.platform() still says Windows-10 on Windows 11; the build number tells them apart
        "os": platform.platform().replace("Windows-10", "Windows-11", 1)
        if sys.platform == "win32" and sys.getwindowsversion().build >= WINDOWS_11_FIRST_BUILD
        else platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "torch": torch.__version__ if torch else None,
        "torch_build": None,
        "backends": {},
        "selected_device": None,
        "selected_dtype": None,
        "adapters": _adapters(bytes_per_gb),
        "ram_gb": round(psutil.virtual_memory().total / bytes_per_gb, 1),
        "disk_free_gb": round(shutil.disk_usage(REPO_ROOT).free / bytes_per_gb, 1),
        "config_backend": cfg["runtime"]["backend"],
    }
    if torch:
        report["torch_build"] = "rocm" if _is_rocm_build(torch) else ("cuda" if torch.version.cuda else "cpu")
    for name in ("cuda", "mps", "directml", "cpu"):
        report["backends"][name] = _available(name, torch)
    try:
        report["selected_device"], report["selected_dtype"] = pick_device(cfg)
        if report["selected_device"].startswith("cuda"):
            report["torch_gpu_name"] = torch.cuda.get_device_name()
    except DeviceError as exc:
        report["device_error"] = str(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description="Print the hardware report")
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    print(json.dumps(hardware_report(cfg), indent=2))


if __name__ == "__main__":
    main()
