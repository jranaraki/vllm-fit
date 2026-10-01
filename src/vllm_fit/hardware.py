import glob
import os
import platform
import shutil
import warnings
from typing import Dict, Optional, Tuple


try:
    import nvidia_ml_py3 as pynvml
except ImportError:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            import pynvml
    except ImportError:
        pynvml = None


def _query_gpu_memory() -> Tuple[Dict[int, Tuple[float, float]], Optional[str]]:
    """Per-GPU ``(total_gb, free_gb)``, plus the error if the query failed."""
    mem: Dict[int, Tuple[float, float]] = {}

    if pynvml is None:
        try:
            import torch

            if not torch.cuda.is_available():
                return mem, None
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                mem[i] = (total / 1024**3, free / 1024**3)
            return mem, None
        except Exception as exc:
            return {}, f"{type(exc).__name__}: {exc}"

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        return mem, f"{type(exc).__name__}: {exc}"
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem[i] = (mem_info.total / 1024**3, mem_info.free / 1024**3)
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return mem, None


def get_vram_info() -> Dict[int, float]:
    """Total memory per GPU, in GiB."""
    return {i: total for i, (total, _) in _query_gpu_memory()[0].items()}


def get_free_vram_info() -> Dict[int, float]:
    """Currently free memory per GPU, in GiB (other processes' usage excluded)."""
    return {i: free for i, (_, free) in _query_gpu_memory()[0].items()}


def get_ram_info() -> float:
    import psutil

    return psutil.virtual_memory().total / (1024**3)


def is_apple_silicon() -> bool:
    """True on Apple Silicon Macs (arm64 Darwin).

    These machines have no CUDA GPU, so ``detect_hardware`` routes them to the CPU
    backend; this flag lets the CLI relabel the unified-memory pool and print
    macOS-specific vLLM guidance.
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class UnsupportedHardwareError(RuntimeError):
    """Hardware that vllm-fit can't size was found (or a requested device is missing)."""


def _undetected_accelerator(nvml_error: Optional[str]) -> Optional[str]:
    """Explain an accelerator that's present but not usable by vllm-fit, if any.

    Without this, a machine whose GPUs can't be queried would silently be sized as a
    CPU-only box, and ``serve`` would launch vLLM's CPU backend on it.
    """
    if os.path.exists("/proc/driver/nvidia/version") or shutil.which("nvidia-smi"):
        detail = f" ({nvml_error})" if nvml_error else ""
        return (
            "An NVIDIA driver is installed, but no GPU could be queried through NVML"
            f"{detail}. Check the driver/library versions, or that the container was "
            "started with GPU access."
        )
    if os.path.exists("/dev/kfd") or shutil.which("rocm-smi") or shutil.which("amd-smi"):
        return "An AMD (ROCm) GPU was detected; vllm-fit currently sizes NVIDIA GPUs and CPUs only."
    if shutil.which("xpu-smi"):
        return "An Intel GPU (XPU) was detected; vllm-fit currently sizes NVIDIA GPUs and CPUs only."
    if glob.glob("/dev/accel*") or shutil.which("hl-smi"):
        return (
            "An accelerator (TPU / Gaudi) was detected; vllm-fit currently sizes NVIDIA "
            "GPUs and CPUs only."
        )
    return None


def detect_hardware(device: str = "auto") -> str:
    """Return the hardware type to target, "gpu" or "cpu".

    ``device="auto"`` picks the GPU when NVIDIA GPUs are visible and the CPU backend
    when no accelerator is present at all; an accelerator that can't be sized raises
    ``UnsupportedHardwareError`` instead of silently falling back to the CPU.
    """
    if device == "cpu":
        return "cpu"
    gpu_memory, nvml_error = _query_gpu_memory()
    if gpu_memory:
        return "gpu"
    if device == "gpu":
        detail = f": {nvml_error}" if nvml_error else ""
        raise UnsupportedHardwareError(f"No usable NVIDIA GPU was found{detail}")
    reason = _undetected_accelerator(nvml_error)
    if reason:
        raise UnsupportedHardwareError(reason)
    return "cpu"
