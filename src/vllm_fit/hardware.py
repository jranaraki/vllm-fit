import platform
import warnings
from typing import Dict, Tuple


try:
    import nvidia_ml_py3 as pynvml
except ImportError:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            import pynvml
    except ImportError:
        pynvml = None


def _query_gpu_memory() -> Dict[int, Tuple[float, float]]:
    """Per-GPU ``(total_gb, free_gb)``; empty when no GPU can be queried."""
    mem = {}

    if pynvml is None:
        try:
            import torch

            if not torch.cuda.is_available():
                return mem
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                mem[i] = (total / 1024**3, free / 1024**3)
            return mem
        except:
            return mem

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count == 0:
            pynvml.nvmlShutdown()
            return mem

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem[i] = (mem_info.total / 1024**3, mem_info.free / 1024**3)
        pynvml.nvmlShutdown()
    except Exception:
        return mem

    return mem


def get_vram_info() -> Dict[int, float]:
    """Total memory per GPU, in GiB."""
    return {i: total for i, (total, _) in _query_gpu_memory().items()}


def get_free_vram_info() -> Dict[int, float]:
    """Currently free memory per GPU, in GiB (other processes' usage excluded)."""
    return {i: free for i, (_, free) in _query_gpu_memory().items()}


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


def detect_hardware() -> str:
    """Return the hardware type to target: "gpu" if any GPU is detected,
    otherwise "cpu" (vLLM supports a CPU backend)."""
    if get_vram_info():
        return "gpu"
    return "cpu"
