import warnings
from typing import Dict


try:
    import nvidia_ml_py3 as pynvml
except ImportError:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            import pynvml
    except ImportError:
        pynvml = None


def get_vram_info() -> Dict[int, float]:
    vram_info = {}

    if pynvml is None:
        try:
            import torch

            if not torch.cuda.is_available():
                return vram_info
            return {
                i: torch.cuda.get_device_properties(i).total_memory / 1024**3
                for i in range(torch.cuda.device_count())
            }
        except:
            return vram_info

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count == 0:
            pynvml.nvmlShutdown()
            return vram_info

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_info[i] = mem_info.total / 1024**3
        pynvml.nvmlShutdown()
    except Exception:
        return vram_info

    return vram_info


def get_ram_info() -> float:
    import psutil

    return psutil.virtual_memory().total / (1024**3)


def detect_hardware() -> str:
    """Return the hardware type to target: "gpu" if any GPU is detected,
    otherwise "cpu" (vLLM supports a CPU backend)."""
    if get_vram_info():
        return "gpu"
    return "cpu"
