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


def check_gpu_availability() -> Tuple[bool, Optional[str]]:
    vram_info = get_vram_info()

    if not vram_info:
        return False, (
            "No GPUs detected on this system. "
            "vllm-fit requires NVIDIA GPU with CUDA support to run."
        )

    return True, None


def get_available_vram(gpu_id: int = 0) -> float:
    if pynvml is None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device available")

        if gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU {gpu_id} not available (only {torch.cuda.device_count()} GPU(s) found)"
            )

        props = torch.cuda.get_device_properties(gpu_id)
        return props.total_memory / 1024**3

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        available = mem_info.free / 1024**3
        pynvml.nvmlShutdown()
        return available
    except ImportError:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device available")

        if gpu_id >= torch.cuda.device_count():
            raise RuntimeError(f"GPU {gpu_id} not available")

        props = torch.cuda.get_device_properties(gpu_id)
        return props.total_memory / 1024**3
