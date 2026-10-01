import glob
import importlib.util
import os
import platform
import shutil
import warnings
from typing import Dict, List, Optional, Tuple


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
    uuids: Dict[int, str] = {}
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem[i] = (mem_info.total / 1024**3, mem_info.free / 1024**3)
            try:
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                uuids[i] = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
            except Exception:
                pass
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    visible = visible_gpu_indices(os.environ.get("CUDA_VISIBLE_DEVICES"), list(mem), uuids)
    if visible is not None:
        mem = {i: mem[i] for i in visible}
        if not mem:
            return mem, (
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} hides every GPU"
            )
    return mem, None


def visible_gpu_indices(
    value: Optional[str], indices: List[int], uuids: Dict[int, str]
) -> Optional[List[int]]:
    """NVML (PCI bus order) indices allowed by a ``CUDA_VISIBLE_DEVICES`` value.

    Accepts integer indices and ``GPU-<uuid>`` (or a unique prefix); like CUDA, parsing
    stops at the first entry that doesn't match a device. Integer indices are taken in
    PCI bus order, which is what launched processes get via ``CUDA_DEVICE_ORDER``.
    Returns None when unset, or when it names MIG devices (not mapped here).
    """
    if value is None:
        return None
    tokens = [t.strip() for t in value.split(",")] if value.strip() else []
    if any(t.upper().startswith("MIG-") for t in tokens):
        return None
    selected: List[int] = []
    for token in tokens:
        match: Optional[int] = None
        if token.lstrip("-").isdigit():
            idx = int(token)
            match = idx if idx in indices else None
        elif token.upper().startswith("GPU-"):
            hits = [i for i, u in uuids.items() if u.upper().startswith(token.upper())]
            match = hits[0] if len(hits) == 1 else None
        if match is None:
            break
        if match not in selected:
            selected.append(match)
    return selected


def gpu_launch_env(gpu_ids: List[int]) -> Dict[str, str]:
    """Environment that pins a vLLM process to the given NVML GPU indices.

    NVML numbers GPUs in PCI bus order, while CUDA defaults to fastest-first; setting
    CUDA_DEVICE_ORDER makes the two agree so the sized GPUs are the ones used.
    """
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
    }


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


# Fallback share of unified memory Metal lets one process keep resident, used only when
# MLX can't report the real limit (it was 0.78 on a 48 GB M-series Mac).
_METAL_WORKING_SET_FALLBACK = 2 / 3


def metal_backend_available() -> bool:
    """True when vLLM can serve on the Apple GPU here: Apple Silicon with the
    vllm-metal plugin installed in this Python environment."""
    return is_apple_silicon() and importlib.util.find_spec("vllm_metal") is not None


def get_metal_working_set() -> Tuple[float, bool]:
    """Metal's recommended max working set in GiB, and whether it was measured.

    vllm-metal sizes its KV cache as a fraction (--gpu-memory-utilization) of this
    limit, not of total RAM. MLX reports it; without MLX, fall back to a conservative
    share of RAM.
    """
    try:
        import mlx.core as mx

        limit = int(mx.device_info().get("max_recommended_working_set_size", 0))
        if limit > 0:
            return limit / 1024**3, True
    except Exception:
        pass
    return get_ram_info() * _METAL_WORKING_SET_FALLBACK, False


def get_available_ram() -> float:
    """Memory the OS can hand out right now, in GiB."""
    import psutil

    return psutil.virtual_memory().available / (1024**3)


def detect_hardware(device: str = "auto") -> str:
    """Return the hardware type to target: "gpu", "metal" or "cpu".

    ``device="auto"`` picks the GPU when NVIDIA GPUs are visible, the Apple GPU when
    the vllm-metal plugin is installed on Apple Silicon, and the CPU backend when no
    accelerator is present at all; an accelerator that can't be sized raises
    ``UnsupportedHardwareError`` instead of silently falling back to the CPU.
    """
    if device == "cpu":
        return "cpu"
    if device == "metal":
        if not is_apple_silicon():
            raise UnsupportedHardwareError("--device metal needs an Apple Silicon Mac")
        return "metal"
    gpu_memory, nvml_error = _query_gpu_memory()
    if gpu_memory:
        return "gpu"
    if device == "gpu":
        detail = f": {nvml_error}" if nvml_error else ""
        raise UnsupportedHardwareError(f"No usable NVIDIA GPU was found{detail}")
    if device == "auto" and metal_backend_available():
        return "metal"
    reason = _undetected_accelerator(nvml_error)
    if reason:
        raise UnsupportedHardwareError(reason)
    return "cpu"
