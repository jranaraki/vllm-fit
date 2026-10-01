import vllm_fit.hardware as hardware
from vllm_fit.hardware import is_apple_silicon, detect_hardware


def test_is_apple_silicon_true(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    assert is_apple_silicon() is True


def test_is_apple_silicon_false_on_intel_mac(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    assert is_apple_silicon() is False


def test_is_apple_silicon_false_on_linux(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    assert is_apple_silicon() is False


def _no_accelerator_files(monkeypatch):
    monkeypatch.setattr(hardware, "metal_backend_available", lambda: False)
    monkeypatch.setattr(hardware.os.path, "exists", lambda path: False)
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    monkeypatch.setattr(hardware.glob, "glob", lambda pattern: [])


def test_detect_hardware_cpu_when_no_vram(monkeypatch):
    monkeypatch.setattr(hardware, "_query_gpu_memory", lambda: ({}, None))
    _no_accelerator_files(monkeypatch)
    assert detect_hardware() == "cpu"


def test_detect_hardware_gpu_when_vram_present(monkeypatch):
    monkeypatch.setattr(hardware, "_query_gpu_memory", lambda: ({0: (24.0, 23.5)}, None))
    assert detect_hardware() == "gpu"


def test_broken_nvidia_driver_is_not_cpu(monkeypatch):
    import pytest

    monkeypatch.setattr(
        hardware, "_query_gpu_memory",
        lambda: ({}, "NVMLError_LibRmVersionMismatch: RM has detected an NVML/RM version mismatch."),
    )
    _no_accelerator_files(monkeypatch)
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    with pytest.raises(hardware.UnsupportedHardwareError, match="version mismatch"):
        detect_hardware()


def test_amd_gpu_is_not_cpu(monkeypatch):
    import pytest

    monkeypatch.setattr(hardware, "_query_gpu_memory", lambda: ({}, "NVMLError_LibraryNotFound"))
    _no_accelerator_files(monkeypatch)
    monkeypatch.setattr(hardware.os.path, "exists", lambda path: path == "/dev/kfd")
    with pytest.raises(hardware.UnsupportedHardwareError, match="AMD"):
        detect_hardware()


def test_device_override(monkeypatch):
    import pytest

    monkeypatch.setattr(hardware, "_query_gpu_memory", lambda: ({}, "NVMLError_LibraryNotFound"))
    monkeypatch.setattr(hardware.os.path, "exists", lambda path: path == "/dev/kfd")
    assert detect_hardware("cpu") == "cpu"
    with pytest.raises(hardware.UnsupportedHardwareError, match="No usable NVIDIA GPU"):
        detect_hardware("gpu")


class _FailingNvml:
    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        self.shutdown = True

    def nvmlDeviceGetCount(self):
        return 2

    def nvmlDeviceGetHandleByIndex(self, i):
        raise RuntimeError("GPU is lost")


def test_nvml_error_is_reported_and_nvml_shut_down(monkeypatch):
    fake = _FailingNvml()
    monkeypatch.setattr(hardware, "pynvml", fake)
    vram, err = hardware._query_gpu_memory()
    assert vram == {}
    assert "GPU is lost" in err
    assert getattr(fake, "shutdown", False)


class _FakeMem:
    def __init__(self, total_gb, free_gb):
        self.total = int(total_gb * 1024**3)
        self.free = int(free_gb * 1024**3)


class _FakeNvml:
    def __init__(self, mems):
        self._mems = mems

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetCount(self):
        return len(self._mems)

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetMemoryInfo(self, handle):
        return self._mems[handle]


def test_vram_total_and_free_from_nvml(monkeypatch):
    monkeypatch.setattr(hardware, "pynvml", _FakeNvml([_FakeMem(24, 22.5), _FakeMem(80, 60)]))
    assert hardware.get_vram_info() == {0: 24.0, 1: 80.0}
    assert hardware.get_free_vram_info() == {0: 22.5, 1: 60.0}


def test_visible_gpu_indices_parsing():
    from vllm_fit.hardware import visible_gpu_indices

    idx = [0, 1, 2, 3]
    uuids = {0: "GPU-aaaa-1", 1: "GPU-bbbb-2", 2: "GPU-cccc-3", 3: "GPU-dddd-4"}
    assert visible_gpu_indices(None, idx, uuids) is None
    assert visible_gpu_indices("2,3", idx, uuids) == [2, 3]
    assert visible_gpu_indices("1,9,0", idx, uuids) == [1]  # stops at the invalid entry
    assert visible_gpu_indices("GPU-cccc,GPU-aaaa-1", idx, uuids) == [2, 0]
    assert visible_gpu_indices("", idx, uuids) == []
    assert visible_gpu_indices("-1", idx, uuids) == []
    assert visible_gpu_indices("MIG-1234", idx, uuids) is None


def test_query_respects_cuda_visible_devices(monkeypatch):
    fake = _FakeNvml([_FakeMem(24, 23), _FakeMem(24, 23), _FakeMem(80, 79), _FakeMem(80, 79)])
    monkeypatch.setattr(hardware, "pynvml", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert hardware.get_vram_info() == {2: 80.0, 3: 80.0}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    vram, err = hardware._query_gpu_memory()
    assert vram == {} and "hides every GPU" in err


def test_gpu_launch_env_pins_pci_order():
    assert hardware.gpu_launch_env([2, 3]) == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "2,3"}


def test_metal_detection(monkeypatch):
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hardware, "_query_gpu_memory", lambda: ({}, "NVMLError_LibraryNotFound"))
    real_metal_check = hardware.metal_backend_available
    _no_accelerator_files(monkeypatch)
    monkeypatch.setattr(hardware, "metal_backend_available", real_metal_check)
    monkeypatch.setattr(hardware.importlib.util, "find_spec",
                        lambda name: object() if name == "vllm_metal" else None)
    assert hardware.metal_backend_available()
    assert detect_hardware() == "metal"
    assert detect_hardware("cpu") == "cpu"

    monkeypatch.setattr(hardware.importlib.util, "find_spec", lambda name: None)
    assert detect_hardware() == "cpu"          # no plugin: CPU backend, as before
    assert detect_hardware("metal") == "metal"  # explicit request still honoured


def test_metal_device_needs_apple_silicon(monkeypatch):
    import pytest

    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    with pytest.raises(hardware.UnsupportedHardwareError, match="Apple Silicon"):
        detect_hardware("metal")


def test_metal_working_set_falls_back_without_mlx(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_mlx(name, *args, **kwargs):
        if name.startswith("mlx"):
            raise ImportError("no mlx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mlx)
    monkeypatch.setattr(hardware, "get_ram_info", lambda: 48.0)
    limit, measured = hardware.get_metal_working_set()
    assert not measured and abs(limit - 32.0) < 1e-9
