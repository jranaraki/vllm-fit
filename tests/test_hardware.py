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


def test_detect_hardware_cpu_when_no_vram(monkeypatch):
    monkeypatch.setattr(hardware, "get_vram_info", lambda: {})
    assert detect_hardware() == "cpu"


def test_detect_hardware_gpu_when_vram_present(monkeypatch):
    monkeypatch.setattr(hardware, "get_vram_info", lambda: {0: 24.0})
    assert detect_hardware() == "gpu"
