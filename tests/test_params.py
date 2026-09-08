import sys
import types

from vllm_fit import params
from vllm_fit.params import (
    WeightInfo,
    resolve_weights,
    _st_dtype_bytes,
    _from_config,
)


def test_dtype_bytes_table():
    assert _st_dtype_bytes("F16") == 2.0
    assert _st_dtype_bytes("BF16") == 2.0
    assert _st_dtype_bytes("F8_E4M3") == 1.0
    assert _st_dtype_bytes("I4") == 0.5
    # Unknown dtype defaults to 2 bytes.
    assert _st_dtype_bytes("WEIRD") == 2.0


def test_weightinfo_gb():
    wi = WeightInfo(source="x", weights_bytes=2 * 1024**3)
    assert abs(wi.weights_gb() - 2.0) < 1e-9
    assert WeightInfo(source="x", total_params=10).weights_gb() is None


def test_from_config():
    wi = _from_config({"num_parameters": 7_000_000_000})
    assert wi is not None and wi.total_params == 7_000_000_000
    assert wi.source == "config"
    assert _from_config({"hidden_size": 4096}) is None


def _install_fake_hub(parameter_count):
    """Install a fake huggingface_hub exposing get_safetensors_metadata."""
    fake = types.ModuleType("huggingface_hub")

    class _Meta:
        def __init__(self, pc):
            self.parameter_count = pc

    fake.get_safetensors_metadata = lambda repo_id: _Meta(parameter_count)
    # The other rungs import lazily; provide stubs that raise so the ladder stops
    # at the first rung when it succeeds.
    fake.hf_hub_download = lambda **kw: (_ for _ in ()).throw(RuntimeError("no net"))

    class _HfApi:
        def model_info(self, repo_id):
            raise RuntimeError("no net")

    fake.HfApi = _HfApi
    sys.modules["huggingface_hub"] = fake
    return fake


def test_ladder_prefers_safetensors_metadata(monkeypatch):
    saved = sys.modules.get("huggingface_hub")
    try:
        _install_fake_hub({"BF16": 6_000_000_000, "F32": 1_000_000_000})
        wi = resolve_weights("some/repo", {})
        assert wi is not None
        assert wi.source == "safetensors_metadata"
        assert wi.total_params == 7_000_000_000
        # 6e9 * 2 bytes + 1e9 * 4 bytes = 16e9 bytes
        assert wi.weights_bytes == 6_000_000_000 * 2 + 1_000_000_000 * 4
        assert wi.per_dtype == {"BF16": 6_000_000_000, "F32": 1_000_000_000}
    finally:
        if saved is not None:
            sys.modules["huggingface_hub"] = saved
        else:
            sys.modules.pop("huggingface_hub", None)


def test_ladder_falls_through_to_config_when_offline(monkeypatch):
    saved = sys.modules.get("huggingface_hub")
    try:
        # A hub with no usable network features -> all network rungs raise/miss.
        fake = types.ModuleType("huggingface_hub")
        fake.get_safetensors_metadata = lambda repo_id: (_ for _ in ()).throw(RuntimeError("offline"))
        fake.hf_hub_download = lambda **kw: (_ for _ in ()).throw(RuntimeError("offline"))

        class _HfApi:
            def model_info(self, repo_id):
                raise RuntimeError("offline")

        fake.HfApi = _HfApi
        sys.modules["huggingface_hub"] = fake

        wi = resolve_weights("some/repo", {"num_parameters": 1_500_000_000})
        assert wi is not None and wi.source == "config"
        assert wi.total_params == 1_500_000_000
    finally:
        if saved is not None:
            sys.modules["huggingface_hub"] = saved
        else:
            sys.modules.pop("huggingface_hub", None)


def test_ladder_returns_none_when_nothing_available():
    saved = sys.modules.get("huggingface_hub")
    try:
        fake = types.ModuleType("huggingface_hub")
        fake.get_safetensors_metadata = lambda repo_id: (_ for _ in ()).throw(RuntimeError("x"))
        fake.hf_hub_download = lambda **kw: (_ for _ in ()).throw(RuntimeError("x"))

        class _HfApi:
            def model_info(self, repo_id):
                raise RuntimeError("x")

        fake.HfApi = _HfApi
        sys.modules["huggingface_hub"] = fake

        assert resolve_weights("some/repo", {}) is None
    finally:
        if saved is not None:
            sys.modules["huggingface_hub"] = saved
        else:
            sys.modules.pop("huggingface_hub", None)
