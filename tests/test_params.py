import json
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


def test_serving_bytes_downcasts_float32_only():
    from vllm_fit.params import _serving_bytes

    assert _serving_bytes("F32") == 2.0
    assert _serving_bytes("F64") == 2.0
    assert _serving_bytes("BF16") == 2.0
    # Packed int4 weights (AWQ/GPTQ qweight) are stored as I32 and stay as-is.
    assert _serving_bytes("I32") == 4.0
    assert _serving_bytes("U8") == 1.0


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
        # F32 tensors are cast to the 16-bit serving dtype: 7e9 * 2 bytes.
        assert wi.weights_bytes == 7_000_000_000 * 2
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


def _write_safetensors(path, tensors):
    import struct

    header = {name: {"dtype": dt, "shape": shape, "data_offsets": [0, 0]}
              for name, (dt, shape) in tensors.items()}
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)


def test_local_dir_weights_from_safetensors_headers(tmp_path):
    _write_safetensors(tmp_path / "model-00001-of-00002.safetensors",
                       {"a": ("BF16", [1000, 1000]), "b": ("F32", [1000])})
    _write_safetensors(tmp_path / "model-00002-of-00002.safetensors",
                       {"c": ("I32", [10, 100])})
    # Consolidated copy of the same weights must not be double-counted.
    _write_safetensors(tmp_path / "consolidated.safetensors", {"a": ("BF16", [1000, 1000])})
    wi = resolve_weights(str(tmp_path), {})
    assert wi.source == "local_safetensors"
    assert wi.total_params == 1_000_000 + 1000 + 1000
    assert wi.weights_bytes == 1_000_000 * 2 + 1000 * 2 + 1000 * 4


def test_mtp_layers_excluded(monkeypatch):
    from types import SimpleNamespace

    from vllm_fit.params import _from_safetensors_metadata, _is_unloaded_mtp

    t = lambda dt, n: SimpleNamespace(dtype=dt, parameter_count=n)
    files = {
        "a.safetensors": SimpleNamespace(tensors={
            "model.layers.0.mlp.weight": t("BF16", 1000),
            "model.layers.45.mlp.weight": t("BF16", 1000),
            "model.layers.46.mlp.weight": t("BF16", 500),       # MTP layer
            "mtp.fc.weight": t("BF16", 200),                    # Qwen3-Next style
            "model.vision_tower.layers.60.weight": t("BF16", 7), # vision, not text stack
        }),
    }
    meta = SimpleNamespace(files_metadata=files, parameter_count={"BF16": 2707})
    fake = types.ModuleType("huggingface_hub")
    fake.get_safetensors_metadata = lambda repo_id: meta
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    wi = _from_safetensors_metadata("org/model", {"num_hidden_layers": 46})
    assert wi.total_params == 2007
    assert wi.weights_bytes == 2007 * 2
    assert not _is_unloaded_mtp("model.layers.45.x", 46)
    assert _is_unloaded_mtp("model.language_model.layers.46.x", 46)


def test_local_dir_skips_mtp_layers(tmp_path):
    _write_safetensors(tmp_path / "model.safetensors",
                       {"model.layers.0.w": ("BF16", [100]),
                        "model.layers.2.w": ("BF16", [50])})  # MTP layer past num_hidden_layers
    wi = resolve_weights(str(tmp_path), {"num_hidden_layers": 2})
    assert wi.total_params == 100
