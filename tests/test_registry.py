import json
import os
import tempfile

from vllm_fit.registry import _load_config_json


def test_load_config_json_valid():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"hidden_size": 4096}, f)
        path = f.name
    try:
        cfg = _load_config_json(path)
        assert cfg == {"hidden_size": 4096}
    finally:
        os.unlink(path)


def test_load_config_json_corrupt_returns_none():
    # A truncated/garbage cached config.json must yield None (caller falls through),
    # not raise a JSONDecodeError traceback.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("{ this is not valid json ")
        path = f.name
    try:
        assert _load_config_json(path) is None
    finally:
        os.unlink(path)


def test_load_config_json_unreadable_returns_none():
    # A path that cannot be opened (missing file) yields None rather than OSError.
    assert _load_config_json("/nonexistent/path/to/config.json") is None


def test_local_model_directory(tmp_path):
    from vllm_fit.registry import get_model_config

    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 1024}))
    cfg, repo = get_model_config(str(tmp_path))
    assert cfg == {"hidden_size": 1024}
    assert repo == str(tmp_path)


def test_local_directory_without_config(tmp_path):
    import pytest

    from vllm_fit.registry import get_model_config

    with pytest.raises(ValueError, match="No readable config.json"):
        get_model_config(str(tmp_path))


def test_gated_repo_gets_actionable_error(monkeypatch):
    import pytest
    from huggingface_hub.errors import GatedRepoError

    import vllm_fit.registry as registry

    class _Gated(GatedRepoError):
        def __init__(self):
            Exception.__init__(self, "401 Client Error: access to model is restricted")

    def gated(**kwargs):
        raise _Gated()

    monkeypatch.setattr(registry, "hf_hub_download", gated)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="is gated.*hf auth login"):
        registry.get_model_config("meta-llama/Llama-3.1-8B-Instruct")
