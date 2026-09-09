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
