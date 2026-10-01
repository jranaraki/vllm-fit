"""Regression tests against real HuggingFace config.json files.

The configs in tests/fixtures/hf_configs are verbatim copies from the Hub, so these
tests run offline. Expected values come from the model cards / vLLM, not from
vllm-fit's own output: the maximum context vLLM accepts and how many layers hold a
full or sliding-window KV cache.
"""

import json
from pathlib import Path

import pytest

from vllm_fit.config_resolver import (
    AttentionLayout,
    attention_layout,
    derive_max_model_len,
    get_field,
    resolve_text_config,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hf_configs"

# file stem -> (max context vLLM accepts, expected attention layout)
EXPECTED = {
    "Qwen__Qwen2.5-7B-Instruct-AWQ": (32768, AttentionLayout(28)),
    "Qwen__Qwen3-8B": (40960, AttentionLayout(36)),
    "Qwen__Qwen3-Next-80B-A3B-Instruct": (262144, AttentionLayout(12)),
    "Qwen__Qwen3.5-35B-A3B": (262144, AttentionLayout(10)),
    "deepseek-ai__DeepSeek-V3": (163840, AttentionLayout(61)),
    "moonshotai__Kimi-K2-Instruct": (131072, AttentionLayout(61)),
    "openai__gpt-oss-20b": (131072, AttentionLayout(12, 12, 128)),
    "unsloth__gemma-2-9b-it": (8192, AttentionLayout(21, 21, 4096)),
    "unsloth__gemma-3-27b-it": (131072, AttentionLayout(10, 52, 1024)),
    "unsloth__gemma-3-4b-it": (131072, AttentionLayout(5, 29, 1024)),
    "nvidia__NVIDIA-Nemotron-Nano-9B-v2": (131072, AttentionLayout(4)),
    "ibm-granite__granite-4.0-h-small": (131072, AttentionLayout(4)),
    "ai21labs__Jamba-v0.1": (262144, AttentionLayout(4)),
    "unsloth__Llama-3.1-8B-Instruct": (131072, AttentionLayout(32)),
    "mistralai__Mixtral-8x7B-Instruct-v0.1": (32768, AttentionLayout(32)),
    "mistralai__Mistral-Small-3.1-24B-Instruct-2503": (131072, AttentionLayout(40)),
    "openai-community__gpt2": (1024, AttentionLayout(12)),
}


def _load(stem):
    with open(FIXTURES / f"{stem}.json") as f:
        return json.load(f)


def test_every_expectation_has_a_fixture():
    missing = [stem for stem in EXPECTED if not (FIXTURES / f"{stem}.json").exists()]
    assert not missing


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_max_model_len_matches_vllm(stem):
    assert derive_max_model_len(_load(stem)) == EXPECTED[stem][0]


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_attention_layout(stem):
    config = _load(stem)
    num_layers = int(get_field(resolve_text_config(config), "num_hidden_layers"))
    assert attention_layout(config, num_layers) == EXPECTED[stem][1]
