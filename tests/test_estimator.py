from vllm_fit.estimator import estimate_parameters
from vllm_fit.registry import extract_repo_id, try_extract_base_model, is_gguf_model


def test_estimate_parameters_basic():
    config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
    }
    vram = 24.0

    result = estimate_parameters(config, vram)

    assert "gpu_memory_utilization" in result
    assert "max_model_len" in result
    assert "tensor_parallel_size" in result
    assert "max_num_seqs" in result
    assert "estimated_weights_memory_gb" in result

    assert 0.5 <= result["gpu_memory_utilization"] <= 0.90
    assert 512 <= result["max_model_len"] <= 32768
    assert result["max_num_seqs"] > 0


def test_estimate_parameters_large_model():
    config = {
        "hidden_size": 8192,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "vocab_size": 128000,
    }
    vram = 24.0

    result = estimate_parameters(config, vram)

    assert result["tensor_parallel_size"] >= 1
    assert result["estimated_weights_memory_gb"] > 10


def test_extract_repo_id():
    assert extract_repo_id("Qwen/Qwen2.5-1.5B") == "Qwen/Qwen2.5-1.5B"
    assert extract_repo_id("Qwen/Qwen3-0.6B-GGUF:Q8_0:Q4_0") == "Qwen/Qwen3-0.6B-GGUF"
    assert extract_repo_id("Qwen/Qwen2.5-1.5B-AWQ") == "Qwen/Qwen2.5-1.5B-AWQ"
    assert (
        extract_repo_id("unsloth/Qwen3-0.6B-GGUF:Q4_K_M") == "unsloth/Qwen3-0.6B-GGUF"
    )
    assert extract_repo_id("model:with:multiple:colons") == "model"


def test_try_extract_base_model():
    candidates = try_extract_base_model("Qwen/Qwen1.5-1.8B-Chat-GGUF")
    assert "Qwen/Qwen1.5-1.8B-Chat-GGUF" in candidates
    assert "Qwen/Qwen1.5-1.8B-Chat" in candidates

    candidates = try_extract_base_model("Qwen/Qwen2.5-1.5B")
    assert "Qwen/Qwen2.5-1.5B" in candidates


def test_is_gguf_model():
    assert is_gguf_model("Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_0")
    assert is_gguf_model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
    assert is_gguf_model("Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    assert is_gguf_model("model:Q4_0")
    assert not is_gguf_model("Qwen/Qwen2.5-1.5B")
    assert not is_gguf_model("Qwen/Qwen2.5-1.5B-Instruct")
