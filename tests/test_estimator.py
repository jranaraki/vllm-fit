from vllm_fit.estimator import estimate_parameters, _estimate_param_count
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


def test_param_count_counts_moe_experts():
    # A Mixtral-8x7B-shaped config should count all 8 experts (~47B), not a
    # single MLP block (~13B).
    moe = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "intermediate_size": 14336,
        "num_local_experts": 8,
        "hidden_act": "silu",
    }
    dense = {**moe}
    del dense["num_local_experts"]

    moe_params = _estimate_param_count(moe)
    dense_params = _estimate_param_count(dense)

    assert moe_params > 40e9
    # Experts dominate the parameter budget, so MoE must be several times dense.
    assert moe_params > dense_params * 3


def test_param_count_untied_head_adds_output_matrix():
    base = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "intermediate_size": 11008,
        "hidden_act": "silu",
    }
    tied = _estimate_param_count(base)
    untied = _estimate_param_count({**base, "tie_word_embeddings": False})

    assert untied > tied


def test_activation_scales_per_gpu_not_total():
    # Large model spread across GPUs: reported activation memory is per-GPU and
    # must not be charged the full-model figure on every shard.
    config = {
        "hidden_size": 8192,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "vocab_size": 128000,
        "intermediate_size": 28672,
    }
    result = estimate_parameters(config, total_vram=80.0, num_gpus=4)

    if result["tensor_parallel_size"] > 1:
        # Per-GPU activation tracks the per-GPU weight shard (~10%), not the
        # whole model's weights.
        assert result["activation_memory_gb"] <= result["per_gpu_weights_gb"] * 0.1 + 0.31


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
