from vllm_fit.estimator import estimate_parameters


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
