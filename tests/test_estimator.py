from vllm_fit.estimator import (
    estimate_parameters,
    _estimate_param_count,
    get_bytes_per_param,
)
from vllm_fit.params import WeightInfo
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


def _qwen35_text_dims():
    return {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "intermediate_size": 27648,
        "vocab_size": 152064,
        "max_position_embeddings": 32768,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
    }


def test_nested_config_reads_text_config_not_defaults():
    # The reported bug: a nested/multimodal config was sized against hard-coded
    # defaults (~17 GB) instead of the real dims under text_config (~65 GB).
    nested = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "vision_config": {"hidden_size": 1152, "num_hidden_layers": 27},
        "text_config": _qwen35_text_dims(),
    }
    flat = _qwen35_text_dims()

    nested_res = estimate_parameters(nested, total_vram=24.0)
    flat_res = estimate_parameters(flat, total_vram=24.0)

    # Resolution works: nested is sized identically to the flattened dims.
    assert nested_res["estimated_weights_memory_gb"] == flat_res["estimated_weights_memory_gb"]
    # And it is the real ~65 GB model, not the ~17 GB defaults result.
    assert nested_res["estimated_weights_memory_gb"] > 40
    # A ~65 GB model cannot fit on a single 24 GB GPU.
    assert nested_res["can_fit"] is False


def test_missing_field_emits_warning():
    # A config genuinely lacking core dims must warn (fail loud), not silently default.
    res = estimate_parameters({"vocab_size": 32000}, total_vram=24.0)
    assert any("hidden_size" in w or "num_hidden_layers" in w for w in res["warnings"])


def test_weight_info_overrides_analytic():
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
           "vocab_size": 32000}
    wi = WeightInfo(source="safetensors_metadata", total_params=1_000_000_000,
                    weights_bytes=int(1.5 * 1024**3))
    res = estimate_parameters(cfg, total_vram=24.0, weight_info=wi)
    assert res["estimated_weights_memory_gb"] == 1.5
    # Exact metadata means no "analytic estimate" warning.
    assert not any("analytic" in w for w in res["warnings"])


def test_tp_respects_head_divisibility():
    cfg = {"hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 32,
           "vocab_size": 128000, "intermediate_size": 28672}
    res = estimate_parameters(cfg, total_vram=48.0, num_gpus=6)
    tp = res["tensor_parallel_size"]
    assert 32 % tp == 0
    assert tp in (1, 2, 4)  # never 3, 5, or 6


def test_mla_gives_more_context_than_dense():
    # MLA stores a tiny latent per token, so the same-size model should support a
    # much longer max_model_len than a naive MHA KV cache.
    base = {"hidden_size": 5120, "num_hidden_layers": 60, "num_attention_heads": 128,
            "num_key_value_heads": 128, "vocab_size": 129280, "intermediate_size": 12288,
            "max_position_embeddings": 163840}
    mla = {**base, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
    wi = WeightInfo(source="test", weights_bytes=int(20 * 1024**3))
    base_res = estimate_parameters(base, total_vram=80.0, weight_info=wi)
    mla_res = estimate_parameters(mla, total_vram=80.0, weight_info=wi)
    assert mla_res["max_model_len"] > base_res["max_model_len"]


def test_hybrid_layers_give_more_context():
    # Only counting full-attention layers frees KV budget vs charging every layer.
    dims = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
            "num_key_value_heads": 8, "vocab_size": 32000, "intermediate_size": 14336,
            "max_position_embeddings": 131072}
    dense = dict(dims)
    hybrid = {**dims, "full_attention_interval": 4}
    wi = WeightInfo(source="test", weights_bytes=int(8 * 1024**3))
    dense_res = estimate_parameters(dense, total_vram=24.0, weight_info=wi)
    hybrid_res = estimate_parameters(hybrid, total_vram=24.0, weight_info=wi)
    assert hybrid_res["max_model_len"] > dense_res["max_model_len"]


def test_max_model_len_capped_at_model_context():
    # Even with abundant VRAM, max_model_len must not exceed the model's real context.
    cfg = {"hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 16,
           "num_key_value_heads": 2, "vocab_size": 32000, "intermediate_size": 5632,
           "max_position_embeddings": 4096}
    wi = WeightInfo(source="test", weights_bytes=int(2 * 1024**3))
    res = estimate_parameters(cfg, total_vram=80.0, weight_info=wi)
    assert res["max_model_len"] <= 4096


def test_enforce_eager_lever_flips_borderline_fit():
    # A model that overflows by less than the CUDA-graph reserve should be made to
    # fit by the enforce-eager lever, and the reported field must say so (no silent
    # inconsistency between can_fit, enforce_eager, and the emitted command).
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
           "num_key_value_heads": 8, "vocab_size": 32000, "intermediate_size": 14336,
           "max_position_embeddings": 32768}
    wi = WeightInfo(source="test", weights_bytes=int(12.8 * 1024**3))
    res = estimate_parameters(cfg, total_vram=16.0, num_gpus=1, weight_info=wi)
    assert res["can_fit"] is True
    assert res["enforce_eager"] is True
    # With the lever engaged, the CUDA-graph workspace is zeroed.
    assert res["compile_workspace_gb"] == 0.0


def test_enforce_eager_field_honest_when_it_cannot_help():
    # 27B bf16 on 4x15GB: even zeroing the CUDA-graph reserve leaves no KV room, so
    # the estimator must NOT claim enforce_eager fixed it.
    cfg = {"hidden_size": 5120, "num_hidden_layers": 64, "num_attention_heads": 40,
           "num_key_value_heads": 8, "vocab_size": 152064, "intermediate_size": 27648,
           "max_position_embeddings": 40960}
    wi = WeightInfo(source="test", weights_bytes=int(51.75 * 1024**3))
    res = estimate_parameters(cfg, total_vram=60.0, num_gpus=4, weight_info=wi)
    assert res["can_fit"] is False
    assert res["enforce_eager"] is False


def test_bytes_per_param_compressed_tensors():
    cfg = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {"weights": {"num_bits": 4}, "input_activations": {"num_bits": 8}}
            },
        }
    }
    assert get_bytes_per_param(cfg) == 0.5


def test_bytes_per_param_compressed_tensors_2of4_sparsity():
    cfg = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": {"num_bits": 8}}},
            "sparsity_config": {"sparsity_structure": "2:4"},
        }
    }
    assert get_bytes_per_param(cfg) == 0.5  # 8 bits -> 1.0 byte, halved by 2:4


def test_bytes_per_param_gguf_effective_bpw():
    assert abs(get_bytes_per_param({}, "org/model-GGUF:Q4_K_M") - 4.89 / 8) < 1e-9
    assert abs(get_bytes_per_param({}, "org/model-GGUF:Q8_0") - 8.5 / 8) < 1e-9


def test_bytes_per_param_fp8_dtype():
    assert get_bytes_per_param({"dtype": "float8_e4m3fn"}) == 1.0
    assert get_bytes_per_param({"dtype": "bfloat16"}) == 2.0
    assert get_bytes_per_param({"torch_dtype": "float32"}) == 4.0


def test_is_gguf_model():
    assert is_gguf_model("Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_0")
    assert is_gguf_model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
    assert is_gguf_model("Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    assert is_gguf_model("model:Q4_0")
    assert not is_gguf_model("Qwen/Qwen2.5-1.5B")
    assert not is_gguf_model("Qwen/Qwen2.5-1.5B-Instruct")
