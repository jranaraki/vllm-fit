from vllm_fit.estimator import (
    estimate_parameters,
    estimate_parameters_cpu,
    _estimate_param_count,
    _select_tensor_parallel,
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


def test_gguf_ignores_base_repo_bf16_bytes():
    # Qwen2.5-7B-Instruct-GGUF has no config.json, so weights resolve against the BF16
    # base repo. The quant tag must decide the size, not the base checkpoint's bytes.
    cfg = {"hidden_size": 3584, "num_hidden_layers": 28, "num_attention_heads": 28,
           "num_key_value_heads": 4, "vocab_size": 152064, "intermediate_size": 18944}
    params = 7_615_616_512
    wi = WeightInfo(source="safetensors_metadata", total_params=params,
                    weights_bytes=params * 2, per_dtype={"BF16": params})
    res = estimate_parameters(cfg, total_vram=24.0,
                              model_id="Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
                              weight_info=wi)
    expected = params * 4.89 / 8 / 1024**3
    assert abs(res["estimated_weights_memory_gb"] - round(expected, 2)) < 0.02
    assert res["estimated_weights_memory_gb"] < 5.0  # not the ~14.2 GB BF16 size


def test_gguf_without_quant_tag_keeps_exact_bytes():
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
           "vocab_size": 32000}
    wi = WeightInfo(source="safetensors_metadata", total_params=1_000_000_000,
                    weights_bytes=int(1.5 * 1024**3))
    res = estimate_parameters(cfg, total_vram=24.0, model_id="org/model-GGUF",
                              weight_info=wi)
    assert res["estimated_weights_memory_gb"] == 1.5


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
    # vLLM serves float32 checkpoints at 16-bit (dtype=auto downcasts).
    assert get_bytes_per_param({"torch_dtype": "float32"}) == 2.0


def test_param_count_handles_string_dims():
    # Some configs store dimensions as strings; the analytic ladder must coerce
    # them, not crash on "vocab" * "hidden".
    cfg = {
        "hidden_size": "4096",
        "num_hidden_layers": "2",
        "num_attention_heads": "32",
        "vocab_size": "32000",
        "intermediate_size": "11008",
    }
    n = _estimate_param_count(cfg)
    assert isinstance(n, int)
    assert n > 0


def test_select_tp_excludes_kv_incompatible_degree():
    # heads=28 is divisible by 7, but kv_heads=4 neither shards (4%7) nor replicates
    # (7%4) evenly across tp=7 — vLLM would refuse to start, so it must not be chosen.
    tp = _select_tensor_parallel(
        weights_gb=100.0, per_gpu_vram=5.0, num_gpus=7, num_heads=28, num_kv_heads=4
    )
    assert tp != 7
    assert 28 % tp == 0
    assert (4 % tp == 0) or (tp % 4 == 0)


def test_estimate_parameters_tolerates_zero_gpus():
    # Public library call with a non-positive GPU count must not divide by zero.
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
           "vocab_size": 32000}
    res = estimate_parameters(cfg, total_vram=24.0, num_gpus=0)
    assert res["tensor_parallel_size"] >= 1


def _small_cpu_cfg():
    return {
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "vocab_size": 32000,
        "intermediate_size": 5632,
        "max_position_embeddings": 32768,
    }


def test_cpu_sizing_scales_with_ram():
    # A tiny model on a small quantized footprint: more RAM must buy more context
    # and never exceed the practical CPU caps.
    cfg = _small_cpu_cfg()
    wi = WeightInfo(source="test", weights_bytes=int(1.0 * 1024**3))
    small = estimate_parameters_cpu(cfg, total_ram=16.0, model_id="", weight_info=wi)
    large = estimate_parameters_cpu(cfg, total_ram=64.0, model_id="", weight_info=wi)

    assert large["kv_cache_space_gb"] >= small["kv_cache_space_gb"]
    assert large["max_model_len"] >= small["max_model_len"]
    assert small["max_model_len"] <= 8192 and large["max_model_len"] <= 8192
    assert 1 <= small["max_num_seqs"] <= 8
    assert 1 <= large["max_num_seqs"] <= 8


def test_cpu_sizing_leaves_headroom():
    # The KV space must be a conservative slice of RAM (<=30%), and the full budget
    # (weights + activation + reserved headroom + KV) must fit inside total RAM so the
    # machine stays responsive once vLLM is serving.
    cfg = _small_cpu_cfg()
    wi = WeightInfo(source="test", weights_bytes=int(3.0 * 1024**3))
    total_ram = 32.0
    res = estimate_parameters_cpu(cfg, total_ram=total_ram, model_id="", weight_info=wi)

    assert res["kv_cache_space_gb"] <= total_ram * 0.30 + 1  # +1 for integer rounding
    assert res["min_required_memory_gb"] <= total_ram
    assert res["can_fit"] is True


def test_cpu_sizing_cannot_fit_when_weights_dominate():
    # A 30 GB model on 16 GB RAM leaves no room for a KV cache: fail loud, no env var.
    cfg = _small_cpu_cfg()
    wi = WeightInfo(source="test", weights_bytes=int(30.0 * 1024**3))
    res = estimate_parameters_cpu(cfg, total_ram=16.0, model_id="", weight_info=wi)

    assert res["can_fit"] is False
    assert res["kv_cache_space_gb"] < 1
    assert res["recommendations"]


def test_cpu_max_model_len_capped_at_model_context():
    # Abundant RAM must not push max_model_len past the model's real context window.
    cfg = {**_small_cpu_cfg(), "max_position_embeddings": 4096}
    wi = WeightInfo(source="test", weights_bytes=int(1.0 * 1024**3))
    res = estimate_parameters_cpu(cfg, total_ram=128.0, model_id="", weight_info=wi)
    assert res["max_model_len"] <= 4096


def test_is_gguf_model():
    assert is_gguf_model("Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_0")
    assert is_gguf_model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
    assert is_gguf_model("Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    assert is_gguf_model("model:Q4_0")
    assert not is_gguf_model("Qwen/Qwen2.5-1.5B")
    assert not is_gguf_model("Qwen/Qwen2.5-1.5B-Instruct")


def test_utilization_capped_by_free_vram():
    # 8 GB card with ~1 GB held by the display: 0.90 x 8 = 7.2 GB > 7.0 GB free, which
    # vLLM rejects at startup. The recommendation must fit inside free memory.
    cfg = {"hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 16,
           "vocab_size": 32000}
    wi = WeightInfo(source="test", weights_bytes=int(2 * 1024**3))
    res = estimate_parameters(cfg, total_vram=8.0, weight_info=wi, free_vram=7.0)
    util = res["gpu_memory_utilization"]
    assert util * 8.0 <= 7.0 - 0.5
    assert any("already in use" in w for w in res["warnings"])


def test_idle_gpu_keeps_default_utilization():
    cfg = {"hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 16,
           "vocab_size": 32000}
    wi = WeightInfo(source="test", weights_bytes=int(2 * 1024**3))
    idle = estimate_parameters(cfg, total_vram=24.0, weight_info=wi, free_vram=23.6)
    unknown = estimate_parameters(cfg, total_vram=24.0, weight_info=wi)
    assert idle["gpu_memory_utilization"] == unknown["gpu_memory_utilization"] == 0.9
    assert not any("already in use" in w for w in idle["warnings"])


def test_busy_gpu_cannot_fit_when_free_memory_too_small():
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
           "vocab_size": 32000}
    wi = WeightInfo(source="test", weights_bytes=int(14 * 1024**3))
    res = estimate_parameters(cfg, total_vram=24.0, weight_info=wi, free_vram=10.0)
    assert res["can_fit"] is False


def test_activation_sized_for_vllm_serve_default_batch():
    from vllm_fit.estimator import _serve_default_batched_tokens

    assert _serve_default_batched_tokens(24.0) == 2048
    assert _serve_default_batched_tokens(79.6) == 8192   # H100 80GB
    assert _serve_default_batched_tokens(178.0) == 16384  # B200
    cfg = {"hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64,
           "num_key_value_heads": 8, "vocab_size": 128256, "intermediate_size": 28672}
    small = estimate_parameters(cfg, total_vram=48.0 * 4, num_gpus=4)
    big = estimate_parameters(cfg, total_vram=79.6 * 4, num_gpus=4)
    assert big["activation_memory_gb"] > small["activation_memory_gb"] * 3


def _gpt_oss_like():
    return {"hidden_size": 2880, "num_hidden_layers": 24, "num_attention_heads": 64,
            "num_key_value_heads": 8, "head_dim": 64, "vocab_size": 201088,
            "intermediate_size": 2880, "max_position_embeddings": 131072,
            "sliding_window": 128,
            "layer_types": ["sliding_attention", "full_attention"] * 12}


def test_sliding_layers_do_not_inflate_concurrency():
    # Full-attention layers still hold every token, so concurrency at full context is
    # bounded by them, not by the 128-token window.
    wi = WeightInfo(source="test", weights_bytes=int(12.8 * 1024**3))
    res = estimate_parameters(_gpt_oss_like(), total_vram=80.0, weight_info=wi)
    full_layer_bytes = 12 * 2 * 8 * 64 * 2 * res["max_model_len"]
    budget = (res["gpu_memory_utilization"] * 80.0 - 12.8) * 1024**3
    assert res["max_num_seqs"] <= budget / full_layer_bytes


def test_sliding_layers_cost_less_than_full():
    wi = WeightInfo(source="test", weights_bytes=int(50 * 1024**3))
    cfg = {"model_type": "gemma3_text", "hidden_size": 5376, "num_hidden_layers": 62,
           "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 128,
           "vocab_size": 262208, "intermediate_size": 21504,
           "max_position_embeddings": 131072, "sliding_window": 1024,
           "sliding_window_pattern": 6}
    hybrid = estimate_parameters(cfg, total_vram=80.0, weight_info=wi)
    all_full = estimate_parameters({**cfg, "sliding_window": None}, total_vram=80.0,
                                   weight_info=wi)
    assert hybrid["max_model_len"] > 3 * all_full["max_model_len"]


def test_idle_gpu_reservation_caps_without_warning():
    # A 4 GB card reports ~0.3 GB used when idle: utilization is still capped to what
    # vLLM's startup check allows, but nobody is told "other processes" hold memory.
    cfg = {"hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16,
           "num_key_value_heads": 8, "vocab_size": 151936}
    wi = WeightInfo(source="test", weights_bytes=int(1.4 * 1024**3))
    res = estimate_parameters(cfg, total_vram=4.0, weight_info=wi, free_vram=3.7)
    assert res["gpu_memory_utilization"] * 4.0 <= 3.7 - 0.5
    assert not any("already in use" in w for w in res["warnings"])


def test_recurrent_state_mamba2_nemotron_shape():
    from vllm_fit.config_resolver import attention_layout
    from vllm_fit.estimator import _recurrent_state_bytes

    pattern = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
    cfg = {"hidden_size": 4480, "num_hidden_layers": 56, "hybrid_override_pattern": pattern,
           "mamba_num_heads": 128, "mamba_head_dim": 80, "ssm_state_size": 128,
           "n_groups": 8, "conv_kernel": 4}
    layout = attention_layout(cfg, 56)
    per_layer = 3 * (128 * 80 + 2 * 8 * 128) * 2 + 128 * 80 * 128 * 2
    assert _recurrent_state_bytes(cfg, 56, layout) == pattern.count("M") * per_layer
    assert _recurrent_state_bytes(cfg, 56, layout, tensor_parallel_size=2) == pattern.count("M") * per_layer / 2


def test_recurrent_state_gated_deltanet_qwen3_next_shape():
    from vllm_fit.config_resolver import attention_layout
    from vllm_fit.estimator import _recurrent_state_bytes

    cfg = {"hidden_size": 2048, "num_hidden_layers": 48, "full_attention_interval": 4,
           "linear_num_value_heads": 32, "linear_num_key_heads": 16,
           "linear_key_head_dim": 128, "linear_value_head_dim": 128,
           "linear_conv_kernel_dim": 4}
    layout = attention_layout(cfg, 48)
    per_layer = (2 * 16 * 128 + 32 * 128) * 3 * 2 + 32 * 128 * 128 * 2
    assert _recurrent_state_bytes(cfg, 48, layout) == 36 * per_layer


def test_pure_attention_has_no_recurrent_state():
    from vllm_fit.config_resolver import attention_layout
    from vllm_fit.estimator import _recurrent_state_bytes

    cfg = {"hidden_size": 4096, "num_hidden_layers": 32}
    assert _recurrent_state_bytes(cfg, 32, attention_layout(cfg, 32)) == 0.0


def test_recurrent_state_reduces_concurrency():
    # A short-context hybrid model: per-request state, not KV, dominates.
    base = {"hidden_size": 4096, "num_hidden_layers": 40, "num_attention_heads": 32,
            "num_key_value_heads": 8, "vocab_size": 100352, "intermediate_size": 8192,
            "max_position_embeddings": 2048,
            "layer_types": ["attention"] * 4 + ["mamba"] * 36}
    mamba = {**base, "mamba_n_heads": 128, "mamba_d_head": 64, "mamba_d_state": 128,
             "mamba_n_groups": 1, "mamba_d_conv": 4}
    wi = WeightInfo(source="test", weights_bytes=int(8 * 1024**3))
    no_state = estimate_parameters({**base, "layer_types": ["attention"] * 4 + ["other"] * 36},
                                   total_vram=24.0, weight_info=wi)
    with_state = estimate_parameters(mamba, total_vram=24.0, weight_info=wi)
    assert with_state["max_num_seqs"] < no_state["max_num_seqs"]


def test_kv_request_rounds_to_blocks_and_reserves_null_block():
    from vllm_fit.config_resolver import AttentionLayout
    from vllm_fit.estimator import (
        KV_BLOCK_SIZE, _kv_bytes_per_request, _kv_pool_bytes, _largest_fitting_len)

    layout = AttentionLayout(32)
    per_token = 2 * 8 * 128 * 2  # one layer's K+V bytes per token
    block = KV_BLOCK_SIZE * per_token * 32
    # 1000 tokens needs 63 blocks of 16, not 62.5.
    assert _kv_bytes_per_request(layout, per_token, 1000, 2048) == 63 * block
    # A pool of exactly 100 blocks leaves 99 for requests (one null block).
    pool = _kv_pool_bytes(layout, per_token, 100 * block)
    assert pool == 99 * block
    best = _largest_fitting_len(pool, 131072, 256,
                                lambda n: _kv_bytes_per_request(layout, per_token, n, 2048))
    assert best == 99 * KV_BLOCK_SIZE


def test_sliding_request_matches_sliding_window_spec():
    from vllm_fit.config_resolver import AttentionLayout
    from vllm_fit.estimator import KV_BLOCK_SIZE, _kv_bytes_per_request

    layout = AttentionLayout(0, 10, 1024)
    # cdiv(min(window - 1 + 2 in-flight batches, len), 16) + 1 blocks per sliding layer.
    expected_blocks = -(-(1024 - 1 + 2 * 2048) // 16) + 1
    assert _kv_bytes_per_request(layout, 1.0, 131072, 2048) == 10 * expected_blocks * KV_BLOCK_SIZE


# Real vllm-metal startups on a 48 GB Apple Silicon Mac (vLLM 0.30, vllm-metal 0.30) at
# --gpu-memory-utilization 0.9: Metal working set 40,200,896,512 bytes; the plugin's
# "Upstream cache layout: reporting N GB KV budget" line gave the measured KV budgets.
_METAL_WORKING_SET_GB = 40200896512 / 1024**3


def _qwen3_0_6b():
    return {"model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28,
            "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
            "intermediate_size": 3072, "vocab_size": 151936,
            "max_position_embeddings": 40960, "tie_word_embeddings": True}


def _qwen3_coder_30b_a3b():
    return {"model_type": "qwen3_moe", "hidden_size": 2048, "num_hidden_layers": 48,
            "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
            "intermediate_size": 5472, "moe_intermediate_size": 768, "num_experts": 128,
            "num_experts_per_tok": 8, "vocab_size": 151936,
            "max_position_embeddings": 262144}


import pytest  # noqa: E402


@pytest.mark.parametrize("config,weights_bytes,measured_kv_bytes", [
    (_qwen3_0_6b(), 1_503_264_768, 34.12e9),
    (_qwen3_coder_30b_a3b(), 17_178_000_000, 18.13e9),
])
def test_metal_kv_budget_matches_real_vllm_metal(config, weights_bytes, measured_kv_bytes):
    from vllm_fit.estimator import estimate_parameters_metal

    wi = WeightInfo(source="test", weights_bytes=weights_bytes)
    res = estimate_parameters_metal(config, _METAL_WORKING_SET_GB, weight_info=wi)
    predicted = res["kv_cache_memory_gb"] * 1024**3
    # Never more than vLLM actually allocated, and within 5% of it.
    assert 0.95 * measured_kv_bytes <= predicted <= measured_kv_bytes
    assert res["can_fit"] and res["tensor_parallel_size"] == 1
    assert res["gpu_memory_utilization"] == 0.9


def test_metal_caps_utilization_by_free_ram():
    from vllm_fit.estimator import estimate_parameters_metal

    wi = WeightInfo(source="test", weights_bytes=1_503_264_768)
    res = estimate_parameters_metal(_qwen3_0_6b(), _METAL_WORKING_SET_GB, weight_info=wi,
                                    available_ram_gb=20.0)
    assert res["gpu_memory_utilization"] * _METAL_WORKING_SET_GB <= 20.0 - 2.0
    assert any("free right now" in w for w in res["warnings"])


def test_metal_unmeasured_working_set_warns_and_too_big_model_doesnt_fit():
    from vllm_fit.estimator import estimate_parameters_metal

    wi = WeightInfo(source="test", weights_bytes=int(40 * 1024**3))
    res = estimate_parameters_metal(_qwen3_0_6b(), 32.0, weight_info=wi,
                                    working_set_measured=False)
    assert not res["can_fit"]
    assert any("working-set limit" in w for w in res["warnings"])



def test_gpt_oss_concurrency_matches_real_vllm_metal():
    # mlx-community/gpt-oss-20b-MXFP4-Q8 on vllm-metal at max_model_len 131072:
    # vLLM reported 59,003 KV blocks and "Maximum concurrency ... 6.98x".
    from vllm_fit.config_resolver import AttentionLayout
    from vllm_fit.estimator import KV_BLOCK_SIZE, _kv_bytes_per_request

    layout = AttentionLayout(12, 12, 128)
    per_layer_token = 2 * 8 * 64 * 2
    block_bytes = KV_BLOCK_SIZE * per_layer_token * 12  # one block across a 12-layer group
    request_blocks = _kv_bytes_per_request(layout, per_layer_token, 131072, 2048) / block_bytes
    assert abs(59003 / request_blocks - 6.98) < 0.005
