from typing import Dict, Any


def estimate_parameters(
    config: Dict[str, Any], total_vram: float, num_gpus: int = 1
) -> Dict[str, Any]:
    hidden_size = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    num_attention_heads = config.get("num_attention_heads", 32)
    vocab_size = config.get("vocab_size", 32000)

    param_count = config.get("num_parameters") or config.get("num_params")
    if not param_count:
        n_embed = config.get("n_embd", hidden_size)
        n_head = config.get("n_head", num_attention_heads)
        intermediate_size = config.get("intermediate_size", n_embed * 4)

        embedding_params = vocab_size * n_embed
        attn_params = num_layers * (4 * n_embed * n_embed)
        mlp_params = num_layers * (2 * n_embed * intermediate_size)
        ln_params = num_layers * 5 * n_embed

        param_count = embedding_params + attn_params + mlp_params + ln_params
        param_count = int(param_count * 1.5)

    weights_memory_gb = param_count * 2 / (1024**3)

    kv_cache_per_token_gb = (2 * num_layers * num_attention_heads * hidden_size * 4) / (
        8 * 1024**3
    )

    activation_buffer_gb = max(0.3, hidden_size * num_layers / (1024**3) * 2)

    compile_workspace_gb = 1.0 if total_vram < 8 else 0.4

    min_reserved_gb = max(0.4, total_vram * 0.1)

    if (
        weights_memory_gb
        + activation_buffer_gb
        + compile_workspace_gb
        + min_reserved_gb
        > total_vram * 0.85
    ):
        gpu_memory_utilization = max(
            0.5,
            (total_vram * 0.85 - min_reserved_gb - compile_workspace_gb) / total_vram,
        )
    else:
        reserved_gb = (
            max(1.2, total_vram * 0.35)
            if total_vram < 8
            else 2.0
            if total_vram >= 24
            else max(1.0, total_vram * 0.25)
        )
        gpu_memory_utilization = min(0.95, (total_vram - reserved_gb) / total_vram)
    gpu_memory_utilization = max(0.5, gpu_memory_utilization)

    per_gpu_vram = total_vram / num_gpus

    tensor_parallel_size = 1
    if weights_memory_gb > per_gpu_vram * 0.85:
        tensor_parallel_size = min(
            num_gpus, max(1, int(weights_memory_gb / (per_gpu_vram * 0.75)) + 1)
        )

    per_gpu_weights = weights_memory_gb / tensor_parallel_size
    available_for_kv = per_gpu_vram * gpu_memory_utilization - (
        per_gpu_weights + activation_buffer_gb
    )

    if kv_cache_per_token_gb > 0:
        max_model_len = int(available_for_kv / kv_cache_per_token_gb)
    else:
        max_model_len = 32768
    max_model_len = max(512, min(max_model_len, 32768))

    max_num_seqs = 8 if total_vram < 8 else 32

    per_gpu_min_memory = (
        per_gpu_weights
        + activation_buffer_gb
        + compile_workspace_gb
        + (min_reserved_gb / num_gpus)
    )

    can_fit = True
    recommendations = []

    if per_gpu_weights > per_gpu_vram * 0.95:
        can_fit = False
        recommendations.append(
            f"Model weights ({weights_memory_gb:.2f} GB) require {tensor_parallel_size}x tensor "
            f"parallel but still exceed 95% of per-GPU VRAM ({per_gpu_vram:.1f} GB)"
        )

    if per_gpu_min_memory > per_gpu_vram * 0.95:
        can_fit = False
        recommendations.append(
            f"Memory requirements ({per_gpu_min_memory:.2f} GB per GPU) exceed available VRAM ({per_gpu_vram:.1f} GB)"
        )

    if not can_fit:
        recommendations.append(
            "Consider using a quantized version of the model (e.g., AWQ, GPTQ, 4-bit/8-bit)"
        )
        recommendations.append(
            "Try a smaller model variant (e.g., 7B instead of 70B, or 0.5B instead of 1.5B)"
        )
        recommendations.append(
            "Use --enforce-eager mode to reduce vLLM's memory footprint (may impact performance)"
        )
        if total_vram < 8:
            recommendations.append(
                "Your GPU has limited VRAM; consider cloud GPU options for larger models"
            )

    return {
        "gpu_memory_utilization": round(gpu_memory_utilization, 2),
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "max_num_seqs": max_num_seqs,
        "estimated_weights_memory_gb": round(weights_memory_gb, 2),
        "per_gpu_weights_gb": round(per_gpu_weights, 2),
        "activation_memory_gb": round(activation_buffer_gb, 2),
        "compile_workspace_gb": round(compile_workspace_gb, 2),
        "min_required_memory_gb": round(per_gpu_min_memory * tensor_parallel_size, 2),
        "can_fit": can_fit,
        "recommendations": recommendations,
    }
