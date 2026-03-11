from typing import Dict, Any
import re


def is_gguf_model(model_id: str) -> bool:
    """Check if model is a GGUF model by looking for GGUF suffix or quantization patterns."""
    repo_id = model_id.split(":")[0].upper()
    model_upper = model_id.upper()
    return "-GGUF" in repo_id or "_GGUF" in repo_id or ":Q" in model_upper


def is_model_quantized(config: Dict[str, Any], model_id: str = "") -> bool:
    quant_config = config.get("quantization_config", {})
    if quant_config:
        return True
    if model_id:
        model_upper = model_id.upper()
        if ":Q" in model_upper or "UD-IQ" in model_upper or "UD-Q" in model_upper:
            return True
    return False


def is_awq_quantized(config: Dict[str, Any]) -> bool:
    """Check if model uses AWQ quantization."""
    quant_method = config.get("quantization_config", {}).get("quant_method", "")
    return "awq" in quant_method.lower()


def get_bytes_per_param(config: Dict[str, Any], model_id: str = "") -> float:
    quant_config = config.get("quantization_config", {})

    if not quant_config:
        if model_id:
            model_upper = model_id.upper()
            match = re.search(r":(?:Q|UD-IQ|UD-Q)(\d+)_|UD-IQ(\d+)_", model_upper)
            if match:
                bits = int(match.group(1) if match.group(1) else match.group(2))
                return bits / 8.0
        return 2.0

    bits = quant_config.get("bits", None)
    if bits is not None:
        return bits / 8.0

    quant_method = quant_config.get("quant_method", "")

    if quant_method.lower() == "bitsandbytes":
        if quant_config.get("load_in_4bit", False):
            return 4.0 / 8.0
        if quant_config.get("load_in_8bit", False):
            return 8.0 / 8.0

    method_bit_mapping = {
        "awq": 4,
        "gptq": 4,
        "fp8": 8,
        "fbgemm_fp8": 8,
        "torchao": 4,
    }

    bits = method_bit_mapping.get(quant_method.lower(), 16)
    return bits / 8.0


def estimate_parameters(
    config: Dict[str, Any], total_vram: float, num_gpus: int = 1, model_id: str = ""
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

    bytes_per_param = get_bytes_per_param(config, model_id)
    weights_memory_gb = param_count * bytes_per_param / (1024**3)

    # awq_marlin kernel adds ~0.18-0.2GB runtime overhead on small GPUs
    if is_awq_quantized(config) and total_vram <= 6:
        weights_memory_gb += 0.2

    kv_cache_per_token_gb = (2 * num_layers * num_attention_heads * hidden_size * 4) / (
        8 * 1024**3
    )

    activation_buffer_gb = max(0.3, hidden_size * num_layers / (1024**3) * 2)

    quantized = is_model_quantized(config, model_id)
    if total_vram < 8:
        compile_workspace_gb = max(1.5, total_vram * 0.4) if quantized else 1.0
    else:
        compile_workspace_gb = max(1.2, total_vram * 0.06) if quantized else 0.4

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

    # Further reduce max_model_len for AWQ on small GPUs due to awq_marlin overhead
    if is_awq_quantized(config) and total_vram <= 4:
        max_model_len = min(max_model_len, 256)

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
            f"Memory requirements ({per_gpu_min_memory:.2f} GB per GPU) exceed 95% of available VRAM ({per_gpu_vram:.1f} GB)"
        )

    quantized = is_model_quantized(config, model_id)
    if not can_fit:
        if not quantized:
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

    enforce_eager = False
    if quantized and total_vram <= 4:
        enforce_eager = True
        if can_fit:
            recommendations.append(
                "Using --enforce-eager to avoid torch.compile memory overhead on limited VRAM"
            )

        # Recommend GGUF as an alternative for AWQ on small GPUs
        if is_awq_quantized(config) and not can_fit:
            recommendations.append(
                "Consider using GGUF format models (valid --load-format gguf, more memory-efficient than AWQ)"
            )
        elif is_awq_quantized(config) and total_vram <= 4 and max_model_len < 512:
            recommendations.append(
                f"AWQ fits but with limited context ({max_model_len} tokens). Consider GGUF for better memory efficiency."
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
        "enforce_eager": enforce_eager,
        "recommendations": recommendations,
    }
