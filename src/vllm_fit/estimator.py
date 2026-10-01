from typing import Dict, Any, Optional
import math
import re

from .config_resolver import (
    resolve_text_config,
    get_field,
    get_head_dim,
    derive_max_model_len,
    detect_mla,
    AttentionLayout,
    attention_layout,
    recurrent_state_layer_count,
    crosscheck_with_transformers,
)


# Memory vLLM's worker allocates for its own CUDA context before it checks that free
# memory covers gpu_memory_utilization x total.
_CUDA_CONTEXT_GB = 0.5

# Memory an otherwise idle GPU reports as used (driver/ECC reservations).
_IDLE_GPU_USED_GB = 0.75


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


# Effective bits-per-weight for GGUF quant types. GGUF k-quants pack scales and
# mins alongside the weights, so the *effective* footprint differs from the
# nominal bit count (e.g. Q4_K_M is 4.89 bpw, not 4). Values from llama.cpp.
_GGUF_BPW = {
    "IQ1_S": 1.56, "IQ1_M": 1.75,
    "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5, "IQ2_M": 2.7, "Q2_K": 2.63, "Q2_K_S": 2.16,
    "IQ3_XXS": 3.06, "IQ3_XS": 3.3, "IQ3_S": 3.44, "IQ3_M": 3.66,
    "Q3_K_S": 3.44, "Q3_K_M": 3.66, "Q3_K_L": 3.9, "Q3_K": 3.66,
    "IQ4_XS": 4.25, "IQ4_NL": 4.5,
    "Q4_0": 4.55, "Q4_1": 5.0, "Q4_K_S": 4.58, "Q4_K_M": 4.89, "Q4_K": 4.89,
    "Q5_0": 5.54, "Q5_1": 6.0, "Q5_K_S": 5.52, "Q5_K_M": 5.69, "Q5_K": 5.69,
    "Q6_K": 6.59,
    "Q8_0": 8.5, "Q8_1": 9.0, "Q8_K": 8.5,
    "F16": 16.0, "BF16": 16.0, "F32": 32.0,
}


def _gguf_bpw_from_id(model_id: str) -> Optional[float]:
    """Effective bits-per-weight for a GGUF model id like 'repo-GGUF:Q4_K_M'."""
    if ":" not in model_id:
        return None
    tag = model_id.split(":")[-1].upper().replace("UD-", "")
    if tag in _GGUF_BPW:
        return _GGUF_BPW[tag]
    # Longest known prefix (handles suffixed community tags).
    for key in sorted(_GGUF_BPW, key=len, reverse=True):
        if tag.startswith(key):
            return _GGUF_BPW[key]
    m = re.match(r"[IU]?Q(\d+)", tag)
    if m:
        return float(m.group(1))
    return None


def _dtype_bytes(config: Dict[str, Any]) -> float:
    """Bytes per weight for an unquantized model, based on the config dtype.

    Reads the current ``dtype`` field (transformers renamed ``torch_dtype`` ->
    ``dtype`` around v4.56) with a ``torch_dtype`` fallback.
    """
    dtype = str(config.get("dtype") or config.get("torch_dtype") or "").lower()
    # float32 checkpoints are served at 16-bit: vLLM's dtype=auto downcasts them.
    if dtype in ("float32", "float", "fp32", "float16", "half", "fp16", "bfloat16", "bf16"):
        return 2.0
    if dtype in ("float8", "fp8", "float8_e4m3fn", "float8_e5m2", "e4m3", "e5m2", "f8_e4m3", "f8_e5m2"):
        return 1.0
    if dtype in ("int8", "uint8", "i8", "u8"):
        return 1.0
    # Default to fp16 (the common serving dtype) when unspecified.
    return 2.0


def _compressed_tensors_bytes(quant_config: Dict[str, Any]) -> Optional[float]:
    """Weight bytes-per-param for a compressed-tensors config.

    compressed-tensors has no top-level ``bits``; the width lives in
    ``config_groups.*.weights.num_bits``. Only the ``weights`` block is stored
    quantized (``input_activations`` is runtime-only). We take the widest group as a
    conservative (memory-safe) estimate.
    """
    groups = quant_config.get("config_groups") or {}
    widths = []
    for group in groups.values():
        if isinstance(group, dict):
            weights = group.get("weights") or {}
            num_bits = weights.get("num_bits")
            if isinstance(num_bits, (int, float)) and num_bits > 0:
                widths.append(float(num_bits))
    if widths:
        return max(widths) / 8.0
    return None


def _apply_sparsity(bytes_pp: float, config: Dict[str, Any], quant_config: Dict[str, Any]) -> float:
    """Halve storage for 2:4 structured sparsity (compressed-tensors)."""
    sparsity = quant_config.get("sparsity_config") or config.get("sparsity_config") or {}
    if isinstance(sparsity, dict):
        structure = str(sparsity.get("sparsity_structure", "")).lower()
        if structure in ("2:4", "2_4"):
            return bytes_pp * 0.5
    return bytes_pp


def get_bytes_per_param(config: Dict[str, Any], model_id: str = "") -> float:
    # GGUF: derive effective bits-per-weight from the quant tag in the id.
    if model_id:
        gguf_bpw = _gguf_bpw_from_id(model_id)
        if gguf_bpw is not None:
            return gguf_bpw / 8.0

    quant_config = config.get("quantization_config", {}) or {}

    if not quant_config:
        # Legacy id-based quant hint (UD-IQ*/UD-Q*), else fall back to dtype.
        if model_id:
            match = re.search(r":(?:Q|UD-IQ|UD-Q)(\d+)_|UD-IQ(\d+)_", model_id.upper())
            if match:
                bits = int(match.group(1) if match.group(1) else match.group(2))
                return bits / 8.0
        return _dtype_bytes(config)

    bits = quant_config.get("bits", None)
    if bits is not None:
        return _apply_sparsity(bits / 8.0, config, quant_config)

    quant_method = str(quant_config.get("quant_method", "")).lower()

    if quant_method == "compressed-tensors":
        ct = _compressed_tensors_bytes(quant_config)
        if ct is not None:
            return _apply_sparsity(ct, config, quant_config)

    if quant_method == "bitsandbytes":
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
        "nvfp4": 4,
        "mxfp4": 4,
    }
    bits = method_bit_mapping.get(quant_method, 16)
    return _apply_sparsity(bits / 8.0, config, quant_config)


# MLP activations that are NOT gated (2-matrix MLP); everything else is assumed
# to be a gated MLP (SwiGLU/GeGLU-style, 3 matrices), which covers most modern
# LLMs (Llama, Qwen, Mistral) and Gemma's gated ``gelu_pytorch_tanh``.
_NON_GATED_ACTS = {"gelu", "gelu_new", "gelu_fast", "quick_gelu", "relu"}


def _moe_layer_split(config: Dict[str, Any], num_layers: int) -> tuple:
    """Return (num_moe_layers, num_dense_layers).

    Only some layers carry the expert stack. Honors the common conventions:
    DeepSeek ``first_k_dense_replace`` (first K layers dense), Qwen
    ``mlp_only_layers`` (listed layers dense) and ``decoder_sparse_step`` /
    ``moe_layer_freq`` (every step-th layer is MoE). Falls back to all-MoE.
    """
    num_experts = (
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
        or (config.get("ffn_config") or {}).get("moe_num_experts")
    )
    if not num_experts or num_experts <= 1:
        return 0, num_layers

    step = config.get("decoder_sparse_step") or config.get("moe_layer_freq") or 1
    step = step if isinstance(step, int) and step > 0 else 1

    fkd = config.get("first_k_dense_replace")
    if isinstance(fkd, int):
        moe = max(0, num_layers - fkd)
        moe = moe // step if step > 1 else moe
        return moe, num_layers - moe

    mlp_only = config.get("mlp_only_layers")
    if isinstance(mlp_only, list):
        dense = len([i for i in mlp_only if isinstance(i, int) and 0 <= i < num_layers])
        moe = num_layers - dense
        moe = moe // step if step > 1 else moe
        return moe, num_layers - moe

    if step > 1:
        moe = num_layers // step
        return moe, num_layers - moe

    return num_layers, 0


def _estimate_param_count(config: Dict[str, Any]) -> int:
    """Estimate total parameter count from architecture fields.

    Accounts for gated MLPs (3 matrices), grouped-query attention (fewer KV heads),
    mixture-of-experts (experts only on MoE layers, plus shared experts) and an
    untied output head. This is the analytic *fallback* used when exact
    safetensors metadata isn't available (offline / GGUF).
    """
    config = resolve_text_config(config)

    param_count = config.get("num_parameters") or config.get("num_params")
    if param_count:
        return int(param_count)

    # Coerce every dimension to int: some configs store these as strings, and the
    # arithmetic below would otherwise raise (e.g. "32000" * "4096").
    n_embed = int(get_field(config, "hidden_size", 4096))
    num_layers = int(get_field(config, "num_hidden_layers", 32))
    num_heads = int(get_field(config, "num_attention_heads", 32))
    num_kv_heads = int(get_field(config, "num_key_value_heads", num_heads))
    head_dim = int(get_head_dim(config) or (n_embed // num_heads if num_heads else n_embed))
    vocab_size = int(get_field(config, "vocab_size", 32000))
    intermediate_size = int(get_field(config, "intermediate_size", n_embed * 4))

    kv_dim = num_kv_heads * head_dim
    q_dim = num_heads * head_dim

    embedding_params = vocab_size * n_embed
    # Q and O projections are full width; K and V shrink under GQA.
    attn_params = num_layers * (2 * n_embed * q_dim + 2 * n_embed * kv_dim)

    hidden_act = str(config.get("hidden_act", "")).lower()
    mlp_matrices = 2 if hidden_act in _NON_GATED_ACTS else 3

    num_experts = int(
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
        or (config.get("ffn_config") or {}).get("moe_num_experts")
        or 1
    )
    moe_layers, dense_layers = _moe_layer_split(config, num_layers)

    mlp_params = dense_layers * (mlp_matrices * n_embed * intermediate_size)

    if moe_layers:
        expert_intermediate = (
            config.get("moe_intermediate_size")
            or (config.get("ffn_config") or {}).get("ffn_hidden_size")
            or intermediate_size
        )
        mlp_params += moe_layers * num_experts * (mlp_matrices * n_embed * expert_intermediate)
        # Shared/always-on experts (DeepSeek, Qwen2-MoE) run every MoE layer.
        shared_inter = config.get("shared_expert_intermediate_size")
        n_shared = config.get("n_shared_experts") or config.get("num_shared_experts")
        if shared_inter:
            mlp_params += moe_layers * (mlp_matrices * n_embed * int(shared_inter))
        elif n_shared:
            mlp_params += moe_layers * int(n_shared) * (mlp_matrices * n_embed * expert_intermediate)

    ln_params = num_layers * 5 * n_embed

    param_count = embedding_params + attn_params + mlp_params + ln_params
    # An untied output head is a second vocab x hidden matrix; when weights are
    # tied it reuses the embedding and adds nothing.
    if config.get("tie_word_embeddings") is False:
        param_count += vocab_size * n_embed
    return int(param_count)


def _activation_peak_gb(hidden_size: int, intermediate_size: int, max_num_batched_tokens: int = 2048) -> float:
    """Transient activation headroom vLLM reserves during a forward pass.

    Scales with the prefill batch (``max_num_batched_tokens``, default 2048), not
    with model weights: a handful of large concurrent tensors of width ~hidden and
    ~intermediate, at 2 bytes.
    """
    dtype_bytes = 2
    inter = intermediate_size or (hidden_size * 4)
    per_token = (4 * hidden_size + 2 * inter) * dtype_bytes
    return max(0.3, max_num_batched_tokens * per_token / (1024**3))


def _select_tensor_parallel(
    weights_gb: float, per_gpu_vram: float, num_gpus: int, num_heads: int, num_kv_heads: int
) -> int:
    """Smallest tensor-parallel degree that fits the weight shard, respecting vLLM's
    head-divisibility constraints on both attention and KV heads.

    vLLM requires ``num_attention_heads % tp == 0`` and, for the KV heads, either
    ``num_kv_heads % tp == 0`` (sharded) or ``tp % num_kv_heads == 0`` (replicated).
    A ``tp`` that violates the KV rule makes vLLM refuse to start, so we exclude it
    from the candidate set. Choosing the smallest fitting degree naturally minimizes
    KV replication."""
    if num_gpus <= 1:
        return 1

    def _valid_tp(tp: int) -> bool:
        if not num_heads or num_heads % tp != 0:
            return False
        if num_kv_heads:
            return num_kv_heads % tp == 0 or tp % num_kv_heads == 0
        return True

    candidates = [tp for tp in range(1, num_gpus + 1) if _valid_tp(tp)]
    if not candidates:
        candidates = [1]
    # Leave ~40% of each GPU for KV + activation + overheads.
    for tp in candidates:
        if weights_gb / tp <= per_gpu_vram * 0.6:
            return tp
    return candidates[-1]


def _kv_bytes_per_layer_token(
    config: Dict[str, Any],
    num_kv_heads: int,
    head_dim: int,
    tensor_parallel_size: int = 1,
    kv_dtype_bytes: int = 2,
) -> float:
    """KV-cache bytes one attention layer stores per token on one GPU.

    Divides KV heads across tensor-parallel ranks (floor 1), and uses the compact MLA
    latent when present (replicated across TP ranks, not sharded; no K/V pair).
    """
    mla = detect_mla(config)
    if mla:
        return (mla["kv_lora_rank"] + mla["qk_rope_head_dim"]) * kv_dtype_bytes
    kv_heads_per_gpu = max(1, num_kv_heads // tensor_parallel_size)
    return 2 * kv_heads_per_gpu * head_dim * kv_dtype_bytes


def _recurrent_state_bytes(
    config: Dict[str, Any],
    num_layers: int,
    layout: AttentionLayout,
    tensor_parallel_size: int = 1,
    dtype_bytes: int = 2,
) -> float:
    """Fixed per-request state of Mamba / linear-attention layers on one GPU.

    Shapes follow vLLM's MambaStateShapeCalculator: a short convolution state plus the
    recurrent (SSM / delta-rule) state, both sharded across tensor-parallel ranks.
    """
    layers = recurrent_state_layer_count(config, num_layers, layout)
    if not layers:
        return 0.0
    tc = resolve_text_config(config)
    tp = max(1, tensor_parallel_size)

    def first(*keys):
        for k in keys:
            v = tc.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
        return None

    hidden = first("hidden_size") or 0
    ssm_bytes = 4 if str(tc.get("mamba_ssm_dtype", "")).lower() == "float32" else dtype_bytes

    if first("linear_num_value_heads"):  # Gated DeltaNet (Qwen3-Next, Qwen3.5)
        v_heads = first("linear_num_value_heads")
        k_heads = first("linear_num_key_heads") or v_heads
        k_dim = first("linear_key_head_dim") or 128
        v_dim = first("linear_value_head_dim") or 128
        kernel = first("linear_conv_kernel_dim") or 4
        conv = (2 * k_heads * k_dim + v_heads * v_dim) * (kernel - 1) * dtype_bytes
        ssm = v_heads * k_dim * v_dim * ssm_bytes
    else:
        d_state = first("ssm_state_size", "mamba_d_state", "mamba_state_dim") or 128
        kernel = first("conv_kernel", "mamba_d_conv") or 4
        heads = first("mamba_num_heads", "mamba_n_heads")
        head_dim = first("mamba_head_dim", "mamba_d_head")
        if heads and head_dim:  # Mamba-2
            groups = first("n_groups", "mamba_n_groups", "mamba_num_groups") or 1
            inner = heads * head_dim
            conv = (kernel - 1) * (inner + 2 * groups * d_state) * dtype_bytes
            ssm = heads * head_dim * d_state * ssm_bytes
        else:  # Mamba-1
            inner = (first("mamba_expand") or 2) * hidden
            conv = inner * (kernel - 1) * dtype_bytes
            ssm = inner * d_state * ssm_bytes
    return layers * (conv + ssm) / tp


# vLLM allocates KV in pages of this many tokens (default block size on GPU and CPU).
KV_BLOCK_SIZE = 16


def _blocks(tokens: int) -> int:
    return -(-tokens // KV_BLOCK_SIZE)


def _kv_bytes_per_request(
    layout: AttentionLayout,
    per_layer_token: float,
    max_model_len: int,
    batched_tokens: int,
    state_bytes: float = 0.0,
) -> float:
    """KV memory one request of ``max_model_len`` tokens needs.

    Full layers hold every token. Sliding-window layers hold at most the window plus
    the tokens scheduled in one step, as in vLLM's SlidingWindowSpec. Mamba /
    linear-attention layers add a fixed ``state_bytes`` regardless of length.
    """
    full_span = _blocks(max_model_len)
    sliding_span = full_span
    if layout.sliding_layers and layout.window:
        # SlidingWindowSpec: the window plus one step, in whole blocks, plus one block.
        sliding_span = _blocks(min(max_model_len, layout.window - 1 + batched_tokens)) + 1
    return state_bytes + KV_BLOCK_SIZE * per_layer_token * (
        layout.full_layers * full_span + layout.sliding_layers * sliding_span
    )


def _kv_pool_bytes(layout: AttentionLayout, per_layer_token: float, usable_bytes: float) -> float:
    """KV memory requests can use: vLLM reserves one null block in every layer."""
    reserved = KV_BLOCK_SIZE * per_layer_token * (layout.full_layers + layout.sliding_layers)
    return max(0.0, usable_bytes - reserved)


def _largest_fitting_len(
    budget_bytes: float,
    cap: int,
    floor: int,
    request_bytes,
) -> int:
    """Largest length in [floor, cap] whose single request fits the budget, or 0."""
    if request_bytes(floor) > budget_bytes:
        return 0
    if request_bytes(cap) <= budget_bytes:
        return cap
    lo, hi = floor, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if request_bytes(mid) <= budget_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _serve_default_batched_tokens(per_gpu_vram_gb: float) -> int:
    """``max_num_batched_tokens`` that ``vllm serve`` picks when it isn't set
    (EngineArgs.get_batch_defaults, OpenAI server context). A100-80GB actually
    keeps 2048; without the device name we assume the larger, safer value."""
    if per_gpu_vram_gb >= 160:
        return 16384
    if per_gpu_vram_gb >= 70:
        return 8192
    return 2048


def estimate_parameters_cpu(
    config: Dict[str, Any], total_ram: float, model_id: str = "", weight_info: Any = None
) -> Dict[str, Any]:
    warnings = []
    tc = resolve_text_config(config)

    weights_memory_gb = _weights_gb(tc, config, model_id, weight_info, warnings)

    # Runtime activation/workspace scales with model size; use a small fraction
    # of the weights footprint with a sensible floor.
    activation_buffer_gb = max(0.5, weights_memory_gb * 0.1)

    tensor_parallel_size = 1

    # Deliberately hold back RAM so the machine stays responsive once vLLM is serving:
    # reserve generous OS/framework headroom, then hand only HALF of what remains to the
    # KV cache, hard-capped at 30% of total RAM. This is the "don't exhaust capacity" rule.
    PRACTICAL_CPU_LEN_CAP = 8192  # CPU inference is slow; very long context isn't practical
    MAX_NUM_SEQS_CAP = 8          # modest concurrency keeps latency and RAM in check

    os_headroom_gb = max(3.0, total_ram * 0.20)
    spare_gb = total_ram - weights_memory_gb - activation_buffer_gb - os_headroom_gb
    kv_cache_space_gb = int(round(max(0.0, min(spare_gb * 0.5, total_ram * 0.30))))

    # Architecture-aware KV bytes/token (tp=1 on CPU); floor of 1 byte guards odd configs.
    hidden_size = int(get_field(tc, "hidden_size", 4096, warnings))
    num_layers = int(get_field(tc, "num_hidden_layers", 32, warnings))
    num_attention_heads = int(get_field(tc, "num_attention_heads", 32, warnings))
    num_kv_heads = int(get_field(tc, "num_key_value_heads", num_attention_heads))
    head_dim = get_head_dim(tc) or (
        hidden_size // num_attention_heads if num_attention_heads else hidden_size
    )
    head_dim = int(head_dim)
    layout = attention_layout(config, num_layers)
    per_layer_token = _kv_bytes_per_layer_token(config, num_kv_heads, head_dim)
    # CPU commands pin max_num_batched_tokens to max_model_len.
    state_bytes = _recurrent_state_bytes(config, num_layers, layout)
    request_bytes = lambda n: max(
        1.0, _kv_bytes_per_request(layout, per_layer_token, n, n, state_bytes)
    )
    kv_budget_bytes = _kv_pool_bytes(layout, per_layer_token, kv_cache_space_gb * (1024**3))

    len_cap = min(derive_max_model_len(config) or PRACTICAL_CPU_LEN_CAP, PRACTICAL_CPU_LEN_CAP)
    max_model_len = _largest_fitting_len(kv_budget_bytes, len_cap, 256, request_bytes) or 256
    max_num_seqs = max(1, min(MAX_NUM_SEQS_CAP, int(kv_budget_bytes // request_bytes(max_model_len))))

    total_required_gb = weights_memory_gb + activation_buffer_gb + os_headroom_gb + kv_cache_space_gb

    can_fit = True
    recommendations = []

    if spare_gb <= 0 or kv_cache_space_gb < 1:
        can_fit = False
        recommendations.append(
            f"Model needs {weights_memory_gb + activation_buffer_gb:.2f} GB for weights+activation "
            f"plus {os_headroom_gb:.1f} GB reserved headroom, leaving no room for a KV cache "
            f"within {total_ram:.1f} GB"
        )

    enforce_eager = True

    if not is_gguf_model(model_id):
        recommendations.append("Consider using GGUF format for better CPU performance")

    if not can_fit:
        if not is_model_quantized(config, model_id):
            recommendations.append(
                "Consider using a quantized version of the model (e.g., AWQ, GPTQ, 4-bit/8-bit)"
            )
        recommendations.append("Try a smaller model variant (e.g., 0.5B instead of 7B)")

    return {
        "gpu_memory_utilization": None,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "max_num_seqs": max_num_seqs,
        "estimated_weights_memory_gb": round(weights_memory_gb, 2),
        "per_gpu_weights_gb": round(weights_memory_gb, 2),
        "activation_memory_gb": round(activation_buffer_gb, 2),
        "compile_workspace_gb": 0.0,
        "kv_cache_space_gb": kv_cache_space_gb,
        "min_required_memory_gb": round(total_required_gb, 2),
        "can_fit": can_fit,
        "enforce_eager": enforce_eager,
        "recommendations": recommendations,
        "warnings": warnings,
    }


def _weights_gb(
    tc: Dict[str, Any],
    config: Dict[str, Any],
    model_id: str,
    weight_info: Any,
    warnings: list,
) -> float:
    """Exact weight footprint from metadata when available, else analytic estimate."""
    # GGUF files aren't safetensors, so any exact byte count came from the unquantized
    # base repo. Keep its parameter count but cost it at the quant tag's bits-per-weight.
    gguf_quantized = is_gguf_model(model_id) and _gguf_bpw_from_id(model_id) is not None
    if weight_info is not None:
        wgb = weight_info.weights_gb() if hasattr(weight_info, "weights_gb") else None
        if wgb is not None and not gguf_quantized:
            return wgb
        total_params = getattr(weight_info, "total_params", None)
        if total_params:
            return total_params * get_bytes_per_param(config, model_id) / (1024**3)
    param_count = _estimate_param_count(tc)
    warnings.append(
        "using analytic parameter estimate (no exact safetensors metadata available); "
        "figures are approximate"
    )
    return param_count * get_bytes_per_param(config, model_id) / (1024**3)


def estimate_parameters(
    config: Dict[str, Any],
    total_vram: float,
    num_gpus: int = 1,
    model_id: str = "",
    weight_info: Any = None,
    free_vram: Optional[float] = None,
) -> Dict[str, Any]:
    """GPU sizing. ``total_vram`` is the summed VRAM to size against; ``free_vram``,
    when known, is the smallest currently-free memory among the selected GPUs."""
    warnings = []
    # Public entry point: tolerate a non-positive GPU count rather than dividing by
    # zero when called as a library (the CLI already guards this upstream).
    if num_gpus < 1:
        num_gpus = 1
    tc = resolve_text_config(config)

    hidden_size = int(get_field(tc, "hidden_size", 4096, warnings))
    num_layers = int(get_field(tc, "num_hidden_layers", 32, warnings))
    num_attention_heads = int(get_field(tc, "num_attention_heads", 32, warnings))
    num_kv_heads = int(get_field(tc, "num_key_value_heads", num_attention_heads))
    head_dim = get_head_dim(tc) or (hidden_size // num_attention_heads if num_attention_heads else hidden_size)
    head_dim = int(head_dim)
    intermediate_size = int(get_field(tc, "intermediate_size", hidden_size * 4))

    crosscheck_with_transformers(
        config,
        {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_attention_heads,
        },
        warnings,
    )

    weights_memory_gb = _weights_gb(tc, config, model_id, weight_info, warnings)

    # awq_marlin kernel adds ~0.2GB runtime overhead on small GPUs
    if is_awq_quantized(config) and total_vram <= 6:
        weights_memory_gb += 0.2

    per_gpu_vram = total_vram / num_gpus
    tensor_parallel_size = _select_tensor_parallel(
        weights_memory_gb, per_gpu_vram, num_gpus, num_attention_heads, num_kv_heads
    )
    per_gpu_weights = weights_memory_gb / tensor_parallel_size

    quantized = is_model_quantized(config, model_id)

    # Recommend a high utilization: weights are fixed, so more headroom => more KV.
    # Leave a small physical margin so vLLM's start-of-run `free >= requested` holds.
    safety_gb = max(0.4, per_gpu_vram * 0.05)
    gpu_memory_utilization = min(0.90, max(0.5, (per_gpu_vram - safety_gb) / per_gpu_vram))
    gpu_memory_utilization = round(gpu_memory_utilization, 2)
    # vLLM refuses to start unless free memory (measured after its own CUDA context is
    # created) covers utilization x total, so memory held by other processes caps it.
    if free_vram is not None and per_gpu_vram > 0:
        free_cap = math.floor((free_vram - _CUDA_CONTEXT_GB) / per_gpu_vram * 100) / 100
        if free_cap < gpu_memory_utilization:
            gpu_memory_utilization = max(0.0, free_cap)
            # An idle GPU still reports a few hundred MB used (driver reservations);
            # only call it out when another process is plausibly holding memory.
            if per_gpu_vram - free_vram > _IDLE_GPU_USED_GB:
                warnings.append(
                    f"{per_gpu_vram - free_vram:.1f} GB of GPU memory is already in use by "
                    f"other processes; gpu_memory_utilization lowered to "
                    f"{gpu_memory_utilization:.2f}"
                )
    requested_gb = gpu_memory_utilization * per_gpu_vram

    # These don't depend on the CUDA-graph lever.
    # The flag is left to vLLM's default (it raises the value itself for models that
    # can't chunk prefill), so size activation for that default.
    batched_tokens = _serve_default_batched_tokens(per_gpu_vram)
    activation_peak_gb = _activation_peak_gb(hidden_size, intermediate_size, batched_tokens)
    non_torch_gb = 0.5 + (0.7 if tensor_parallel_size > 1 else 0.0)

    # KV cache, architecture-aware (independent of enforce_eager).
    kv_dtype_bytes = 2  # fp16/bf16 KV (vLLM default); fp8 KV would halve this.
    layout = attention_layout(config, num_layers)
    per_layer_token = _kv_bytes_per_layer_token(
        config, num_kv_heads, head_dim, tensor_parallel_size, kv_dtype_bytes
    )
    state_bytes = _recurrent_state_bytes(
        config, num_layers, layout, tensor_parallel_size, kv_dtype_bytes
    )
    request_bytes = lambda n: _kv_bytes_per_request(
        layout, per_layer_token, n, batched_tokens, state_bytes
    )

    derived_cap = derive_max_model_len(config)
    absolute_cap = derived_cap if derived_cap else 131072

    def _compute_fit(enforce_eager: bool) -> Dict[str, Any]:
        # torch.compile / CUDA-graph capture; enforce_eager zeroes it (the tight-VRAM lever).
        cudagraph_gb = 0.0 if enforce_eager else min(3.0, max(0.5, per_gpu_weights * 0.1))
        non_kv_per_gpu = per_gpu_weights + activation_peak_gb + non_torch_gb + cudagraph_gb
        usable_for_kv_gb = requested_gb - non_kv_per_gpu

        kv_budget_bytes = _kv_pool_bytes(
            layout, per_layer_token, max(0.0, usable_for_kv_gb) * (1024**3)
        )

        # Longest context one request can use, then how many such requests fit at once.
        fitting_len = _largest_fitting_len(kv_budget_bytes, absolute_cap, 256, request_bytes)
        max_model_len = fitting_len or 256
        if fitting_len:
            max_num_seqs = max(1, min(256, int(kv_budget_bytes // request_bytes(max_model_len))))
        else:
            max_num_seqs = 1

        can_fit = non_kv_per_gpu < requested_gb and fitting_len > 0
        return {
            "enforce_eager": enforce_eager,
            "cudagraph_gb": cudagraph_gb,
            "non_kv_per_gpu": non_kv_per_gpu,
            "kv_budget_bytes": kv_budget_bytes,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "can_fit": can_fit,
        }

    recommendations = []

    # Weights alone exceeding the card is unrecoverable — no lever helps.
    weights_dont_fit = per_gpu_weights > per_gpu_vram * 0.95

    # Start with the heuristic default, then try the enforce-eager lever if tight.
    initial_eager = quantized and per_gpu_vram <= 4
    fit = _compute_fit(initial_eager)
    eager_lever_applied = False
    if not weights_dont_fit and not fit["can_fit"] and not fit["enforce_eager"]:
        eager_fit = _compute_fit(True)
        if eager_fit["can_fit"] or eager_fit["kv_budget_bytes"] > fit["kv_budget_bytes"]:
            fit = eager_fit
            eager_lever_applied = True

    can_fit = fit["can_fit"] and not weights_dont_fit
    enforce_eager = fit["enforce_eager"]

    if weights_dont_fit:
        recommendations.append(
            f"Model weights ({weights_memory_gb:.2f} GB) require {tensor_parallel_size}x tensor "
            f"parallel but still exceed 95% of per-GPU VRAM ({per_gpu_vram:.1f} GB)"
        )
    elif not can_fit:
        recommendations.append(
            f"Memory requirements ({fit['non_kv_per_gpu']:.2f} GB per GPU for weights+overhead) "
            f"leave no room for KV cache within {gpu_memory_utilization:.2f}×{per_gpu_vram:.1f} GB"
            + (" even with --enforce-eager" if enforce_eager else "")
        )

    if not can_fit:
        if not quantized:
            recommendations.append(
                "Consider using a quantized version of the model (e.g., AWQ, GPTQ, 4-bit/8-bit)"
            )
        recommendations.append(
            "Try a smaller model variant (e.g., 7B instead of 70B, or 0.5B instead of 1.5B)"
        )
        if not enforce_eager:
            recommendations.append(
                "Use --enforce-eager mode to reduce vLLM's memory footprint (may impact performance)"
            )
        recommendations.append(
            "Use --kv-cache-dtype fp8 to roughly halve KV-cache memory and extend context"
        )
        if per_gpu_vram < 8:
            recommendations.append(
                "Your GPU has limited VRAM; consider cloud GPU options for larger models"
            )

    if enforce_eager and can_fit:
        reason = (
            "Enabled --enforce-eager: without it, weights+overhead leave no KV-cache room"
            if eager_lever_applied
            else "Using --enforce-eager to avoid torch.compile/CUDA-graph memory overhead on limited VRAM"
        )
        recommendations.append(reason)

    return {
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": fit["max_model_len"],
        "tensor_parallel_size": tensor_parallel_size,
        "max_num_seqs": fit["max_num_seqs"],
        "estimated_weights_memory_gb": round(weights_memory_gb, 2),
        "per_gpu_weights_gb": round(per_gpu_weights, 2),
        "activation_memory_gb": round(activation_peak_gb, 2),
        "compile_workspace_gb": round(fit["cudagraph_gb"], 2),
        "min_required_memory_gb": round(fit["non_kv_per_gpu"] * tensor_parallel_size, 2),
        "can_fit": can_fit,
        "enforce_eager": enforce_eager,
        "recommendations": recommendations,
        "warnings": warnings,
    }
