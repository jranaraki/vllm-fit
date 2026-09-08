"""Structure-aware resolution of HuggingFace model configs.

The estimator needs a handful of architecture fields (hidden size, layer count,
head counts, intermediate size, vocab, max positions). Those fields live in
different places and under different names across the HF ecosystem:

  * flat top-level (most text-only models),
  * one level down under ``text_config`` / ``llm_config`` / ``language_config`` /
    ``language_model`` (multimodal and some community configs),
  * two levels down under ``thinker_config.text_config`` (Qwen-Omni),

and individual fields carry synonyms (``hidden_size`` vs ``n_embd`` vs ``d_model``).
Reading a field with a plain ``config.get()`` therefore silently misses the real
value and falls back to a wrong default — which is exactly how the estimator used
to mis-size nested/multimodal models.

This module resolves the text sub-config first, then reads fields through alias
maps, returning ``None`` (never a silent default) when a field genuinely cannot be
found, so callers can *warn* instead of quietly guessing.

Everything here is pure and dependency-free. The one optional touch of the outside
world is :func:`crosscheck_with_transformers`, which — only when vLLM/transformers
are importable — validates our resolution against ``PretrainedConfig.get_text_config``
(the exact call vLLM makes internally) and warns on divergence. It never raises and
never requires the network.
"""

from typing import Any, Dict, List, Optional


# Sub-config keys that hold the language-model fields for multimodal / composite
# configs. Mirrors transformers' ``get_text_config`` plus common community names.
_TEXT_CONFIG_KEYS = ("text_config", "llm_config", "language_config", "language_model")

# Sub-configs we must never descend into when looking for LM dimensions.
_NON_TEXT_KEYS = (
    "vision_config",
    "audio_config",
    "vision_tower",
    "audio_tower",
    "image_config",
    "speech_config",
)

# Canonical field name -> ordered aliases (canonical first). First present wins.
_ALIASES: Dict[str, tuple] = {
    "hidden_size": ("hidden_size", "n_embd", "d_model", "embed_dim", "hidden_dim"),
    "num_hidden_layers": ("num_hidden_layers", "n_layer", "n_layers", "num_layers"),
    "num_attention_heads": (
        "num_attention_heads",
        "n_head",
        "n_heads",
        "num_heads",
        "attention_heads",
    ),
    "num_key_value_heads": (
        "num_key_value_heads",
        "num_kv_heads",
        "n_head_kv",
        "multi_query_group_num",
        "num_attention_groups",
    ),
    "intermediate_size": (
        "intermediate_size",
        "n_inner",
        "ffn_dim",
        "ffn_hidden_size",
    ),
    "vocab_size": ("vocab_size", "n_vocab", "padded_vocab_size"),
    "head_dim": ("head_dim", "attention_head_dim", "v_head_dim"),
}

# max-position field names scanned by vLLM's max-len derivation (smallest wins,
# except model_max_length which takes precedence when present).
_MAX_LEN_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_seq_len",
    "seq_length",
    "max_sequence_length",
    "max_seq_length",
    "max_target_positions",
    "seq_len",
)

_CORE_DIM_ALIASES = _ALIASES["hidden_size"] + _ALIASES["num_hidden_layers"]


def _has_core_dims(cfg: Any) -> bool:
    """True if the dict carries the fields that mark it as a language-model config."""
    if not isinstance(cfg, dict):
        return False
    has_hidden = any(cfg.get(k) is not None for k in _ALIASES["hidden_size"])
    has_layers = any(cfg.get(k) is not None for k in _ALIASES["num_hidden_layers"])
    return has_hidden and has_layers


def resolve_text_config(config: Any) -> Dict[str, Any]:
    """Return the sub-config holding the language-model fields.

    Precedence mirrors transformers' ``get_text_config``: descend into exactly one
    nested text sub-config when present; otherwise return the config unchanged.
    Handles the two-level ``thinker_config.text_config`` case (Qwen-Omni) and never
    descends into vision/audio sub-configs.
    """
    if not isinstance(config, dict):
        return config

    # Two-level: thinker_config wraps its own text_config (Qwen-Omni family).
    thinker = config.get("thinker_config")
    if isinstance(thinker, dict):
        inner = resolve_text_config(thinker)
        if inner is not thinker:
            return inner
        if _has_core_dims(thinker):
            return thinker

    matches = [
        config[k]
        for k in _TEXT_CONFIG_KEYS
        if isinstance(config.get(k), dict)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Ambiguous: prefer the single sub-config that actually carries LM dims.
        with_dims = [m for m in matches if _has_core_dims(m)]
        if len(with_dims) == 1:
            return with_dims[0]
        # Still ambiguous — stay at this level; callers warn on missing fields.
    return config


def get_field(
    config: Dict[str, Any],
    canonical: str,
    default: Any = None,
    warn: Optional[List[str]] = None,
) -> Any:
    """Read a field by canonical name, trying every known alias.

    Returns ``default`` when the field is genuinely absent. When ``warn`` is a list
    and the field is missing, appends a human-readable note to it so the caller can
    surface that the estimate rests on a fallback rather than the model's real value.
    """
    aliases = _ALIASES.get(canonical, (canonical,))
    for key in aliases:
        val = config.get(key)
        if val is not None:
            return val

    # MPT-style FFN sizing: intermediate = expansion_ratio * hidden_size.
    if canonical == "intermediate_size":
        ratio = config.get("expansion_ratio")
        hidden = get_field(config, "hidden_size")
        if ratio and hidden:
            return int(ratio * hidden)

    if warn is not None:
        warn.append(
            f"config field '{canonical}' not found (looked for {', '.join(aliases)}); "
            f"using fallback {default!r} — estimate may be unreliable"
        )
    return default


def get_head_dim(config: Dict[str, Any]) -> Optional[int]:
    """Explicit head_dim if present, else hidden_size // num_attention_heads."""
    head_dim = get_field(config, "head_dim")
    if head_dim:
        return int(head_dim)
    hidden = get_field(config, "hidden_size")
    heads = get_field(config, "num_attention_heads")
    if hidden and heads:
        return int(hidden) // int(heads)
    return None


def derive_max_model_len(config: Dict[str, Any]) -> Optional[int]:
    """Model's true maximum context, mirroring vLLM's derivation.

    Scans the text config for the smallest positional limit (``model_max_length``
    wins outright when present), then applies RoPE scaling with the yarn / longrope /
    gemma3 exceptions. Returns ``None`` when no positional field is present.
    """
    tc = resolve_text_config(config)
    if not isinstance(tc, dict):
        return None

    candidates: List[int] = []
    model_max: Optional[int] = None
    for key in _MAX_LEN_KEYS:
        val = tc.get(key)
        if isinstance(val, (int, float)) and val > 0:
            candidates.append(int(val))
    mm = tc.get("model_max_length")
    if isinstance(mm, (int, float)) and 0 < mm < 1e12:
        model_max = int(mm)

    if model_max is not None:
        derived = model_max
    elif candidates:
        derived = min(candidates)
    else:
        return None

    rope = tc.get("rope_scaling")
    if isinstance(rope, dict):
        rtype = str(rope.get("rope_type") or rope.get("type") or "").lower()
        factor = rope.get("factor")
        orig = rope.get("original_max_position_embeddings")
        if rtype == "yarn":
            base = int(orig) if orig else derived
            if isinstance(factor, (int, float)):
                derived = int(base * factor)
        elif rtype in ("longrope", "su"):
            if orig:
                derived = int(orig)
        elif rtype in ("gemma3", "llama3"):
            pass  # already scaled / no plain factor applied
        elif isinstance(factor, (int, float)) and factor > 0:
            derived = int(derived * factor)

    return derived


def detect_mla(config: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Multi-head Latent Attention (DeepSeek-V2/V3). Detected via ``kv_lora_rank``.

    Returns the latent dims needed to size the (much smaller) MLA KV cache, or None.
    """
    tc = resolve_text_config(config)
    if not isinstance(tc, dict):
        return None
    lora = tc.get("kv_lora_rank")
    if lora:
        return {
            "kv_lora_rank": int(lora),
            "qk_rope_head_dim": int(tc.get("qk_rope_head_dim", 0) or 0),
        }
    return None


def full_attention_layer_count(config: Dict[str, Any], num_layers: int) -> int:
    """Number of layers that hold a full (global) KV cache.

    Hybrid models interleave full-attention layers with linear-attention / SSM /
    Mamba layers that keep little or no KV cache. Counting all layers as full (the
    old behavior) massively overestimates KV memory. When the pattern can't be read,
    fall back to ``num_layers`` (a safe overestimate rather than a risky undercount).
    """
    tc = resolve_text_config(config)
    if not isinstance(tc, dict) or not num_layers:
        return num_layers

    layer_types = tc.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        n = sum(
            1
            for t in layer_types
            if isinstance(t, str) and t.lower() in ("full_attention", "attention", "full")
        )
        return n if n > 0 else num_layers

    indices = tc.get("attn_layer_indices")
    if isinstance(indices, list) and indices:
        return len(indices)

    interval = tc.get("full_attention_interval")
    if isinstance(interval, int) and interval > 0:
        return max(1, num_layers // interval)

    period = tc.get("attn_layer_period")
    if isinstance(period, int) and period > 0:
        offset = int(tc.get("attn_layer_offset", 0) or 0)
        return max(1, len([i for i in range(num_layers) if i % period == offset % period]))

    type_list = tc.get("attn_type_list")
    if isinstance(type_list, list) and type_list:
        n = sum(1 for t in type_list if t in (1, True, "1", "attention", "full"))
        return n if n > 0 else num_layers

    return num_layers


def sliding_window(config: Dict[str, Any]) -> Optional[int]:
    """Effective sliding-window size (caps KV footprint), or None if not windowed."""
    tc = resolve_text_config(config)
    if not isinstance(tc, dict):
        return None
    window = tc.get("sliding_window")
    use = tc.get("use_sliding_window")
    if window and (use is None or use):
        return int(window)
    return None


def crosscheck_with_transformers(
    config: Dict[str, Any],
    resolved_dims: Dict[str, Any],
    warn: Optional[List[str]] = None,
) -> None:
    """Best-effort cross-check against transformers' own ``get_text_config``.

    Decision A: the self-contained resolver above is authoritative. When vLLM is
    installed we additionally validate against transformers (the library vLLM uses
    to resolve configs): build the concrete config class *offline* from ``model_type``
    via ``AutoConfig.for_model`` and compare its ``get_text_config`` dims to ours.
    Any divergence is appended to ``warn``. This never raises and never hits the
    network — if the model_type isn't registered in the installed transformers (e.g.
    a brand-new architecture), we simply skip and trust the self-contained result.
    """
    try:
        import vllm  # noqa: F401  (gate on the user's stated condition: vLLM present)
    except Exception:
        return

    try:
        from transformers import AutoConfig
    except Exception:
        return

    model_type = config.get("model_type")
    if not model_type:
        return

    try:
        cfg = AutoConfig.for_model(model_type, **config)
        text = cfg.get_text_config()
        reference = {
            "hidden_size": getattr(text, "hidden_size", None),
            "num_hidden_layers": getattr(text, "num_hidden_layers", None),
            "num_attention_heads": getattr(text, "num_attention_heads", None),
        }
    except Exception:
        return

    for key, ref_val in reference.items():
        ours = resolved_dims.get(key)
        try:
            if ref_val is not None and ours is not None and int(ref_val) != int(ours):
                if warn is not None:
                    warn.append(
                        f"resolver disagrees with transformers on '{key}': "
                        f"{ours} (ours) vs {ref_val} (transformers)"
                    )
        except (TypeError, ValueError):
            continue
