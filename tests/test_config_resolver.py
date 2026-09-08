from vllm_fit.config_resolver import (
    resolve_text_config,
    get_field,
    get_head_dim,
    derive_max_model_len,
    detect_mla,
    full_attention_layer_count,
    sliding_window,
)


# --- resolve_text_config -------------------------------------------------

def test_resolve_flat_config_returns_self():
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32}
    assert resolve_text_config(cfg) is cfg


def test_resolve_text_config_descends():
    cfg = {
        "architectures": ["FooForConditionalGeneration"],
        "vision_config": {"hidden_size": 1152, "num_hidden_layers": 27},
        "text_config": {"hidden_size": 5120, "num_hidden_layers": 64},
    }
    tc = resolve_text_config(cfg)
    assert tc["hidden_size"] == 5120
    assert tc["num_hidden_layers"] == 64


def test_resolve_llm_config_key():
    cfg = {"llm_config": {"hidden_size": 2048, "num_hidden_layers": 24}}
    assert resolve_text_config(cfg)["hidden_size"] == 2048


def test_resolve_thinker_config_two_level():
    cfg = {
        "thinker_config": {
            "text_config": {"hidden_size": 3584, "num_hidden_layers": 28}
        }
    }
    tc = resolve_text_config(cfg)
    assert tc["hidden_size"] == 3584
    assert tc["num_hidden_layers"] == 28


def test_resolve_never_descends_into_vision_only():
    # A single vision_config is not a text sub-config; stay at top level.
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "vision_config": {"hidden_size": 1152}}
    tc = resolve_text_config(cfg)
    assert tc["hidden_size"] == 4096


# --- get_field aliases + fail-loud --------------------------------------

def test_get_field_alias_n_embd():
    assert get_field({"n_embd": 768}, "hidden_size") == 768


def test_get_field_alias_num_kv_heads():
    assert get_field({"multi_query_group_num": 2}, "num_key_value_heads") == 2


def test_get_field_mpt_expansion_ratio():
    cfg = {"d_model": 4096, "expansion_ratio": 4}
    assert get_field(cfg, "intermediate_size") == 16384


def test_get_field_warns_on_missing():
    warn = []
    val = get_field({}, "hidden_size", default=4096, warn=warn)
    assert val == 4096
    assert warn and "hidden_size" in warn[0]


def test_get_head_dim_explicit_and_derived():
    assert get_head_dim({"head_dim": 128}) == 128
    assert get_head_dim({"hidden_size": 4096, "num_attention_heads": 32}) == 128


# --- derive_max_model_len -----------------------------------------------

def test_max_len_plain():
    assert derive_max_model_len({"max_position_embeddings": 8192}) == 8192


def test_max_len_model_max_length_wins():
    cfg = {"max_position_embeddings": 8192, "model_max_length": 4096}
    assert derive_max_model_len(cfg) == 4096


def test_max_len_yarn_scales_from_original():
    cfg = {
        "max_position_embeddings": 32768,
        "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                         "original_max_position_embeddings": 8192},
    }
    assert derive_max_model_len(cfg) == 32768  # 8192 * 4


def test_max_len_longrope_uses_original():
    cfg = {
        "max_position_embeddings": 131072,
        "rope_scaling": {"rope_type": "longrope", "factor": 8.0,
                         "original_max_position_embeddings": 4096},
    }
    assert derive_max_model_len(cfg) == 4096


def test_max_len_gemma3_no_plain_factor():
    cfg = {
        "max_position_embeddings": 8192,
        "rope_scaling": {"rope_type": "gemma3", "factor": 8.0},
    }
    assert derive_max_model_len(cfg) == 8192


def test_max_len_linear_factor():
    cfg = {"max_position_embeddings": 4096, "rope_scaling": {"type": "linear", "factor": 2.0}}
    assert derive_max_model_len(cfg) == 8192


def test_max_len_none_when_absent():
    assert derive_max_model_len({"hidden_size": 4096}) is None


# --- architecture detectors ---------------------------------------------

def test_detect_mla():
    cfg = {"kv_lora_rank": 512, "qk_rope_head_dim": 64}
    mla = detect_mla(cfg)
    assert mla == {"kv_lora_rank": 512, "qk_rope_head_dim": 64}


def test_detect_mla_none():
    assert detect_mla({"hidden_size": 4096}) is None


def test_full_attention_layer_count_layer_types():
    cfg = {"layer_types": ["full_attention", "linear_attention",
                           "full_attention", "linear_attention"]}
    assert full_attention_layer_count(cfg, 4) == 2


def test_full_attention_interval():
    cfg = {"full_attention_interval": 4}
    assert full_attention_layer_count(cfg, 32) == 8


def test_full_attention_fallback_all_layers():
    # Unknown pattern -> safe overestimate: count all layers.
    assert full_attention_layer_count({}, 32) == 32


def test_sliding_window():
    assert sliding_window({"sliding_window": 4096}) == 4096
    assert sliding_window({"sliding_window": 4096, "use_sliding_window": False}) is None
    assert sliding_window({}) is None
