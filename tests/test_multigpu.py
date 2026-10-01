from vllm_fit.cli import parse_gpu_ids


def test_parse_gpu_ids_all():
    """Test parsing 'all' for GPU IDs"""
    vram_info = {0: 24.0, 1: 24.0, 2: 24.0, 3: 24.0}
    result = parse_gpu_ids("all", vram_info)
    assert result == [0, 1, 2, 3], f"Expected [0, 1, 2, 3], got {result}"


def test_parse_gpu_ids_single():
    """Test parsing single GPU ID"""
    vram_info = {0: 24.0, 1: 24.0, 2: 24.0, 3: 24.0}
    result = parse_gpu_ids("0", vram_info)
    assert result == [0], f"Expected [0], got {result}"


def test_parse_gpu_ids_multiple():
    """Test parsing multiple GPU IDs"""
    vram_info = {0: 24.0, 1: 24.0, 2: 24.0, 3: 24.0}
    result = parse_gpu_ids("0,1", vram_info)
    assert result == [0, 1], f"Expected [0, 1], got {result}"


def test_parse_gpu_ids_multiple_gpus():
    """Test parsing multiple comma-separated GPU IDs"""
    vram_info = {0: 24.0, 1: 24.0, 2: 24.0, 3: 24.0}
    result = parse_gpu_ids("0,1,2", vram_info)
    assert result == [0, 1, 2], f"Expected [0, 1, 2], got {result}"


def test_parse_gpu_ids_invalid():
    """Test parsing invalid GPU ID returns empty list"""
    vram_info = {0: 24.0, 1: 24.0, 2: 24.0, 3: 24.0}
    result = parse_gpu_ids("5", vram_info)
    assert result == [], f"Expected empty list for invalid GPU ID, got {result}"


def test_parse_gpu_ids_empty_vram_info():
    """Test parsing with no available GPUs"""
    vram_info = {}
    result = parse_gpu_ids("all", vram_info)
    assert result == [], f"Expected empty list when no GPUs available, got {result}"


def test_sizing_vram_uses_smallest_gpu():
    from vllm_fit.cli import sizing_vram

    vram_info = {0: 24.0, 1: 80.0}
    assert sizing_vram(vram_info, [0, 1]) == 48.0
    assert sizing_vram(vram_info, [1]) == 80.0


def test_mixed_gpus_do_not_fit_13b_fp16_on_24gb_card():
    # 13B fp16 (~24 GB of weights) on a 24 GB + 80 GB pair: averaging the cards made
    # TP=1 look like it fit, but the 24 GB card bounds every shard.
    from vllm_fit.cli import sizing_vram
    from vllm_fit.estimator import estimate_parameters
    from vllm_fit.params import WeightInfo

    cfg = {"hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 40,
           "vocab_size": 32000, "intermediate_size": 13824}
    wi = WeightInfo(source="test", weights_bytes=int(24.24 * 1024**3))
    vram_info = {0: 24.0, 1: 80.0}
    res = estimate_parameters(cfg, sizing_vram(vram_info, [0, 1]), 2, weight_info=wi)
    assert res["tensor_parallel_size"] == 2
    assert res["per_gpu_weights_gb"] < 0.9 * 24.0


def test_free_vram_binds_on_busiest_gpu_ratio(monkeypatch):
    # 24 GB idle + 80 GB with 50 GB used: the 80 GB card can only take ~0.37 of its
    # total, even though the 24 GB card has less free memory in absolute terms.
    import vllm_fit.cli as cli
    from vllm_fit.estimator import estimate_parameters
    from vllm_fit.params import WeightInfo

    vram_info = {0: 24.0, 1: 80.0}
    monkeypatch.setattr(cli, "get_free_vram_info", lambda: {0: 23.5, 1: 30.0})
    free = cli.free_vram_for_sizing(vram_info, [0, 1])
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
           "vocab_size": 32000}
    wi = WeightInfo(source="test", weights_bytes=int(4 * 1024**3))
    res = estimate_parameters(cfg, cli.sizing_vram(vram_info, [0, 1]), 2,
                              weight_info=wi, free_vram=free)
    util = res["gpu_memory_utilization"]
    assert util * 80.0 <= 30.0 - 0.5
    assert util * 24.0 <= 23.5 - 0.5
