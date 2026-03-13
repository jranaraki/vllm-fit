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
