import sys

sys.path.insert(0, "src")

from vllm_fit.cli import parse_gpu_ids

# Test GPU ID parsing
vram_info = {0: 24.0, 1: 24.0, 2: 24.0, 3: 24.0}

# Test case 1: 'all'
print("Test 'all':", parse_gpu_ids("all", vram_info))

# Test case 2: single GPU
print("Test '0':", parse_gpu_ids("0", vram_info))

# Test case 3: multiple GPUs
print("Test '0,1':", parse_gpu_ids("0,1", vram_info))

# Test case 4: range
print("Test '0-2':", parse_gpu_ids("0-2", vram_info))
