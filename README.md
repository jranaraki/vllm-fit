# vLLM-Fit

<p align="center">
    <img src="logo.png" alt="vllm-fit" style="width:30%; height:auto;">
</p>

A CLI tool designed to simply _recommend_ (conservative), and/or _profile_ (to maximize resource utilization) vLLM engine arguments for any HuggingFace model on the user's current hardware.

## Features

- **Static Estimation**: Instant, architecture-aware parameter recommendations from model config (accounts for GQA, mixture-of-experts, and quantized weights)
- **Dynamic Profiling**: Tests real memory usage to find actual limits
- **Multi-GPU Support**: Automatic tensor parallel configuration
- **CPU & Apple Silicon Fallback**: Auto-detects when no NVIDIA GPU is present, sizes limits conservatively from available (unified) RAM, and emits a complete CPU serve command
- **Smart Fail Handling**: Graceful errors when VRAM insufficient
- **Optimized Output**: Clean logs without vLLM noise

## Quick Start

```bash
# Create and activate uv environment
uv venv --seed --python 3.10
source .venv/bin/activate

# Install vLLM
uv pip install vllm --torch-backend=auto

# Install vllm-fit
uv pip install git+https://github.com/jranaraki/vllm-fit
```

## Commands

### `recommend` - Quick parameter suggestions

```
vllm-fit recommend <model_id> [--gpuid <ids>]
```

Returns estimated optimal parameters without running the model.

### `profile` - Find actual memory limits

```
vllm-fit profile <model_id> [--gpuid <ids>]
```

Tests different configurations to find what actually fits your GPU.

### `serve` - Start vLLM server

```
vllm-fit serve <model_id> [--gpuid <ids>]
```

Profiles, then starts an optimized vLLM OpenAI-compatible server.

### GPU Selection

Use `--gpuid` to control which GPUs are used:

- `--gpuid 0` - Use GPU 0 only
- `--gpuid 0,1,2` - Use GPUs 0, 1, and 2
- `--gpuid all` (default) - Use all available GPUs

### CPU & Apple Silicon Mode

All three commands auto-detect your hardware. When no NVIDIA GPU is found,
vllm-fit runs in CPU mode automatically (no flag needed; `--gpuid` is ignored):

- **Conservative, RAM-aware sizing** — `max_model_len` and `max_num_seqs` are
  derived from available system RAM and deliberately leave headroom, so the machine
  stays responsive while vLLM serves rather than consuming all free memory.
- **Complete CPU command** — the emitted command drops the GPU-only flags
  (`--gpu_memory_utilization`, `--tensor_parallel_size`) and is prefixed with the
  `VLLM_CPU_KVCACHE_SPACE=<GB>` the CPU backend needs to size its KV cache:

  ```
  VLLM_CPU_KVCACHE_SPACE=10 vllm serve <model_id> --max_model_len 8192 --max_num_seqs 8 --max_num_batched_tokens 2048 --enforce-eager
  ```

- **Apple Silicon** — Macs are detected and reported against their unified-memory
  pool. Note that native vLLM on macOS is a source build, CPU-only (no Apple-GPU
  acceleration), and experimental; for Metal GPU inference, see the community
  [`vllm-metal`](https://github.com/vllm-project/vllm-metal) (MLX) plugin.

## Output

### `recommend`

Running `vllm-fit recommend Qwen/Qwen3-0.6B` generates the following output:

```
GPU VRAM: 4.0 GB

╭────────────────────────────────── Static Estimation ──────────────────────────────────╮
│ Recommended Parameters                                                                │                                                                                                                                                          │
╰───────────────────────────────────────────────────────────────────────────────────────╯
model_id: Qwen/Qwen3-0.6B
gpu_memory_utilization: 0.65
max_model_len: 2444
tensor_parallel_size: 1
max_num_seqs: 8
estimated_weights_memory_gb: 1.26

Run this command:
vllm serve Qwen/Qwen3-0.6B --gpu_memory_utilization 0.65 --tensor_parallel_size 1 --max_model_len 2444 --max_num_seqs 8
```

### `profile`

Running `vllm-fit profile Qwen/Qwen2.5-7B-Instruct` generates the following output:

```
✓ Profiling completed successfully!

Summary:
  • Attempted 3 configurations
  • Final test: Memory=0.80, Len=4096
  • Strategy: Used --enforce-eager mode for memory efficiency
  • Time elapsed: 45s

╭────────────────────────────────── Dynamic Profiling ──────────────────────────────────╮
│ Optimized Parameters                                                                  │
╰───────────────────────────────────────────────────────────────────────────────────────╯
model_id: Qwen/Qwen2.5-7B-Instruct
gpu_memory_utilization: 0.80
max_model_len: 4096
tensor_parallel_size: 1
max_num_seqs: 16
enforce_eager: True

Run this command:
vllm serve Qwen/Qwen2.5-7B-Instruct --gpu_memory_utilization 0.80 --tensor_parallel_size 1 --max_model_len 4096 --max_num_seqs 16 --enforce-eager
```

## Requirements

- **vLLM** - Install separately (GPU or CPU version, see vLLM installation docs)
- **Hardware**:
  - GPU: NVIDIA GPU with CUDA support, 4GB+ VRAM recommended
  - CPU / Apple Silicon: 16GB+ (unified) RAM recommended (varies by model size)
- Python 3.9+

## Troubleshooting

### "No GPU detected"

1. Check `nvidia-smi` shows your GPU
2. Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
4. Update NVIDIA drivers from https://developer.nvidia.com/cuda-downloads

### CPU mode not working?

1. Ensure vLLM CPU version is installed (not GPU version)
2. Check available RAM: `python -c "import psutil; print(psutil.virtual_memory().total / 1024**3)"` (need 16GB+ recommended)
3. Try a smaller model or quantized GGUF format
4. On macOS, vLLM has no prebuilt wheel — build it from source (CPU-only), or use the `vllm-metal` plugin for Apple-GPU inference

### Model won't fit?

Try:
- Use `--enforce-eager` flag (added automatically by vllm-fit)
- Try a quantized model (AWQ, GPTQ, 4-bit/8-bit)
- Use a smaller model variant
- Get more GPUs for larger models

### "Downloaded GGUF files not found"

Try:
- For some of the older GGUF models on HuggingFace, there might be ones that did not follow the naming convention accurately. Therefore, you might need to rename them before deployment. For instance, when you run `vllm-fit recommend Qwen/Qwen3-0.6B-GGUF:Q8_0`, the recommended arguments `vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 --gpu_memory_utilization 0.65 --tensor_parallel_size 1 --max_model_len 3914 --max_num_seqs 8 --hf-config-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --enforce-eager` works just fine since the model name in the `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots/23749fefcc72300e3a2ad315e1317431b06b590a` is as expected (see `Qwen3-0.6B-Q8_0.gguf`), where the quantization substring is `Q8_0`. However, when you run `vllm-fit recommend Qwen/Qwen2-0.5B-Instruct-GGUF:Q8_0`, although the provided recommendation is correct, `vllm serve Qwen/Qwen2-0.5B-Instruct-GGUF:Q8_0 --gpu_memory_utilization 0.65 --tensor_parallel_size 1 --max_model_len 6098 --max_num_seqs 8 --hf-config-path Qwen/Qwen2-0.5B-Instruct --tokenizer Qwen/Qwen2-0.5B-Instruct --enforce-eager`, since the downloaded model's name did not follow the naming convention, you will get this error from vLLM `ValueError: Downloaded GGUF files not found in /home/USER/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857 for quant_type Q8_0` since the model name is `qwen2-0_5b-instruct-q8_0.gguf`. To fix this, rename the model, in the cache `/home/USER/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857`, from `qwen2-0_5b-instruct-q8_0.gguf` to `qwen2-0_5b-instruct-Q8_0.gguf`.

## How It Works

1. **Fetches model config** from Hugging Face
2. **Estimates parameters** based on model architecture and available VRAM
3. **Profiles** by testing the actual vLLM engine with different settings
4. **Iteratively adjusts** memory, sequence length, and batch size until successful
5. **Returns exact command** to run with optimal parameters

## Citing

If you find vllm-fit useful and are interested in citing this work, please use the following BibTex entry:

```
@software{vllmfit2026,
  author = {Javad Anaraki},
  title = {vllm-fit: Hardware-Aware vLLM Argument Recommendation and Profiling},
  url = {https://github.com/jranaraki/vllm-fit},
  version = {0.5.0},
  year = {2026},
}
```

## Acknowledgments

- [llmfit](https://github.com/AlexsJones/llmfit) for the inspiration 
