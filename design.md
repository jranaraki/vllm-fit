# Design Document: vLLM Parameter Optimizer (vllm-fit)

## 1. Overview
`vllm-fit` is a CLI tool designed to automate the discovery of optimal `vLLM` engine arguments for any Hugging Face model on the user's current hardware. It eliminates the trial-and-error process of fitting models into GPU VRAM.

## 2. Core Logic Flow
The application follows a "Refine-and-Test" strategy:
1. **Hardware Discovery:** Detect GPU count and total VRAM (via `pynvml` or `torch`).
2. **Metadata Fetching:** Download the model's `config.json` from Hugging Face to get hidden dimensions, layer counts, and default max context.
3. **Static Estimation:** Calculate a baseline `gpu_memory_utilization` and `max_model_len` using a physics-based memory model (Weights + KV Cache + Activation buffers).
4. **Dynamic Profiling (The "Dry Run"):**
   - Attempt to initialize the `vLLM` engine with the estimated parameters.
   - If it fails (OOM), catch the exception, reduce `max_model_len` (or `gpu_memory_utilization`), and retry.
   - If it succeeds, attempt to increase `max_num_seqs` until throughput plateaus or memory is exhausted.

## 3. Architecture
- **CLI Layer:** Uses `Typer` or `Click` for a modern interface.
- **Hardware Module:** Interface with `NVIDIA Management Library (NVML)` to get real-time VRAM availability.
- **Registry Module:** Uses `huggingface_hub` to fetch model configs without downloading full weights.
- **Engine Runner:** A subprocess wrapper that initializes `vllm.LLM` in a separate process (to safely handle OOM crashes without killing the CLI).

## 4. Key Parameters to Tune
| Parameter | Tuning Logic |
| :--- | :--- |
| `gpu_memory_utilization` | Start at 0.90. Reduce if system/display memory is high. |
| `max_num_seqs` | Dependent on throughput requirements. Tuned after model fits. |
| `max_model_len` | **The Master Fallback.** If model + KV Cache > VRAM, decrement by 512 until fit. |
| `tensor_parallel_size` | Auto-set based on (Model Size / VRAM per GPU). |

## 5. Proposed CLI Commands
- `vllm-fit recommend <model_id>`: Quick static estimation based on HF config.
- `vllm-fit profile <model_id> --gpu-id 0`: Active testing (dry runs) to find the absolute ceiling of the current hardware.
- `vllm-fit serve <model_id>`: Starts the vLLM server using the discovered optimal parameters.

## 6. Technical Requirements
- **Python:** 3.9+
- **Libraries:** `huggingface_hub`, `pynvml`, `typer`, `rich` (for beautiful progress bars) - **Note:** `vllm` must be installed by the user separately.
- **Safety:** Must use `multiprocessing` to spawn the vLLM engine. vLLM often cannot be "re-initialized" in the same process after a CUDA OOM.

## 7. Success Criteria
- User provides `meta-llama/Llama-3-70B` on a single 24GB GPU.
- Tool realizes it won't fit in FP16.
- Tool checks for 4-bit/AWQ versions or suggests a `max_model_len` of 2048 and `gpu_memory_utilization` of 0.8.
