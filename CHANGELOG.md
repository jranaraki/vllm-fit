# CHANGELOG

<!-- version list -->

## v0.6.1 (2026-10-01)

### Bug Fixes

- **profile**: Cap the length search at the model's maximum context
  ([`74b2c44`](https://github.com/jranaraki/vllm-fit/commit/74b2c44e9b2377f6a9eb5d81da9304cc4b764f7f))


## v0.6.0 (2026-10-01)

### Continuous Integration

- Test on Python 3.12 and 3.13
  ([`e6ffacc`](https://github.com/jranaraki/vllm-fit/commit/e6ffacc667c6a3fe6b3fc7a5e4d9bbcedaf1e0da))

### Features

- Size for the Apple GPU when vllm-metal is installed
  ([`9857fc9`](https://github.com/jranaraki/vllm-fit/commit/9857fc95fc2b6f55d380656ee24c2bab3580ee1c))

### Testing

- Add ground-truth fixtures and a harness for real vLLM runs
  ([`bd19451`](https://github.com/jranaraki/vllm-fit/commit/bd19451906af813b346adf3e16818cca73435e27))


## v0.5.18 (2026-10-01)

### Bug Fixes

- **estimator**: Size KV in 16-token blocks and reserve vLLM's null block
  ([`12fe567`](https://github.com/jranaraki/vllm-fit/commit/12fe567bf8631cb9f2f76f600e7bc25994e99fed))


## v0.5.17 (2026-10-01)

### Bug Fixes

- **estimator**: Count Mamba / linear-attention state per request
  ([`b4e7c60`](https://github.com/jranaraki/vllm-fit/commit/b4e7c6099010abb5d45bc1b06d9d45b4ba6b9575))


## v0.5.16 (2026-10-01)

### Bug Fixes

- **profile**: Probe with the same engine arguments as the served command
  ([`0270ad4`](https://github.com/jranaraki/vllm-fit/commit/0270ad4ba9e2bc73949a67d3a5f410a121d03a12))

### Documentation

- Document GPU pinning, CUDA_VISIBLE_DEVICES, local and gated models
  ([`18ddbcd`](https://github.com/jranaraki/vllm-fit/commit/18ddbcdef0dd40e48f825c000c1944a62eba3456))

- Make the README match current behaviour and state its assumptions
  ([`64c4934`](https://github.com/jranaraki/vllm-fit/commit/64c4934715b36238f8def332fb4394655044ede6))


## v0.5.15 (2026-10-01)

### Bug Fixes

- **estimator**: Don't warn about other processes on an idle GPU
  ([`6a76e13`](https://github.com/jranaraki/vllm-fit/commit/6a76e132e9cc62efe6f684edd6f1408489c08be8))


## v0.5.14 (2026-10-01)

### Bug Fixes

- **params**: Don't count multi-token-prediction weights vLLM skips
  ([`0778222`](https://github.com/jranaraki/vllm-fit/commit/07782224f456fc2bd08e2527f383afb5d340297c))


## v0.5.13 (2026-10-01)

### Bug Fixes

- **config**: Derive max_model_len the way current vLLM does
  ([`e8ea31b`](https://github.com/jranaraki/vllm-fit/commit/e8ea31b897af5316bc14811bae2082b43cddbfc0))


## v0.5.12 (2026-10-01)

### Bug Fixes

- **registry**: Support local model directories and explain gated repos
  ([`cfc5bb7`](https://github.com/jranaraki/vllm-fit/commit/cfc5bb700709dc1bad76975df71288234132d67f))


## v0.5.11 (2026-10-01)

### Bug Fixes

- **gpu**: Honor CUDA_VISIBLE_DEVICES and pin the GPUs that were sized
  ([`7656af8`](https://github.com/jranaraki/vllm-fit/commit/7656af80002f0f2f516c5e03b0da44bf18e4288b))


## v0.5.10 (2026-10-01)

### Bug Fixes

- **hardware**: Don't silently fall back to CPU when GPUs can't be sized
  ([`829f96c`](https://github.com/jranaraki/vllm-fit/commit/829f96c1f65c33e6b8d282dd7d8bc0e264291680))


## v0.5.9 (2026-10-01)

### Bug Fixes

- **estimator**: Model sliding-window and hybrid layers per request
  ([`4136d0d`](https://github.com/jranaraki/vllm-fit/commit/4136d0d86e5f34aa1b2c715e4cbf4d051fd410e7))


## v0.5.8 (2026-10-01)

### Bug Fixes

- Stop pinning --max_num_batched_tokens 2048
  ([`0d63cda`](https://github.com/jranaraki/vllm-fit/commit/0d63cda0d6d51472a412a0172ba2579d8401900c))


## v0.5.7 (2026-10-01)

### Bug Fixes

- **profile**: Surface vLLM errors instead of treating every failure as OOM
  ([`d8a8eb4`](https://github.com/jranaraki/vllm-fit/commit/d8a8eb4e4e5243065adcbe3dc7a1185f90d77174))


## v0.5.6 (2026-10-01)

### Bug Fixes

- **cli**: Print the vllm serve command on one line
  ([`d9603c3`](https://github.com/jranaraki/vllm-fit/commit/d9603c35eac9c9b26e7f9adf82c550d0d32abc03))

### Build System

- Require huggingface-hub>=0.25 and make `uv run pytest` work
  ([`44bfae7`](https://github.com/jranaraki/vllm-fit/commit/44bfae73868533e57dd4ee449b66f377133e9fa8))


## v0.5.5 (2026-10-01)

### Bug Fixes

- Cost float32 weights at vLLM's 16-bit serving size
  ([`d5b5b39`](https://github.com/jranaraki/vllm-fit/commit/d5b5b39c6046328a5164c5c83c2ef92aaf6637af))


## v0.5.4 (2026-10-01)

### Bug Fixes

- Cap gpu_memory_utilization by currently free VRAM
  ([`d5e0238`](https://github.com/jranaraki/vllm-fit/commit/d5e0238231ecb39626bb5401c07c30a0b575d9c3))


## v0.5.3 (2026-10-01)

### Bug Fixes

- **cli**: Size multi-GPU setups against the smallest card
  ([`84973a2`](https://github.com/jranaraki/vllm-fit/commit/84973a21f8b5eae03d569f4474027ede1ab87752))


## v0.5.2 (2026-10-01)

### Bug Fixes

- **estimator**: Size GGUF models by quant bits-per-weight
  ([`6507308`](https://github.com/jranaraki/vllm-fit/commit/6507308f7463df6faaba4ef906dab122b920bd20))


## v0.5.1 (2026-10-01)

### Bug Fixes

- **config**: Don't apply Gemma-3 rope factor to max_model_len
  ([`e59cb21`](https://github.com/jranaraki/vllm-fit/commit/e59cb2168d51ddab6b936a8c87ebebe1b1f742eb))


## v0.5.0 (2026-09-09)

### Features

- **cpu,macos**: Apple Silicon detection, VLLM_CPU_KVCACHE_SPACE, RAM-aware CPU sizing
  ([`ec5855c`](https://github.com/jranaraki/vllm-fit/commit/ec5855cfd97bdccf8f9a5f7fac375a0514718e50))


## v0.4.2 (2026-09-09)

### Bug Fixes

- **estimator,cli,registry**: Harden config loading and estimator edge cases
  ([`8914e85`](https://github.com/jranaraki/vllm-fit/commit/8914e85fce017e8150a21054da04877a74ba2dd0))


## v0.4.1 (2026-09-09)

### Bug Fixes

- **estimator**: Apply enforce-eager lever before declaring no-fit
  ([`61ccf65`](https://github.com/jranaraki/vllm-fit/commit/61ccf65ce34c5e9ff7cfe75012d5d3b2de07b26f))


## v0.4.0 (2026-09-08)

### Bug Fixes

- **estimator**: VLLM-accurate memory model and quant-aware sizing (Layers 3-4)
  ([`b1ed902`](https://github.com/jranaraki/vllm-fit/commit/b1ed90215fe6f3018f3a4d4bcbbeb8a85c9b57a1))

- **registry**: Clean offline cache-miss error, dedupe candidates
  ([`69741d1`](https://github.com/jranaraki/vllm-fit/commit/69741d1a4e2cb5f9a0d23b7189b4fc22100e1bda))

### Documentation

- Note architecture-aware estimation, CPU mode, and fix arg ordering
  ([`a25385d`](https://github.com/jranaraki/vllm-fit/commit/a25385d7c0d4c11ecc1507fedcf89c2d7ac7f285))

### Features

- **cli**: Wire exact param resolution and surface warnings
  ([`d2d6275`](https://github.com/jranaraki/vllm-fit/commit/d2d62753565055e0d75375a98cb38d3d7f90c1a7))

- **params**: Exact model-size ladder from HF metadata (Layer 2)
  ([`2f9c544`](https://github.com/jranaraki/vllm-fit/commit/2f9c544e0b0744904f6bc4f6c3ae33672490c682))

- **resolver**: Structure-aware config resolution (Layer 1)
  ([`2b05d1e`](https://github.com/jranaraki/vllm-fit/commit/2b05d1ea2011aa64fbf011871dd14b5e73908183))

### Testing

- Cover resolver, param ladder, and new estimator behavior
  ([`9602c31`](https://github.com/jranaraki/vllm-fit/commit/9602c31b0e7861c404cf6a10864dd8a9793048fd))


## v0.3.2 (2026-09-08)

### Bug Fixes

- **estimator**: Count MoE experts and untied head, scale activation per-GPU
  ([`44afa29`](https://github.com/jranaraki/vllm-fit/commit/44afa294ffdac1dd234c50c682b0e9b7a1683e12))

- **profiler**: Terminate engine-test child on interrupt and always restore stdio
  ([`5953346`](https://github.com/jranaraki/vllm-fit/commit/595334691f9ff973b1cac7bde6212c83ac061b33))

### Refactoring

- **cli**: Drop unused is_gguf_model import
  ([`f58d775`](https://github.com/jranaraki/vllm-fit/commit/f58d775d5597404bbdcd08431acb35490eeaae6d))


## v0.3.1 (2026-09-08)

### Bug Fixes

- **cli**: Repair serve command and honor --gpuid in recommend
  ([`16832ae`](https://github.com/jranaraki/vllm-fit/commit/16832ae261f667e39dfffe446b1059917e81fe97))

- **estimator**: Correct memory math for KV cache, params and dtype
  ([`ec3adcb`](https://github.com/jranaraki/vllm-fit/commit/ec3adcb92a00d18484c5c437d7b643e7b00e73a6))

- **hardware**: Add honest detect_hardware and drop dead code
  ([`7b7fee1`](https://github.com/jranaraki/vllm-fit/commit/7b7fee1feb94756d8b1879d78920e6e09788051c))

- **profiler**: Count all tests, handle interrupts, restore output streams
  ([`3bcf444`](https://github.com/jranaraki/vllm-fit/commit/3bcf444854c6dc7788775e26eeb7e33a1ca7e20c))

- **registry**: Handle genuine HF lookup errors and dedupe is_gguf_model
  ([`8cdd260`](https://github.com/jranaraki/vllm-fit/commit/8cdd2600acd79ca707e79c20a25fe631e1aecb50))


## v0.3.0 (2026-03-13)

### Features

- - Added CPU support for vLLM inference
  ([`52920cc`](https://github.com/jranaraki/vllm-fit/commit/52920ccbc129db2ac1c6ee3064a35fd8bc1e1087))


## v0.2.0 (2026-03-11)

### Features

- Added support for quantized models | Improved the estimations, specifically for the smaller GPUs
  (VRAM <= 4GB)
  ([`ba30698`](https://github.com/jranaraki/vllm-fit/commit/ba306981210ed2852c156f8f9b431d71c2ccafb0))


## v0.1.1 (2026-03-02)

### Bug Fixes

- Updated pyproject.toml
  ([`6c3db3c`](https://github.com/jranaraki/vllm-fit/commit/6c3db3cb151202c3e41e2fc6fe967423775bce39))


## v0.1.0 (2026-03-02)

### Features

- Initial release
  ([`ad2100d`](https://github.com/jranaraki/vllm-fit/commit/ad2100d9a163b6a82beca74112b07152533cb841))

- Initial release
  ([`3efc729`](https://github.com/jranaraki/vllm-fit/commit/3efc729868bfdbbf0b52a1d75dbe8845e4a6f356))

- Updated release.yml
  ([`89a8e8e`](https://github.com/jranaraki/vllm-fit/commit/89a8e8e9ffba176035b7b89e9e359da5a00cfcd5))

- Updated release.yml to see the logs
  ([`860a975`](https://github.com/jranaraki/vllm-fit/commit/860a975f993d94c252cd91efb47bdd82e6e05fce))


## v1.0.0 (2026-03-02)

### Features

- Initial release
  ([`3efc729`](https://github.com/jranaraki/vllm-fit/commit/3efc729868bfdbbf0b52a1d75dbe8845e4a6f356))

- Updated release.yml
  ([`89a8e8e`](https://github.com/jranaraki/vllm-fit/commit/89a8e8e9ffba176035b7b89e9e359da5a00cfcd5))

- Updated release.yml to see the logs
  ([`860a975`](https://github.com/jranaraki/vllm-fit/commit/860a975f993d94c252cd91efb47bdd82e6e05fce))


## v0.0.0 (2026-03-02)

- Initial Release
