# CHANGELOG

<!-- version list -->

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
