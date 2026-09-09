from vllm_fit.cli import _format_vllm_command, _build_vllm_args


def _cpu_params():
    return {
        "gpu_memory_utilization": None,
        "max_model_len": 4096,
        "tensor_parallel_size": 1,
        "max_num_seqs": 4,
        "kv_cache_space_gb": 12,
    }


def _gpu_params():
    return {
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
        "max_num_seqs": 32,
    }


def test_cpu_command_prefixes_kvcache_env():
    cmd = _format_vllm_command("some/model", _cpu_params(), hardware_type="cpu")
    assert cmd.startswith("VLLM_CPU_KVCACHE_SPACE=12 ")
    assert "vllm serve some/model" in cmd


def test_cpu_argv_has_no_env_token():
    # The argv fed to subprocess must NOT carry the env assignment (it would become argv[0]).
    args = _build_vllm_args("some/model", _cpu_params(), hardware_type="cpu")
    assert args[0] == "vllm"
    assert not any(a.startswith("VLLM_CPU_KVCACHE_SPACE=") for a in args)


def test_gpu_command_has_no_kvcache_env():
    cmd = _format_vllm_command("some/model", _gpu_params(), hardware_type="gpu")
    assert "VLLM_CPU_KVCACHE_SPACE" not in cmd
    assert cmd.startswith("vllm serve some/model")
