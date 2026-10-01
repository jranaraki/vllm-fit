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


def test_printed_command_is_not_wrapped(monkeypatch):
    import io

    from rich.console import Console

    import vllm_fit.cli as cli

    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=40, force_terminal=False))
    cmd = _format_vllm_command("org/[weird]-model", _gpu_params(), hardware_type="gpu")
    cli._print_command(cmd)
    assert buf.getvalue() == cmd + "\n"


def test_profiling_failure_exits_nonzero():
    import pytest
    import typer

    from vllm_fit.cli import _report_profiling_failure

    with pytest.raises(typer.Exit) as exc:
        _report_profiling_failure({"profiling_success": False, "error": "boom"}, 1.0)
    assert exc.value.exit_code == 1
    with pytest.raises(typer.Exit) as exc:
        _report_profiling_failure({"profiling_success": False, "interrupted": True})
    assert exc.value.exit_code == 130


def test_gpu_command_leaves_batched_tokens_to_vllm():
    # Pinning 2048 below max_model_len breaks models without chunked prefill.
    args = _build_vllm_args("some/model", _gpu_params(), hardware_type="gpu")
    assert "--max_num_batched_tokens" not in args


def test_cpu_command_batched_tokens_cover_max_model_len():
    args = _build_vllm_args("some/model", _cpu_params(), hardware_type="cpu")
    i = args.index("--max_num_batched_tokens")
    assert int(args[i + 1]) >= _cpu_params()["max_model_len"]
