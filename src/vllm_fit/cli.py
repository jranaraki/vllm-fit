import time
from typing import List, Optional

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel

from .engine_tester import DEFAULT_TIMEOUT, profile_parameters, profile_parameters_cpu
from .estimator import _CUDA_CONTEXT_GB, estimate_parameters, estimate_parameters_cpu
from .hardware import (
    UnsupportedHardwareError,
    detect_hardware,
    get_free_vram_info,
    get_ram_info,
    get_vram_info,
    gpu_launch_env,
    is_apple_silicon,
)
from .params import resolve_weights
from .registry import get_model_config

app = typer.Typer()
console = Console()


def _load_config(model_id: str):
    """Fetch the model's config.json, surfacing lookup/offline failures as a clean
    CLI error (actionable message + exit 1) instead of a raw Python traceback."""
    try:
        return get_model_config(model_id)
    except ValueError as exc:
        print("[red]❌ Could not load model configuration:[/red]")
        console.print(str(exc), markup=False)
        raise typer.Exit(1)


def _resolve_weight_info(config_repo_id: str, config: dict):
    """Best-effort exact weight metadata; None on any failure (offline, old hub)."""
    try:
        return resolve_weights(config_repo_id, config)
    except Exception:
        return None


def _print_mac_guidance() -> None:
    """On Apple Silicon, explain the vLLM-on-macOS reality (informational only)."""
    if not is_apple_silicon():
        return
    print("[cyan]🍎 Apple Silicon detected — vLLM runs on the CPU backend here.[/cyan]")
    print(
        "[dim]  • Native vLLM on macOS is a source build (no prebuilt wheels), CPU-only "
        "(no Apple-GPU acceleration), FP32/FP16, and experimental.\n"
        "  • For Metal GPU inference, see the community 'vllm-metal' (MLX) plugin instead.\n"
        "  • Memory below is the shared unified-memory pool.[/dim]"
    )
    print()


def _ram_label() -> str:
    return "Unified memory" if is_apple_silicon() else "CPU RAM"


def _print_warnings(params: dict) -> None:
    """Surface estimator warnings (fail-loud missing field, offline analytic, vLLM divergence)."""
    warnings = params.get("warnings") or []
    for warning in warnings:
        print(f"[yellow]⚠️  {warning}[/yellow]")
    if warnings:
        print()


def parse_gpu_ids(gpuid: str, vram_info: dict) -> list[int]:
    if gpuid == "all":
        return list(vram_info.keys())

    try:
        gpu_ids = [int(x.strip()) for x in gpuid.split(",")]
        valid_ids = [gid for gid in gpu_ids if gid in vram_info]
        return valid_ids
    except ValueError:
        return []


def sizing_vram(vram_info: dict, gpuids: list[int]) -> float:
    """Total VRAM to size against: smallest selected GPU x count.

    vLLM applies gpu_memory_utilization to each device's own memory and shards
    evenly under tensor parallelism, so the smallest card bounds every shard.
    """
    return min(vram_info[gid] for gid in gpuids) * len(gpuids)


def _print_mixed_gpu_warning(vram_info: dict, gpuids: list[int]) -> None:
    sizes = [vram_info[gid] for gid in gpuids]
    if max(sizes) - min(sizes) > 0.5:
        print(
            f"[yellow]⚠ Mixed GPU sizes ({', '.join(f'{s:.1f}' for s in sizes)} GB); "
            f"sizing every GPU as {min(sizes):.1f} GB[/yellow]"
        )


def free_vram_for_sizing(vram_info: dict, gpuids: list[int]) -> Optional[float]:
    """Free memory to size against, expressed on the smallest selected GPU.

    vLLM requires free >= utilization x total on every device, so the binding GPU is
    the one with the lowest (free - CUDA context) / total ratio, not simply the one
    with the least free memory. Returns None if free memory can't be read.
    """
    free = get_free_vram_info()
    if not all(gid in free for gid in gpuids):
        return None
    ratio = min((free[g] - _CUDA_CONTEXT_GB) / vram_info[g] for g in gpuids)
    return ratio * min(vram_info[g] for g in gpuids) + _CUDA_CONTEXT_GB


def _detect_hardware(device: str, gpuid: str) -> str:
    """Resolve the target hardware, exiting with a clear message when it can't be sized."""
    if device not in ("auto", "gpu", "cpu"):
        print(f"[red]--device must be 'auto', 'gpu' or 'cpu', not '{device}'[/red]")
        raise typer.Exit(2)
    try:
        hardware_type = detect_hardware(device)
    except UnsupportedHardwareError as exc:
        print(f"[red]❌ {exc}[/red]")
        if device == "auto":
            print("[dim]Pass --device cpu to size for vLLM's CPU backend instead.[/dim]")
        raise typer.Exit(1)
    if hardware_type == "cpu" and gpuid != "all":
        print(f"[yellow]⚠ --gpuid {gpuid} ignored: sizing for the CPU backend[/yellow]")
    return hardware_type


def _show_no_hardware_error():
    print("[red]❌ No compatible hardware detected[/red]")
    print()
    print("[dim]Alternatives:[/dim]")
    print("  • GPU: Install NVIDIA drivers and CUDA")
    print("  • CPU: Ensure sufficient RAM (16GB+ recommended)")
    print("  • Cloud: Use RunPod, Lambda Labs, Google Colab")


def _build_vllm_args(
    model_id: str,
    params: dict,
    enforce_eager: bool = False,
    config_repo_id: Optional[str] = None,
    hardware_type: str = "gpu",
) -> List[str]:
    """Build the `vllm serve` argument list for the given parameters."""
    args = ["vllm", "serve", model_id]

    if hardware_type == "gpu":
        args += ["--gpu_memory_utilization", str(params["gpu_memory_utilization"])]
        args += ["--tensor_parallel_size", str(params["tensor_parallel_size"])]

    args += ["--max_model_len", str(params["max_model_len"])]
    args += ["--max_num_seqs", str(params["max_num_seqs"])]
    # GPU: leave max_num_batched_tokens to vLLM. Its default is what the estimator sized
    # activation memory for, and vLLM raises it itself for models that can't chunk
    # prefill (pinning a value below max_model_len breaks those at startup).
    # CPU: some vLLM CPU builds disable chunked prefill, which requires
    # max_num_batched_tokens >= max_model_len; pinning it to max_model_len is valid
    # either way.
    if hardware_type == "cpu":
        args += ["--max_num_batched_tokens", str(params["max_model_len"])]

    if config_repo_id and config_repo_id != model_id.split(":")[0]:
        args += ["--hf-config-path", config_repo_id]
        args += ["--tokenizer", config_repo_id]

    if enforce_eager:
        args.append("--enforce-eager")

    return args


def _format_vllm_command(
    model_id: str,
    params: dict,
    enforce_eager: bool = False,
    config_repo_id: Optional[str] = None,
    hardware_type: str = "gpu",
    gpu_ids: Optional[List[int]] = None,
) -> str:
    args = _build_vllm_args(
        model_id, params, enforce_eager, config_repo_id, hardware_type
    )
    # The CPU KV cache is sized via an env var, not a CLI flag: show it as an inline
    # assignment prefix so the copy-pasteable command matches what `serve` actually runs.
    kv_space = params.get("kv_cache_space_gb")
    if hardware_type == "cpu" and kv_space:
        return f"VLLM_CPU_KVCACHE_SPACE={kv_space} " + " ".join(args)
    if hardware_type == "gpu" and gpu_ids:
        # Pin the GPUs that were sized, so running the command elsewhere in the shell
        # doesn't land on cuda:0 or on GPUs outside the user's allocation.
        prefix = " ".join(f"{k}={v}" for k, v in gpu_launch_env(gpu_ids).items())
        return f"{prefix} " + " ".join(args)
    return " ".join(args)


def _print_command(cmd: str) -> None:
    """Print a copy-pasteable command on one line: no hard wrapping at the terminal
    width (pasting a wrapped command runs a truncated one) and no markup parsing."""
    console.print(cmd, style="dim", soft_wrap=True, markup=False, highlight=False)


def _report_profiling_failure(params: dict, elapsed_time: Optional[float] = None) -> None:
    """Explain why profiling failed and exit non-zero; never print an untested command."""
    if params.get("interrupted"):
        print("[yellow]Profiling interrupted; no configuration was verified.[/yellow]")
        raise typer.Exit(130)
    if params.get("error"):
        print(f"[red]❌ Profiling stopped: {params['error']}[/red]")
        if params.get("error_detail"):
            console.print(params["error_detail"], style="dim", markup=False, highlight=False)
    else:
        print("[red]⚠️  No configuration fits in memory, even at the smallest settings tried[/red]")
    print()
    print(f"  • Attempted {params.get('attempts_made', '?')} configurations")
    if elapsed_time is not None:
        print(f"  • Time elapsed: {elapsed_time:.0f}s")
    raise typer.Exit(1)


@app.command()
def recommend(
    model_id: str,
    gpuid: str = typer.Option(
        "all", "--gpuid", help="GPU ID(s) to use (e.g., '0', '0,1', 'all')"
    ),
    device: str = typer.Option(
        "auto", "--device", help="Hardware to size for: 'auto', 'gpu' or 'cpu'"
    ),
) -> None:
    hardware_type = _detect_hardware(device, gpuid)

    config, config_repo_id = _load_config(model_id)

    weight_info = _resolve_weight_info(config_repo_id, config)

    if hardware_type == "cpu":
        total_ram = get_ram_info()
        params = estimate_parameters_cpu(config, total_ram, model_id, weight_info=weight_info)
        _print_mac_guidance()
        print(f"{_ram_label()}: {total_ram:.1f} GB")
        print()
        print("[dim]Using CPU mode (no GPU detected)[/dim]")
        print()
        _print_warnings(params)
    else:
        vram_info = get_vram_info()
        gpuids = parse_gpu_ids(gpuid, vram_info)
        if not gpuids:
            print("[red]No valid GPUs specified[/red]")
            _show_no_hardware_error()
            raise typer.Exit(1)

        total_vram = sum(vram_info[gid] for gid in gpuids)
        num_gpus = len(gpuids)
        params = estimate_parameters(
            config, sizing_vram(vram_info, gpuids), num_gpus, model_id,
            weight_info=weight_info, free_vram=free_vram_for_sizing(vram_info, gpuids),
        )

        gpu_info = f"{total_vram:.1f} GB"
        if num_gpus > 1:
            gpu_info += f" ({', '.join(f'{vram_info[g]:.1f}' for g in gpuids)} GB)"
        print(f"GPU VRAM: {gpu_info}")
        _print_mixed_gpu_warning(vram_info, gpuids)
        print()
        _print_warnings(params)

    if not params["can_fit"]:
        if hardware_type == "cpu":
            print("[red]⚠️  WARNING: Model may not fit in available RAM[/red]")
        else:
            print("[red]⚠️  WARNING: Model may not fit in available GPU memory[/red]")
        print()
        for reason in params["recommendations"]:
            if any(keyword in reason for keyword in ["exceed", "requires"]):
                print(f"  [red]• {reason}[/red]")
        print()
        print("[yellow]Recommendations:[/yellow]")
        for rec in params["recommendations"]:
            if not any(keyword in rec for keyword in ["exceed", "requires"]):
                print(f"  [dim]• {rec}[/dim]")
        print()
        print(
            "[yellow]Proceeding with theoretical parameters - actual results may vary![/yellow]"
        )
        print()

    print(
        Panel(
            f"[bold green]Recommended Parameters[/bold green]",
            title="Static Estimation",
        )
    )
    print(f"model_id: {model_id}")
    if hardware_type == "gpu":
        print(f"gpu_memory_utilization: {params['gpu_memory_utilization']}")
    print(f"max_model_len: {params['max_model_len']}")
    print(f"tensor_parallel_size: {params['tensor_parallel_size']}")
    print(f"max_num_seqs: {params['max_num_seqs']}")
    print(f"estimated_weights_memory_gb: {params['estimated_weights_memory_gb']}")
    if hardware_type == "cpu" and params.get("kv_cache_space_gb"):
        print(f"VLLM_CPU_KVCACHE_SPACE: {params['kv_cache_space_gb']} GB")
    print()
    print("[bold cyan]Run this command:[/bold cyan]")
    _print_command(
        _format_vllm_command(model_id, params, params.get('enforce_eager', False), config_repo_id, hardware_type, gpuids if hardware_type == 'gpu' else None)
    )


@app.command()
def profile(
    model_id: str,
    gpuid: str = typer.Option(
        "all", "--gpuid", help="GPU ID(s) to use (e.g., '0', '0,1', 'all')"
    ),
    timeout: int = typer.Option(
        DEFAULT_TIMEOUT, "--timeout",
        help="Seconds to wait for each vLLM test launch before giving up",
    ),
    device: str = typer.Option(
        "auto", "--device", help="Hardware to size for: 'auto', 'gpu' or 'cpu'"
    ),
) -> None:
    hardware_type = _detect_hardware(device, gpuid)

    config, config_repo_id = _load_config(model_id)

    if hardware_type == "cpu":
        total_ram = get_ram_info()
        _print_mac_guidance()
        print(f"[yellow]Using CPU mode ({total_ram:.1f} GB {_ram_label()})[/yellow]")
        print("[yellow]🔍 Starting simplified profiling...[/yellow]")
        print(
            "[dim]Note: CPU profiling is slower and uses conservative estimates[/dim]"
        )
        print()

        start_time = time.time()

        weight_info = _resolve_weight_info(config_repo_id, config)
        initial_params = estimate_parameters_cpu(
            config, total_ram, model_id, weight_info=weight_info
        )
        _print_warnings(initial_params)
        params = profile_parameters_cpu(
            model_id,
            initial_params,
            progress_callback=lambda msg: print(f"[dim]  {msg}[/dim]"),
            timeout=timeout,
        )

        elapsed_time = time.time() - start_time
        print()

        if not params.get("profiling_success", False):
            _report_profiling_failure(params, elapsed_time)
        else:
            print("[green]✓ Profiling completed successfully![/green]")
            print()
            print("[yellow]Summary:[/yellow]")
            print(f"  • Attempted {params.get('attempts_made', '?')} configurations")
            print(f"  • Final test: Len={params['max_model_len']}")
            if params.get("enforce_eager", False):
                print("  • Strategy: Used --enforce-eager mode for memory efficiency")
            print(f"  • Time elapsed: {elapsed_time:.0f}s")
            print()

        print(
            Panel(
                f"[bold green]Optimized Parameters[/bold green]",
                title="Dynamic Profiling",
            )
        )
        print(f"model_id: {model_id}")
        print(f"max_model_len: {params['max_model_len']}")
        print(f"tensor_parallel_size: {params['tensor_parallel_size']}")
        print(f"max_num_seqs: {params['max_num_seqs']}")
        if params.get("enforce_eager", False):
            print(f"enforce_eager: True")
        print()
        print("[bold cyan]Run this command:[/bold cyan]")
        _print_command(
            _format_vllm_command(model_id, params, params.get('enforce_eager', False), config_repo_id, hardware_type, gpuids if hardware_type == 'gpu' else None)
        )
    else:
        vram_info = get_vram_info()
        gpuids = parse_gpu_ids(gpuid, vram_info)

        if not gpuids:
            print("[red]No valid GPUs specified[/red]")
            _show_no_hardware_error()
            raise typer.Exit(1)

        num_gpus = len(gpuids)
        total_vram = sum(vram_info[gid] for gid in gpuids)
        weight_info = _resolve_weight_info(config_repo_id, config)
        initial_params = estimate_parameters(
            config, sizing_vram(vram_info, gpuids), num_gpus=num_gpus,
            model_id=model_id, weight_info=weight_info,
            free_vram=free_vram_for_sizing(vram_info, gpuids),
        )
        initial_params["gpu_ids"] = gpuids

        print(
            f"[yellow]Using {num_gpus} GPU(s): {gpuids} ({total_vram:.1f} GB total)[/yellow]"
        )
        _print_mixed_gpu_warning(vram_info, gpuids)
        print("[yellow]🔍 Starting dynamic profiling...[/yellow]")
        print()
        _print_warnings(initial_params)

        start_time = time.time()

        params = profile_parameters(
            model_id,
            initial_params,
            progress_callback=lambda msg: print(f"[dim]  {msg}[/dim]"),
            timeout=timeout,
        )

        elapsed_time = time.time() - start_time
        print()

        if not params.get("profiling_success", False):
            _report_profiling_failure(params, elapsed_time)
        else:
            print("[green]✓ Profiling completed successfully![/green]")
            print()
            print("[yellow]Summary:[/yellow]")
            print(f"  • Attempted {params.get('attempts_made', '?')} configurations")
            print(
                f"  • Final test: Memory={params['gpu_memory_utilization']}, Len={params['max_model_len']}"
            )
            if params.get("enforce_eager", False):
                print("  • Strategy: Used --enforce-eager mode for memory efficiency")
            print(f"  • Time elapsed: {elapsed_time:.0f}s")
            print()

        print(
            Panel(
                f"[bold green]Optimized Parameters[/bold green]",
                title="Dynamic Profiling",
            )
        )
        print(f"model_id: {model_id}")
        print(f"gpu_memory_utilization: {params['gpu_memory_utilization']}")
        print(f"max_model_len: {params['max_model_len']}")
        print(f"tensor_parallel_size: {params['tensor_parallel_size']}")
        print(f"max_num_seqs: {params['max_num_seqs']}")
        if params.get("enforce_eager", False):
            print(f"enforce_eager: True")
        print()
        print("[bold cyan]Run this command:[/bold cyan]")
        _print_command(
            _format_vllm_command(model_id, params, params.get('enforce_eager', False), config_repo_id, hardware_type, gpuids if hardware_type == 'gpu' else None)
        )


@app.command()
def serve(
    model_id: str,
    gpuid: str = typer.Option(
        "all", "--gpuid", help="GPU ID(s) to use (e.g., '0', '0,1', 'all')"
    ),
    timeout: int = typer.Option(
        DEFAULT_TIMEOUT, "--timeout",
        help="Seconds to wait for each vLLM test launch before giving up",
    ),
    device: str = typer.Option(
        "auto", "--device", help="Hardware to size for: 'auto', 'gpu' or 'cpu'"
    ),
) -> None:
    import os
    import subprocess

    hardware_type = _detect_hardware(device, gpuid)

    config, config_repo_id = _load_config(model_id)

    env = os.environ.copy()

    weight_info = _resolve_weight_info(config_repo_id, config)

    if hardware_type == "cpu":
        total_ram = get_ram_info()
        _print_mac_guidance()
        print(f"[yellow]Using CPU mode ({total_ram:.1f} GB {_ram_label()})[/yellow]")
        print("[yellow]🔍 Profiling optimal parameters...[/yellow]")
        print()

        initial_params = estimate_parameters_cpu(
            config, total_ram, model_id, weight_info=weight_info
        )
        _print_warnings(initial_params)
        params = profile_parameters_cpu(
            model_id,
            initial_params,
            progress_callback=lambda msg: print(f"[dim]  {msg}[/dim]"),
            timeout=timeout,
        )
        kv_space = params.get("kv_cache_space_gb")
        if kv_space:
            env["VLLM_CPU_KVCACHE_SPACE"] = str(kv_space)
    else:
        vram_info = get_vram_info()
        gpuids = parse_gpu_ids(gpuid, vram_info)

        if not gpuids:
            print("[red]No valid GPUs specified[/red]")
            _show_no_hardware_error()
            raise typer.Exit(1)

        num_gpus = len(gpuids)
        total_vram = sum(vram_info[gid] for gid in gpuids)
        initial_params = estimate_parameters(
            config, sizing_vram(vram_info, gpuids), num_gpus=num_gpus,
            model_id=model_id, weight_info=weight_info,
            free_vram=free_vram_for_sizing(vram_info, gpuids),
        )
        initial_params["gpu_ids"] = gpuids

        print(
            f"[yellow]Using {num_gpus} GPU(s): {gpuids} ({total_vram:.1f} GB total)[/yellow]"
        )
        _print_mixed_gpu_warning(vram_info, gpuids)
        print("[yellow]🔍 Profiling optimal parameters...[/yellow]")
        print()
        _print_warnings(initial_params)

        params = profile_parameters(
            model_id,
            initial_params,
            progress_callback=lambda msg: print(f"[dim]  {msg}[/dim]"),
            timeout=timeout,
        )
        env.update(gpu_launch_env(gpuids))

    if not params.get("profiling_success", False):
        print()
        print("[red]Not starting the server.[/red]")
        _report_profiling_failure(params)

    cmd = _build_vllm_args(
        model_id,
        params,
        params.get("enforce_eager", False),
        config_repo_id,
        hardware_type,
    )

    print()
    print("[green]Starting vLLM server with optimal parameters[/green]")
    print(f"[dim]{' '.join(cmd)}[/dim]")
    print()

    try:
        raise typer.Exit(subprocess.call(cmd, env=env))
    except FileNotFoundError:
        print(
            "[red]Could not find the 'vllm' executable on PATH.[/red] "
            "Install vLLM (e.g. `uv pip install vllm`) and ensure it is available."
        )
        raise typer.Exit(1)
