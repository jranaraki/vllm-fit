import time
from typing import List, Optional

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel

from .engine_tester import profile_parameters, profile_parameters_cpu
from .estimator import estimate_parameters, estimate_parameters_cpu
from .hardware import get_vram_info, detect_hardware, get_ram_info, is_apple_silicon
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
    # The estimator sizes activation headroom for a 2048-token prefill batch; pin
    # vLLM to the same bound so the served config matches what we reserved for.
    args += ["--max_num_batched_tokens", "2048"]

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
) -> str:
    args = _build_vllm_args(
        model_id, params, enforce_eager, config_repo_id, hardware_type
    )
    # The CPU KV cache is sized via an env var, not a CLI flag: show it as an inline
    # assignment prefix so the copy-pasteable command matches what `serve` actually runs.
    kv_space = params.get("kv_cache_space_gb")
    if hardware_type == "cpu" and kv_space:
        return f"VLLM_CPU_KVCACHE_SPACE={kv_space} " + " ".join(args)
    return " ".join(args)


@app.command()
def recommend(
    model_id: str,
    gpuid: str = typer.Option(
        "all", "--gpuid", help="GPU ID(s) to use (e.g., '0', '0,1', 'all')"
    ),
) -> None:
    hardware_type = detect_hardware()

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
            config, total_vram, num_gpus, model_id, weight_info=weight_info
        )

        gpu_info = f"{total_vram:.1f} GB"
        if num_gpus > 1:
            gpu_info += f" ({num_gpus}x ~{total_vram / num_gpus:.1f} GB each)"
        print(f"GPU VRAM: {gpu_info}")
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
    print(
        f"[dim]{_format_vllm_command(model_id, params, params.get('enforce_eager', False), config_repo_id, hardware_type)}[/dim]"
    )


@app.command()
def profile(
    model_id: str,
    gpuid: str = typer.Option(
        "all", "--gpuid", help="GPU ID(s) to use (e.g., '0', '0,1', 'all')"
    ),
) -> None:
    hardware_type = detect_hardware()

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
        )

        elapsed_time = time.time() - start_time
        print()

        if not params.get("profiling_success", False):
            print("[red]⚠️  Profiling could not find a successful configuration[/red]")
            print()
            print("[yellow]Summary:[/yellow]")
            print(f"  • Attempted {params.get('attempts_made', '?')} configurations")
            print(f"  • Time elapsed: {elapsed_time:.0f}s")
            if params.get("enforce_eager", False):
                print(
                    "  • Strategy: Enabled --enforce-eager to reduce memory (disabled torch.compile)"
                )
            print("  • Parameters below are our best attempt")
            print()
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
        print(
            f"[dim]{_format_vllm_command(model_id, params, params.get('enforce_eager', False), config_repo_id, hardware_type)}[/dim]"
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
        small_gpu = total_vram / num_gpus < 8
        weight_info = _resolve_weight_info(config_repo_id, config)
        initial_params = estimate_parameters(
            config, total_vram, num_gpus=num_gpus, model_id=model_id, weight_info=weight_info
        )
        initial_params["gpu_ids"] = gpuids

        print(
            f"[yellow]Using {num_gpus} GPU(s): {gpuids} ({total_vram:.1f} GB total)[/yellow]"
        )
        print("[yellow]🔍 Starting dynamic profiling...[/yellow]")
        print()
        _print_warnings(initial_params)

        start_time = time.time()

        params = profile_parameters(
            model_id,
            initial_params,
            progress_callback=lambda msg: print(f"[dim]  {msg}[/dim]"),
        )

        elapsed_time = time.time() - start_time
        print()

        if not params.get("profiling_success", False):
            print("[red]⚠️  Profiling could not find a successful configuration[/red]")
            print()
            print("[yellow]Summary:[/yellow]")
            print(f"  • Attempted {params.get('attempts_made', '?')} configurations")
            print(f"  • Time elapsed: {elapsed_time:.0f}s")
            if params.get("enforce_eager", False):
                print(
                    "  • Strategy: Enabled --enforce-eager to reduce memory (disabled torch.compile)"
                )
            print("  • Parameters below are our best attempt")
            print()
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
        print(
            f"[dim]{_format_vllm_command(model_id, params, params.get('enforce_eager', False), config_repo_id, hardware_type)}[/dim]"
        )


@app.command()
def serve(
    model_id: str,
    gpuid: str = typer.Option(
        "all", "--gpuid", help="GPU ID(s) to use (e.g., '0', '0,1', 'all')"
    ),
) -> None:
    import os
    import subprocess

    hardware_type = detect_hardware()

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
            config, total_vram, num_gpus=num_gpus, model_id=model_id, weight_info=weight_info
        )
        initial_params["gpu_ids"] = gpuids

        print(
            f"[yellow]Using {num_gpus} GPU(s): {gpuids} ({total_vram:.1f} GB total)[/yellow]"
        )
        print("[yellow]🔍 Profiling optimal parameters...[/yellow]")
        print()
        _print_warnings(initial_params)

        params = profile_parameters(
            model_id,
            initial_params,
            progress_callback=lambda msg: print(f"[dim]  {msg}[/dim]"),
        )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpuids))

    if not params.get("profiling_success", False):
        print()
        print(
            "[red]Profiling did not find a working configuration; not starting the server.[/red]"
        )
        raise typer.Exit(1)

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
