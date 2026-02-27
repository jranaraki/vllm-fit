import time

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel

from .engine_tester import profile_parameters
from .estimator import estimate_parameters
from .hardware import get_vram_info, check_gpu_availability
from .registry import get_model_config

app = typer.Typer()
console = Console()


def parse_gpu_ids(gpuid: str, vram_info: dict) -> list[int]:
    if gpuid == "all":
        return list(vram_info.keys())

    try:
        gpu_ids = [int(x.strip()) for x in gpuid.split(",")]
        valid_ids = [gid for gid in gpu_ids if gid in vram_info]
        return valid_ids
    except ValueError:
        return []


def _show_no_gpu_error():
    print("[red]❌ No GPU detected[/red]")
    print()
    print("[yellow]Required: NVIDIA GPU with CUDA support[/yellow]")
    print()
    print("[dim]Troubleshooting steps:[/dim]")
    print("  1. Verify GPU is present: 'nvidia-smi' should list your GPU")
    print(
        "  2. Check PyTorch CUDA: python -c 'import torch; print(torch.cuda.is_available())'"
    )
    print(
        "  3. Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118"
    )
    print(
        "  4. Update/install NVIDIA drivers from https://developer.nvidia.com/cuda-downloads"
    )
    print("  5. Check container GPU passthrough if using Docker/VM")
    print()
    print("[dim]Alternatives:[/dim]")
    print("  • Use cloud GPU services: RunPod, Lambda Labs, Google Colab, etc.")
    print("  • Try CPU-only inference: HuggingFace Transformers pipeline")


def _format_vllm_command(
    model_id: str, params: dict, small_gpu: bool = False, enforce_eager: bool = False
) -> str:
    cmd = f"vllm serve {model_id}"
    cmd += f" --gpu_memory_utilization {params['gpu_memory_utilization']}"
    cmd += f" --max_model_len {params['max_model_len']}"
    cmd += f" --tensor_parallel_size {params['tensor_parallel_size']}"
    cmd += f" --max_num_seqs {params['max_num_seqs']}"
    if small_gpu:
        cmd += " --disable-frontend-multiprocessing"
    if enforce_eager:
        cmd += " --enforce-eager"
    return cmd


@app.command()
def recommend(model_id: str) -> None:
    has_gpu, error_msg = check_gpu_availability()
    if not has_gpu:
        _show_no_gpu_error()
        raise typer.Exit(1)

    config = get_model_config(model_id)
    vram_info = get_vram_info()

    if not vram_info:
        _show_no_gpu_error()
        raise typer.Exit(1)

    total_vram = sum(vram_info.values())
    num_gpus = len(vram_info)
    small_gpu = total_vram < 8
    params = estimate_parameters(config, total_vram, num_gpus)

    gpu_info = f"{total_vram:.1f} GB"
    if num_gpus > 1:
        gpu_info += f" ({num_gpus}x ~{total_vram / num_gpus:.1f} GB each)"
    print(f"GPU VRAM: {gpu_info}")
    print()

    if not params["can_fit"]:
        print("[red]⚠️  WARNING: Model may not fit in available GPU memory[/red]")
        print()
        for reason in params["recommendations"]:
            if any(keyword in reason for keyword in ["exceed", "requirements"]):
                print(f"  [red]• {reason}[/red]")
        print()
        print("[yellow]Recommendations:[/yellow]")
        for rec in params["recommendations"]:
            if not any(keyword in rec for keyword in ["exceed", "requirements"]):
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
    print(f"gpu_memory_utilization: {params['gpu_memory_utilization']}")
    print(f"max_model_len: {params['max_model_len']}")
    print(f"tensor_parallel_size: {params['tensor_parallel_size']}")
    print(f"max_num_seqs: {params['max_num_seqs']}")
    print(f"estimated_weights_memory_gb: {params['estimated_weights_memory_gb']}")
    print()
    print("[bold cyan]Run this command:[/bold cyan]")
    print(
        f"[dim]{_format_vllm_command(model_id, params, small_gpu, not params['can_fit'])}[/dim]"
    )


@app.command()
def profile(
    model_id: str,
    gpuid: str = typer.Option(
        "all", "--gpuid", help="GPU ID(s) to use (e.g., '0', '0,1', 'all')"
    ),
) -> None:
    has_gpu, error_msg = check_gpu_availability()
    if not has_gpu:
        _show_no_gpu_error()
        raise typer.Exit(1)

    config = get_model_config(model_id)
    vram_info = get_vram_info()
    gpuids = parse_gpu_ids(gpuid, vram_info)

    if not gpuids:
        print("[red]No valid GPUs specified[/red]")
        _show_no_gpu_error()
        raise typer.Exit(1)

    num_gpus = len(gpuids)
    total_vram = sum(vram_info[gid] for gid in gpuids)
    small_gpu = total_vram / num_gpus < 8
    initial_params = estimate_parameters(config, total_vram, num_gpus=num_gpus)
    initial_params["gpu_ids"] = gpuids

    print(
        f"[yellow]Using {num_gpus} GPU(s): {gpuids} ({total_vram:.1f} GB total)[/yellow]"
    )
    print("[yellow]🔍 Starting dynamic profiling...[/yellow]")
    print()

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
            f"[bold green]Optimized Parameters[/bold green]", title="Dynamic Profiling"
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
        f"[dim]{_format_vllm_command(model_id, params, small_gpu, params.get('enforce_eager', False))}[/dim]"
    )


@app.command()
def serve(
    model_id: str,
    gpuid: str = typer.Option(
        "all", "--gpuid", help="GPU ID(s) to use (e.g., '0', '0,1', 'all')"
    ),
) -> None:
    has_gpu, error_msg = check_gpu_availability()
    if not has_gpu:
        _show_no_gpu_error()
        raise typer.Exit(1)

    from vllm.entrypoints.openai.api_server import serve

    config = get_model_config(model_id)
    vram_info = get_vram_info()
    gpuids = parse_gpu_ids(gpuid, vram_info)

    if not gpuids:
        print("[red]No valid GPUs specified[/red]")
        _show_no_gpu_error()
        raise typer.Exit(1)

    num_gpus = len(gpuids)
    total_vram = sum(vram_info[gid] for gid in gpuids)
    small_gpu = total_vram / num_gpus < 8
    initial_params = estimate_parameters(config, total_vram, num_gpus=num_gpus)
    initial_params["gpu_ids"] = gpuids

    print(
        f"[yellow]Using {num_gpus} GPU(s): {gpuids} ({total_vram:.1f} GB total)[/yellow]"
    )
    print("[yellow]🔍 Profiling optimal parameters...[/yellow]")
    print()

    params = profile_parameters(
        model_id,
        initial_params,
        progress_callback=lambda msg: print(f"[dim]  {msg}[/dim]"),
    )

    print()
    print("[green]Starting vLLM server with optimal parameters[/green]")
    print()
    import uvicorn

    uvicorn.run(
        lambda: serve(
            model=model_id,
            gpu_memory_utilization=params["gpu_memory_utilization"],
            max_model_len=params["max_model_len"],
            tensor_parallel_size=params["tensor_parallel_size"],
            max_num_seqs=params["max_num_seqs"],
            enforce_eager=params.get("enforce_eager", False),
        )
    )
