import io
import multiprocessing
import os
import sys
from typing import Optional, Callable, List


def _test_engine_worker(
    model_id: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    tensor_parallel_size: int,
    max_num_seqs: int,
    enforce_eager: bool = False,
    gpu_ids: Optional[List[int]] = None,
) -> int:
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["GLOG_v"] = "3"
    os.environ["GLOO_DEBUG"] = "WARN"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    sys.stdout = devnull
    sys.stderr = devnull

    from vllm import LLM

    llm = LLM(
        model=model_id,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
    )
    devnull.close()
    return max_num_seqs


def _test_configuration(
    model_id: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    tensor_parallel_size: int,
    max_num_seqs: int,
    enforce_eager: bool,
    gpu_ids: List[int],
    timeout: int = 180,
) -> tuple[bool, bool]:
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=_test_engine_worker,
        args=(
            model_id,
            gpu_memory_utilization,
            max_model_len,
            tensor_parallel_size,
            max_num_seqs,
            enforce_eager,
            gpu_ids,
        ),
    )

    devnull = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull

    p.start()
    p.join(timeout=timeout)

    sys.stdout = old_stdout
    sys.stderr = old_stderr
    devnull.close()

    if p.is_alive():
        p.terminate()
        return False, True  # Failed, timeout

    if p.exitcode == 0:
        return True, False  # Success, no timeout

    return False, False  # Failed, no timeout


def _binary_search_max_num_seqs(
    model_id: str,
    fixed_params: dict,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> int:
    gpu_ids = fixed_params["gpu_ids"]
    gpu_memory_utilization = fixed_params["gpu_memory_utilization"]
    max_model_len = fixed_params["max_model_len"]
    tensor_parallel_size = fixed_params["tensor_parallel_size"]
    enforce_eager = fixed_params["enforce_eager"]

    low = 1
    high = fixed_params["max_num_seqs"] * 4
    best = fixed_params["max_num_seqs"]

    while low <= high:
        mid = (low + high) // 2

        if progress_callback:
            progress_callback(
                f"  Binary search Seqs: testing {mid} (range {low}-{high})"
            )

        success, timeout = _test_configuration(
            model_id,
            gpu_memory_utilization,
            max_model_len,
            tensor_parallel_size,
            mid,
            enforce_eager,
            gpu_ids,
        )

        if success and not timeout:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best


def _binary_search_max_model_len(
    model_id: str,
    fixed_params: dict,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> int:
    gpu_ids = fixed_params["gpu_ids"]
    gpu_memory_utilization = fixed_params["gpu_memory_utilization"]
    tensor_parallel_size = fixed_params["tensor_parallel_size"]
    max_num_seqs = fixed_params["max_num_seqs"]
    enforce_eager = fixed_params["enforce_eager"]

    low = fixed_params["max_model_len"]
    high = max(32768, fixed_params["max_model_len"] * 2)
    best = fixed_params["max_model_len"]

    while low <= high:
        mid = (low + high) // 2

        if progress_callback:
            progress_callback(
                f"  Binary search Len: testing {mid} (range {low}-{high})"
            )

        success, timeout = _test_configuration(
            model_id,
            gpu_memory_utilization,
            mid,
            tensor_parallel_size,
            max_num_seqs,
            enforce_eager,
            gpu_ids,
        )

        if success and not timeout:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best


def profile_parameters(
    model_id: str,
    initial_params: dict,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    gpu_ids = initial_params.get("gpu_ids", [0])
    gpu_memory_utilization = initial_params["gpu_memory_utilization"]
    max_model_len = initial_params["max_model_len"]
    tensor_parallel_size = initial_params["tensor_parallel_size"]
    max_num_seqs = initial_params["max_num_seqs"]
    enforce_eager = False
    total_attempts = 0

    old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    def _log_attempt(msg: str):
        nonlocal total_attempts
        total_attempts += 1
        if progress_callback:
            progress_callback(f"Attempt {total_attempts} • {msg}")

    try:
        _log_attempt(
            f"Mem={gpu_memory_utilization:.2f} • Len={max_model_len} • TP={tensor_parallel_size} • Seqs={max_num_seqs} • Eager={'ON' if enforce_eager else 'OFF'}"
        )

        success, timeout = _test_configuration(
            model_id,
            gpu_memory_utilization,
            max_model_len,
            tensor_parallel_size,
            max_num_seqs,
            enforce_eager,
            gpu_ids,
        )

        if not success:
            if progress_callback:
                progress_callback("[red]✗ Initial config failed[/red]")

            enforce_eager = True
            _log_attempt(
                f"Enabling enforce_eager: Mem={gpu_memory_utilization:.2f} • Len={max_model_len} • TP={tensor_parallel_size} • Seqs={max_num_seqs} • Eager=ON"
            )

            success, timeout = _test_configuration(
                model_id,
                gpu_memory_utilization,
                max_model_len,
                tensor_parallel_size,
                max_num_seqs,
                enforce_eager,
                gpu_ids,
            )

        if not success:
            if progress_callback:
                progress_callback("[yellow]Finding baseline configuration...[/yellow]")

            while not success:
                if max_num_seqs > 1:
                    previous_seqs = max_num_seqs
                    max_num_seqs = max(1, max_num_seqs // 2)
                    if progress_callback:
                        progress_callback(
                            f"[yellow]  Reducing Seqs {previous_seqs} → {max_num_seqs}[/yellow]"
                        )
                elif max_model_len > 512:
                    previous_len = max_model_len
                    max_model_len = max(512, max_model_len - 512)
                    if progress_callback:
                        progress_callback(
                            f"[yellow]  Reducing Len {previous_len} → {max_model_len}[/yellow]"
                        )
                elif gpu_memory_utilization > 0.5:
                    previous_mem = gpu_memory_utilization
                    gpu_memory_utilization = max(0.5, gpu_memory_utilization - 0.05)
                    if progress_callback:
                        progress_callback(
                            f"[yellow]  Reducing Mem {previous_mem:.2f} → {gpu_memory_utilization:.2f}[/yellow]"
                        )
                else:
                    if progress_callback:
                        progress_callback(
                            "[red]Cannot find working configuration[/red]"
                        )
                    break

                _log_attempt(
                    f"Baseline: Mem={gpu_memory_utilization:.2f} • Len={max_model_len} • TP={tensor_parallel_size} • Seqs={max_num_seqs} • Eager=ON"
                )

                success, timeout = _test_configuration(
                    model_id,
                    gpu_memory_utilization,
                    max_model_len,
                    tensor_parallel_size,
                    max_num_seqs,
                    enforce_eager,
                    gpu_ids,
                )

        if not success:
            if progress_callback:
                progress_callback("[red]✗ No configuration found[/red]")
            return {
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
                "tensor_parallel_size": tensor_parallel_size,
                "max_num_seqs": max_num_seqs,
                "enforce_eager": enforce_eager,
                "profiling_success": False,
                "attempts_made": total_attempts,
            }

        if progress_callback:
            progress_callback("[green]✓ Baseline found![/green]")
            progress_callback(
                "[cyan]Optimizing parameters with binary search...[/cyan]"
            )

        fixed_params = {
            "gpu_ids": gpu_ids,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
        }

        max_num_seqs = _binary_search_max_num_seqs(
            model_id, fixed_params, progress_callback
        )
        fixed_params["max_num_seqs"] = max_num_seqs

        max_model_len = _binary_search_max_model_len(
            model_id, fixed_params, progress_callback
        )
        fixed_params["max_model_len"] = max_model_len

        if progress_callback:
            progress_callback("[green]✓ Optimization complete![/green]")

        return {
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "profiling_success": True,
            "attempts_made": total_attempts,
        }

    finally:
        if old_cuda_visible:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
