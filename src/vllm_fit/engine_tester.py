import importlib.util
import multiprocessing
import os
import re
import signal
import sys
import tempfile
import traceback
from typing import Optional, Callable, List, Tuple

from .hardware import gpu_launch_env


# Default per-launch timeout. A launch includes weight loading and, on GPU, torch.compile
# and CUDA-graph capture, so large models need minutes even when already downloaded.
DEFAULT_TIMEOUT = 900

# Failure signatures meaning "this configuration needs more memory than is available".
# Anything else (missing package, gated repo, unsupported architecture, invalid argument)
# can't be fixed by shrinking the configuration, so profiling stops and reports it.
_MEMORY_ERROR_PATTERNS = [
    r"out of memory",
    r"OutOfMemoryError",
    r"No available memory for the cache blocks",
    r"larger than the available KV cache memory",
    r"KV cache is needed",
    r"less than desired GPU memory utilization",
    r"Insufficient memory",
    r"Cannot allocate memory",
    r"VLLM_CPU_KVCACHE_SPACE",
]
_MEMORY_ERROR_RE = re.compile("|".join(_MEMORY_ERROR_PATTERNS), re.IGNORECASE)


class ProfilingAborted(Exception):
    """A probe failed for a reason other than insufficient memory (or timed out)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _is_memory_failure(log_text: str, exitcode: Optional[int]) -> bool:
    # Killed by SIGKILL with no Python traceback: almost always the host OOM killer.
    if exitcode is not None and exitcode == -getattr(signal, "SIGKILL", 9):
        return True
    return bool(_MEMORY_ERROR_RE.search(log_text))


def _log_tail(log_text: str, lines: int = 15) -> str:
    return "\n".join(log_text.strip().splitlines()[-lines:])


def check_vllm_installed() -> None:
    if importlib.util.find_spec("vllm") is None:
        raise ProfilingAborted(
            "vLLM is not installed in this environment",
            "Install it (e.g. `uv pip install vllm`) and run profiling again.",
        )


def _redirect_output(log_path: str) -> None:
    # Become a process-group leader so a timeout can kill vLLM's engine and
    # tensor-parallel worker processes along with this one.
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    log = open(log_path, "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = log
    sys.stderr = log


def _build_llm(log_path: str, env: dict, **llm_kwargs) -> None:
    _redirect_output(log_path)
    os.environ.update(env)
    try:
        from vllm import LLM

        LLM(**llm_kwargs)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)


def _kill_process_group(p) -> None:
    if hasattr(os, "killpg"):
        try:
            os.killpg(p.pid, getattr(signal, "SIGKILL", 9))
        except (ProcessLookupError, PermissionError):
            pass
    if p.is_alive():
        p.kill()
    p.join()


def _run_probe(env: dict, llm_kwargs: dict, timeout: int) -> Tuple[bool, str]:
    """Launch one vLLM engine in a child process.

    Returns ``(started, log_text)``: ``started`` is False only when the failure looks
    like insufficient memory. Raises ``ProfilingAborted`` on a timeout or any other
    failure. ``log_text`` is vLLM's captured stdout/stderr either way, so a caller can
    read the figures vLLM logged on a successful start (see ``_measure_concurrency``).
    """
    fd, log_path = tempfile.mkstemp(prefix="vllm-fit-", suffix=".log")
    os.close(fd)
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_build_llm, args=(log_path, env), kwargs=llm_kwargs)
    try:
        try:
            p.start()
            p.join(timeout=timeout)
        except BaseException:
            # On Ctrl-C (or any error) don't leave processes holding memory.
            _kill_process_group(p)
            raise

        if p.is_alive():
            _kill_process_group(p)
            raise ProfilingAborted(
                f"vLLM did not start within {timeout}s (result inconclusive)",
                "The model may still be downloading or compiling. Run again once it is "
                "cached, or raise the limit with --timeout.",
            )
        # Reap any engine/worker processes left in the group.
        _kill_process_group(p)

        with open(log_path, errors="replace") as f:
            log_text = f.read()

        if p.exitcode == 0:
            return True, log_text
        if _is_memory_failure(log_text, p.exitcode):
            return False, log_text
        raise ProfilingAborted(
            f"vLLM failed for a reason other than memory (exit code {p.exitcode})",
            _log_tail(log_text),
        )
    finally:
        try:
            os.unlink(log_path)
        except OSError:
            pass


def _config_override_kwargs(model_id: str, config_repo_id: Optional[str]) -> dict:
    """The same --hf-config-path / --tokenizer the served command gets (GGUF repos
    without their own config.json borrow the base model's)."""
    if config_repo_id and config_repo_id != model_id.split(":")[0]:
        return {"hf_config_path": config_repo_id, "tokenizer": config_repo_id}
    return {}


def _test_configuration_cpu(
    model_id: str,
    max_model_len: int,
    max_num_seqs: int,
    enforce_eager: bool,
    timeout: int = DEFAULT_TIMEOUT,
    kv_cache_space_gb: Optional[int] = None,
    config_repo_id: Optional[str] = None,
) -> Tuple[bool, bool]:
    env = {"OMP_NUM_THREADS": str(os.cpu_count() or 4)}
    # Size the CPU KV cache to match what we recommend, so the probe is representative.
    if kv_cache_space_gb and kv_cache_space_gb > 0:
        env["VLLM_CPU_KVCACHE_SPACE"] = str(kv_cache_space_gb)
    started, _log_text = _run_probe(
        env,
        dict(
            model=model_id,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            # Matches the served CPU command (valid with or without chunked prefill).
            max_num_batched_tokens=max_model_len,
            **_config_override_kwargs(model_id, config_repo_id),
        ),
        timeout,
    )
    return started, False


def _test_configuration(
    model_id: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    tensor_parallel_size: int,
    max_num_seqs: int,
    enforce_eager: bool,
    gpu_ids: List[int],
    timeout: int = DEFAULT_TIMEOUT,
    config_repo_id: Optional[str] = None,
) -> Tuple[bool, bool]:
    env = {
        "NCCL_DEBUG": "WARN",
        "GLOG_v": "3",
        "GLOO_DEBUG": "WARN",
        "TORCH_CPP_LOG_LEVEL": "ERROR",
    }
    if gpu_ids:
        env.update(gpu_launch_env(gpu_ids))
    started, _log_text = _run_probe(
        env,
        dict(
            model=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            **_config_override_kwargs(model_id, config_repo_id),
        ),
        timeout,
    )
    return started, False


def _measure_concurrency(
    model_id: str,
    gpu_memory_utilization: Optional[float],
    max_model_len: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    gpu_ids: List[int],
    timeout: int = DEFAULT_TIMEOUT,
    config_repo_id: Optional[str] = None,
    kv_cache_space_gb: Optional[int] = None,
) -> Optional[float]:
    """One launch at the verified ``max_model_len``, reading vLLM's own
    "Maximum concurrency for N tokens per request: X.XXx" line.

    vLLM only checks that a *single* max_model_len-sized request fits in KV at
    startup — it never checks ``max_num_seqs`` against capacity — so searching for
    "the largest max_num_seqs that still starts" mostly confirmed whatever value was
    asked for, regardless of how many such requests actually fit. This reads the
    authoritative figure from vLLM's own log instead.

    Returns ``None`` if the probe fails or the figure can't be parsed; the caller
    then keeps its own (already KV-aware) estimate.
    """
    is_cpu = gpu_memory_utilization is None
    if is_cpu:
        env = {"OMP_NUM_THREADS": str(os.cpu_count() or 4)}
        if kv_cache_space_gb and kv_cache_space_gb > 0:
            env["VLLM_CPU_KVCACHE_SPACE"] = str(kv_cache_space_gb)
    else:
        env = {
            "NCCL_DEBUG": "WARN",
            "GLOG_v": "3",
            "GLOO_DEBUG": "WARN",
            "TORCH_CPP_LOG_LEVEL": "ERROR",
        }
        if gpu_ids:
            env.update(gpu_launch_env(gpu_ids))

    kwargs = dict(
        model=model_id,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        # The probe's own max_num_seqs doesn't affect whether vLLM starts or how much
        # KV it reports (vLLM derives concurrency from capacity, not the other way
        # round), so a generous placeholder is fine here.
        max_num_seqs=256,
        **_config_override_kwargs(model_id, config_repo_id),
    )
    if is_cpu:
        kwargs["max_num_batched_tokens"] = max_model_len
    else:
        kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        kwargs["tensor_parallel_size"] = tensor_parallel_size

    try:
        started, log_text = _run_probe(env, kwargs, timeout)
    except ProfilingAborted:
        return None
    if not started:
        return None

    from .vllm_log import parse_startup_log

    return parse_startup_log(log_text).get("max_concurrency")


def _binary_search_max_model_len(
    model_id: str,
    fixed_params: dict,
    progress_callback: Optional[Callable[[str], None]] = None,
    on_test: Optional[Callable[[], None]] = None,
) -> int:
    gpu_ids = fixed_params["gpu_ids"]
    gpu_memory_utilization = fixed_params["gpu_memory_utilization"]
    tensor_parallel_size = fixed_params["tensor_parallel_size"]
    max_num_seqs = fixed_params["max_num_seqs"]
    enforce_eager = fixed_params["enforce_eager"]

    low = fixed_params["max_model_len"]
    high = max(32768, fixed_params["max_model_len"] * 2)
    # vLLM rejects lengths above the model's own limit outright (not a memory failure).
    if fixed_params.get("max_len_cap"):
        high = min(high, fixed_params["max_len_cap"])
    best = fixed_params["max_model_len"]

    while low <= high:
        mid = (low + high) // 2

        if progress_callback:
            progress_callback(
                f"  Binary search Len: testing {mid} (range {low}-{high})"
            )

        if on_test:
            on_test()
        success, timeout = _test_configuration(
            model_id,
            gpu_memory_utilization,
            mid,
            tensor_parallel_size,
            max_num_seqs,
            enforce_eager,
            gpu_ids,
            timeout=fixed_params.get("timeout", DEFAULT_TIMEOUT),
            config_repo_id=fixed_params.get("config_repo_id"),
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
    timeout: int = DEFAULT_TIMEOUT,
    config_repo_id: Optional[str] = None,
    max_len_cap: Optional[int] = None,
) -> dict:
    probe_timeout = timeout
    gpu_ids = initial_params.get("gpu_ids", [0])
    gpu_memory_utilization = initial_params["gpu_memory_utilization"]
    max_model_len = initial_params["max_model_len"]
    tensor_parallel_size = initial_params["tensor_parallel_size"]
    max_num_seqs = initial_params["max_num_seqs"]
    enforce_eager = False
    total_attempts = 0

    def _log_attempt(msg: str):
        nonlocal total_attempts
        total_attempts += 1
        if progress_callback:
            progress_callback(f"Attempt {total_attempts} • {msg}")

    def _count_test():
        nonlocal total_attempts
        total_attempts += 1

    try:
        check_vllm_installed()
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
            timeout=probe_timeout,
            config_repo_id=config_repo_id,
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
                timeout=probe_timeout,
                config_repo_id=config_repo_id,
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
                    timeout=probe_timeout,
                    config_repo_id=config_repo_id,
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
            "timeout": probe_timeout,
            "config_repo_id": config_repo_id,
            "max_len_cap": max_len_cap,
        }

        max_model_len = _binary_search_max_model_len(
            model_id, fixed_params, progress_callback, _count_test
        )
        fixed_params["max_model_len"] = max_model_len

        if progress_callback:
            progress_callback("[cyan]Measuring real concurrency at that length...[/cyan]")
        total_attempts += 1
        measured = _measure_concurrency(
            model_id,
            gpu_memory_utilization,
            max_model_len,
            tensor_parallel_size,
            enforce_eager,
            gpu_ids,
            timeout=probe_timeout,
            config_repo_id=config_repo_id,
        )
        if measured is not None:
            max_num_seqs = max(1, min(256, int(measured)))

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

    except ProfilingAborted as exc:
        if progress_callback:
            progress_callback(f"[red]✗ {exc.reason}[/red]")
        return {
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "profiling_success": False,
            "attempts_made": total_attempts,
            "error": exc.reason,
            "error_detail": exc.detail,
        }

    except KeyboardInterrupt:
        if progress_callback:
            progress_callback("[yellow]Profiling interrupted by user[/yellow]")
        return {
            "interrupted": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "profiling_success": False,
            "attempts_made": total_attempts,
        }


def profile_parameters_cpu(
    model_id: str,
    initial_params: dict,
    progress_callback: Optional[Callable[[str], None]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    config_repo_id: Optional[str] = None,
    max_len_cap: Optional[int] = None,
) -> dict:
    probe_timeout = timeout
    max_model_len = initial_params["max_model_len"]
    max_num_seqs = initial_params["max_num_seqs"]
    enforce_eager = initial_params["enforce_eager"]
    kv_cache_space_gb = initial_params.get("kv_cache_space_gb")
    total_attempts = 0

    def _log_attempt(msg: str):
        nonlocal total_attempts
        total_attempts += 1
        if progress_callback:
            progress_callback(f"Attempt {total_attempts} • {msg}")

    try:
        check_vllm_installed()
        _log_attempt(
            f"Len={max_model_len} • Seqs={max_num_seqs} • Eager={'ON' if enforce_eager else 'OFF'}"
        )

        success, timeout = _test_configuration_cpu(
            model_id,
            max_model_len,
            max_num_seqs,
            enforce_eager,
            kv_cache_space_gb=kv_cache_space_gb,
            timeout=probe_timeout,
            config_repo_id=config_repo_id,
        )

        if not success and not enforce_eager:
            if progress_callback:
                progress_callback("[red]✗ Initial config failed[/red]")

            enforce_eager = True
            _log_attempt(
                f"Enabling enforce_eager: Len={max_model_len} • Seqs={max_num_seqs} • Eager=ON"
            )

            success, timeout = _test_configuration_cpu(
                model_id,
                max_model_len,
                max_num_seqs,
                enforce_eager,
                kv_cache_space_gb=kv_cache_space_gb,
                timeout=probe_timeout,
                config_repo_id=config_repo_id,
            )

        if not success:
            if progress_callback:
                progress_callback("[yellow]Finding baseline configuration...[/yellow]")

            while not success:
                if max_model_len > 128:
                    previous_len = max_model_len
                    max_model_len = max(128, int(max_model_len * 0.6))
                    if progress_callback:
                        progress_callback(
                            f"[yellow]  Reducing Len {previous_len} → {max_model_len}[/yellow]"
                        )
                else:
                    if progress_callback:
                        progress_callback(
                            "[red]Cannot find working configuration[/red]"
                        )
                    break

                _log_attempt(
                    f"Baseline: Len={max_model_len} • Seqs={max_num_seqs} • Eager=ON"
                )

                success, timeout = _test_configuration_cpu(
                    model_id,
                    max_model_len,
                    max_num_seqs,
                    enforce_eager,
                    kv_cache_space_gb=kv_cache_space_gb,
                    timeout=probe_timeout,
                    config_repo_id=config_repo_id,
                )

        if not success:
            if progress_callback:
                progress_callback("[red]✗ No configuration found[/red]")
            return {
                "gpu_memory_utilization": None,
                "max_model_len": max_model_len,
                "tensor_parallel_size": 1,
                "max_num_seqs": max_num_seqs,
                "enforce_eager": enforce_eager,
                "kv_cache_space_gb": kv_cache_space_gb,
                "profiling_success": False,
                "attempts_made": total_attempts,
            }

        if progress_callback:
            progress_callback("[green]✓ Baseline found![/green]")
            progress_callback(
                "[cyan]Optimizing parameters with binary search...[/cyan]"
            )

        # The baseline already works; search upward from it.
        low = max_model_len + 1
        high = max_model_len * 2
        if max_len_cap:
            high = min(high, max_len_cap)
        best = max_model_len

        while low <= high:
            mid = (low + high) // 2
            total_attempts += 1

            if progress_callback:
                progress_callback(
                    f"  Binary search Len: testing {mid} (range {low}-{high})"
                )

            success, timeout = _test_configuration_cpu(
                model_id,
                mid,
                max_num_seqs,
                enforce_eager,
                kv_cache_space_gb=kv_cache_space_gb,
                timeout=probe_timeout,
                config_repo_id=config_repo_id,
            )

            if success and not timeout:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        max_model_len = best

        if progress_callback:
            progress_callback("[cyan]Measuring real concurrency at that length...[/cyan]")
        total_attempts += 1
        measured = _measure_concurrency(
            model_id,
            None,
            max_model_len,
            1,
            enforce_eager,
            [],
            timeout=probe_timeout,
            config_repo_id=config_repo_id,
            kv_cache_space_gb=kv_cache_space_gb,
        )
        if measured is not None:
            max_num_seqs = max(1, min(256, int(measured)))

        if progress_callback:
            progress_callback("[green]✓ Optimization complete![/green]")

        return {
            "gpu_memory_utilization": None,
            "max_model_len": max_model_len,
            "tensor_parallel_size": 1,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "kv_cache_space_gb": kv_cache_space_gb,
            "profiling_success": True,
            "attempts_made": total_attempts,
        }

    except ProfilingAborted as exc:
        if progress_callback:
            progress_callback(f"[red]✗ {exc.reason}[/red]")
        return {
            "gpu_memory_utilization": None,
            "max_model_len": max_model_len,
            "tensor_parallel_size": 1,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "kv_cache_space_gb": kv_cache_space_gb,
            "profiling_success": False,
            "attempts_made": total_attempts,
            "error": exc.reason,
            "error_detail": exc.detail,
        }

    except KeyboardInterrupt:
        if progress_callback:
            progress_callback("[yellow]Profiling interrupted by user[/yellow]")
        return {
            "interrupted": True,
            "gpu_memory_utilization": None,
            "max_model_len": max_model_len,
            "tensor_parallel_size": 1,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "kv_cache_space_gb": kv_cache_space_gb,
            "profiling_success": False,
            "attempts_made": total_attempts,
        }
