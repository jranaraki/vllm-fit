#!/usr/bin/env python
"""Record a real vLLM startup as a ground-truth fixture for vllm-fit's tests.

Run on a machine with an NVIDIA GPU and vLLM installed:

    python scripts/collect_ground_truth.py Qwen/Qwen2.5-7B-Instruct \
        --gpu-memory-utilization 0.9 --max-model-len 8192

It launches `vllm serve` with the given arguments, waits until vLLM has logged its KV
cache size (or exits), stops the server, and writes
tests/fixtures/vllm_runs/<gpu>__<model>__tp<N>__<vllm-version>.json with the model's
config, exact weight size, GPU, arguments and the memory figures vLLM logged.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from vllm_fit.hardware import get_vram_info
from vllm_fit.params import resolve_weights
from vllm_fit.registry import get_model_config
from vllm_fit.vllm_log import parse_startup_log

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "vllm_runs"


def _gpu_name(index: int) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown-gpu"


def _vllm_version() -> str:
    try:
        import vllm

        return vllm.__version__
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--gpu-ids", default=None, help="comma-separated NVML indices (default: first N)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    opts = ap.parse_args()

    tp = opts.tensor_parallel_size
    vram = get_vram_info()
    gpu_ids = [int(g) for g in opts.gpu_ids.split(",")] if opts.gpu_ids else sorted(vram)[:tp]
    if len(gpu_ids) < tp or any(g not in vram for g in gpu_ids):
        print(f"Need {tp} visible GPU(s); found {sorted(vram)}", file=sys.stderr)
        return 2

    config, config_repo = get_model_config(opts.model)
    weight_info = resolve_weights(config_repo, config)
    if weight_info is None or weight_info.weights_bytes is None:
        print("Exact weight size unavailable (needs Hub access); not recording.", file=sys.stderr)
        return 2

    cmd = [
        "vllm", "serve", opts.model,
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(opts.gpu_memory_utilization),
        "--max-model-len", str(opts.max_model_len),
    ]
    if opts.enforce_eager:
        cmd.append("--enforce-eager")
    env = {
        **os.environ,
        # NVML indices are PCI-bus ordered; make CUDA agree.
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
    }

    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    lines = []
    deadline = time.time() + opts.timeout
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            lines.append(line)
            if re.search(r"Application startup complete|Graph capturing finished", line):
                break
            if time.time() > deadline:
                break
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)

    log = "".join(lines)
    result = parse_startup_log(log)
    result["started"] = "kv_tokens" in result
    if not result["started"]:
        result["error"] = "".join(lines[-15:])

    record = {
        "model": opts.model,
        "vllm_version": _vllm_version(),
        "gpu_name": _gpu_name(gpu_ids[0]),
        "gpu_total_gib": round(min(vram[g] for g in gpu_ids), 3),
        "args": {
            "tensor_parallel_size": tp,
            "gpu_memory_utilization": opts.gpu_memory_utilization,
            "max_model_len": opts.max_model_len,
            "enforce_eager": opts.enforce_eager,
        },
        "config": config,
        "weights": {"weights_bytes": weight_info.weights_bytes,
                    "total_params": weight_info.total_params},
        "result": result,
    }
    slug = lambda s: re.sub(r"[^A-Za-z0-9.]+", "-", s).strip("-")
    name = f"{slug(record['gpu_name'])}__{slug(opts.model)}__tp{tp}__{slug(record['vllm_version'])}.json"
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    (opts.out_dir / name).write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nWrote {opts.out_dir / name}: {result}")
    return 0 if result["started"] else 1


if __name__ == "__main__":
    sys.exit(main())
