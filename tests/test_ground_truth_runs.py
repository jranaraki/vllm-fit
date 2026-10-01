"""Check estimates against real vLLM startups recorded on real hardware.

Each JSON file in tests/fixtures/vllm_runs is written by scripts/collect_ground_truth.py
and holds the model's config, its exact weight size, the GPU, the vLLM arguments used,
and what vLLM logged. Nothing here needs a GPU or the network.

Tolerances:
- weights: within 5% of "Model loading took ... GiB";
- KV memory (safety): never predict more than 5% above vLLM's "Available KV cache
  memory" - over-prediction is what makes recommended configs fail to start;
- KV memory (usefulness): at least 70% of it, so recommendations aren't wasteful.
"""

import json
from pathlib import Path

import pytest

from vllm_fit.estimator import estimate_parameters, kv_cache_memory_gb
from vllm_fit.params import WeightInfo

RUNS = sorted((Path(__file__).parent / "fixtures" / "vllm_runs").glob("*.json"))

pytestmark = pytest.mark.skipif(not RUNS, reason="no recorded vLLM runs yet")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _started(path):
    run = _load(path)
    return run["result"].get("started") and "weights_gib" in run["result"]


@pytest.mark.parametrize("path", [p for p in RUNS if _started(p)], ids=lambda p: p.stem)
def test_weights_match_vllm(path):
    run = _load(path)
    tp = run["args"]["tensor_parallel_size"]
    wi = WeightInfo(source="fixture", **run["weights"])
    est = estimate_parameters(run["config"], run["gpu_total_gib"] * tp, tp,
                              run["model"], weight_info=wi)
    predicted = est["estimated_weights_memory_gb"] / tp
    measured = run["result"]["weights_gib"]
    assert abs(predicted - measured) <= 0.05 * measured, (predicted, measured)


@pytest.mark.parametrize("path", [p for p in RUNS if _started(p)], ids=lambda p: p.stem)
def test_kv_memory_is_safe_and_useful(path):
    run = _load(path)
    args = run["args"]
    measured = run["result"]["available_kv_gib"]
    wi = WeightInfo(source="fixture", **run["weights"])
    weights = estimate_parameters(run["config"], run["gpu_total_gib"], 1, run["model"],
                                  weight_info=wi)["estimated_weights_memory_gb"]
    predicted = kv_cache_memory_gb(
        run["config"], weights, run["gpu_total_gib"], args["gpu_memory_utilization"],
        args["tensor_parallel_size"], args.get("enforce_eager", False),
    )
    assert predicted <= measured * 1.05, f"over-predicts KV: {predicted:.2f} > {measured:.2f} GiB"
    assert predicted >= measured * 0.70, f"too conservative: {predicted:.2f} vs {measured:.2f} GiB"
