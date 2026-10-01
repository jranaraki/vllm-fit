"""Parse the memory figures vLLM logs while starting up.

These lines are the ground truth vllm-fit's estimates are checked against (see
scripts/collect_ground_truth.py). Formats follow vLLM V1:

    Model loading took 14.99 GiB memory and 5.3 seconds
    Available KV cache memory: 52.31 GiB
    GPU KV cache size: 190,464 tokens, Maximum concurrency for 32,768 tokens per request: 5.81x
    Graph capturing finished in 12 secs, took 0.53 GiB
"""

import re
from typing import Any, Dict

_PATTERNS = {
    "weights_gib": re.compile(r"Model loading took ([\d.]+) GiB"),
    "available_kv_gib": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
    "kv_tokens": re.compile(r"KV cache size: ([\d,]+) tokens"),
    "max_concurrency": re.compile(
        r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"
    ),
    "cudagraph_gib": re.compile(r"Graph capturing finished in [\d.]+ secs?, took ([\d.]+) GiB"),
}


def parse_startup_log(text: str) -> Dict[str, Any]:
    """Memory figures found in a vLLM startup log; missing ones are omitted."""
    result: Dict[str, Any] = {}
    for key, pattern in _PATTERNS.items():
        matches = pattern.findall(text)
        if not matches:
            continue
        value = matches[-1].replace(",", "")
        result[key] = int(value) if key == "kv_tokens" else float(value)
    return result
