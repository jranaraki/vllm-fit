"""Exact model-size resolution from HuggingFace metadata (Layer 2).

The analytic parameter estimate is fragile across MoE / MLA / tied-head variants
and, worse, can't see how weights are actually stored (quantized, mixed precision).
When the network (or the local HF cache) can answer, we should ask it instead of
guessing.

:func:`resolve_weights` walks a fallback ladder, exact first:

  1. ``get_safetensors_metadata`` — per-dtype parameter counts read from safetensors
     *headers* (Range GETs, no weight download). Exact, family-agnostic, and — because
     it's per-dtype — it yields exact storage **bytes**, which already accounts for
     quantization and mixed precision (e.g. MXFP4) with zero format parsing.
  2. safetensors index ``metadata.total_size`` — exact storage bytes.
  3. ``model_info().safetensors`` — per-dtype counts / total from the Hub API.
  4. config ``num_parameters`` — a plain count (bytes then come from bytes-per-param).
  5. ``None`` — caller falls back to the analytic estimate.

Every rung is guarded and cache-first: reads hit ``~/.cache/huggingface`` first and
only touch the network on a miss. Under ``HF_HUB_OFFLINE=1`` or a blocked network,
network rungs raise and are caught, so the ladder degrades cleanly to ``None``.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Optional


# safetensors dtype string -> bytes per element.
_ST_DTYPE_BYTES: Dict[str, float] = {
    "F64": 8.0, "I64": 8.0, "U64": 8.0,
    "F32": 4.0, "I32": 4.0, "U32": 4.0,
    "F16": 2.0, "BF16": 2.0, "I16": 2.0, "U16": 2.0,
    "F8_E4M3": 1.0, "F8_E5M2": 1.0, "I8": 1.0, "U8": 1.0, "BOOL": 1.0,
    "I4": 0.5, "U4": 0.5, "F4": 0.5,
}


def _st_dtype_bytes(dtype: str) -> float:
    return _ST_DTYPE_BYTES.get(str(dtype).upper(), 2.0)


@dataclass
class WeightInfo:
    """Result of the size ladder. ``total_params`` and/or ``weights_bytes`` may be set;
    at least one is present when this is returned (callers get ``None`` otherwise)."""

    source: str
    total_params: Optional[int] = None
    weights_bytes: Optional[int] = None
    per_dtype: Dict[str, int] = field(default_factory=dict)

    def weights_gb(self) -> Optional[float]:
        if self.weights_bytes is not None:
            return self.weights_bytes / (1024**3)
        return None


def _from_safetensors_metadata(repo_id: str) -> Optional[WeightInfo]:
    try:
        from huggingface_hub import get_safetensors_metadata
    except Exception:
        return None
    try:
        meta = get_safetensors_metadata(repo_id)
    except Exception:
        return None

    per_dtype = dict(getattr(meta, "parameter_count", {}) or {})
    if not per_dtype:
        return None
    total = int(sum(per_dtype.values()))
    weights_bytes = int(sum(c * _st_dtype_bytes(dt) for dt, c in per_dtype.items()))
    return WeightInfo(
        source="safetensors_metadata",
        total_params=total,
        weights_bytes=weights_bytes,
        per_dtype={str(k): int(v) for k, v in per_dtype.items()},
    )


def _from_safetensors_index(repo_id: str) -> Optional[WeightInfo]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors.index.json",
            force_download=False,
        )
        with open(path, "r") as f:
            index = json.load(f)
    except Exception:
        return None

    total_size = (index.get("metadata") or {}).get("total_size")
    if not isinstance(total_size, (int, float)) or total_size <= 0:
        return None
    return WeightInfo(source="safetensors_index", weights_bytes=int(total_size))


def _from_model_info(repo_id: str) -> Optional[WeightInfo]:
    try:
        from huggingface_hub import HfApi
    except Exception:
        return None
    try:
        info = HfApi().model_info(repo_id)
    except Exception:
        return None

    st = getattr(info, "safetensors", None)
    if st is None:
        return None
    per_dtype = dict(getattr(st, "parameters", {}) or {})
    total = getattr(st, "total", None)
    if per_dtype:
        total = int(total) if total else int(sum(per_dtype.values()))
        weights_bytes = int(sum(c * _st_dtype_bytes(dt) for dt, c in per_dtype.items()))
        return WeightInfo(
            source="model_info",
            total_params=total,
            weights_bytes=weights_bytes,
            per_dtype={str(k): int(v) for k, v in per_dtype.items()},
        )
    if total:
        return WeightInfo(source="model_info", total_params=int(total))
    return None


def _from_config(config: dict) -> Optional[WeightInfo]:
    count = config.get("num_parameters") or config.get("num_params")
    if count:
        return WeightInfo(source="config", total_params=int(count))
    return None


def resolve_weights(repo_id: str, config: dict) -> Optional[WeightInfo]:
    """Best available model-size information, exact first. ``None`` when nothing but the
    analytic estimate is possible (offline cache miss, or repo has no usable metadata)."""
    for rung in (
        lambda: _from_safetensors_metadata(repo_id),
        lambda: _from_safetensors_index(repo_id),
        lambda: _from_model_info(repo_id),
        lambda: _from_config(config),
    ):
        info = rung()
        if info is not None:
            return info
    return None
