import json
import os
from typing import Dict, Any, Tuple, Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HFValidationError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

try:  # available on modern hub; guarded for old versions
    from huggingface_hub.errors import LocalEntryNotFoundError
except Exception:  # pragma: no cover - depends on installed hub version
    class LocalEntryNotFoundError(Exception):
        pass

# is_gguf_model lives in the dependency-free estimator module; re-exported here
# for backwards compatibility (and used by the config lookup below).
from .estimator import is_gguf_model

# Errors that mean "this candidate repo/file genuinely isn't there" — we swallow
# them so the lookup can fall through to the next candidate and, ultimately, to
# the informative ValueError below. Transient transport failures (generic
# HfHubHTTPError such as 5xx/429, connection/timeout errors) are deliberately
# NOT included, so they propagate instead of being misreported as "not found".
_LOOKUP_ERRORS = (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    GatedRepoError,
    HFValidationError,
    # Offline + not-in-cache: treat as a clean miss so we fall through to the
    # actionable ValueError rather than leaking a raw traceback.
    LocalEntryNotFoundError,
)


def extract_repo_id(model_id: str) -> str:
    return model_id.split(":")[0]


def try_extract_base_model(repo_id: str) -> list:
    base_candidates = [
        repo_id,
        repo_id.replace("-GGUF", ""),
        repo_id.replace("-GGML", ""),
        repo_id.replace("-gguf", ""),
        repo_id.replace("-ggml", ""),
        repo_id.replace("_GGUF", ""),
        repo_id.replace("_GGML", ""),
    ]
    # Preserve order but drop duplicates (identical when there's no GGUF suffix),
    # so we don't repeat the same cache/network lookup or list it seven times.
    seen = set()
    unique = []
    for candidate in base_candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def try_find_non_gguf_base(repo_id: str) -> Optional[str]:
    """Try to find a non-GGUF base repo for tokenizer files."""
    base_candidates = [
        repo_id.replace("-GGUF", ""),
        repo_id.replace("-GGML", ""),
        repo_id.replace("-gguf", ""),
        repo_id.replace("-ggml", ""),
        repo_id.replace("_GGUF", ""),
        repo_id.replace("_GGML", ""),
    ]

    for candidate in base_candidates:
        if candidate != repo_id:
            try:
                hf_hub_download(
                    repo_id=candidate,
                    filename="config.json",
                    force_download=False,
                )
                return candidate
            except _LOOKUP_ERRORS:
                continue

    return None


def gguf_repo_has_config(repo_id: str) -> bool:
    """Check if the GGUF repo has its own config.json."""
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            force_download=False,
        )
        return True
    except _LOOKUP_ERRORS:
        return False


def get_model_config(model_id: str) -> Tuple[Dict[str, Any], str]:
    repo_id = extract_repo_id(model_id)

    if is_gguf_model(model_id):
        has_config = gguf_repo_has_config(repo_id)

        for candidate in try_extract_base_model(repo_id):
            try:
                config_path = hf_hub_download(
                    repo_id=candidate,
                    filename="config.json",
                    force_download=False,
                )
                with open(config_path, "r") as f:
                    config = json.load(f)

                    if has_config and candidate == repo_id:
                        return config, repo_id

                    if not has_config and candidate != repo_id:
                        return config, candidate
            except _LOOKUP_ERRORS:
                continue

    for candidate in try_extract_base_model(repo_id):
        try:
            config_path = hf_hub_download(
                repo_id=candidate,
                filename="config.json",
                force_download=False,
            )
            with open(config_path, "r") as f:
                config = json.load(f)
                return config, candidate
        except _LOOKUP_ERRORS:
            continue

    offline = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
    offline_hint = (
        "\nOffline mode is set (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE) and this model's "
        "config.json is not in the local HuggingFace cache. Fetch it once with network "
        "access, or unset offline mode."
        if offline
        else "\nIf you are behind a firewall/proxy that blocks HuggingFace, download the "
        "config.json into ~/.cache/huggingface while on an open network first."
    )
    raise ValueError(
        f"Could not find config.json for model '{model_id}'. "
        f"Tried: {', '.join(try_extract_base_model(repo_id))}.\n"
        f"The model may not be compatible or may be a GGUF-only format without standard config.\n"
        f"Try specifying the base model directly (e.g., 'Qwen/Qwen1.5-1.8B-Chat' instead of "
        f"'Qwen/Qwen1.5-1.8B-Chat-GGUF')."
        f"{offline_hint}"
    )
