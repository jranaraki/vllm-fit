import json
from typing import Dict, Any

from huggingface_hub import hf_hub_download


def get_model_config(model_id: str) -> Dict[str, Any]:
    config_path = hf_hub_download(
        repo_id=model_id,
        filename="config.json",
        force_download=False,
    )
    with open(config_path, "r") as f:
        return json.load(f)
