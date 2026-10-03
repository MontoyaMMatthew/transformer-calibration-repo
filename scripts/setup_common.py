"""Shared paths and serialization for the independent Step 001 tools."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_paths():
    defaults = {
        "HF_HOME": ROOT / "cache/huggingface",
        "HF_HUB_CACHE": ROOT / "cache/huggingface/hub",
        "HF_DATASETS_CACHE": ROOT / "cache/huggingface/datasets",
        "TRANSFORMER_CALIBRATION_ARTIFACTS": ROOT / "artifacts",
        "MPLCONFIGDIR": ROOT / "cache/matplotlib",
        "PIP_CACHE_DIR": ROOT / "cache/pip",
    }
    paths = {}
    for key, default in defaults.items():
        path = Path(os.environ.get(key, str(default)))
        if not path.is_absolute():
            raise ValueError(f"{key} must be an absolute path")
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
        paths[key] = path
    paths["reports"] = paths["TRANSFORMER_CALIBRATION_ARTIFACTS"] / "environment"
    paths["reports"].mkdir(parents=True, exist_ok=True)
    return paths


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resource_pins():
    pins = json.loads((ROOT / "config/resources.json").read_text())
    for entry in [pins["model"], *pins["datasets"]]:
        if not re.fullmatch(r"[0-9a-f]{40}", entry["revision"]):
            raise ValueError(f"Immutable revision required: {entry['repo_id']}")
    if pins["model"]["tokenizer_revision"] != pins["model"]["revision"]:
        raise ValueError("These setup tools require model and tokenizer from the same snapshot")
    return pins
