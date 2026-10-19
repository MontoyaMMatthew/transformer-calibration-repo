"""Atomic records, immutable input checks, and code/environment provenance."""

import fcntl
import importlib.metadata
import json
import os
import platform
import subprocess
from contextlib import contextmanager
from pathlib import Path

from scripts.setup_common import ROOT, configure_paths, now, resource_pins
from scripts.validate_gpu import digest, file_hash, save


def read(path):
    return json.loads(Path(path).read_text())


def persistent_path(path):
    path = Path(path).resolve()
    root = os.environ.get("TRANSFORMER_CALIBRATION_PERSISTENT_ROOT", "/workspace")
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Output must be under persistent root {root}: {path}")
    return path


@contextmanager
def writer_lock(run):
    run = persistent_path(run)
    run.mkdir(parents=True, exist_ok=True)
    with (run / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def code_files():
    return sorted([*(ROOT / "src").glob("*.py"), *(ROOT / "scripts").glob("*.py"),
                   *(ROOT / "config").glob("*.json"), *(ROOT.glob("requirements.*")),
                   ROOT / "scripts/runpod", ROOT / "scripts/setup_runpod.sh"])


def environment():
    return {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformers", "datasets", "tokenizers", "numpy",
                      "huggingface_hub", "safetensors", "accelerate"]},
        "sources": {str(p.relative_to(ROOT)): file_hash(p) for p in code_files()},
        "resources": resource_pins(),
    }


def save_manifest(run, config, plan, questions, streams, selection, resources):
    """Inputs become immutable before any model inference can start."""
    inputs = {"config.json": config, "plan.json": plan, "questions.json": questions,
              "distractors.json": streams, "selection.json": selection,
              "resources.json": resources}
    for name, value in inputs.items():
        save(run / name, value)
    for p in code_files():
        target = run / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    git = lambda *args: subprocess.run(["git", "-C", str(ROOT), *args],
                                      capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        "schema_version": 1, "created_utc": now(), "purpose": config["purpose"],
        "run_id": run.name, "run_directory": str(run), "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"), "environment": environment(),
        "input_sha256": {name: file_hash(run / name) for name in inputs},
        "paths": {k: str(v) for k, v in configure_paths().items()},
        "inference_settings": config,
    }
    manifest["identity"] = digest(manifest)
    save(run / "manifest.json", manifest)
    return manifest


def load_run(run, check_environment=False):
    manifest = read(run / "manifest.json")
    if digest({k: v for k, v in manifest.items() if k != "identity"}) != manifest["identity"]:
        raise ValueError("Run manifest identity changed")
    for name, expected in manifest["input_sha256"].items():
        if file_hash(run / name) != expected:
            raise ValueError(f"Saved input changed: {name}")
    if check_environment and environment() != manifest["environment"]:
        raise ValueError("Resume rejected: scientific code, packages, Python, or resource pins changed")
    return manifest, read(run / "config.json"), read(run / "plan.json")


def verify_record(record, case, identity):
    import numpy as np
    for key, value in case.items():
        if record.get(key) != value:
            raise ValueError(f"Record differs from plan: {key}")
    if record["plan_identity"] != identity or record["status"] != "success":
        raise ValueError("Record identity or status differs")
    k = len(case["labels"])
    z, lp, p = [np.asarray(record[n], dtype=float) for n in
                ["choice_logits", "log_probabilities", "probabilities"]]
    if any(a.shape != (k,) or not np.isfinite(a).all() for a in [z, lp, p]):
        raise ValueError("Invalid choice score shape or nonfinite score")
    expected_lp = z - np.logaddexp.reduce(z)
    if (not np.allclose(lp, expected_lp, atol=3e-5, rtol=1e-5) or
            not np.allclose(p, np.exp(lp), atol=1e-7, rtol=1e-5) or
            abs(p.sum() - 1) > 1e-6 or np.any(p < 0) or np.any(p > 1) or
            record["prediction"] != case["labels"][int(z.argmax())]):
        raise ValueError("Saved scores, probabilities, or prediction disagree")


def load_records(run, plan, identity):
    cases = {c["evaluation_id"]: c for c in plan}
    records = {}
    for p in sorted((run / "records").glob("*.json")):
        if p.stem not in cases:
            raise ValueError(f"Unexpected evaluation record {p}")
        record = read(p)
        verify_record(record, cases[p.stem], identity)
        records[p.stem] = record
    return records
