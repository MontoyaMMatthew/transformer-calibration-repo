"""Report the host; optionally validate pinned resources and tiny FP32 CPU scoring."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

from setup_common import ROOT, configure_paths, now, resource_pins, write_json


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def validate_resources(paths, report, cpu_smoke, offline):
    from datasets import load_dataset, load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    manifest = json.loads((paths["reports"] / "resources.json").read_text())
    pins = resource_pins()
    if manifest["model"]["revision"] != pins["model"]["revision"]:
        raise ValueError("Model manifest differs from tracked pin")
    samples = {}
    checks = []
    for entry in manifest["datasets"]:
        expected = next(p for p in pins["datasets"] if p["key"] == entry["key"])
        if entry["revision"] != expected["revision"] or entry["config"] != expected["config"]:
            raise ValueError("Dataset manifest differs from tracked pins")
        for split, info in entry["splits"].items():
            data = load_from_disk(info["prepared_path"])
            if len(data) != info["rows"]:
                raise ValueError("Prepared dataset row count changed")
            # In offline mode prepare again from local parquet in a distinct cache.
            if offline:
                rebuilt = load_dataset(
                    "parquet", data_files={split: info["parquet_files"]}, split=split,
                    cache_dir=str(paths["HF_DATASETS_CACHE"] / "offline-rebuild"),
                )
                if len(rebuilt) != len(data) or rebuilt[0] != data[0]:
                    raise ValueError("Offline dataset preparation differs")
            rows = [data[i] for i in range(min(3, len(data)))]
            for row in rows:
                if entry["key"] == "mmlu_pro":
                    if not row["question"] or len(row["options"]) < 2:
                        raise ValueError("Invalid MMLU-Pro question/choices")
                    if row["answer"] != chr(65 + row["answer_index"]):
                        raise ValueError("MMLU-Pro gold mapping differs")
                elif entry["key"] == "arc_challenge":
                    if not row["question"] or row["answerKey"] not in row["choices"]["label"]:
                        raise ValueError("Invalid ARC question/choice/gold mapping")
                elif not isinstance(row["text"], str):
                    raise ValueError("Invalid WikiText text field")
            checks.append({"dataset": entry["key"], "revision": entry["revision"],
                           "split": split, "rows": len(data), "sampled_rows": len(rows),
                           "columns": data.column_names, "offline_rebuilt": offline})
            if split == "validation" and entry["key"] != "wikitext":
                samples[entry["key"]] = rows[0]
    report["dataset_checks"] = checks
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"]["snapshot_path"], local_files_only=True, trust_remote_code=False)
    prompts = []
    for key, row in samples.items():
        if key == "mmlu_pro":
            choices = row["options"]
            gold = row["answer_index"]
            question_id = row["question_id"]
        else:
            choices = row["choices"]["text"]
            gold = row["choices"]["label"].index(row["answerKey"])
            question_id = row["id"]
        labels = [chr(65 + i) for i in range(len(choices))]
        question = ("Answer the multiple-choice question using only its choice letter.\n\n" +
                    row["question"] + "\n" +
                    "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices)))
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        assistant = rendered.rsplit("<|im_start|>assistant\n", 1)[-1]
        if not re.fullmatch(r"<think>\s*</think>\s*", assistant):
            raise ValueError("Expected explicit empty thinking block in non-thinking template")
        prompt = rendered + "Answer:"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        label_tokens = []
        for label in labels:
            extended = tokenizer.encode(prompt + " " + label, add_special_tokens=False)
            if extended[:-1] != ids or len(extended) != len(ids) + 1:
                raise ValueError(f"Label {label} is not one token in the exact answer context")
            label_tokens.append(extended[-1])
        if len(set(label_tokens)) != len(labels):
            raise ValueError("Answer tokens are not distinct")
        prompts.append({"dataset": key, "split": "validation", "question_id": question_id,
                        "prompt": prompt, "input_ids": ids, "labels": labels,
                        "label_completions": [" " + label for label in labels],
                        "label_token_ids": label_tokens, "gold_index": gold})
    report["tokenizer_checks"] = [
        {k: v for k, v in p.items() if k != "input_ids"} | {"prompt_tokens": len(p["input_ids"])}
        for p in prompts]
    if not cpu_smoke:
        return
    if len(prompts) != 2:
        raise ValueError("Expected one validation prompt from each question dataset")
    print("Loading Qwen3-4B in FP32 on CPU", flush=True)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model"]["snapshot_path"], local_files_only=True,
        dtype=torch.float32, device_map="cpu", attn_implementation="sdpa",
        trust_remote_code=False)
    model.eval()
    if next(model.parameters()).dtype != torch.float32 or next(model.parameters()).device.type != "cpu":
        raise ValueError("Expected FP32 CPU model")
    report["model_load_seconds"] = time.perf_counter() - started
    report["cpu_choice_scoring"] = []
    for prompt in prompts:
        tokens = torch.tensor([prompt["input_ids"]], device="cpu")
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                           use_cache=False, logits_to_keep=1)
            scores = output.logits[0, -1, prompt["label_token_ids"]].float()
            log_probs = torch.log_softmax(scores, dim=-1)
            probs = log_probs.exp()
            nll = -log_probs[prompt["gold_index"]]
        if not (torch.isfinite(scores).all() and torch.isfinite(log_probs).all()
                and torch.isfinite(nll) and torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)):
            raise ValueError("Choice scoring finite/normalization checks failed")
        report["cpu_choice_scoring"].append({
            "dataset": prompt["dataset"], "split": prompt["split"],
            "question_id": prompt["question_id"], "prompt_tokens": tokens.shape[1],
            "labels": prompt["labels"], "label_token_ids": prompt["label_token_ids"],
            "label_completions": prompt["label_completions"],
            "gold_index": prompt["gold_index"], "choice_logits": scores.tolist(),
            "log_probabilities": log_probs.tolist(), "probabilities": probs.tolist(),
            "probability_sum": probs.sum().item(), "nll_nats": nll.item(),
            "prediction": prompt["labels"][scores.argmax().item()],
            "forward_seconds": time.perf_counter() - started,
        })
        print(f"Scored {prompt['dataset']}/{prompt['question_id']}", flush=True)
    report["inference_settings"] = {
        "device": "cpu", "dtype": "float32", "batch_size": 1, "enable_thinking": False,
        "evaluation_mode": True, "gradient_tracking": False, "attention": "sdpa",
        "use_cache": False, "logits_to_keep": 1, "context_extension": False,
        "quantization": False, "sampling": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    paths = configure_paths()
    output = Path(args.output) if args.output else paths["reports"] / (
        "cpu-smoke-offline.json" if args.offline and args.cpu_smoke else
        "cpu-smoke.json" if args.cpu_smoke else "environment.json")
    report = {
        "created_utc": now(), "purpose": "development validation; not study results",
        "python": {"executable": sys.executable, "version": sys.version,
                   "isolated_venv": sys.prefix != sys.base_prefix},
        "git_revision": command("git", "-C", str(ROOT), "rev-parse", "HEAD")["stdout"],
        "git_status": command("git", "-C", str(ROOT), "status", "--short")["stdout"],
        "os": platform.platform(), "architecture": platform.machine(), "cpu_count": os.cpu_count(),
        "meminfo": Path("/proc/meminfo").read_text().splitlines()[:3],
        "paths": {k: str(v) for k, v in paths.items()},
        "wheel_index": "https://download.pytorch.org/whl/cu118",
        "offline": args.offline,
        "limitations": ["CUDA/A100 execution unverified", "32K inference unverified",
                        "Independent Mac backup requires transfer and checksum verification"],
        "status": "running",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([* (ROOT / "scripts").glob("*.py"), ROOT / "config/resources.json",
                             ROOT / "requirements.in"])
        },
    }
    write_json(output, report)
    try:
        packages = {"torch": "torch", "transformers": "transformers", "datasets": "datasets",
                    "huggingface_hub": "huggingface-hub", "safetensors": "safetensors",
                    "accelerate": "accelerate", "numpy": "numpy", "pandas": "pandas",
                    "scipy": "scipy", "matplotlib": "matplotlib", "yaml": "PyYAML", "pytest": "pytest"}
        report["packages"] = {}
        for module, distribution in packages.items():
            importlib.import_module(module)
            report["packages"][distribution] = importlib.metadata.version(distribution)
        report["pip_check"] = command(sys.executable, "-m", "pip", "check")
        if report["pip_check"]["exit_code"]:
            raise RuntimeError("pip check failed")
        report["storage"] = {name: command("findmnt", "-T", str(path), "-o", "SOURCE,FSTYPE,TARGET")
                             for name, path in {"repository": ROOT, "venv": Path(sys.prefix),
                                                "cache": paths["HF_HOME"], "artifacts": paths["reports"]}.items()}
        report["disk_free"] = command("df", "-hT", str(ROOT))
        report["nvidia_smi"] = command("nvidia-smi")
        import torch
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        report["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads()}
        tensor = torch.ones(2, device="cpu") + 1
        if not torch.equal(tensor, torch.tensor([2.0, 2.0])):
            raise ValueError("CPU tensor check failed")
        report["cpu_tensor_check"] = "passed"
        from transformers import Qwen3ForCausalLM
        report["qwen3_support"] = Qwen3ForCausalLM.__name__
        if args.resources or args.cpu_smoke:
            validate_resources(paths, report, args.cpu_smoke, args.offline)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = now()
        write_json(output, report)
        print(f"Report: {output} ({report['status']})", flush=True)


if __name__ == "__main__":
    main()
