"""Offline A100/BF16/context validation, with durable records and a recovery check."""

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from setup_common import ROOT, configure_paths, now, resource_pins


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def save(path, value):
    """Replace an entire record atomically, flushing file and directory to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity():
    files = [ROOT / "scripts/validate_gpu.py", ROOT / "scripts/verify_environment.py",
             ROOT / "scripts/setup_common.py", ROOT / "config/resources.json",
             ROOT / "config/gpu_validation.json", ROOT / "requirements.cpu-validated.txt"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def render(tokenizer, row, key, background, verify_labels=True):
    if key == "mmlu_pro":
        choices, gold, qid = row["options"], row["answer_index"], row["question_id"]
    else:
        choices = row["choices"]["text"]
        gold, qid = row["choices"]["label"].index(row["answerKey"]), row["id"]
    labels = [chr(65 + i) for i in range(len(choices))]
    content = ("Answer the multiple-choice question using only its choice letter.\n\n"
               "Irrelevant background:\n" + background + "\n\nQuestion:\n" +
               row["question"] + "\n" +
               "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices)))
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) + "Answer:"
    assistant = prompt.rsplit("<|im_start|>assistant\n", 1)[-1]
    if not re.fullmatch(r"<think>\s*</think>\s*Answer:", assistant):
        raise ValueError("Expected an empty thinking block and fixed answer prefix")
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    label_ids = []
    for label in labels if verify_labels else []:
        extended = tokenizer.encode(prompt + " " + label, add_special_tokens=False)
        if len(extended) != len(ids) + 1 or extended[:-1] != ids:
            raise ValueError(f"Choice {label} is not one appended token")
        label_ids.append(extended[-1])
    if (verify_labels and len(set(label_ids)) != len(labels)) or not 0 <= gold < len(labels):
        raise ValueError("Invalid choice tokens or gold mapping")
    return {"dataset": key, "question_id": qid, "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_tokens": len(ids), "labels": labels, "label_token_ids": label_ids,
            "label_completions": [" " + label for label in labels], "gold_index": gold}


def prepare_plan(paths, config, manifest, tokenizer):
    from datasets import load_from_disk
    wiki_entry = next(e for e in manifest["datasets"] if e["key"] == "wikitext")
    wiki = load_from_disk(wiki_entry["splits"]["train"]["prepared_path"])
    cases, streams = [], {}
    for key in config["question_datasets"]:
        entry = next(e for e in manifest["datasets"] if e["key"] == key)
        row = load_from_disk(entry["splits"][config["question_split"]]["prepared_path"])[config["question_row"]]
        qid = row["question_id"] if key == "mmlu_pro" else row["id"]
        seed = int(digest([key, qid, config["distractor_seed"]]), 16)
        rng = random.Random(seed)
        selected, chunks, count = [], [], 0
        while count < 40000:
            index = rng.randrange(len(wiki))
            text = wiki[index]["text"]
            if not text.strip():
                continue
            selected.append(index)
            chunks.append(text)
            count += len(tokenizer.encode(text, add_special_tokens=False))
        stream = "\n".join(chunks)
        offsets = tokenizer(stream, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        stream_id = digest([key, qid, config["distractor_seed"]])[:16]
        streams[stream_id] = {
            "text": stream, "text_sha256": hashlib.sha256(stream.encode()).hexdigest(),
            "source": "Salesforce/wikitext", "revision": wiki_entry["revision"],
            "config": wiki_entry["config"], "split": "train", "row_indices": selected,
            "seed": config["distractor_seed"],
        }
        baseline = render(tokenizer, row, key, "")
        if baseline["input_tokens"] + 1 > min(b for b in config["context_budgets"] if b):
            raise ValueError("Fixed question exceeds the smallest nonzero budget")
        for budget in config["context_budgets"]:
            if budget:
                limit = budget - config["reserved_answer_tokens"]
                lo, hi = 0, len(offsets)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    text = stream[:offsets[mid - 1][1]] if mid else ""
                    measured = render(tokenizer, row, key, text, verify_labels=False)
                    if measured["input_tokens"] <= limit:
                        lo = mid
                    else:
                        hi = mid - 1
                text = stream[:offsets[lo - 1][1]] if lo else ""
                case = render(tokenizer, row, key, text)
                if not 0 <= limit - case["input_tokens"] <= 4:
                    raise ValueError("Assembled prompt did not closely fill its token budget")
            else:
                text, case = "", dict(baseline)
            case.update(split=config["question_split"], context_budget=budget,
                        reserved_answer_tokens=config["reserved_answer_tokens"],
                        total_context_tokens=case["input_tokens"] + config["reserved_answer_tokens"],
                        target_shortfall=(budget - case["input_tokens"] - 1) if budget else None,
                        distractor_tokens=len(tokenizer.encode(text, add_special_tokens=False)),
                        distractor_characters=len(text),
                        distractor_stream_id=stream_id if budget else None,
                        distractor_seed=config["distractor_seed"] if budget else None)
            case["evaluation_id"] = digest([key, qid, budget, case["distractor_seed"]])[:24]
            cases.append(case)
            print(f"Prepared {key}/{budget or '0K'}: {case['input_tokens']} input tokens", flush=True)
    return cases, streams


def verify_record(record, case, identity):
    if (record["evaluation_id"] != case["evaluation_id"] or
            record["plan_identity"] != identity or record["prompt_sha256"] != case["prompt_sha256"]):
        raise ValueError("Saved inference record differs from immutable plan")
    probabilities = record["probabilities"]
    if (len(probabilities) != len(case["labels"]) or
            not all(0 <= p <= 1 and math.isfinite(p) for p in probabilities) or
            abs(sum(probabilities) - 1) > 1e-6 or not math.isfinite(record["nll_nats"]) or
            not all(math.isfinite(x) for x in record["choice_logits"] + record["log_probabilities"])):
        raise ValueError("Invalid saved choice probabilities, scores, or NLL")


def worker(args, paths):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verify_environment import command, validate_resources

    run = args.run_dir
    report_path = run / "report.json"
    report = {"started_utc": now(), "purpose": "development GPU validation; not study results",
              "status": "running", "offline": True, "run_directory": str(run)}
    save(report_path, report)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use a shell with direct GPU access")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Validation requires one GPU")
        properties = torch.cuda.get_device_properties(0)
        if "A100" not in properties.name or properties.total_memory < 79 * 1024**3:
            raise RuntimeError("Expected the planned A100 80 GB")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unavailable")
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        software = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        pip_check = command(sys.executable, "-m", "pip", "check")
        if pip_check["exit_code"]:
            raise RuntimeError("Package consistency check failed")
        report.update(
            python=sys.version, os=platform.platform(), cpu_count=os.cpu_count(),
            memory=Path("/proc/meminfo").read_text().splitlines()[:3], packages=software,
            pip_check=pip_check,
            nvidia_smi=command("nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                               "--format=csv,noheader,nounits"),
            torch={"version": torch.__version__, "runtime_cuda": torch.version.cuda,
                   "cuda_available": True, "bf16_supported": True},
            gpu={"name": properties.name, "total_memory_bytes": properties.total_memory,
                 "compute_capability": list(torch.cuda.get_device_capability(0))},
            paths={k: str(v) for k, v in paths.items()},
            storage={k: command("findmnt", "-T", str(p), "-o", "SOURCE,FSTYPE,TARGET")
                     for k, p in {"repository": ROOT, "venv": Path(sys.prefix),
                                  "cache": paths["HF_HOME"], "run": run}.items()},
        )
        config = json.loads((ROOT / "config/gpu_validation.json").read_text())
        if (config["device"] != "cuda:0" or config["dtype"] != "bfloat16" or
                config["enable_thinking"] or config["quantization"] or config["use_cache"] or
                config["batch_size"] != 1 or config["logits_to_keep"] != 1 or
                config["sdpa_backend"] != "FLASH_ATTENTION" or config["attention"] != "sdpa" or
                config["reserved_answer_tokens"] != 1 or config["question_split"] != "validation" or
                config["context_budgets"] != [0, 2048, 4096, 8192, 16384, 32768]):
            raise ValueError("GPU validation settings differ from the required scope")
        pins = resource_pins()
        resource_checks = {}
        validate_resources(paths, resource_checks, cpu_smoke=False, offline=True)
        report["resource_checks"] = resource_checks
        manifest = json.loads((paths["reports"] / "resources.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(manifest["model"]["snapshot_path"],
                                                  local_files_only=True, trust_remote_code=False)
        identity_inputs = {
            "config": config, "resources": pins, "sources": source_identity(),
            "python": platform.python_version(), "software": software,
            "gpu_name": properties.name, "compute_capability": report["gpu"]["compute_capability"],
            "driver": report["nvidia_smi"]["stdout"].split(",")[-2].strip(),
        }
        plan_path = run / "plan.json"
        if args.resume:
            plan = json.loads(plan_path.read_text())
            if plan["identity_inputs"] != identity_inputs:
                raise ValueError("Resume rejected: code, software, hardware, or settings changed")
        else:
            if plan_path.exists():
                raise ValueError("Run already exists; use --resume or choose a new directory")
            cases, streams = prepare_plan(paths, config, manifest, tokenizer)
            plan = {"identity_inputs": identity_inputs, "cases": cases,
                    "distractor_streams_sha256": digest(streams)}
            plan["identity"] = digest(plan)
            save(run / "distractor-streams.json", streams)
            save(plan_path, plan)
        check_plan = {k: v for k, v in plan.items() if k != "identity"}
        if digest(check_plan) != plan["identity"]:
            raise ValueError("Saved plan hash differs")
        if digest(json.loads((run / "distractor-streams.json").read_text())) != plan["distractor_streams_sha256"]:
            raise ValueError("Saved distractor source identity differs")
        report.update(plan_identity=plan["identity"], config=config,
                      git_revision=command("git", "-C", str(ROOT), "rev-parse", "HEAD")["stdout"],
                      source_sha256=identity_inputs["sources"])
        records = run / "records"
        records.mkdir(exist_ok=True)
        expected = {c["evaluation_id"] for c in plan["cases"]}
        if any(p.stem not in expected for p in records.glob("*.json")):
            raise ValueError("Unexpected inference records in run directory")
        completed = []
        for case in plan["cases"]:
            path = records / (case["evaluation_id"] + ".json")
            if path.exists():
                record = json.loads(path.read_text())
                verify_record(record, case, plan["identity"])
                completed.append(record)
        report["resumed_completed_records"] = len(completed)
        save(report_path, report)
        print(f"Validated cached inputs; skipping {len(completed)} completed records", flush=True)
        pending = [c for c in plan["cases"] if not (records / (c["evaluation_id"] + ".json")).exists()]
        if pending:
            print("Loading cached Qwen3-4B in BF16 on CUDA", flush=True)
            started = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                manifest["model"]["snapshot_path"], dtype=torch.bfloat16, device_map="cuda:0",
                attn_implementation="sdpa", local_files_only=True, trust_remote_code=False)
            model.eval()
            if any(p.device.type != "cuda" or p.dtype != torch.bfloat16 for p in model.parameters()):
                raise ValueError("Expected all model weights on CUDA in BF16")
            if model.config.rope_scaling is not None or model.config._attn_implementation != "sdpa":
                raise ValueError("Expected native context and SDPA")
            torch.cuda.synchronize()
            report["model_load_seconds"] = time.perf_counter() - started

            def forward(case):
                encoded = tokenizer.encode(case["prompt"], add_special_tokens=False)
                if (len(encoded) != case["input_tokens"] or
                        hashlib.sha256(case["prompt"].encode()).hexdigest() != case["prompt_sha256"]):
                    raise ValueError("Prompt differs from saved evaluation plan")
                for label, token in zip(case["labels"], case["label_token_ids"], strict=True):
                    extended = tokenizer.encode(case["prompt"] + " " + label, add_special_tokens=False)
                    if extended != encoded + [token]:
                        raise ValueError("Exact answer token context changed")
                if case["total_context_tokens"] > (case["context_budget"] or 32768):
                    raise ValueError("Context budget exceeded")
                tokens = torch.tensor([encoded], device="cuda:0")
                mask = torch.ones_like(tokens)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    output = model(input_ids=tokens, attention_mask=mask,
                                   use_cache=False, logits_to_keep=1)
                    if output.logits.shape[1] != 1:
                        raise ValueError("Expected answer-position-only vocabulary logits")
                    scores = output.logits[0, -1, case["label_token_ids"]].float()
                    log_probs = torch.log_softmax(scores, dim=-1)
                    probabilities = log_probs.exp()
                torch.cuda.synchronize()
                record = {k: v for k, v in case.items() if k != "prompt"}
                record.update(
                    plan_identity=plan["identity"], completed_utc=now(),
                    forward_and_score_seconds=time.perf_counter() - start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    choice_logits=scores.tolist(), log_probabilities=log_probs.tolist(),
                    probabilities=probabilities.tolist(), nll_nats=-log_probs[case["gold_index"]].item(),
                    prediction=case["labels"][scores.argmax().item()],
                )
                verify_record(record, case, plan["identity"])
                return record

            forward(plan["cases"][0])  # Untimed warm-up; no durable result or study observation.
            if not (run / "attention-check.json").exists():
                profile_case = next(c for c in plan["cases"] if c["context_budget"] == 2048)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    forward(profile_case)
                operators = {e.key: e.count for e in profile.key_averages() if "scaled_dot_product" in e.key}
                if operators.get("aten::_scaled_dot_product_flash_attention", 0) != model.config.num_hidden_layers:
                    raise RuntimeError(f"Expected flash attention in every layer; observed {operators}")
                save(run / "attention-check.json", {
                    "status": "passed", "operators": operators,
                    "model_layers": model.config.num_hidden_layers,
                    "backend": "PyTorch SDPA FLASH_ATTENTION", "separate_flash_attn_package": False,
                    "profiled_input_tokens": profile_case["input_tokens"],
                    "profile_excluded_from_primary_timings": True,
                })
            newly_saved = 0
            for case in pending:
                torch.cuda.empty_cache()
                record = forward(case)
                save(records / (case["evaluation_id"] + ".json"), record)
                completed.append(record)
                newly_saved += 1
                print(f"Passed {case['dataset']}/{case['context_budget'] or '0K'}: "
                      f"{record['forward_and_score_seconds']:.3f}s, "
                      f"{record['peak_allocated_bytes'] / 1024**3:.2f} GiB peak allocated", flush=True)
                if args.pause_after and newly_saved == args.pause_after:
                    save(run / "interruption-ready.json", {"durable_records": len(completed), "utc": now()})
                    while True:
                        time.sleep(1)
        report["attention_check"] = json.loads((run / "attention-check.json").read_text())
        report["results"] = sorted(completed, key=lambda r: (r["dataset"], r["context_budget"]))
        report["completed_records"] = len(completed)
        report["planned_records"] = len(plan["cases"])
        if len(completed) != len(plan["cases"]):
            raise ValueError("Incomplete validation coverage")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = now()
        save(report_path, report)
        print(f"GPU report: {report_path} ({report['status']})", flush=True)


def recovery_check(args):
    run = args.run_dir
    if (run / "plan.json").exists():
        raise ValueError("Recovery check requires a new run directory")
    run.mkdir(parents=True, exist_ok=True)
    child = None
    with (run / "interrupted-process.log").open("w") as log:
        try:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                      "--run-dir", str(run), "--pause-after", "2"],
                                     stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 240
            while not (run / "interruption-ready.json").exists():
                if child.poll() is not None:
                    raise RuntimeError(f"Initial GPU process failed; inspect {log.name}")
                if time.monotonic() > deadline:
                    raise TimeoutError("GPU recovery checkpoint timed out")
                time.sleep(0.5)
            before = {p.name: file_hash(p) for p in (run / "records").glob("*.json")}
            if len(before) != 2:
                raise ValueError("Expected precisely two durable records before interruption")
            child.kill()
            returncode = child.wait(timeout=30)
        finally:
            if child is not None and child.poll() is None:
                child.kill()
                child.wait(timeout=30)
    print("Terminated initial process with SIGKILL after two durable records; resuming", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--run-dir", str(run), "--resume"], check=True)
    after = {p.name: file_hash(p) for p in (run / "records").glob("*.json")}
    plan = json.loads((run / "plan.json").read_text())
    if len(after) != len(plan["cases"]) or not all(after.get(k) == v for k, v in before.items()):
        raise ValueError("Recovery changed completed records or left incomplete coverage")
    save(run / "recovery-check.json", {
        "status": "passed", "finished_utc": now(), "interruption_signal": "SIGKILL",
        "interrupted_process_returncode": returncode, "durable_records_before": len(before),
        "completed_records_after": len(after), "prior_records_byte_identical": True,
        "duplicate_evaluations": 0,
    })
    print("GPU recovery check passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/gpu_validation/a100-initial")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--recovery-check", action="store_true")
    parser.add_argument("--pause-after", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.recovery_check:
        recovery_check(args)
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    paths = configure_paths()
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    with (run / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker(args, paths)


if __name__ == "__main__":
    main()
