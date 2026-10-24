"""Prepare on CPU, explicitly execute on GPU, or resume an immutable pilot."""

import argparse
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

from scripts.setup_common import ROOT, configure_paths, resource_pins
from src.artifacts import (environment, load_records, load_run, now, persistent_path, read,
                           save, save_manifest, verify_record, writer_lock)
from src.data import prepare
from src.prompts import HEADING, reconstruct


def validate_config(config):
    expected = read(ROOT / "config/pilot.json")
    if set(config) != set(expected):
        raise ValueError("Unsupported or missing pilot configuration keys")
    mutable = {"sample_counts", "selection_seed", "batch_size", "minimum_vram_gib"}
    for key in set(expected) - mutable:
        if config[key] != expected[key]:
            raise ValueError(f"Unsupported pilot setting: {key}")
    required = {"question_split": "validation", "device": "cuda:0", "dtype": "bfloat16",
                "attention": "sdpa", "sdpa_backend": "FLASH_ATTENTION", "enable_thinking": False,
                "use_cache": False, "quantization": False, "logits_to_keep": 1,
                "reserved_answer_tokens": 1, "prompt_heading": HEADING,
                "context_budgets": [0, 2048, 4096, 8192, 16384, 32768],
                "distractor_seeds": [11, 22, 33],
                "max_attempts": 3}
    if any(config[key] != value for key, value in required.items()):
        raise ValueError("Scientific settings differ from the supported pilot protocol")
    if (set(config["sample_counts"]) != {"mmlu_pro", "arc_challenge"} or
            any(type(v) is not int or v < 1 for v in config["sample_counts"].values()) or
            config["batch_size"] not in [1, 2, 4] or config["minimum_vram_gib"] <= 0):
        raise ValueError("Invalid sample, batch, or hardware configuration")


def prepare_run(run, config, paths):
    from transformers import AutoTokenizer
    if (run / "manifest.json").exists():
        raise ValueError("Run exists; choose a fresh directory or use --resume")
    validate_config(config)
    resources = read(paths["reports"] / "resources.json")
    pins = resource_pins()
    if any(resources["model"][k] != v for k, v in pins["model"].items()):
        raise ValueError("Model/tokenizer pins differ from resource manifest")
    if [{k: e[k] for k in p} for e, p in zip(resources["datasets"], pins["datasets"], strict=True)] != pins["datasets"]:
        raise ValueError("Dataset pins differ from resource manifest")
    tokenizer = AutoTokenizer.from_pretrained(resources["model"]["snapshot_path"], local_files_only=True)
    questions, selection, streams, plan = prepare(tokenizer, resources, config)
    # Reconstruct every input before sealing; save human-readable examples for review.
    for case in plan:
        prompt = reconstruct(tokenizer, case, questions, streams)
        if case["context_budget"] in [0, 32768] and case["question_key"] == sorted(questions)[0]:
            audit = run / "audit_prompts" / f"{case['evaluation_id']}.txt"
            audit.parent.mkdir(parents=True, exist_ok=True)
            audit.write_text(prompt)
    save_manifest(run, config, plan, questions, streams, selection, resources)
    save(run / "preparation.json", {"status": "prepared; inference not started", "created_utc": now(),
                                    "evaluations": len(plan), "questions": len(questions),
                                    "heading": HEADING, "all_prompts_reconstructed": True})
    print(f"Prepared {len(plan)} evaluations at {run}; no model inference started", flush=True)


def execute(run, max_new_records=None, pause_after=0, benchmark=None):
    from transformers import AutoTokenizer
    from src.inference import Scorer, hardware
    manifest, config, plan = load_run(run, check_environment=True)
    validate_config(config)
    gpu = hardware(config)
    host_path = run / "execution-hardware.json"
    if host_path.exists() and read(host_path) != gpu:
        raise ValueError("Resume rejected: GPU, driver, or CUDA runtime changed")
    save(host_path, gpu)
    if config["batch_size"] > 1:
        if benchmark is None:
            raise ValueError("Batches above one require an explicit accepted benchmark report")
        accepted = read(benchmark)
        if (accepted["plan_identity"] != manifest["identity"] or
                config["batch_size"] not in accepted["accepted_batch_sizes"] or
                accepted["hardware"] != gpu or accepted["environment"] != environment()):
            raise ValueError("Batch benchmark does not certify this run and environment")
    resources = read(run / "resources.json")
    tokenizer = AutoTokenizer.from_pretrained(resources["model"]["snapshot_path"], local_files_only=True)
    questions, streams = read(run / "questions.json"), read(run / "distractors.json")
    records = load_records(run, plan, manifest["identity"])
    pending = [c for c in plan if c["evaluation_id"] not in records]
    if not pending:
        print(f"All {len(plan)} evaluations already saved; no model loaded", flush=True)
        return
    scorer = Scorer(resources["model"]["snapshot_path"], tokenizer, config)
    save(run / "attention-check.json", scorer.attention_check(
        next(c for c in plan if c["context_budget"] == 2048),
        reconstruct(tokenizer, next(c for c in plan if c["context_budget"] == 2048), questions, streams)))
    log = logging.getLogger("pilot")
    log.setLevel(logging.INFO)
    handler = logging.FileHandler(run / "run.log")
    log.addHandler(handler)
    (run / "records").mkdir(exist_ok=True)
    (run / "failures").mkdir(exist_ok=True)
    buckets = defaultdict(list)
    for case in pending:
        buckets[(case["context_budget"], case["input_tokens"])].append(case)
    newly_saved, stopped_early = 0, False
    start = time.perf_counter()
    for bucket in buckets.values():
        for offset in range(0, len(bucket), config["batch_size"]):
            batch = bucket[offset:offset + config["batch_size"]]
            # Count attempts durably, including attempts interrupted before a result.
            attempts = {c["evaluation_id"]: len(list((run / "failures").glob(c["evaluation_id"] + "-*.json")))
                        for c in batch}
            while any(attempts[c["evaluation_id"]] < config["max_attempts"] for c in batch):
                active = [c for c in batch if attempts[c["evaluation_id"]] < config["max_attempts"]]
                for c in active:
                    eid = c["evaluation_id"]
                    attempts[eid] += 1
                    save(run / "failures" / f"{eid}-{attempts[eid]}.json",
                         {"evaluation_id": eid, "attempt": attempts[eid], "started_utc": now(),
                          "status": "started; may have been interrupted"})
                try:
                    prompts = [reconstruct(tokenizer, c, questions, streams) for c in active]
                    results = scorer.score(active, prompts)
                    for case, result in zip(active, results, strict=True):
                        result.update(plan_identity=manifest["identity"], run_id=run.name,
                                      attempt=attempts[case["evaluation_id"]])
                        verify_record(result, case, manifest["identity"])
                        save(run / "records" / f"{case['evaluation_id']}.json", result)
                        save(run / "failures" / f"{case['evaluation_id']}-{result['attempt']}.json",
                             {"evaluation_id": case["evaluation_id"], "attempt": result["attempt"],
                              "status": "success", "completed_utc": now()})
                        records[case["evaluation_id"]] = result
                        newly_saved += 1
                        log.info("saved %s (%s/%s)", case["evaluation_id"], len(records), len(plan))
                    print(f"Saved {len(records)}/{len(plan)}; {active[0]['context_budget']} tokens", flush=True)
                    break
                except Exception as error:
                    for c in active:
                        eid = c["evaluation_id"]
                        save(run / "failures" / f"{eid}-{attempts[eid]}.json",
                             {"evaluation_id": eid, "attempt": attempts[eid], "status": "failed",
                              "completed_utc": now(), "error": f"{type(error).__name__}: {error}"})
                    if isinstance(error, (MemoryError, ValueError)) or "out of memory" in str(error).lower():
                        raise
                    log.exception("Batch failed; retrying unchanged pending inputs")
            if pause_after and newly_saved >= pause_after:
                save(run / "interruption-ready.json", {"saved": newly_saved})
                while True:  # Internal recovery-test hook; parent kills this process.
                    time.sleep(0.5)
            if max_new_records is not None and newly_saved >= max_new_records:
                stopped_early = True
                break
        if stopped_early:
            break
    save(run / "execution-summary.json", {
        "finished_utc": now(), "expected": len(plan), "saved": len(records),
        "missing": [c["evaluation_id"] for c in plan if c["evaluation_id"] not in records],
        "status": "complete" if len(records) == len(plan) else "incomplete",
        "model_load_seconds": scorer.load_seconds, "execution_wall_seconds": time.perf_counter() - start,
        "hardware": gpu})
    if len(records) < len(plan) and not stopped_early:
        raise RuntimeError("Pilot incomplete; inspect attempts and completeness before analysis")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Tokenize and seal inputs; no inference")
    parser.add_argument("--max-new-records", type=int)
    parser.add_argument("--pause-after", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-report", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    paths = configure_paths()
    if args.config and not args.run_dir:
        parser.error("--config requires --run-dir")
    if args.resume and (args.run_dir or args.prepare_only):
        parser.error("--resume cannot be combined with --run-dir or --prepare-only")
    run = persistent_path(args.resume or args.run_dir)
    with writer_lock(run):
        if args.config:
            prepare_run(run, read(args.config), paths)
        if not args.prepare_only:
            execute(run, args.max_new_records, args.pause_after, args.benchmark_report)


if __name__ == "__main__":
    main()
