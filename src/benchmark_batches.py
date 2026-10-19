"""Pilot batch screening in isolated processes; never an automatic OOM search."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.setup_common import configure_paths
from src.artifacts import environment, load_run, now, persistent_path, read, save
from src.prompts import reconstruct

PROBABILITY_ATOL = 0.005


def worker(run, output, size):
    import torch
    from transformers import AutoTokenizer
    from src.inference import Scorer
    manifest, config, plan = load_run(run, check_environment=True)
    resources = read(run / "resources.json")
    tokenizer = AutoTokenizer.from_pretrained(resources["model"]["snapshot_path"], local_files_only=True)
    scorer = Scorer(resources["model"]["snapshot_path"], tokenizer, config)
    questions, streams = read(run / "questions.json"), read(run / "distractors.json")
    # Identical input set for all candidates; group only equal-length inputs (no padding).
    reference = read(output / "batch-1.json") if size > 1 else None
    result = {"batch_size": size, "created_utc": now(), "plan_identity": manifest["identity"],
              "hardware": scorer.hardware, "environment": environment(),
              "probability_absolute_tolerance": PROBABILITY_ATOL, "prediction_changes": 0,
              "maximum_probability_difference": 0.0, "batches": [], "reference_scores": {},
              "resident_device_used_bytes": scorer.hardware["total_memory_bytes"] - torch.cuda.mem_get_info()[0],
              "status": "running"}
    path = output / f"batch-{size}.json"
    save(path, result)
    first = next(c for c in plan if c["context_budget"] == 2048)
    result["attention_check"] = scorer.attention_check(first, reconstruct(tokenizer, first, questions, streams))
    # Warm-up excluded from candidate timing.
    scorer.score([first], [reconstruct(tokenizer, first, questions, streams)])
    for budget in [32768, 0, 2048, 4096, 8192, 16384]:
        for dataset in config["sample_counts"]:
            cases = [c for c in plan if c["context_budget"] == budget and c["dataset"] == dataset
                     and c["distractor_seed"] in [None, config["distractor_seeds"][0]]][:4]
            grouped = {}
            for c in cases:
                grouped.setdefault(c["input_tokens"], []).append(c)
            for bucket in grouped.values():
                # Repeat inputs only inside this throughput check; they are never scientific records.
                repeats = 3 if budget == 32768 else 1
                expanded = bucket * size
                for repeat in range(repeats):
                    for start in range(0, len(expanded), size):
                        batch = expanded[start:start + size]
                        torch.cuda.empty_cache()
                        rows = scorer.score(batch, [reconstruct(tokenizer, c, questions, streams) for c in batch])
                        first_row = rows[0]
                        result["batches"].append({"dataset": dataset, "context_budget": budget,
                                                   "repeat": repeat, "actual_batch_size": len(batch),
                                                   "seconds": first_row["batch_forward_and_score_seconds"],
                                                   "questions_per_second": len(batch) / first_row["batch_forward_and_score_seconds"],
                                                   **{k: v for k, v in first_row.items() if "memory" in k or "headroom" in k or "peak" in k}})
                        for r in rows:
                            eid = r["evaluation_id"]
                            scores = {"probabilities": r["probabilities"], "prediction": r["prediction"]}
                            # Includes single-batch repeat stability.
                            baseline = (reference or result)["reference_scores"].get(eid, scores)
                            diff = float(np.max(np.abs(np.array(scores["probabilities"]) - baseline["probabilities"])))
                            result["maximum_probability_difference"] = max(result["maximum_probability_difference"], diff)
                            result["prediction_changes"] += int(scores["prediction"] != baseline["prediction"])
                            result["reference_scores"][eid] = scores
                        save(path, result)
                        if first_row["minimum_headroom_fraction"] < 0.25:
                            result["status"] = "rejected: insufficient device memory headroom"
                            save(path, result)
                            return
    result["status"] = ("passed" if result["prediction_changes"] == 0 and
                        result["maximum_probability_difference"] <= PROBABILITY_ATOL else
                        "rejected: investigate numerical or prediction changes")
    save(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-size", type=int, choices=[1, 2, 4], help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_paths()
    os.environ["HF_HUB_OFFLINE"] = "1"
    run, output = args.run.resolve(), persistent_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.worker_size:
        worker(run, output, args.worker_size)
        return
    if (output / "screening.json").exists() or list(output.glob("batch-*.json")):
        raise ValueError("Choose a fresh benchmark directory")
    manifest, _, _ = load_run(run, check_environment=True)
    accepted, estimates = [], []
    previous = None
    for size in [1, 2, 4]:
        if previous:
            peak = max(r["conservative_peak_device_used_bytes"] for r in previous["batches"])
            resident = previous["resident_device_used_bytes"]
            # Double measured activation/reservation growth plus 2 GiB uncertainty allowance.
            estimate = resident + max(peak - resident, 0) * (size / previous["batch_size"]) + 2 * 1024**3
            estimates.append({"batch_size": size, "estimated_peak_bytes": estimate})
            if estimate > previous["hardware"]["total_memory_bytes"] * 0.75:
                break
        with (output / f"batch-{size}.log").open("w") as log:
            completed = subprocess.run([sys.executable, "-m", "src.benchmark_batches", "--run", str(run),
                                        "--output", str(output), "--worker-size", str(size)],
                                       stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            save(output / f"batch-{size}-failure.json", {"returncode": completed.returncode,
                                                        "action": "stop screening; inspect durable candidate log"})
            break
        candidate = read(output / f"batch-{size}.json")
        if candidate["status"] != "passed":
            break
        accepted.append(size)
        previous = candidate
    save(output / "screening.json", {
        "plan_identity": manifest["identity"], "created_utc": now(),
        "environment": environment(), "hardware": (read(output / "batch-1.json")["hardware"]
                                                     if (output / "batch-1.json").exists() else None),
        "accepted_batch_sizes": accepted, "next_candidate_estimates": estimates,
        "scientific_sweep_batch_size": 1,
        "selection_policy": "Keep pilot at one; inspect measured throughput and stability before freezing main execution",
        "probability_absolute_tolerance": PROBABILITY_ATOL,
        "padding_policy": "Exact-length buckets; no padding; final token is each answer position"})
    if 1 not in accepted:
        raise RuntimeError("Batch one failed screening; inspect reports before running the sweep")
    print(f"Screened batch sizes {accepted}; pilot configuration remains at batch one", flush=True)


if __name__ == "__main__":
    main()
