"""Regenerate tables, paired bootstrap intervals and figures without model inference."""

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.setup_common import ROOT, configure_paths
from src.artifacts import (file_hash, load_records, load_run, now, persistent_path,
                           read, save)
from src.metrics import METRICS, aggregate, calibration_bins, example, seed_mean


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def completeness(plan, records, questions):
    required = defaultdict(set)
    for c in plan:
        required[c["question_key"]].add(c["evaluation_id"])
    complete = sorted(q for q, ids in required.items() if ids <= records.keys())
    return {"expected_records": len(plan), "saved_records": len(records),
            "missing_evaluation_ids": sorted({c["evaluation_id"] for c in plan} - records.keys()),
            "planned_questions": len(required), "paired_complete_questions": len(complete),
            "complete_question_ids": complete,
            "excluded_questions": [{"question_key": q, "dataset": questions[q]["dataset"],
                                    "subject": questions[q]["subject"],
                                    "missing": sorted(ids - records.keys())}
                                   for q, ids in required.items() if q not in complete],
            "status": "complete" if len(records) == len(plan) else "incomplete",
            "policy": "Primary analysis uses only questions complete at every planned length and seed"}


def analyze(run, output, settings):
    manifest, config, plan = load_run(run)
    records = load_records(run, plan, manifest["identity"])
    questions = read(run / "questions.json")
    output = persistent_path(output)
    if output.exists():
        raise ValueError("Choose a fresh analysis directory to preserve earlier outputs")
    output.mkdir(parents=True)
    report = completeness(plan, records, questions)
    save(output / "completeness_report.json", report)
    save(output / "analysis_config.json", settings)
    save(output / "source_runs.json", {"run": str(run), "identity": manifest["identity"],
                                       "manifest_sha256": file_hash(run / "manifest.json"),
                                       "record_sha256": {eid: file_hash(run / "records" / f"{eid}.json")
                                                         for eid in records},
                                       "created_utc": now(), "purpose": config["purpose"]})
    all_seed, summary, deltas, diagnostics, bins_rows, timing = [], [], [], [], [], []
    per_example = [{"evaluation_id": r["evaluation_id"], "question_key": r["question_key"],
                    "dataset": r["dataset"], "context_budget": r["context_budget"],
                    "distractor_seed": r["distractor_seed"], **example(r)} for r in records.values()]
    write_csv(output / "per_example_metrics.csv", per_example)
    for dataset in config["sample_counts"]:
        ids = [q for q in report["complete_question_ids"] if questions[q]["dataset"] == dataset]
        if not ids:
            continue
        selected = [r for r in records.values() if r["question_key"] in ids]
        grouped_records = defaultdict(list)
        groups = defaultdict(dict)
        for r in selected:
            key = (r["context_budget"], r["distractor_seed"])
            grouped_records[key].append(r)
            groups[key][r["question_key"]] = example(r)
        for (budget, seed), rows in grouped_records.items():
            for bins in settings["ece_bins"]:
                all_seed.append({"dataset": dataset, "context_budget": budget, "seed": seed,
                                 "ece_bins": bins, "unique_questions": len(rows), **aggregate(rows, bins)})
                bins_rows.extend({"dataset": dataset, "context_budget": budget, "seed": seed,
                                  "ece_bins": bins, **b} for b in calibration_bins([example(r) for r in rows], bins))
            errors = [r for r in rows if r["prediction"] != r["labels"][r["gold_index"]]]
            baseline = {r["question_key"]: r for r in grouped_records[(0, None)]}
            transitions = {"correct_to_incorrect": 0, "incorrect_to_correct": 0,
                           "incorrect_to_different_incorrect": 0, "prediction_changes": 0}
            for r in rows:
                b = baseline[r["question_key"]]
                b_ok = b["prediction"] == b["labels"][b["gold_index"]]
                r_ok = r["prediction"] == r["labels"][r["gold_index"]]
                changed = r["prediction"] != b["prediction"]
                transitions["prediction_changes"] += int(changed)
                transitions["correct_to_incorrect"] += int(b_ok and not r_ok)
                transitions["incorrect_to_correct"] += int(not b_ok and r_ok)
                transitions["incorrect_to_different_incorrect"] += int(not b_ok and not r_ok and changed)
            diagnostics.append({"dataset": dataset, "context_budget": budget, "seed": seed,
                                "questions": len(rows), "incorrect_count": len(errors),
                                "confidence_on_incorrect": float(np.mean([max(r["probabilities"]) for r in errors])) if errors else None,
                                "prediction_change_fraction": transitions["prediction_changes"] / len(rows),
                                **transitions})
        budgets, seeds = config["context_budgets"], config["distractor_seeds"]
        primary_bins = settings["ece_bins"][0]
        point = {b: seed_mean(groups, ids, b, seeds, primary_bins) for b in budgets}
        strata = defaultdict(list)
        for q in ids:
            strata[questions[q]["subject"] if dataset == "mmlu_pro" else "all"].append(q)
        rng = np.random.default_rng(settings["bootstrap_seed"])
        bootstrap = {b: [] for b in budgets}
        for _ in range(settings["bootstrap_resamples"]):
            # One resample shared by every length and seed; duplicate IDs keep their multiplicity.
            sampled = [str(q) for s in sorted(strata)
                       for q in rng.choice(strata[s], size=len(strata[s]), replace=True)]
            for b in budgets:
                stats = seed_mean(groups, sampled, b, seeds, primary_bins)
                bootstrap[b].append([stats[m] for m in METRICS])
        tail = (1 - settings["confidence_level"]) / 2
        for b in budgets:
            draws = np.asarray(bootstrap[b])
            delta_draws = draws - np.asarray(bootstrap[0])
            row = {"dataset": dataset, "context_budget": b, "unique_questions": len(ids),
                   "planned_questions": config["sample_counts"][dataset],
                   "analysis_population": "paired_complete_questions", "run_completeness": report["status"]}
            delta = dict(row)
            for i, metric in enumerate(METRICS):
                lo, hi = np.quantile(draws[:, i], [tail, 1-tail])
                dlo, dhi = np.quantile(delta_draws[:, i], [tail, 1-tail])
                row.update({metric: point[b][metric], metric + "_ci_low": float(lo), metric + "_ci_high": float(hi)})
                delta.update({metric: point[b][metric] - point[0][metric],
                              metric + "_ci_low": float(dlo), metric + "_ci_high": float(dhi)})
                seed_stats = [seed_mean(groups, ids, b, [s], primary_bins)[metric] for s in seeds] if b else [point[0][metric]]
                row[metric + "_seed_min"] = min(seed_stats)
                row[metric + "_seed_max"] = max(seed_stats)
            summary.append(row)
            deltas.append(delta)
        print(f"Analyzed {dataset}: {len(ids)} complete questions", flush=True)
    # Timing/memory includes every saved observation, including incomplete-question data.
    for dataset in config["sample_counts"]:
        for budget in config["context_budgets"]:
            rows = [r for r in records.values() if r["dataset"] == dataset and r["context_budget"] == budget]
            if rows:
                mean_seconds = float(np.mean([r["amortized_forward_and_score_seconds"] for r in rows]))
                timing.append({"dataset": dataset, "context_budget": budget, "observations": len(rows),
                               "mean_amortized_seconds": mean_seconds,
                               "questions_per_second": 1 / mean_seconds,
                               "max_peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rows),
                               "min_headroom_fraction": min(r["minimum_headroom_fraction"] for r in rows),
                               "estimated_main_forward_hours": mean_seconds * (1000 if dataset == "mmlu_pro" else 500) * (1 if budget == 0 else len(config["distractor_seeds"])) / 3600})
    for name, rows in [("metrics_by_seed", all_seed), ("metrics_summary", summary),
                       ("baseline_deltas", deltas), ("diagnostics", diagnostics),
                       ("calibration_bins", bins_rows), ("timing_memory", timing)]:
        write_csv(output / f"{name}.csv", rows)
    if summary:
        from src.plotting import plot
        plot(output, summary, deltas, bins_rows, timing, config["purpose"], report["status"])
    save(output / "SHA256SUMS.json", {str(p.relative_to(output)): file_hash(p)
                                       for p in sorted(output.rglob("*")) if p.is_file()})
    print(f"Analysis saved to {output}; source run is {report['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/analysis.json")
    args = parser.parse_args()
    configure_paths()
    settings = read(args.config)
    if (settings["ece_bins"] != [10, 15, 20] or settings["bootstrap_resamples"] < 2 or
            not 0 < settings["confidence_level"] < 1):
        raise ValueError("Invalid analysis configuration")
    analyze(args.run.resolve(), args.output, settings)


if __name__ == "__main__":
    main()
