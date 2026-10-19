"""Scientific/record correctness tests; synthetic results are never study observations."""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from scripts.setup_common import ROOT, configure_paths
from src.artifacts import (digest, load_records, load_run, read, save, save_manifest,
                           verify_record, writer_lock)
from src.analyze_results import analyze, completeness
from src.metrics import aggregate, calibration_bins, example, seed_mean
from src.prompts import make_case, make_stream, reconstruct, render_question


def record(case, probabilities=(0.75, 0.25), identity="test"):
    return {**case, "status": "success", "plan_identity": identity,
            "probabilities": list(probabilities), "choice_logits": np.log(probabilities).tolist(),
            "log_probabilities": np.log(probabilities).tolist(),
            "prediction": case["labels"][int(np.argmax(probabilities))],
            "amortized_forward_and_score_seconds": 0.1,
            "peak_reserved_bytes": 1000, "minimum_headroom_fraction": 0.8}


def test_metrics_known_values_and_bin_edges():
    a = record({"labels": ["A", "B"], "gold_index": 0})
    b = record({"labels": ["A", "B"], "gold_index": 1})
    result = aggregate([a, b])
    assert result["accuracy"] == 0.5
    assert result["mean_confidence"] == 0.75
    assert result["ece"] == 0.25
    assert result["brier"] == 0.625  # Sum over classes, no division by two.
    assert result["nll"] == pytest.approx((-math.log(.75)-math.log(.25))/2)
    assert example(a)["normalized_entropy"] == pytest.approx(example(a)["entropy"]/math.log(2))
    bins = calibration_bins([{"mean_confidence": 0, "accuracy": 1},
                             {"mean_confidence": .1, "accuracy": 0},
                             {"mean_confidence": 1, "accuracy": 1}])
    assert [bins[i]["count"] for i in [0, 1, 9]] == [1, 1, 1]
    assert bins[2]["accuracy"] is None


def test_seed_statistics_do_not_score_an_ensemble():
    case = {"labels": ["A", "B"], "gold_index": 0}
    groups = {(2048, 11): {"q": example(record(case, (.9, .1)))},
              (2048, 22): {"q": example(record(case, (.1, .9)))}}
    result = seed_mean(groups, ["q"], 2048, [11, 22])
    assert result["accuracy"] == .5
    assert result["mean_confidence"] == .9
    assert result["brier"] == pytest.approx(.82)
    assert result["nll"] == pytest.approx((-math.log(.9) - math.log(.1))/2)
    assert result["brier"] != aggregate([record(case, (.5, .5))])["brier"]


@pytest.fixture
def synthetic_run(tmp_path):
    run = tmp_path / "synthetic-run"
    run.mkdir()
    config = read(ROOT / "config/pilot.json")
    config["purpose"] = "SYNTHETIC TEST ONLY; no model inference"
    config["sample_counts"] = {"mmlu_pro": 3, "arc_challenge": 3}
    questions, plan = {}, []
    for dataset in config["sample_counts"]:
        for i in range(3):
            qkey = f"{dataset}:{i}"
            questions[qkey] = {"dataset": dataset, "subject": str(i % 2)}
            for budget in config["context_budgets"]:
                for seed in config["distractor_seeds"] if budget else [None]:
                    plan.append({"evaluation_id": digest([qkey, budget, seed])[:24],
                                 "question_key": qkey, "dataset": dataset, "gold_index": i % 2,
                                 "labels": ["A", "B"], "context_budget": budget, "distractor_seed": seed})
    manifest = save_manifest(run, config, plan, questions, {}, {}, {})
    for case in plan:
        save(run / "records" / f"{case['evaluation_id']}.json", record(case, identity=manifest["identity"]))
    return run


def test_saved_analysis_all_artifacts_and_paired_zero(synthetic_run, tmp_path):
    settings = read(ROOT / "config/analysis.json")
    settings["bootstrap_resamples"] = 30
    output = tmp_path / "synthetic-analysis"
    analyze(synthetic_run, output, settings)
    import csv
    rows = list(csv.DictReader((output / "baseline_deltas.csv").open()))
    # Identical repeated inputs must have exactly zero paired changes, including nonlinear ECE.
    for row in rows:
        for metric in ["accuracy", "ece", "nll", "brier"]:
            assert float(row[metric]) == 0
            assert float(row[metric + "_ci_low"]) == 0
            assert float(row[metric + "_ci_high"]) == 0
    assert read(output / "completeness_report.json")["status"] == "complete"
    assert len(list((output / "figures").glob("*.pdf"))) == 10
    assert len(list((output / "figures").glob("*.png"))) == 10
    assert len(list(csv.DictReader((output / "calibration_bins.csv").open()))) == 1440


def test_incomplete_question_excluded_and_corruption_rejected(synthetic_run):
    manifest, _, plan = load_run(synthetic_run)
    records = load_records(synthetic_run, plan, manifest["identity"])
    first = plan[0]
    del records[first["evaluation_id"]]
    report = completeness(plan, records, read(synthetic_run / "questions.json"))
    assert report["status"] == "incomplete"
    assert report["paired_complete_questions"] == 5
    assert first["question_key"] not in report["complete_question_ids"]
    altered = record(first, identity=manifest["identity"])
    altered["probabilities"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="disagree"):
        verify_record(altered, first, manifest["identity"])
    with (synthetic_run / "plan.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Saved input changed"):
        load_run(synthetic_run)


def test_atomic_kill_recovery_and_writer_lock(tmp_path):
    run = tmp_path / "recovery"
    run.mkdir()
    child_code = """
import sys, time
from pathlib import Path
from src.artifacts import save, writer_lock
run = Path(sys.argv[1])
with writer_lock(run):
    save(run/'one.json', {'evaluation_id': 'one', 'status': 'success'})
    save(run/'ready.json', {'ready': True})
    time.sleep(90)
"""
    child = subprocess.Popen([sys.executable, "-c", child_code, str(run)])
    try:
        deadline = time.monotonic() + 20
        while not (run / "ready.json").exists():
            if child.poll() is not None or time.monotonic() > deadline:
                pytest.fail("Child did not save durable checkpoint")
            time.sleep(.05)
        before = (run / "one.json").read_bytes()
        with pytest.raises(BlockingIOError):
            with writer_lock(run):
                pass
        child.kill()
        child.wait(timeout=10)
        with writer_lock(run):
            save(run / "two.json", {"evaluation_id": "two", "status": "success"})
        assert (run / "one.json").read_bytes() == before
        assert read(run / "two.json")["status"] == "success"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_runner_resume_skips_success_and_rejects_changed_code(synthetic_run, monkeypatch):
    from src import run_experiments
    from src import inference
    import transformers
    manifest, config, plan = load_run(synthetic_run)
    config["purpose"] = read(ROOT / "config/pilot.json")["purpose"]
    # Seal an executable test configuration, keeping all mocked results in the synthetic directory.
    manifest = save_manifest(synthetic_run, config, plan, read(synthetic_run / "questions.json"),
                             {}, {}, {"model": {"snapshot_path": "mock"}})
    for c in plan:
        save(synthetic_run / "records" / f"{c['evaluation_id']}.json", record(c, identity=manifest["identity"]))
    missing = plan[-1]
    (synthetic_run / "records" / f"{missing['evaluation_id']}.json").unlink()
    # Exact-length bucketing is an execution detail; all mock examples have the same length.
    for c in plan:
        c["input_tokens"] = 10
    manifest = save_manifest(synthetic_run, config, plan, read(synthetic_run / "questions.json"),
                             {}, {}, {"model": {"snapshot_path": "mock"}})
    for c in plan[:-1]:
        save(synthetic_run / "records" / f"{c['evaluation_id']}.json", record(c, identity=manifest["identity"]))
    before = {p.name: p.read_bytes() for p in (synthetic_run / "records").glob("*.json")}
    seen = []

    class FakeScorer:
        load_seconds = 0.01
        def __init__(self, *args):
            pass
        def attention_check(self, *args):
            return {"status": "passed"}
        def score(self, cases, prompts):
            seen.extend(c["evaluation_id"] for c in cases)
            return [record(c) for c in cases]

    monkeypatch.setattr(inference, "Scorer", FakeScorer)
    monkeypatch.setattr(inference, "hardware", lambda c: {"mock": True})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(run_experiments, "reconstruct", lambda *args: "mock")
    run_experiments.execute(synthetic_run)
    assert seen == [missing["evaluation_id"]]
    assert all((synthetic_run / "records" / name).read_bytes() == contents for name, contents in before.items())
    run_experiments.execute(synthetic_run)
    assert seen == [missing["evaluation_id"]]  # Completed resume does not load the model or score again.
    monkeypatch.setattr("src.artifacts.environment", lambda: {"changed": True})
    with pytest.raises(ValueError, match="Resume rejected"):
        run_experiments.execute(synthetic_run)


def test_real_tokenizer_new_prompt_budget_prefix_and_reconstruction():
    from transformers import AutoTokenizer
    paths = configure_paths()
    resources = read(paths["reports"] / "resources.json")
    tokenizer = AutoTokenizer.from_pretrained(resources["model"]["snapshot_path"], local_files_only=True)
    row = {"question_id": 42, "question": "What is two plus two?", "options": ["Four", "Five"],
           "answer_index": 0, "answer": "A", "category": "math"}
    question = {"dataset": "mmlu_pro", "question_key": "mmlu_pro:42", "subject": "math",
                "split": "validation", "row": row}
    wiki = [{"text": "This encyclopedia entry discusses mountain landscapes and their history. " * 25}]
    stream = make_stream(tokenizer, wiki, "mmlu_pro", "42", 11, "test-revision")
    previous = ""
    for budget in [0, 2048, 4096, 8192, 16384, 32768]:
        case = make_case(tokenizer, question, "stream", stream, budget, 11)
        prompt = reconstruct(tokenizer, case, {"mmlu_pro:42": question}, {"stream": stream})
        assert "Background text:\n" in prompt
        assert "Irrelevant background:" not in prompt
        assert "What is two plus two?" in prompt
        if budget:
            assert 0 <= budget - case["total_context_tokens"] <= 4
            prefix = stream["text"][:case["distractor_characters"]]
            assert prefix.startswith(previous)
            previous = prefix
        for label, token in zip(case["labels"], case["label_token_ids"]):
            assert tokenizer.encode(prompt + " " + label, add_special_tokens=False) == tokenizer.encode(prompt, add_special_tokens=False) + [token]
    corrupted = dict(case, prompt_sha256="changed")
    with pytest.raises(ValueError, match="reconstruction changed"):
        reconstruct(tokenizer, corrupted, {"mmlu_pro:42": question}, {"stream": stream})
