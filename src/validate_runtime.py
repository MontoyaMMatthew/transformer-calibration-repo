"""Short synthetic-prompt GPU check for the pilot scorer; no benchmark data is evaluated."""

import argparse
from pathlib import Path

import numpy as np

from scripts.setup_common import ROOT, configure_paths
from src.artifacts import now, persistent_path, read, save
from src.inference import Scorer
from src.prompts import make_case, reconstruct


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = configure_paths()
    resources = read(paths["reports"] / "resources.json")
    config = read(ROOT / "config/pilot.json")
    tokenizer = AutoTokenizer.from_pretrained(resources["model"]["snapshot_path"], local_files_only=True)
    q = {"dataset": "mmlu_pro", "question_key": "synthetic:runtime", "subject": "synthetic",
         "split": "synthetic", "row": {"question_id": "synthetic:runtime",
         "question": "What is two plus two?", "options": ["Four", "Five"], "answer_index": 0}}
    case = make_case(tokenizer, q, None, None, 0, None)
    prompt = reconstruct(tokenizer, case, {q["question_key"]: q}, {})
    scorer = Scorer(resources["model"]["snapshot_path"], tokenizer, config)
    attention = scorer.attention_check(case, prompt)
    reference = scorer.score([case], [prompt])[0]
    results = []
    for size in [1, 2, 4]:
        rows = scorer.score([case] * size, [prompt] * size)
        differences = [float(np.max(np.abs(np.array(r["probabilities"]) - reference["probabilities"]))) for r in rows]
        if max(differences) > .005 or any(r["prediction"] != reference["prediction"] for r in rows):
            raise ValueError("Batched synthetic scoring differs from batch one")
        results.append({"batch_size": size, "maximum_probability_difference": max(differences),
                        "records": rows})
    save(persistent_path(args.output), {
        "purpose": "Synthetic short-prompt runtime check; no pilot or benchmark dataset inference",
        "status": "passed", "finished_utc": now(), "hardware": scorer.hardware,
        "attention_check": attention, "batch_checks": results,
        "limitation": "Higher-batch long-context memory and throughput screening remains part of the unstarted pilot"})
    print("Synthetic scorer runtime check passed; pilot remains unstarted", flush=True)


if __name__ == "__main__":
    main()
