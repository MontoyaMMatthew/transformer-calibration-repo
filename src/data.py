"""Eligibility first, then deterministic proportional subject sampling."""

import math
import random
from collections import defaultdict

from src.prompts import make_case, make_stream, render_question
from scripts.validate_gpu import digest


def select_questions(tokenizer, datasets, config):
    selected, report = {}, {}
    for key, data in datasets.items():
        strata, excluded, ids = defaultdict(list), [], set()
        for index, row in enumerate(data):
            if key == "mmlu_pro":
                choices = row["options"]
                if row["answer"] != chr(65 + row["answer_index"]):
                    raise ValueError("MMLU-Pro gold label differs from gold index")
            else:
                choices = row["choices"]["text"]
                labels = row["choices"]["label"]
                if len(set(labels)) != len(choices) or row["answerKey"] not in labels:
                    raise ValueError("Invalid ARC original labels or gold mapping")
            if not row["question"].strip() or not 2 <= len(choices) <= 26 or any(not c.strip() for c in choices):
                raise ValueError("Invalid question or choice text/count")
            qid = str(row["question_id"] if key == "mmlu_pro" else row["id"])
            qkey = f"{key}:{qid}"
            if qkey in ids:
                raise ValueError(f"Duplicate source question ID: {qkey}")
            ids.add(qkey)
            q = {"question_key": qkey, "dataset": key, "question_id": qid, "row": row,
                 "source_row": index, "split": config["question_split"],
                 "subject": row["category"] if key == "mmlu_pro" else "all"}
            baseline = render_question(tokenizer, q, "")
            if baseline["input_tokens"] + 1 > 2048:
                excluded.append({"question_id": qid, "source_row": index,
                                 "fixed_input_tokens": baseline["input_tokens"],
                                 "reason": "fixed prompt exceeds smallest nonzero budget"})
            else:
                strata[q["subject"]].append(q)
        total = sum(map(len, strata.values()))
        n = config["sample_counts"][key]
        if not 0 < n <= total:
            raise ValueError(f"Cannot sample {n} eligible questions from {key}: {total}")
        quotas = {s: n * len(rows) / total for s, rows in strata.items()}
        allocation = {s: math.floor(v) for s, v in quotas.items()}
        for s in sorted(quotas, key=lambda s: (-(quotas[s] - allocation[s]), s))[:n-sum(allocation.values())]:
            allocation[s] += 1
        rng = random.Random(config["selection_seed"])
        for s in sorted(strata):
            rows = sorted(strata[s], key=lambda q: q["question_id"])
            for q in rng.sample(rows, allocation[s]):
                selected[q["question_key"]] = q
        report[key] = {"source_rows": len(data), "eligible": total, "excluded": excluded,
                       "eligible_subject_counts": {s: len(v) for s, v in strata.items()},
                       "sample_subject_counts": allocation,
                       "selected_ids": sorted(q["question_id"] for q in selected.values()
                                              if q["dataset"] == key)}
    return selected, report


def prepare(tokenizer, resources, config):
    from datasets import load_from_disk
    entries = {e["key"]: e for e in resources["datasets"]}
    data = {key: load_from_disk(entries[key]["splits"][config["question_split"]]["prepared_path"])
            for key in config["sample_counts"]}
    questions, selection = select_questions(tokenizer, data, config)
    wiki = load_from_disk(entries["wikitext"]["splits"]["train"]["prepared_path"])
    streams, plan = {}, []
    for qkey, q in sorted(questions.items()):
        plan.append(make_case(tokenizer, q, None, None, 0, None))
        for seed in config["distractor_seeds"]:
            sid = digest([qkey, seed])[:24]
            stream = make_stream(tokenizer, wiki, q["dataset"], q["question_id"], seed,
                                 entries["wikitext"]["revision"])
            streams[sid] = stream
            offsets = tokenizer(stream["text"], add_special_tokens=False,
                                return_offsets_mapping=True)["offset_mapping"]
            for budget in config["context_budgets"][1:]:
                plan.append(make_case(tokenizer, q, sid, stream, budget, seed, offsets))
        print(f"Prepared {qkey} ({len(plan)} evaluations)", flush=True)
    if len({c["evaluation_id"] for c in plan}) != len(plan):
        raise ValueError("Duplicate planned evaluation")
    return questions, selection, streams, plan
