"""Deterministic, saved-source background prefixes and exact label scoring."""

import hashlib
import random

from scripts.validate_gpu import digest, render

HEADING = "Background text:"


def render_question(tokenizer, question, background, verify_labels=True):
    return render(tokenizer, question["row"], question["dataset"], background,
                  verify_labels=verify_labels, heading=HEADING)


def make_stream(tokenizer, wiki, dataset, qid, seed, revision):
    rng = random.Random(int(digest([dataset, qid, seed]), 16))
    chunks, sources, tokens, offset = [], [], 0, 0
    while tokens < 40000:
        index = rng.randrange(len(wiki))
        text = wiki[index]["text"]
        if not text.strip():
            continue
        if chunks:
            offset += 1  # The joining newline.
        sources.append({"row_index": index, "stream_char_start": offset,
                        "stream_char_end": offset + len(text), "source_char_start": 0,
                        "source_char_end": len(text)})
        chunks.append(text)
        offset += len(text)
        tokens += len(tokenizer.encode(text, add_special_tokens=False))
    text = "\n".join(chunks)
    return {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source": "Salesforce/wikitext", "revision": revision,
            "config": "wikitext-103-raw-v1", "split": "train", "seed": seed,
            "sources": sources}


def make_case(tokenizer, question, stream_id, stream, budget, seed, offsets=None):
    if budget:
        text = stream["text"]
        if offsets is None:
            offsets = tokenizer(text, add_special_tokens=False,
                                return_offsets_mapping=True)["offset_mapping"]
        lo, hi = 0, len(offsets)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            prefix = text[:offsets[mid - 1][1]] if mid else ""
            if render_question(tokenizer, question, prefix, False)["input_tokens"] + 1 <= budget:
                lo = mid
            else:
                hi = mid - 1
        background = text[:offsets[lo - 1][1]] if lo else ""
    else:
        background = ""
    case = render_question(tokenizer, question, background)
    case.pop("prompt")
    shortfall = budget - case["input_tokens"] - 1 if budget else None
    if budget and not 0 <= shortfall <= 4:
        raise ValueError(f"Token target not closely filled: {shortfall}")
    case.update(
        question_key=question["question_key"], subject=question["subject"],
        split=question["split"], context_budget=budget, distractor_seed=seed if budget else None,
        distractor_stream_id=stream_id if budget else None,
        distractor_characters=len(background),
        distractor_tokens=len(tokenizer.encode(background, add_special_tokens=False)),
        target_shortfall=shortfall, reserved_answer_tokens=1,
        total_context_tokens=case["input_tokens"] + 1, placement="before_question",
    )
    case["evaluation_id"] = digest([question["question_key"], budget, case["distractor_seed"]])[:24]
    return case


def reconstruct(tokenizer, case, questions, streams):
    stream_id = case["distractor_stream_id"]
    background = streams[stream_id]["text"][:case["distractor_characters"]] if stream_id else ""
    rendered = render_question(tokenizer, questions[case["question_key"]], background)
    for key in ["prompt_sha256", "input_tokens", "labels", "gold_index", "label_token_ids"]:
        if rendered[key] != case[key]:
            raise ValueError(f"Prompt reconstruction changed {key}")
    if case["total_context_tokens"] > (case["context_budget"] or 32768):
        raise ValueError("Token budget exceeded")
    return rendered["prompt"]
