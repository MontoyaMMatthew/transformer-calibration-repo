"""Metrics calculated from saved full-choice probabilities, never from an ensemble."""

import numpy as np

METRICS = ["accuracy", "mean_confidence", "nll", "ece", "brier", "entropy",
           "normalized_entropy", "confidence_gap"]


def example(record):
    p = np.asarray(record["probabilities"], dtype=float)
    lp = np.asarray(record["log_probabilities"], dtype=float)
    gold = record["gold_index"]
    target = np.zeros_like(p)
    target[gold] = 1
    entropy = float(-(p * lp).sum())
    return {"accuracy": float(int(p.argmax()) == gold), "mean_confidence": float(p.max()),
            "nll": float(-lp[gold]), "brier": float(((p-target)**2).sum()),
            "entropy": entropy, "normalized_entropy": entropy / np.log(len(p))}


def calibration_bins(values, bins=10):
    conf = np.asarray([v["mean_confidence"] for v in values])
    acc = np.asarray([v["accuracy"] for v in values])
    index = np.minimum((conf * bins).astype(int), bins - 1)
    result = []
    for b in range(bins):
        mask = index == b
        n = int(mask.sum())
        result.append({"bin": b, "lower": b / bins, "upper": (b + 1) / bins,
                       "count": n, "accuracy": float(acc[mask].mean()) if n else None,
                       "confidence": float(conf[mask].mean()) if n else None})
    return result


def aggregate_values(values, bins=10):
    if not values:
        raise ValueError("Cannot aggregate zero observations")
    result = {k: float(np.mean([v[k] for v in values])) for k in example_keys()}
    reliability = calibration_bins(values, bins)
    result["ece"] = sum(b["count"] / len(values) * abs(b["accuracy"] - b["confidence"])
                        for b in reliability if b["count"])
    result["confidence_gap"] = result["mean_confidence"] - result["accuracy"]
    return result


def example_keys():
    return [k for k in METRICS if k not in ["ece", "confidence_gap"]]


def aggregate(records, bins=10):
    return aggregate_values([example(r) for r in records], bins)


def seed_mean(groups, question_ids, budget, seeds, bins=10):
    actual_seeds = [None] if budget == 0 else seeds
    statistics = [aggregate_values([groups[(budget, s)][q] for q in question_ids], bins)
                  for s in actual_seeds]
    return {k: float(np.mean([v[k] for v in statistics])) for k in METRICS}
