"""CPU figures exported to PNG for inspection and PDF for the paper."""

import numpy as np


def plot(output, summary, deltas, bins_rows, timing, purpose, completeness):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "figures"
    figures.mkdir()
    run_label = "synthetic validation" if "synthetic" in purpose.lower() else "pilot"

    def finish(fig, dataset, name):
        fig.suptitle(f"{dataset} — {run_label} ({completeness})", fontsize=11)
        fig.tight_layout()
        for extension in ["pdf", "png"]:
            fig.savefig(figures / f"{dataset}_{name}.{extension}", dpi=170)
        plt.close(fig)

    def trajectory(ax, rows, metric, label=None):
        values = [r[metric] for r in rows]
        low = [r[metric + "_ci_low"] for r in rows]
        high = [r[metric + "_ci_high"] for r in rows]
        # Percentile CIs need not contain the point estimate; draw intervals directly.
        x = np.arange(len(rows))
        line, = ax.plot(x, values, marker="o", label=label or metric)
        ax.vlines(x, low, high, color=line.get_color(), alpha=0.65)
        ax.set_xticks(x, ["0K" if r["context_budget"] == 0 else f"{r['context_budget']//1024}K" for r in rows])
        ax.set_xlabel("Total context budget (0K has no background)")
        ax.grid(alpha=0.2)

    for dataset in sorted({r["dataset"] for r in summary}):
        rows = sorted([r for r in summary if r["dataset"] == dataset], key=lambda r: r["context_budget"])
        fig, ax = plt.subplots(figsize=(7, 4))
        trajectory(ax, rows, "accuracy", "Accuracy")
        trajectory(ax, rows, "mean_confidence", "Mean confidence")
        ax.set_ylim(0, 1)
        ax.legend()
        finish(fig, dataset, "accuracy_confidence")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, metric in zip(axes, ["ece", "nll", "brier"], strict=True):
            trajectory(ax, rows, metric)
            ax.set_title("Cross-entropy (NLL), nats" if metric == "nll" else metric.upper())
        finish(fig, dataset, "calibration_metrics")
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        delta_rows = sorted([r for r in deltas if r["dataset"] == dataset], key=lambda r: r["context_budget"])
        for ax, metric in zip(axes.flat, ["accuracy", "mean_confidence", "nll", "ece", "brier", "entropy", "normalized_entropy", "confidence_gap"], strict=True):
            trajectory(ax, delta_rows, metric)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_title(f"Δ {metric} from 0K")
        finish(fig, dataset, "baseline_deltas")
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for col, budget in enumerate([0, 8192, 32768]):
            rows_b = [r for r in bins_rows if r["dataset"] == dataset and
                      r["context_budget"] == budget and r["ece_bins"] == 10]
            for seed in sorted({str(r["seed"]) for r in rows_b}):
                seed_rows = [r for r in rows_b if str(r["seed"]) == seed]
                filled = [r for r in seed_rows if r["count"]]
                axes[0, col].plot([r["confidence"] for r in filled], [r["accuracy"] for r in filled],
                                  "o-", label=f"seed {seed}" if budget else "baseline")
                axes[1, col].plot([(r["lower"]+r["upper"])/2 for r in seed_rows],
                                  [r["count"] for r in seed_rows], "o-", label=f"seed {seed}")
            axes[0, col].plot([0, 1], [0, 1], "k--", linewidth=0.7)
            axes[0, col].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy",
                              title=f"{budget//1024}K reliability")
            axes[0, col].legend()
            axes[1, col].set(xlim=(0, 1), xlabel="Confidence bin", ylabel="Question count")
        finish(fig, dataset, "reliability_diagrams")
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        time_rows = [r for r in timing if r["dataset"] == dataset]
        x = [r["context_budget"] / 1024 for r in time_rows]
        axes[0].plot(x, [r["mean_amortized_seconds"] for r in time_rows], "o-")
        axes[0].set(xlabel="Context (K)", ylabel="Seconds per question")
        axes[1].plot(x, [r["max_peak_reserved_bytes"] / 1024**3 for r in time_rows], "o-")
        axes[1].set(xlabel="Context (K)", ylabel="Peak reserved GPU memory (GiB)")
        finish(fig, dataset, "timing_memory")
