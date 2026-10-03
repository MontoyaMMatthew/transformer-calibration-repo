# Experiment design and result generation plan

Status: proposed design, recorded October 3, 2026. This document describes the intended implementation; the Python modules, configurations, backup process, and resume behavior are not yet implemented. Sample counts remain subject to pilot timing.

This plan supports *Quantifying Transformer Confidence and Accuracy Trajectories Under Non-Ideal Conditions* and complements [RESOURCE_DEFINITION.md](RESOURCE_DEFINITION.md).

## Research question and scope

Measure how the accuracy and confidence of a fixed Qwen3-4B model change as irrelevant context increases. Test the hypothesis that confidence becomes less reliable with increasing distractor context.

Use MMLU-Pro as the primary evaluation dataset, WikiText-103 as the distractor source, and ARC-Challenge for verification. There is no training or fine-tuning.

Falling accuracy alone does not establish worsening calibration. Evidence should consider accuracy, confidence, calibration error, and probability quality together. Report the two evaluation datasets separately.

## Expected final results

Produce one summary table per dataset, with measured values and 95% confidence intervals:

| Context condition | Unique questions | Accuracy | Mean confidence | Cross-entropy (NLL) | ECE | Brier score | Entropy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0K: no distractor | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2K | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 4K | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 8K | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 16K | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 32K | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Higher accuracy is better. Lower NLL, ECE, and Brier scores are better. Confidence and entropy have no universally preferred direction: confidence should reflect correctness, and uncertainty should increase when warranted.

Generate these figures:

- Accuracy and mean confidence versus context condition on the same axes.
- ECE, NLL, and Brier trajectories in separate panels.
- Reliability diagrams for 0K, 8K, and 32K, with confidence-bin counts.
- Changes from the 0K baseline with paired confidence intervals.

The results are unknown until the experiment runs. Sustained confidence while accuracy falls would be consistent with the hypothesis; it must be interpreted alongside the calibration measures.

## Metrics and conventions

For question `i`, let `p[i,k]` denote the probability assigned to valid choice `k`, `y[i]` the gold choice, and `c[i] = max_k p[i,k]`. The prediction is the highest-probability valid choice. Use a fixed tie-breaking rule, such as the first choice in the saved label order.

| Metric | Calculation | Interpretation |
| --- | --- | --- |
| Accuracy | Mean of prediction equals gold | Task performance |
| Mean confidence | Mean of `c[i]` | Average confidence in the selected answer |
| Cross-entropy (NLL) | Mean of `-log p[i,y[i]]` | Probability quality; strongly penalizes confident mistakes |
| Expected calibration error (ECE) | Sum over bins of bin frequency times absolute difference between bin accuracy and bin confidence | Calibration of the highest-probability choice |
| Multiclass Brier score | Mean of `sum_k (p[i,k] - 1[k=y[i]])^2` | Quality of the full choice distribution |
| Predictive entropy | Mean of `-sum_k p[i,k] log p[i,k]` | Uncertainty across answer choices |
| Signed confidence gap | Mean confidence minus accuracy | Aggregate overconfidence or underconfidence |

Conventions:

- Use 10 equal-width confidence bins for primary ECE. Specify boundary handling consistently, include confidence 1 in the final bin, and give empty bins zero weight. Save bin counts, accuracy, and mean confidence. Use 15 and 20 bins only as secondary sensitivity checks.
- Sum Brier errors over classes, then average over examples. Do not divide by the number of classes.
- Use natural logarithms for NLL and entropy, giving units of nats.
- Also calculate normalized entropy `H / log(K)` for each example, where `K` is its number of valid choices.
- Treat signed confidence gap as descriptive: positive and negative errors can cancel.
- NLL and Brier evaluate probability quality beyond calibration alone. ECE depends on binning. Interpret them together.

### Cross-entropy and NLL

For a single correct answer with no class weighting or label smoothing, categorical cross-entropy and NLL are identical:

```text
CE(y, p) = -sum_k 1[k=y] log p[k] = -log p[y] = NLL.
```

Report one column named **Cross-entropy (NLL)**. There is no mathematical preference for NLL and no reason to report duplicate columns. The NLL terminology emphasizes the probability assigned to the correct answer.

PyTorch cross-entropy takes raw logits; NLL loss takes log-probabilities. Do not pass already-softmaxed probabilities into `CrossEntropyLoss`. Compute stable log-probabilities from the choice logits and preserve them in the output.

This is answer-choice NLL, conditional on the supplied choices. It is not language-model loss over the prompt or distractor text.

### Diagnostic results supported by the saved records

- **Confidence on incorrect answers:** select records whose prediction differs from gold, then average maximum choice probability. Report the number of incorrect answers. If there are none, report the metric as undefined, not zero.
- **Fraction of predictions changing from baseline:** join each condition to the same question's 0K result and calculate the fraction with a different prediction.
- **Transition breakdown:** correct to incorrect, incorrect to correct, and incorrect to a different incorrect answer.

These diagnostics require no additional inference. Missing condition or baseline records must be reported, not counted as unchanged predictions.

## Experiment matrix

| Parameter | Proposed setting |
| --- | --- |
| Model | `Qwen/Qwen3-4B`; pin exact model and tokenizer revisions |
| Primary sample | 1,000 MMLU-Pro test questions, proportionally stratified by subject |
| Verification sample | 500 ARC-Challenge test questions |
| Distractor source | WikiText-103, `wikitext-103-raw-v1`, training split |
| Conditions | 0K, 2K, 4K, 8K, 16K, 32K |
| Distractor realizations | Three per question at each nonzero condition |
| Distractor seeds | Proposed values: 11, 22, 33 |
| Question-selection seed | Separate fixed seed, proposed value: 42 |
| Prompting | Zero-shot, fixed instructions and answer format |
| Placement | Distractor before question and choices |
| Choice ordering | Preserve dataset order |
| Thinking | Explicitly disabled with `enable_thinking=False` |
| Precision | BF16 for main study; no quantization |
| Execution | Evaluation mode, gradient tracking disabled |
| Initial batch size | One; freeze the tested execution policy after the pilot |
| Attention | Tested memory-efficient implementation, fixed for main runs |
| Scoring | Direct answer-choice logits; no sampled answer generation |
| Probability transformation | FP32 choice-logit softmax/log-softmax; no fitted temperature or sampling filters |
| Context extension | Disabled |

The three seeds change distractor sampling, not the question subset, model weights, or sampled answers. Repeating identical deterministic prompts three times is unnecessary.

For each question and distractor seed, prepare a reproducible stream of distractor text. Use progressively longer prefixes of that stream across nonzero conditions. This preserves pairing as context grows. Derive randomness from a stable combination of question ID and seed, independent of execution order; do not rely on a process-randomized hash.

The same questions appear at every condition. Compute the no-distractor baseline once per question and reuse it across seed comparisons.

At the proposed sample sizes, the main study requires:

```text
(1,000 + 500) questions × (1 baseline + 5 nonzero lengths × 3 seeds)
= 24,000 forward passes.
```

WikiText is a distractor source, not a guarantee that every passage is semantically irrelevant. Inspect pilot examples for accidental helpful overlap. Freeze any screening rules before the main run; do not filter based on observed model performance.

## Prompt layout and token accounting

The fixed layout is the input supplied to the model during each inference:

```text
Chat-template formatting
Task instructions
Distractor text
Question
Answer choices
Answer prefix
Other required chat-template formatting
```

Freeze the exact rendered template after the pilot. The experiment concerns increasing distractor context placed before the question; varying question placement would be a separate experiment.

Every input token counts, including instructions, choices, answer prefix, separators, and chat-template tokens.

| Condition | Definition |
| --- | --- |
| 0K | Full fixed prompt, without distractor text |
| 2K | 2,048-token total budget |
| 4K | 4,096-token total budget |
| 8K | 8,192-token total budget |
| 16K | 16,384-token total budget |
| 32K | 32,768-token total budget |

For nonzero conditions, reserve one answer token inside the budget:

```text
complete rendered input tokens + 1 reserved answer token <= condition budget
```

For example, a 350-token fixed portion gives an initial distractor allowance of `2048 - 350 - 1 = 1697` tokens at 2K. Re-tokenize the entire assembled prompt and adjust the distractor portion: token counts of separately tokenized pieces are not guaranteed to add exactly at boundaries. Record actual prompt and distractor counts and any target shortfall.

Direct scoring does not require appending a generated answer. The one-token reserve is a conservative, consistent budgeting convention.

Before selecting the final sample, identify questions whose fixed prompt cannot fit the smallest nonzero budget. Exclude them from the entire paired sweep, report the exclusions, and sample from the eligible population. Never silently truncate a question or its choices.

## Inference and analysis workflow

### Stage 1: inference on GPU

For each evaluation:

1. Reconstruct the planned prompt and verify its token budget.
2. Run the fixed model with thinking disabled.
3. Extract vocabulary logits only at the answer position, then select valid answer-label logits.
4. Calculate choice log-softmax and probabilities in FP32.
5. Save logits, log-probabilities, probabilities, prediction, and execution metadata immediately.

Verify during the pilot that each answer label is one token in the exact answer-prefix context. If this fails, revise and validate the label format before the main study. No free-text explanations or sampled answers are required.

### Stage 2: analysis on CPU

Read saved inference records to compute metrics, confidence intervals, tables, and plots. Running metrics during inference are optional progress indicators; regenerate all final metrics from saved records.

Changing ECE bins, diagnostic calculations, or plot styling creates a new analysis, not a new inference run. Preserve log-probabilities so very small probabilities do not cause numerical problems in NLL calculations.

## Aggregation and uncertainty

Calculate metrics separately by dataset, context condition, and distractor seed. Report the mean of the three seed-specific metrics and retain individual seed results. The baseline is one measurement per question, not three independent observations.

Do not average probability vectors across seeds before calculating metrics: that evaluates an ensemble rather than individual distractor realizations.

Use approximately 2,000 question-level bootstrap resamples for 95% confidence intervals:

- Resample question IDs, preserving MMLU-Pro subject stratification.
- Retain all context conditions and seed realizations for each resampled question.
- Recompute the entire statistic, including ECE, in each resample.
- Use identical resampled question IDs for paired changes from baseline.

These intervals describe question-sampling uncertainty conditional on the sampled distractor realizations. Report seed variation separately; three seeds provide limited information about the wider distractor distribution.

Predefine 32K minus 0K ECE as the primary calibration comparison. Treat the full trajectory and changes in NLL and Brier as supporting evidence. Report completeness before interpreting comparisons; do not silently compare different question populations after failures.

## Proposed Python and configuration structure

Keep `src` mostly flat initially:

```text
transformer-calibration-repo/
├── configs/
│   ├── smoke.yaml
│   ├── pilot.yaml
│   ├── mmlu_pro.yaml
│   ├── arc_challenge.yaml
│   └── analysis.yaml
├── src/
│   ├── __init__.py
│   ├── run_experiments.py   # CLI, config loading, execution loop
│   ├── analyze_results.py   # CLI for analysis of saved outputs
│   ├── config.py            # Configuration validation
│   ├── data.py              # Loading, normalization, question sampling
│   ├── prompts.py           # Distractors, rendering, token budgets
│   ├── inference.py         # Model loading and choice scoring
│   ├── artifacts.py         # Writing, manifests, resume handling
│   ├── metrics.py           # Per-example and aggregate metrics
│   └── plotting.py          # Tables and figures
├── scripts/
│   ├── download_resources.py
│   └── verify_environment.py
├── artifacts/
│   ├── prepared/
│   ├── runs/
│   └── analyses/
├── tests/
└── pyproject.toml
```

Use standard Python `logging`, configured by the CLI, writing to the console and a run-specific log file. A separate logging package or experiment-tracking service is unnecessary initially.

Experiment YAML files specify model and dataset revisions, sample selection, context budgets, distractor seeds, prompt settings, execution settings, and artifact paths. Analysis YAML specifies metric conventions, bootstrap settings, and figure options. Both datasets use the same runner and scoring modules, with dataset-specific normalization.

Proposed CLI:

```bash
python -m src.run_experiments --config configs/pilot.yaml

python -m src.run_experiments \
  --config configs/mmlu_pro.yaml configs/arc_challenge.yaml

python -m src.run_experiments --resume artifacts/runs/<run_id>

python -m src.analyze_results \
  --run artifacts/runs/<run_id> \
  --config configs/analysis.yaml
```

Each experiment configuration produces its own run directory. Multiple configurations can be executed sequentially by one CLI invocation.

## Artifact locations and record structure

Store authoritative working artifacts under the repository's `artifacts/` directory on the VM's persistent managed disk. Record its absolute path in the run manifest. Do not use the VM's temporary disk for authoritative results.

The corresponding local path is:

```text
/Users/matthewmontoya/A1_JHU_School/DLT_744/transformer-calibration-repo/artifacts/
```

Proposed contents:

```text
artifacts/prepared/<preparation_id>/
├── manifest.json
├── questions.jsonl
└── distractor_sources.jsonl

artifacts/runs/<run_id>/
├── config.resolved.yaml
├── manifest.json
├── evaluation_plan.jsonl
├── questions.jsonl
├── distractor_manifest.jsonl
├── prompt_template.txt
├── predictions/
│   ├── baseline.jsonl
│   ├── context_02048_seed_11.jsonl
│   └── ...
├── logs/
│   └── run.log
└── failures.jsonl

artifacts/analyses/<analysis_id>/
├── analysis_config.yaml
├── source_runs.json
├── metrics_by_seed.csv
├── metrics_summary.csv
├── baseline_deltas.csv
├── diagnostics.csv
├── calibration_bins.csv
├── completeness_report.json
└── figures/
    ├── accuracy_confidence.pdf
    ├── calibration_metrics.pdf
    └── reliability_diagrams.pdf
```

JSONL provides one incrementally written record per evaluation. CSV is suitable for analysis tables. Runs reference prepared snapshots by identity and checksum; preserve those dependencies with backups.

Each successful inference record contains:

| Category | Fields |
| --- | --- |
| Identity | Run ID, stable evaluation ID, dataset, question ID, subject where applicable |
| Gold answer | Valid labels in order, gold choice |
| Condition | Context budget, distractor seed or baseline marker, placement |
| Input audit | Actual prompt/distractor token counts, prompt hash, distractor reference |
| Scores | Choice logits, log-probabilities, probabilities, predicted choice |
| Execution | Duration, status, timestamp |

Record timing definitions consistently and collect peak GPU memory by context during the pilot. The run manifest captures model/tokenizer and dataset revisions, software versions, hardware, code revision and any relevant uncommitted source snapshot, prompt identity, schema version, and all inference settings.

Save normalized question text once per run and retain prepared distractor source material with source IDs and offsets. Seeds alone do not guarantee prompt reconstruction. Preserve tokenizer/template versions and enough material to reconstruct and hash-check every prompt. Save selected fully rendered audit prompts without duplicating every expanded prompt.

Keep model weights and download caches outside individual run directories, on persistent storage. Avoid duplicating large model files per experiment.

## Resume behavior

`--resume` is mutually exclusive with `--config`. It loads the saved resolved configuration and evaluation plan; it does not resample questions or distractors.

Within a run, identify each evaluation using:

```text
dataset + question ID + context condition + distractor realization
```

The run fixes the model, prompt, and other scientific settings. Baseline evaluations have no distractor realization and occur once per question.

On resume:

1. Validate the configuration, prepared input identities, and relevant model/software settings. Reject incompatible changes rather than mixing results.
2. Inspect saved prediction records and identify complete, valid successes.
3. Compare their evaluation IDs with the saved evaluation plan.
4. Reload the model, skip completed evaluations, and retry pending or previously failed evaluations. Bound retries and retain the failure history.
5. Persist each completed result before marking that evaluation complete.

Recover an incomplete trailing JSONL write to the last complete record; preserve evidence of recovery and reject other corruption rather than silently discarding it. Use one writer per run and prevent duplicate evaluation IDs from being counted twice. Save the plan and configuration before starting inference.

Resume operates between examples, not inside a forward pass. An interrupted evaluation may need repeating, but successfully persisted evaluations do not. Reloading the model after restart is expected.

Scientific-setting changes create a new run. Analysis-setting changes reuse existing records in a new analysis directory. Failed or missing evaluations remain visible in the completeness report; an incomplete run is not silently presented as a complete comparison.

## Independent backup and cache recovery

No independent backup destination is configured yet. The VM's persistent disk survives resizing but is not an independent backup.

Proposed initial policy:

| Material | Working location | Independent recovery |
| --- | --- | --- |
| Prepared inputs, results, configs, manifests, source records, analyses | Persistent VM storage | Copy to the Mac after each experiment session and verify completeness/checksums |
| Model weights and original dataset download caches | Persistent VM cache | Record exact revisions for re-download |
| Exact pinned model and dataset snapshots, if self-contained recovery is desired | Persistent VM cache | Optional archive to an external drive or separate object storage |

Re-download is a recovery strategy, not a backup; it relies on the source remaining available. Back up prepared question and distractor snapshots even if original download caches are not archived. There is no need to copy temporary cache files or unused model versions.

Choose and record the Mac backup directory when implementing the transfer process. Verify the independent copy before deleting VM resources. Full cache backup and its destination remain optional decisions, not completed setup.

## Execution sequence and validation

1. **Smoke test:** use a few questions to verify loading, scoring, saved records, and analysis.
2. **Pilot:** approximately 20 held-out questions per evaluation dataset across all lengths and three distractor seeds. Keep pilot questions outside the main sample. Verify thinking is disabled, label tokenization, token budgets, prompt reconstruction, memory use, timing, and interrupted-run recovery.
3. **Freeze the design:** finalize sample counts from measured throughput; pin revisions and execution settings; save question IDs, distractor plans, metric conventions, and analysis seeds.
4. **Main study:** run MMLU-Pro and then ARC-Challenge using the same protocol.
5. **CPU analysis:** generate metrics, diagnostics, paired intervals, tables, and figures from saved outputs; back up the results.

Estimate runtime separately at each context length, including model loading and operational overhead. Do not extrapolate 32K timing from short prompts. If the planned workload exceeds the existing provisional GPU allocation, reduce question counts before the main study while preserving the paired sweep and three distractor realizations.

Implementation validation should cover metric calculations on known examples, normalized probability records, exact prompt reconstruction, token-budget enforcement, baseline joins, complete evaluation coverage, and recovery without duplicate successes.

## References

- [Project proposal](../../DLT_topic_pick/mmonto16_Topic_Pick.pdf).
- [Research resource definition](RESOURCE_DEFINITION.md).
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
- [Guo et al., On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).
- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B).
- [MMLU-Pro dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro).
- [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext).
- [ARC dataset](https://huggingface.co/datasets/allenai/ai2_arc).
