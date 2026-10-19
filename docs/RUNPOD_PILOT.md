# A40 pilot preparation and launch

Prepared October 18, 2026 (Denver). **The scientific pilot and long-context batch
screening have not started. No changes have been committed.** Review and commit
the source changes before executing the launch commands below.

## Current machine and persistent paths

The actual checkout is
`/workspace/transformer-calibration/transformer-calibration-repo`, already on
`dev` (starting commit `d1fa0c7`). `/workspace` is a RunPod Standard network
volume, mounted from `ca-mtl-1-vast.runpod.net`; it has approximately 94 GiB
capacity. About 83 GiB remained after installation and resource downloads.

The Pod exposes one NVIDIA A40: 47,697,690,624 usable bytes (44.42 GiB), compute
capability 8.6, driver 580.173.02. The container is Ubuntu 24.04.3 with Python
3.12.3 and preinstalled PyTorch 2.8.0+cu128 / CUDA runtime 12.8. The host reports
503 GiB RAM; this is shared host information, not a guaranteed Pod reservation.
The exact template image tag/digest was not available in the inspected records.

The simplest working environment reuses the template's PyTorch in
`.venv` created with `--system-site-packages`. Research packages are installed
only in this persistent venv, with Transformers 4.57.6 and Datasets 4.8.5.
`requirements.runpod-overlay.txt` pins its resolved overlay; the full inventory
including inherited packages is `artifacts/environment/runpod-packages.txt`.
The Azure CPU/GPU dependency snapshots are preserved unchanged.

All generated research files are under the checkout on `/workspace`:

| Material | Location relative to checkout |
| --- | --- |
| Python environment | `.venv/` |
| Model/tokenizer and datasets | `cache/huggingface/` |
| Package, plotting, compiler and temporary caches | `cache/` |
| Environment and verification reports | `artifacts/environment/` |
| Separate 12-case A40 runtime validation | `artifacts/gpu_validation/a40-initial/` |
| CPU-only pilot preview | `artifacts/prepared/pilot-a40-review/` |
| Future pilot and analyses | `artifacts/runs/`, `artifacts/analyses/` |

Ignored `.env` supplies absolute paths, including `TMPDIR`, Torch/Triton/CUDA
caches and the `/workspace` output restriction. It disables the template's
inherited `HF_HUB_ENABLE_HF_TRANSFER=1`, which otherwise fails because the
optional accelerator is absent. `scripts/runpod` loads these settings and
executes the persistent venv's Python from the repository root.

On an identical fresh template, recreate the environment with
`bash scripts/setup_runpod.sh`. This checks the Python/PyTorch base rather than
silently installing a different CUDA stack. The persistent overlay still
depends on that compatible container base after Pod replacement.

## Checks completed before the pilot

- Imports, package consistency, CUDA and hardware BF16 checks passed.
- All pinned model/data resources downloaded and passed offline reconstruction
  and resource verification on this volume.
- The historical runtime-validation prompt was preserved. Its A40 run scored
  both development questions across all six lengths (12 records). At 32K,
  synchronized forward/scoring took 6.708 and 6.732 seconds; peak allocation
  was approximately 9.92 GiB. These timings describe two validation inputs,
  not pilot throughput.
- The profiler observed flash SDPA in all 36 model layers. A forced SIGKILL
  after two durable results resumed to 12 unique records; the prior two
  remained byte-identical. See `recovery-check.json` in that validation run.
- The new scorer passed a separate short **synthetic arithmetic prompt** at
  batches 1, 2 and 4, including probability/prediction comparison and the
  neutral `Background text:` heading. This does not certify larger batches
  at 32K. The scorer passes an explicit all-ones mask to avoid Transformers'
  packed-sequence path and the incompatible 4D mask it creates.
- Tests exercise known metric values and bin endpoints, complete saved-record
  analysis and PNG/PDF generation, paired bootstrap behavior, exclusion of
  incomplete questions, corruption rejection, writer locking, forced-kill
  durability, resume skipping, and real-tokenizer reconstruction through 32K.

The CPU preview checks every pilot prompt and seals its inputs without loading
model weights or making pilot predictions. Review `selection.json` and
`audit_prompts/` there for background quality and accidental helpful overlap.
It is a review artifact from the uncommitted source state. Prepare a fresh run
after committing so its manifest records the committed revision.

The validator now defaults to A40 and the configured artifact root. To reproduce
the earlier A100 check on that hardware, pass `--expected-gpu A100
--minimum-vram-gib 79` and a new run directory explicitly; old reports are never
overwritten by the A40 check.

## Implemented pilot

`config/pilot.json` follows the experiment plan and LaTeX method: 20 validation
questions per dataset, proportional MMLU-Pro subject allocation, selection seed
42, distractor seeds 11/22/33, and total budgets 0/2048/4096/8192/16384/32768.
Eligibility is checked before sampling. No question or choice is truncated.
There are **640 evaluations**: 40 questions × (one baseline + 15 backgrounds).

The default sweep remains batch one. The model uses BF16, FP32 choice
log-softmax, disabled thinking, native context, no quantization or KV caching,
and forced flash SDPA. Each input reserves one answer token, and each actual
label completion must append exactly one distinct token. Distractor prefixes
share a saved question/seed stream across lengths.

`src.benchmark_batches` assesses 1, 2 and 4 in separate processes, starting with
repeated 32K cases from both datasets and checking all other lengths. A larger
candidate is skipped if a conservative estimate would cross 75% of device
memory. Reports include sampled device usage, peak allocator reservation,
non-allocator overhead, headroom, throughput, numerical differences and
prediction changes. The probability tolerance is 0.005 absolute with identical
predictions required. Exact-length buckets avoid padding and preserve each
answer position. A passing candidate is evidence to review before choosing
the main-study batch policy; it does not automatically change the pilot.

## Launch only after review and commit

Run from `/workspace/transformer-calibration/transformer-calibration-repo`.
These commands are documented for the next step; they have **not** been run.

```bash
# Seal fresh inputs under the committed code; still no inference.
scripts/runpod -m src.run_experiments --config config/pilot.json \
  --run-dir artifacts/runs/pilot-a40 --prepare-only

# This begins pilot batch screening. It writes separate candidate logs/results.
scripts/runpod -m src.benchmark_batches --run artifacts/runs/pilot-a40 \
  --output artifacts/benchmarks/pilot-a40

# Start the batch-one scientific sweep (or resume after interruption).
scripts/runpod -m src.run_experiments --resume artifacts/runs/pilot-a40

# Recompute all metrics and figures on CPU from saved results.
scripts/runpod -m src.analyze_results --run artifacts/runs/pilot-a40 \
  --config config/analysis.json --output artifacts/analyses/pilot-a40-v1
```

Use a persistent terminal session for execution. Resume keeps the saved
question/distractor plan, validates inputs and score records, and rejects
changed code, packages, pins or hardware. Each attempt is recorded before
execution; an evaluation has at most three attempts (initial plus two retries).
OOM or insufficient device headroom stops the run without silently reducing
batch size. A partial result stays visible in completeness reporting.

## Saved data and analysis coverage

Each run contains the resolved config, pinned resource manifest, sample and
exclusion report, normalized original question rows, saved background text and
source row/character offsets, evaluation plan and hashes, exact label token
mapping, audit prompts, code/dependency snapshots, software/hardware records,
attention proof, execution logs and attempt history. Successful evaluations
are separate atomic JSON files, flushed with their directory to disk. This
avoids partially appended JSONL records. A one-writer lock protects each run.

Each result preserves all valid-choice logits, stable log-probabilities and
probabilities, gold and predicted labels, actual token counts, seed, prompt
hash, synchronized timing, batch IDs and member evaluation IDs, and
device-memory measurements. Original all-token vocabulary logits are not
needed for the planned analysis and are not stored. Prompts reconstruct from
saved question/background snapshots and the pinned tokenizer; seeds alone
are not the reconstruction record.

CPU analysis writes:

- `metrics_by_seed.csv`, `metrics_summary.csv`, `per_example_metrics.csv`:
  accuracy, confidence, cross-entropy/NLL, ECE, Brier, raw/normalized entropy
  and signed confidence gap. ECE sensitivity uses 10/15/20 bins.
- `baseline_deltas.csv`: paired changes and percentile 95% intervals from
  2,000 question-level resamples, preserving MMLU-Pro subject strata and
  sharing the resampled IDs across all contexts and seeds. Nonzero-condition
  summaries average seed-specific metrics, never probability vectors.
- `calibration_bins.csv`, `diagnostics.csv`: bin counts/accuracy/confidence,
  confidence on errors with denominators, prediction changes and correctness
  transitions. Undefined confidence on errors is blank rather than zero.
- `completeness_report.json`: every missing evaluation and excluded question.
  Primary analysis uses the same fully observed questions at every condition.
- `timing_memory.csv`: measured time and memory per length, plus provisional
  forward-only main-workload timing; loading/operational overhead is separate.
- Per-dataset PNG and PDF figures: accuracy/confidence, ECE/NLL/Brier,
  reliability diagrams at 0K/8K/32K with counts, all metric baseline deltas,
  and timing/memory. Figures are marked as pilot results.
- Analysis settings, source-run/record checksums, and output checksums.

You can change plot style, ECE conventions or diagnostics and produce a new
analysis directory from the same scores. Preserve the complete run directory,
not only summary tables. The main-study runner/sample freeze remains later
work; this CLI deliberately requires the validation split for the pilot.

## Backup after execution

```bash
scripts/runpod scripts/package_run.py \
  --include artifacts/runs/pilot-a40 artifacts/analyses/pilot-a40-v1 \
            artifacts/benchmarks/pilot-a40 \
  --output artifacts/backups/pilot-a40.tar.gz
```

The archive has a verified member-checksum index and `.sha256` sidecar. Copy
all three to the independent Mac backup destination, then verify the archive
checksum there. The archive includes saved source/question/background records
but excludes model weights and bulk caches, whose immutable revisions support
re-download. Persistent RunPod storage is the working copy; an independent
transfer has not been performed by this setup.
