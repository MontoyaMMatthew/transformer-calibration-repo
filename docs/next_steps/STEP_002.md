# Step 002: validate the retained environment on the A100

Created: October 4, 2026. Status: **GPU validation complete October 4, 2026**. All 12 context evaluations, flash-attention verification, and forced-interruption recovery passed with unchanged project packages.

This step implements the early GPU checks in section 6 of `NEXT_STEPS.md`.
It uses the existing DSVM, project venv, immutable resource revisions, and
managed OS disk. It validates the GPU runtime before the full experiment
runner and larger scientific pilot are built.

## Checklist

- [x] Update `NEXT_STEPS.md` from completed Step 001 evidence and owner confirmations.
- [x] Confirm actual `Standard_NC24ads_A100_v4` allocation in South Central US.
- [x] Verify the identical managed OS disk and pinned DSVM image survived resizing.
- [x] Confirm one NVIDIA A100 80 GB PCIe, driver 535.230.02, PyTorch CUDA access, and BF16 tensor operations.
- [x] Locate retained model weights, tokenizer, datasets, CPU reports, and dependency snapshot on persistent storage.
- [x] Prepare held-out validation questions, paired WikiText distractor prefixes, single-token answer completions, and exact context budgets with thinking disabled.
- [x] Load the cached model in BF16 on CUDA and score both validation questions across all six context conditions.
- [x] Prove the memory-efficient attention backend is active and validate finite scores, normalized probabilities, and stable NLL.
- [x] Record synchronized inference time and peak GPU memory for each context length.
- [x] Terminate the validation process after two durable records; resume and verify prior records remain byte-identical and no evaluations are duplicated.
- [x] Save the GPU-validated dependency snapshot and document the observed hardware/software combination.
- [x] Prepare a separate, checksummed GPU-validation archive for the existing Mac backup destination.

## Observed host and access

Azure metadata confirms `Standard_NC24ads_A100_v4` and the same 256 GiB
`StandardSSD_LRS` managed OS disk and image
`microsoft-dsvm:ubuntu-2204:2204-gen2:25.06.18`. Driver 535.230.02 reports
CUDA 12.2 support; the project wheel is PyTorch 2.7.1+cu118 with CUDA 11.8
runtime. Direct device access passes PyTorch CUDA and BF16 checks.

The Codex restricted shell hides `/dev/nvidia*`, so it reports CUDA unavailable.
GPU checks must run with approved direct device access. An ordinary SSH shell
on this VM has direct access. This restriction does not indicate a broken
driver; no driver/toolkit or project-package changes are needed for preflight.

## Validation scope and command

The initial validation ran from `/home/azureuser/transformer-calibration-repo` on the GPU VM with:

```bash
source .venv/bin/activate
set -a
source .env
set +a
python scripts/validate_gpu.py --recovery-check
```

The recovery check requires a fresh run directory. The default is
`artifacts/gpu_validation/a100-initial`, which now contains the completed run. To use another directory:

```bash
python scripts/validate_gpu.py --run-dir artifacts/gpu_validation/another-check --recovery-check
```

Resume an interrupted validation with:

```bash
python scripts/validate_gpu.py --run-dir artifacts/gpu_validation/a100-initial --resume
```

The test uses one MMLU-Pro validation question and one ARC-Challenge validation
question. The 12 planned evaluations cover no distractor, 2K, 4K, 8K, 16K,
and 32K for both questions. WikiText-103 raw training text is sampled with seed
11 and a stable identity derived from dataset/question/seed. Each question uses
progressively longer prefixes of the same saved source stream. This is a small
runtime validation, not the proposed 20-question-per-dataset scientific pilot
or a main evaluation sample.

`config/gpu_validation.json` fixes BF16, CUDA, batch size one, evaluation mode,
no gradients, thinking disabled, no quantization, no sampling, no KV caching,
and answer-position-only vocabulary logits. Choice scores and log-softmax are
computed in FP32. The answer prefix is `Answer:` with completions such as ` A`.
Complete prompts are re-tokenized; each nonzero input is budget minus one,
reserving one answer token. At 32K this means 32,767 input tokens and one reserved
answer token. The entire question and choices remain intact, and YaRN is disabled.

## Attention, timing, and recovery evidence

The supported [PyTorch SDPA backend selector](https://docs.pytorch.org/docs/2.7/generated/torch.nn.attention.sdpa_kernel.html)
forces `SDPBackend.FLASH_ATTENTION` for every forward pass. A separate 2K
profiler check requires a flash-attention operator for every model layer.
The separate `flash_attn` package is not installed. Math-attention fallback is
disabled during these forwards.

Timing includes the synchronized model forward and choice scoring, after
inputs are transferred to the GPU. Model loading, prompt preparation, warm-up,
and the separate profiler pass are excluded. Peak allocated and reserved CUDA
memory include resident model weights; these are PyTorch allocator measures,
not the entire NVIDIA process footprint. Batch size one and disabled KV caching
are part of the tested configuration.

The immutable plan saves complete validation prompts, token counts, prompt
hashes, label tokens, dataset revisions, source-stream identities, execution
settings, software versions, and source-file hashes before inference. A writer
lock prevents concurrent writers. Each successful result is atomically replaced
and flushed to persistent storage. Resume rejects incompatible plan/settings
or source changes and skips valid completed evaluations.

The recovery test sends SIGKILL only to its own validation subprocess after
two complete records are flushed. It then launches a fresh process to reload
the cached model and finish the remaining evaluations. The two prior record
files must retain their exact checksums, with complete coverage and zero
duplicates. This proves recovery for the validation harness; the full research
runner still needs its own resume implementation and tests.

## Outputs and follow-up

Reports and durable records are ignored under
`artifacts/gpu_validation/a100-initial/`. Host/device preflight records are in
`artifacts/environment/gpu-host.json` and `gpu-preflight.json`.

The owner has already verified the Step 001 archive on the Mac. New GPU records
will need their own independent copy at
`/Volumes/X9/projects/transformer-calibration-backups`. The verified CPU archive
is preserved. The local infrastructure checkout should also record the A100
size so a later Terraform apply agrees with the actual VM; this remote checkout
has no personal Terraform configuration or Azure login.

## Measured results (October 4, 2026)

| Context | Actual input tokens | Forward + scoring (seconds) | Peak allocated (GiB) |
| --- | ---: | ---: | ---: |
| 0K | 100 / 131 | 0.037–0.045 | 7.51 |
| 2K | 2047 | 0.116–0.122 | 7.65 |
| 4K | 4095 | 0.239–0.240 | 7.80 |
| 8K | 8191 | 0.524–0.525 | 8.11 |
| 16K | 16383 | 1.254–1.260 | 8.71 |
| 32K | 32767 | 3.373–3.379 | 9.92 |

Each range contains two validation forwards, one per dataset, with batch size
one. These timings are runtime checks rather than study throughput estimates.
All 12 records contain finite logits and NLL and valid normalized probabilities.
Every nonzero condition has zero target shortfall, including exactly 32,767
input tokens plus one reserved token at 32K. Peak reserved memory at 32K was
11.15 GiB; peak allocated memory was 9.92 GiB. Memory excludes driver/context
allocations outside the PyTorch allocator and includes resident model weights.

The profiler observed 36 `aten::_scaled_dot_product_flash_attention` operators,
one per model layer. The forced-kill test retained two byte-identical records
and resumed ten pending evaluations, completing 12 unique evaluations with zero
duplicates. Cached model reload after interruption took 1.536 seconds; model
loading is excluded from the table. The first process's loading progress is
preserved in `interrupted-process.log`.

`requirements.gpu-validated.txt` contains the same 77 exact distributions as
the CPU snapshot. No packages, driver, or CUDA toolkit were changed. The GPU
and BF16/native-context path are now validated for this exact batch-one direct
scoring configuration. Larger batches, quantization, KV-cached generation,
other attention backends, and the full scientific pilot were not tested.

## Copy the new GPU records to the Mac

The independent CPU backup remains verified. The new GPU archive is prepared
on the VM, with every member and the archive checksum verified locally. Its
independent Mac copy has **not yet been confirmed**. Run on the Mac, substituting
your existing SSH host and adding your usual key option to `scp` if needed:

```bash
cd /Volumes/X9/projects/transformer-calibration-backups
scp azureuser@YOUR_EXISTING_VM_SSH_HOST:/home/azureuser/transformer-calibration-repo/artifacts/environment/step-002-gpu.tar.gz .
scp azureuser@YOUR_EXISTING_VM_SSH_HOST:/home/azureuser/transformer-calibration-repo/artifacts/environment/step-002-gpu.tar.gz.sha256 .
shasum -a 256 -c step-002-gpu.tar.gz.sha256
mkdir -p step-002-gpu
tar -xzf step-002-gpu.tar.gz -C step-002-gpu
cd step-002-gpu
shasum -a 256 -c SHA256SUMS
```

The archive includes the plan, source-stream audit data, all 12 result records,
GPU reports, recovery proof, dependency snapshot, scripts, and documentation.
Model weights, bulk datasets, venvs, `.env`, and authentication state are excluded.
