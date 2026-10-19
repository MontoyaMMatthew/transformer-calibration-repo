# Azure development and Runpod GPU execution

Decision notes: October 18, 2026, Denver time.

This document records the storage decision and proposed Runpod execution workflow discussed after an Azure A100 allocation failure. No Runpod Pod, volume, installation, or GPU validation has been completed by this documentation change. Earlier Azure validation remains historical evidence, not validation of Runpod.

## Decisions and candidate hardware

- Keep the Azure CPU development workflow; use Runpod GPU Pods as an alternative for validation and experiments when Azure GPU allocation is unavailable.
- Use a **network volume**, starting with **100 GB, Standard storage**. Global volumes were considered and declined.
- An **A40 48 GB** is an acceptable candidate to validate. An A100 80 GB remains the previously tested GPU. Other GPUs require their own compatibility and performance checks.
- A Canadian Runpod data center is acceptable for the current public-data workload, subject to any applicable university or funding requirements. No particular Canadian data center or Pod has been recorded as selected.
- Keep model, tokenizer, dataset revisions, BF16 precision, disabled thinking, scoring, and attention behavior fixed according to [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

The Azure VM's configured resize succeeded, but a subsequent start failed with `AllocationFailed`: insufficient capacity for `Standard_NC24ads_A100_v4` in South Central US. Its deployed configuration had no availability zone, availability set, proximity placement group, dedicated host, or capacity reservation. The pinned DSVM image does not reserve physical A100 hardware. Quota permits a request; it does not guarantee allocation. Azure publishes no reliable best hour to retry.

## Choose storage and location together

A network volume survives Pod termination and normally mounts at `/workspace`. It is tied to one data center: you can select another compatible GPU there, but moving to another data center requires another volume and a data transfer. Network volumes for Pods are currently available in Secure Cloud. Inspect alternative GPU choices in the chosen location before creating the volume.

The 100 GB recommendation is a planning estimate, not measured Runpod usage:

| Contents | Allowance |
| --- | ---: |
| Qwen3-4B weights and tokenizer | 10 GB |
| Python environment and installation caches | 15–25 GB |
| Datasets and prepared inputs | 5–10 GB |
| Results, logs, and temporary files | 10–20 GB |
| Remaining headroom | 35–60 GB |

There is no need to duplicate Azure's full 256 GiB operating-system disk. Runpod provides the container separately. Standard network storage is currently listed at $0.07/GB/month for the first TB, approximately $7/month for 100 GB, excluding GPU and container storage. Volume capacity can increase but cannot decrease. Recheck pricing when deploying; storage remains billable without running GPU compute.

Global volumes would allow switching data centers, but are currently beta and lack file locking, atomic rename, and full POSIX semantics. This project's JSON writers use temporary files followed by replacement, and its validation runner uses file locking. A global volume is therefore not a drop-in working filesystem for this repository.

Sources: [network volumes](https://docs.runpod.io/storage/network-volumes), [storage lifecycle](https://docs.runpod.io/pods/storage/types), [global-volume limitations](https://docs.runpod.io/storage/globalvolume/overview).

## Prepare a Runpod environment

1. Select an on-demand GPU Pod in the volume's data center, with one full A40 48 GB or A100 80 GB GPU. Verify host RAM and free disk space as well as VRAM; GPU identity alone does not specify the complete host. Select a documented CUDA/PyTorch container and record its image tag and, where possible, digest.
2. Attach the network volume during Pod creation. Confirm its actual mount path before copying files. Enable SSH using a public key; do not copy Azure credentials, Terraform state, or private SSH keys to the Pod.
3. Clone the repository under `/workspace/transformer-calibration-repo` and check out the exact development commit. Transfer any required prepared inputs separately: Git excludes caches, artifacts, `.env`, and `.venv`.
4. Recreate the virtual environment; do not copy Azure's `.venv`. The Azure baseline is CPython 3.10.18, PyTorch `2.7.1+cu118`, and Transformers `4.57.6`. The exact package snapshot is [requirements.gpu-validated.txt](../requirements.gpu-validated.txt), with installation provenance in [ENVIRONMENT.md](ENVIRONMENT.md). A Runpod template's preinstalled packages are not proof of a matching project environment.
5. If using a compatible Python 3.10 interpreter, the baseline installation is:

   ```bash
   cd /workspace/transformer-calibration-repo
   python3.10 -m venv .venv
   source .venv/bin/activate
   python -m pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   python -m pip install -r requirements.gpu-validated.txt
   python -m pip check
   ```

   Check the actual interpreter and NVIDIA driver first. If the baseline cannot be recreated, record the alternative environment and validate it separately; do not silently upgrade dependencies or alter the earlier validated snapshot. The host driver comes from Runpod, not the Python environment.

6. Create a Pod-specific ignored `.env` from [.env.example](../.env.example), replacing every placeholder path with the Runpod repository path. Keep the repository, environment, caches, and artifacts on the mounted volume. Activate these settings in each new shell:

   ```bash
   source .venv/bin/activate
   set -a
   source .env
   set +a
   ```

7. Download the pinned resources and regenerate their local manifest:

   ```bash
   python scripts/download_resources.py
   python scripts/verify_environment.py --resources
   python scripts/download_resources.py --offline
   python scripts/verify_environment.py --resources --offline
   ```

   Pins live in [config/resources.json](../config/resources.json). The generated `artifacts/environment/resources.json` contains absolute model and prepared-dataset paths; do not reuse an Azure manifest unchanged. Model/data caches can be transferred deliberately, but manifests must point to the new locations. Keep authentication files out of transferred research artifacts.

## Validate the A40 before experiments

The A40 has 48 GB GDDR6 memory and hardware BF16 support. The earlier A100 batch-one validation reached 32K context with 9.92 GiB peak allocated and 11.15 GiB peak reserved PyTorch memory. This suggests the workload should fit an A40, but is not an A40 measurement. Expect different throughput; compare cost per completed workload rather than hourly price alone. See [NVIDIA's A40 datasheet](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf) and [historical validation results](next_steps/STEP_002.md).

**Required code change before A40 validation:** [scripts/validate_gpu.py](../scripts/validate_gpu.py) currently rejects any GPU whose name does not contain `A100` or whose memory is below approximately 79 GiB. Its description and default output directory also refer to A100. Generalize the hardware expectation explicitly and record the actual GPU; retain the BF16, attention, resource-identity, output-integrity, and recovery checks. This documentation does not implement that change.

After that change:

- Verify one CUDA GPU, the actual GPU model and memory, driver/runtime compatibility, and BF16 support.
- Use a new A40 validation directory; preserve historical A100 reports. Run representative cases across all six context conditions through 32K at batch size one.
- Confirm the intended PyTorch SDPA flash-attention backend runs in every layer; do not accept a silent fallback without assessment.
- Check finite logits/NLL, normalized probabilities, prompt and token-budget correctness, and disabled thinking. Earlier runtime records use the old prompt wording; validate the accepted `Background text:` wording for the pilot.
- Measure peak device memory and throughput, then test interrupted-run recovery and persistent output access from a fresh session.
- Record Git commit, package versions, model/data revisions, GPU, driver, attention backend, and batch settings. Mark results as Runpod/A40 validation, not historical Azure/A100 validation or main-study results.

The scientific pilot and main experiment remain separate work. GPU availability is not authorization to begin them. Follow the existing pilot policy: start at batch one, assess higher batches conservatively, and retain at least 25% observed device-memory headroom. Do not carry an A100 batch-size choice over without measuring it on the replacement GPU.

Prefer one validated GPU type for the main study. If switching is necessary, retain all inference settings, rerun validation, compare a fixed overlap sample for numerical differences, and record hardware per run. Avoid assigning different context conditions exclusively to different GPUs. Do not assume bit-for-bit equality across hardware or kernels.

## Daily workflow and persistence

1. Develop on Azure CPU, run appropriate CPU checks, and commit the code.
2. Launch a compatible Runpod GPU in the network volume's data center; check out that commit and activate the recorded environment.
3. Confirm mounted storage, pins, and GPU validation before starting an authorized experiment. Use `tmux` or another persistent session for long commands; this protects against SSH disconnection, not Pod failure.
4. Save durable per-example results and run manifests. Back up completed results and prepared-input identities to Azure, the Mac, or independent storage, and verify the copy.
5. Release GPU compute when finished. Files on the container disk are disposable; verify authoritative files are actually on the network volume before stopping or terminating. Keep sufficient account balance for storage retention.

Azure's shutdown schedule applies only to the Azure VM. Configure and verify Runpod's own cost controls separately. If returning Azure to `Standard_E8s_v5`, update the local Terraform size setting, review a fresh plan, and apply only after confirming the intended in-place resize; provisioning Runpod does not resize Azure automatically.

## Outstanding work

- Select the exact data center, GPU, Pod template, and host RAM allocation; confirm live pricing and capacity.
- Create and mount the 100 GB Standard network volume and establish SSH access.
- Recreate the environment and regenerate manifests on Runpod.
- Generalize the A100-specific validation guard and output naming, then complete A40 validation.
- Update the environment records and broader infrastructure documentation after the new workflow has actually been verified. Existing Azure completion records remain historical.
