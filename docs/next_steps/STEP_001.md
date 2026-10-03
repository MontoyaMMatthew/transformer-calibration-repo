# Step 001: establish the CPU development environment

Created: October 3, 2026.

Status: remote CPU work validated; **step remains open** until the owner verifies the independent Mac backup and current shutdown schedule. See [manual instructions](STEP_001_MANUAL.md).

## Starting point and objective

The Azure VM has been provisioned, as confirmed by the project owner. Initial development will use `Standard_E8s_v5` (8 vCPUs, 64 GiB RAM). CPU environment setup, recreation, resource preparation, and online/offline FP32 validation passed October 3, 2026. The independent Mac backup and live shutdown setting still require owner action.

Finish this step with a reproducible Python environment, persistent resource caches, and a small CPU validation result. Follow [ENVIRONMENT.md](../ENVIRONMENT.md) for setup decisions and record actual versions there. The scientific design remains in [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md).

Older provisioning checklists in the repository may still describe the VM as undeployed. Do not provision another VM based on those historical entries.

## Scope

This step covers connecting to the existing VM, verifying its storage and runtime, creating an isolated Python environment, recording dependencies, preparing resource downloads, and checking short-context inference on CPU.

Defer the full experiment CLI, main dataset sampling, distractor sweeps, analysis pipeline, GPU resize, 32K inference, and final experiments to later steps. Do not change the research design to accommodate a development smoke test.

## Ordered work

### 1. Inspect the existing VM

- [x] Connect using the existing SSH configuration and place or locate the repository on persistent storage.
- [x] Record the checkout path and Git revision, OS, architecture, CPU count, memory, Python interpreters, disk mounts, and free space.
- [x] Verify the checkout, future `.venv`, caches, and artifacts are on the managed disk, not temporary storage.
- [ ] Confirm the configured shutdown schedule so downloads and checks can complete.

Deliverable: the observed-host section of `docs/ENVIRONMENT.md`, with paths and facts verified on the VM. No deployment or image replacement is needed.

### 2. Create and record the Python environment

- [x] Select an installed Python interpreter supported by the chosen stable PyTorch and Transformers versions; record its exact version.
- [x] Create a project `.venv` without inheriting system packages. Keep the DSVM's system Python and driver installation intact.
- [x] Select a CUDA-enabled PyTorch wheel that can execute on CPU now; document its version and wheel source. GPU compatibility remains unverified until an A100 check.
- [x] Install the minimal dependencies listed in `docs/ENVIRONMENT.md` and verify imports and package consistency.
- [x] Add `requirements.in` for direct dependencies and `requirements.cpu-validated.txt` for the exact resolved environment. Record the PyTorch index/build separately so the installation is reproducible.
- [x] Record the commands used to install the environment. Verify recreation in a separate temporary virtual environment before declaring this step complete.

Deliverable: a CPU-validated dependency snapshot and installation instructions. This is not yet the final GPU-validated research environment.

### 3. Set persistent paths and add small setup scripts

- [x] Configure cache and artifact paths according to `docs/ENVIRONMENT.md`; add a tracked `.env.example` containing non-secret settings, and use an ignored `.env` for machine-specific values.
- [x] Add `scripts/verify_environment.py` to report interpreter/package versions, CPU/CUDA availability, paths, and basic tensor checks to `artifacts/environment/`.
- [x] Add `scripts/download_resources.py` to download the selected model/tokenizer and datasets at explicit revisions into the configured cache and save a resource manifest.
- [x] Keep download and verification scripts independently runnable; do not build the full experiment runner in this step.

Deliverable: two focused setup scripts, local path configuration, and a machine-readable environment report. Do not include credentials in reports.

### 4. Validate resources and a tiny CPU inference

- [x] Pin and record model/tokenizer and dataset revisions, configurations, and splits before downloading.
- [x] Verify that the three datasets load and that a few examples expose the expected question, choices, gold label, or distractor text fields.
- [x] Load the pinned tokenizer, render a non-thinking prompt, and verify single-token answer labels in the exact answer-prefix context.
- [x] Load Qwen3-4B in FP32 on CPU and score one or two short prompts, batch size one, with gradient tracking disabled.
- [x] Verify finite answer-choice scores, normalized probabilities, and stable NLL calculation. Save a small report under `artifacts/environment/`, clearly marked as development validation.
- [x] Repeat resource loading from cache with network access disabled at the library level to verify the required files are locally available.

Deliverable: a successful short-context CPU validation. CPU runtime is not an estimate of GPU performance, and these outputs are not paper results. No long-context sweep is required.

### 5. Preserve the setup and close the step

- [x] Select and record an independent backup directory on the Mac.
- [ ] Copy the environment report, resource manifest, and relevant prepared artifacts there and verify the transferred files.
- [x] Confirm `.venv`, cache, `.env`, and generated artifacts remain ignored by Git; commit only source scripts, dependency specifications, examples, and documentation when committing is requested.
- [x] Update `docs/ENVIRONMENT.md` with actual commands, versions, validation date, limitations, and backup location.

Deliverable: documented recovery instructions and one verified independent copy of the setup records. Model-weight backup is optional; pinned re-download is the initial recovery strategy.

## Completion criteria

This step is complete when all of the following hold:

1. The repository, environment, and caches have verified persistent locations.
2. A fresh shell can activate the environment and load the same path settings.
3. Dependencies can be recreated from the recorded specifications and wheel sources.
4. Imports, package consistency checks, resource loading, and short CPU choice scoring pass.
5. Environment and resource reports are saved and independently copied off the VM.
6. Documentation explicitly labels CUDA execution and the A100/32K path as unverified.

## Next scope after completion

Plan Step 002 around a small end-to-end CPU development slice: configuration validation, normalized questions, prompt construction, one saved inference record, resume identity, and metrics from saved output. Schedule A100 validation before substantial development depends on long-context behavior. Define that step after the environment findings are available.
