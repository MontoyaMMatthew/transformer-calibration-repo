# Development and experiment environment

Created: October 3, 2026. Status: Step 001 CPU setup complete; Step 002 GPU validation complete October 4, 2026. The existing environment passed A100 BF16 scoring across all six context conditions through 32K and forced-interruption recovery. The independent CPU backup is verified; new GPU records are prepared for a separate Mac copy.

The Azure VM is already provisioned. Initial CPU development used `Standard_E8s_v5`; the active VM is now `Standard_NC24ads_A100_v4`. CPU setup is recorded in [Step 001](next_steps/STEP_001.md) and the completed GPU validation in [Step 002](next_steps/STEP_002.md). Experiment behavior is defined in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md); infrastructure details are in [RESOURCE_DEFINITION.md](RESOURCE_DEFINITION.md).

## Environment policy

- Develop on the existing CPU VM using a project-local virtual environment.
- Keep the repository, Python environment, model and dataset caches, and artifacts on persistent managed storage.
- Preserve the DSVM system environment. Do not install project packages into system Python or install drivers as routine CPU setup.
- Use a CUDA-enabled PyTorch distribution that also runs CPU operations, so the same environment can later be validated on the A100. Its installed CUDA runtime does not prove GPU compatibility.
- Validate and pin the CPU environment first. After A100 validation, record a separate GPU-validated dependency snapshot and any changes required.
- Avoid automatic dependency upgrades during experiments. Changes to Python, PyTorch, Transformers, drivers, or attention backends require relevant revalidation.

## Intended host and observed configuration

The image and storage below are intended settings from the resource definition, not observations from this session. Fill the observed column during setup.

| Item | Intended setting | Observed value / verification |
| --- | --- | --- |
| Provisioning | Existing Azure VM | Verified via Azure instance metadata October 3, 2026 |
| Initial size | `Standard_E8s_v5`, 8 vCPUs, 64 GiB RAM | Verified `Standard_E8s_v5`; 8 CPUs, 62.8 GiB OS-visible RAM |
| OS/image | Ubuntu 22.04 DSVM; planned image `25.06.18` | Ubuntu 22.04.5; Azure metadata confirms `microsoft-dsvm:ubuntu-2204:2204-gen2:25.06.18` |
| Architecture | x86-64 | Verified x86_64 |
| Region | South Central US | Metadata confirms `southcentralus` |
| Persistent disk | Planned 256 GiB managed SSD | 256 GiB `StandardSSD_LRS` managed OS disk; `/dev/sda1` ext4 root, 172 GiB free before setup; no Azure resource disk |
| Repository path | Existing checkout on persistent disk | `/home/azureuser/transformer-calibration-repo`; managed OS disk verified against metadata and mounts |
| Python executable/version | Installed interpreter compatible with selected packages | `/anaconda/envs/azureml_py310_sdkv2/bin/python`, CPython 3.10.18 |
| Virtual environment | `<repo>/.venv` | `/home/azureuser/transformer-calibration-repo/.venv`; `include-system-site-packages = false` |
| PyTorch version/build/index | Stable CUDA-enabled wheel; CPU validated first | `2.7.1+cu118`, official `https://download.pytorch.org/whl/cu118`; CUDA 11.8 bundled runtime |
| Transformers version | Stable version supporting Qwen3 | `4.57.6`; Qwen3 imports, tokenizer, FP32 inference passed |
| Cache location | `<repo>/cache/huggingface` | `/home/azureuser/transformer-calibration-repo/cache/huggingface` on managed OS disk |
| Artifact location | `<repo>/artifacts` | `/home/azureuser/transformer-calibration-repo/artifacts` on managed OS disk |
| Shutdown schedule | Inspect current Azure setting | Owner confirmed daily 23:00, reported as MST, October 3, 2026. Repository time-zone default is Azure `Mountain Standard Time` (Denver); the exact deployed zone ID was not read by the agent |
| Independent backup | Directory on the Mac, separate from VM | `/Volumes/X9/projects/transformer-calibration-backups`; owner confirmed copy and verification October 3, 2026 |
| CPU validation date | After Step 001 checks pass | 2026-10-03 UTC; imports, consistency, schema/tokenizer, online/offline FP32 choice scoring passed |
| GPU validation | A100 BF16/native context through 32K | Passed 2026-10-04: two validation questions across all six conditions; forced flash attention and interruption recovery passed |
| Current VM size | `Standard_NC24ads_A100_v4` | Verified in metadata; 24 CPUs, approximately 216.3 GiB OS-visible RAM, one A100 80 GB PCIe |
| GPU dependency snapshot | Exact validated package versions | `requirements.gpu-validated.txt`; same 77 distributions as CPU setup |

## 1. Inspect the host and storage

Run the following on the VM, from the existing repository checkout:

```bash
pwd
git rev-parse HEAD
cat /etc/os-release
uname -m
lscpu
free -h
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS
findmnt -T "$PWD"
df -h "$PWD"
command -v python3
python3 --version
```

Verify the backing disk against the deployed Azure disk configuration; a path under a home directory alone is not proof of persistence. Record available disk space before downloading model and dataset snapshots. Do not place authoritative files on a temporary mount such as `/mnt` without checking its backing storage.

No GPU is expected on E8s_v5. An unavailable `nvidia-smi` device or `torch.cuda.is_available() == False` is not a CPU setup failure. Any installed NVIDIA driver version can be recorded, but operational GPU checks wait until resize.

## 2. Create an isolated Python environment

Inspect available interpreters and select one supported by the package versions chosen at setup time. Prefer a compatible installed interpreter; do not replace the OS Python merely to select a newer release. Record its exact path and version.

From the repository root, the following uses `python3` if it is the selected interpreter:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -c 'import sys; print(sys.executable); print(sys.version)'
python -m pip --version
```

If the interpreter lacks `venv` support, install only the matching OS support package during setup. Do not use `--system-site-packages`. Virtual environments are recreated from specifications rather than copied between machines. See the [Python venv documentation](https://docs.python.org/3/library/venv.html).

## 3. Select and record dependencies

| Dependency | Purpose |
| --- | --- |
| `torch` | CPU development and later GPU inference |
| `transformers` | Qwen3 model and tokenizer |
| `datasets`, `huggingface_hub` | Dataset loading and pinned downloads |
| `safetensors`, `accelerate` | Model loading support |
| `numpy`, `pandas`, `scipy` | Numerical analysis and result tables |
| `matplotlib` | Paper figures |
| `PyYAML` | Experiment configuration |
| `pytest` | Focused implementation checks |

Use standard-library logging. Jupyter is optional. Do not add compiled FlashAttention, a tracking service, or a distributed inference framework for initial setup; begin with supported PyTorch attention.

Select the PyTorch installation command from the [official installation selector](https://pytorch.org/get-started/locally/), choosing Linux, pip, and a CUDA build compatible with the intended driver stack. Install only the required `torch` package; vision and audio packages are not needed for this project. Verify Python and Transformers compatibility before resolving the remaining packages. Record the exact command and wheel index used; do not assume the DSVM's preinstalled stack is the project environment.

The CUDA build can execute CPU tensor operations on E8s_v5. Driver/runtime compatibility and A100 execution remain provisional until tested on that GPU. Do not install a separate CUDA toolkit just to run prebuilt wheels without an identified requirement.

Create these files during implementation:

- `requirements.in`: direct dependencies and deliberate version constraints.
- `requirements.cpu-validated.txt`: exact resolved package versions after CPU checks pass.
- Later, `requirements.gpu-validated.txt`: exact environment after A100 validation.

Record installation sources and Python version alongside the snapshot. A `pip freeze` file is a version snapshot, not a complete specification of wheel provenance. Check it for local paths, editable installs, or private URLs before tracking it.

After installation, run:

```bash
python -m pip check
python -c 'import torch, transformers, datasets, huggingface_hub, numpy, pandas, scipy, matplotlib, yaml; print("Imports passed"); print("torch", torch.__version__); print("transformers", transformers.__version__); print("CUDA build", torch.version.cuda); print("CUDA available", torch.cuda.is_available()); print(torch.ones(2, device="cpu") + 1)'
```

Recreate the environment in a separate temporary virtual environment using the documented Python version, wheel source, and dependency snapshot. Repeat imports and `pip check` before declaring the dependency setup reproducible. Do not overwrite the working environment to test recreation.

## 4. Configure persistent paths

Use these initial locations under the verified persistent repository root:

```text
.venv/                       Python environment
cache/huggingface/hub/        Downloaded model and dataset source snapshots
cache/huggingface/datasets/   Prepared datasets cache
artifacts/environment/       Setup reports and resource manifest
artifacts/prepared/          Frozen experimental inputs, later
artifacts/runs/              Inference outputs, later
artifacts/analyses/          Tables and plots, later
```

For the current shell, from the repository root:

```bash
export HF_HOME="$PWD/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMER_CALIBRATION_ARTIFACTS="$PWD/artifacts"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMER_CALIBRATION_ARTIFACTS/environment"
```

Set these before importing Hugging Face libraries. `HF_HOME` can also contain authentication state; do not indiscriminately back up its entire contents into research artifacts. See the [Hugging Face environment-variable documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

During implementation, add `.env.example` with placeholder absolute paths and save actual machine paths in the ignored `.env`. Make it shell-compatible and explicitly source it; Python and virtual-environment activation do not automatically read `.env`. The project-specific artifact variable must be read by the setup scripts and future runner.

Fresh-shell workflow, from the repository root after `.env` exists:

```bash
source .venv/bin/activate
set -a
source .env
set +a
```

The existing `.gitignore` excludes `.venv/`, `cache/`, `artifacts/`, and `.env`, while allowing `.env.example`. Verify generated files remain ignored. Keep credentials out of tracked examples and environment reports.

## 5. Download and validate pinned resources

The planned download script must record immutable revision IDs, resource names, dataset configurations/splits, and cache locations in `artifacts/environment/resources.json`.

| Resource | Selection |
| --- | --- |
| Model/tokenizer | `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c` |
| Primary questions | `TIGER-Lab/MMLU-Pro`, test; separate development examples for checks |
| Distractors | `Salesforce/wikitext`, `wikitext-103-raw-v1`, train |
| Verification questions | `allenai/ai2_arc`, `ARC-Challenge`, test; separate development examples for checks |

Use development/validation questions for setup checks where available and record their IDs. Do not select or tune on the main evaluation sample in this step. Full model weights are needed for the final CPU smoke test, but no full-dataset inference is needed.

The smoke test uses explicit `device=cpu`, FP32, batch size one, evaluation mode, no gradients, disabled thinking, and one or two short prompts. Check finite choice logits, probabilities summing to one, correct gold-label mapping, and finite NLL. Label its output as development validation.

Once resources are cached, verify loading with Hugging Face offline settings and local-only model/tokenizer loading. Record whether dataset preparation also works offline with the selected library version. Never claim offline recovery is verified based only on files existing in a cache.

## 6. Reports and independent backup

Save a report under `artifacts/environment/` containing the validation date, Git revision, Python/package versions, wheel source, CPU information, persistent paths, resource revisions, check outcomes, and limitations. Do not dump the entire process environment: it may contain credentials.

Select an absolute backup path on the Mac and enter it in the observed-configuration table. Copy the setup report, resource manifest, dependency specifications, and any prepared inputs there; verify checksums or file contents. This independent copy protects against VM deletion. Persistent VM storage by itself is not a backup.

Keep model weights cached on the VM initially and preserve exact revisions for re-download. Re-download depends on upstream availability and is not an independent backup. A self-contained archive of pinned model/dataset snapshots on an external drive or object storage is optional. Exclude Hugging Face authentication files from such archives.

## 7. GPU validation (completed October 4, 2026)

The following checks were completed on `Standard_NC24ads_A100_v4` in Step 002 before relying on long-context experiments:

- Verify actual GPU identity, driver, CUDA availability, and BF16 execution.
- Verify persistent files survived the resize and cached resources load without another download.
- Run representative prompts through 32K with the fixed scoring method and memory-efficient attention.
- Record memory, timings, package changes, and the final attention backend.
- Create `requirements.gpu-validated.txt` after the checks pass. Completed October 4 with unchanged package versions.

FP32 CPU outputs are development checks, not interchangeable with BF16 GPU study results. Do not silently fall back to CPU when a future experiment configuration requests CUDA.

## Setup history

| Date | Action | Outcome |
| --- | --- | --- |
| 2026-10-03 | Created environment plan; owner confirmed VM provisioned and E8s_v5 development target | Initial plan, followed by remote validation below |
| 2026-10-03 | Inspected all bundled environments; verified host and managed storage; isolated Python setup; pinned resources; CPU checks; recreated dependencies | Online/offline CPU scoring passed; all 77 recreated package versions match; independent Mac copy and live shutdown confirmation pending |
| 2026-10-03 | Owner confirmed live daily shutdown at 11pm MST and independent Mac backup copy/verification | Step 001 complete; GPU checks deferred at that time |
| 2026-10-04 | Verified A100 allocation and retained disk/cache; ran BF16 context checks and forced-kill recovery | All 12 cases passed through 32K; flash attention in all 36 layers; packages and driver unchanged; new GPU backup copy pending |

Append actual setup commands, selected versions, validation evidence, and subsequent environment changes here as work proceeds.

## Remote setup record (October 3, 2026)

The clean checkout initially pointed at `main` (`ded8b25`). The requested
Step 001 and environment plan existed on `origin/dev` (`2b60f9c`), a descendant
of main. Setup uses local branch `codex/cpu-environment` based on that revision.
The initial setup task left source changes uncommitted and included them in
the independent-backup archive. The owner subsequently committed and pushed
the setup to `origin/dev` as `378be6a` (`S001 env,dwnld scripts,inference smoke`).

### Why the bundled environments are insufficient

The active `py38_default` environment actually contains Python 3.10.11 and
PyTorch 2.5.1, but no Transformers or Datasets. `azureml_py38` and
`azureml_py38_PT_and_TF` contain Transformers 4.48.0, which predates Qwen3
support (the [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B) requires
Transformers 4.51.0 or newer). The latter has PyTorch 2.7.1, but neither is the
isolated, reproducible project environment required here. Full observed
package inventories are in `artifacts/environment/host-inventory.json`.

The selected installed interpreter is CPython 3.10.18 in
`/anaconda/envs/azureml_py310_sdkv2/bin/python`. Its standard-library venv
support works without any OS package installation. The OS interpreter's
`ensurepip` support is absent, so it was not selected. The project venv does
not inherit the DSVM's Python packages.

### Installation and fresh-shell commands

Run from `/home/azureuser/transformer-calibration-repo`:

```bash
/anaconda/envs/azureml_py310_sdkv2/bin/python -m venv .venv
source .venv/bin/activate
set -a
source .env
set +a
python -m pip install --cache-dir cache/pip pip==25.3
python -m pip install --cache-dir cache/pip torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118 --report artifacts/environment/torch-install.json
python -m pip install --cache-dir cache/pip -r requirements.in --report artifacts/environment/dependencies-install.json
python scripts/verify_environment.py
python scripts/download_resources.py
python scripts/verify_environment.py --cpu-smoke
python scripts/download_resources.py --offline
python scripts/verify_environment.py --cpu-smoke --offline
```

On a new checkout, create `.env` from `.env.example` and enter the absolute
persistent paths before sourcing it. The scripts also use repository-local
persistent defaults when those variables are unset. The installed pip 23.0.1
failed to normalize `typing_extensions` wheel metadata from the PyTorch index;
upgrading **only the venv's pip** to 25.3 resolved the failure.

PyTorch 2.7.1+cu118 was selected from the
[official previous-release instructions](https://pytorch.org/get-started/previous-versions/).
CUDA 11.8 wheels include their own runtime; they do not require changing the
DSVM's CUDA 12.2 toolkit. The installed driver package is 535.230.02;
CUDA 11.x is supported by newer drivers through
[NVIDIA backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
This is a conservative compatibility choice, **not an operational GPU check**.
`nvidia-smi` cannot communicate with a GPU on this E8s_v5 VM.

### Immutable resources and offline recovery

The authoritative revisions, selected configurations, split mappings, and
parquet file patterns are tracked in `config/resources.json`. Revisions were
queried and saved before downloads. The tokenizer is pinned to the same
revision as model weights. Download scripts never resolve a moving `main`
revision. They download model safetensors/tokenizer files and only the selected
dataset parquet files, then prepare each split under
`cache/huggingface/datasets/prepared/<dataset>/<revision>/<split>`.

`artifacts/environment/resources.json` records resolved snapshot/prepared
paths, source parquet files, row counts, columns, and preparation fingerprints.
Model/tokenizer loading uses local snapshot paths and `local_files_only=True`.
For offline checks, `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` are set
before imports. Dataset loading is checked both from saved prepared copies and
by rebuilding local parquet files in a separate preparation cache. This avoids
relying on a moving Hub alias or only on an existing prepared-cache entry.

The tiny inference uses one MMLU-Pro validation question and one ARC-Challenge
validation question. Test-split schema checks use a few rows without inference
or tuning. No experiment sample, distractor realization, or scientific result
is selected here. Choice labels are checked by tokenizing the entire exact
prompt-plus-label and requiring precisely one appended token, with an unchanged
prefix. The tested answer prefix ends in `Answer:` (no trailing space), and each
completion includes one leading space, such as ` A`. Directly appending `A`
failed because the tokenizer merged the colon and letter. This setup-format
choice is recorded for later scoring validation; it does not finalize the
experiment prompt. FP32 choice log-probabilities use `log_softmax`; NLL is the negative
gold log-probability. Only answer-position vocabulary logits are materialized.

### Owner confirmations and completion

The backup destination is `/Volumes/X9/projects/transformer-calibration-backups`.
The owner confirmed on October 3, 2026 that the backup was copied and verified
there and the live shutdown schedule was confirmed at 11pm MST. These are owner
confirmations; the remote agent cannot inspect the Mac-mounted volume and has
no authenticated Azure CLI account. Step 001 is **complete**. Retain
[the manual instructions](next_steps/STEP_001_MANUAL.md) for future backups.
At Step 001 completion, A100/CUDA/BF16/32K and the resize path were still
unverified. The later Step 002 results below now validate the CPU-to-A100
resize, retained files, and batch-one BF16 native-context direct scoring path.

### CPU validation evidence

The 2026-10-03 online and offline checks passed for MMLU-Pro validation
question `0` (125 prompt tokens, 10 valid choices) and ARC-Challenge validation
question `Mercury_SC_407695` (94 prompt tokens, 4 valid choices). Inference used
FP32, CPU, batch size one, evaluation mode, no gradients, SDPA, no quantization,
no context extension, no generation/sampling, and an explicitly non-thinking
chat template. Both probability vectors sum to 1 and have finite NLL. Reports
include raw choice logits and log-probabilities, prompt text, label completions,
and exact answer token IDs. These are **development validation**, not paper
measurements. Runtime does not estimate GPU performance.

Cached dataset counts: MMLU-Pro test 12,032 / validation 70; WikiText-103 raw
train 1,801,350; ARC-Challenge test 1,172 / validation 299. Sources and processed
splits loaded offline, including preparation in a separate cache. The tracked
pins record every selected revision/configuration/split. Bulk resource files
are ignored by Git.

The exact resolved dependency snapshot contains 77 distributions, including
pip 25.3 and setuptools 65.5.0. Core versions: torch 2.7.1+cu118, Transformers
4.57.6, Datasets 4.8.5, huggingface-hub 0.36.2, accelerate 1.15.0, safetensors
0.8.0, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, Matplotlib 3.10.9, PyYAML
6.0.3, pytest 9.1.1. All resolved versions are in
`requirements.cpu-validated.txt`.

### Recreate the pinned environment

For recovery with network access, on Linux x86_64 with the recorded interpreter:

```bash
/anaconda/envs/azureml_py310_sdkv2/bin/python -m venv .venv
source .venv/bin/activate
python -m pip install --cache-dir cache/pip pip==25.3
python -m pip install --cache-dir cache/pip torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --cache-dir cache/pip -r requirements.cpu-validated.txt
```

The validation also downloads the exact wheels to ignored `cache/wheels` and
recreates a separate environment without network access. To repeat that check,
choose an unused temporary venv path (the validation already created the path
below):

```bash
python -m pip download --cache-dir cache/pip --only-binary=:all: --dest cache/wheels -r requirements.cpu-validated.txt --extra-index-url https://download.pytorch.org/whl/cu118
/anaconda/envs/azureml_py310_sdkv2/bin/python -m venv cache/recreation/.venv
cache/recreation/.venv/bin/python -m pip install --no-index --find-links cache/wheels -r requirements.cpu-validated.txt --report artifacts/environment/recreation-install.json
cache/recreation/.venv/bin/python scripts/verify_environment.py --resources --offline --output artifacts/environment/recreation-check.json
```

The working `.venv` is retained throughout. Wheels and the temporary environment
are disposable and ignored. `wheel-manifest.json` records wheel SHA256 hashes;
installation reports record package URLs/build sources. The CPU snapshot is a
version snapshot for this interpreter/platform, not a portable GPU lock or a
claim of validated CUDA execution.

Recreation validation passed with the same 77 exact package versions as the
working environment. Imports, `pip check`, Qwen3 tokenizer support, local
resource loading, and offline dataset preparation passed in that environment.
Online and offline CPU choice logits were compared within `1e-5` tolerance;
the measured maximum absolute difference was 0.0.

Report inventory (all ignored under `artifacts/environment/`):

- `azure-host.json`, `host-inventory.json`: observed host/image/storage and bundled packages.
- `environment.json`: fresh-shell import, package consistency, and tensor checks.
- `resources.json`, `resources-offline.json`: pinned model and dataset snapshots/prepared splits.
- `cpu-smoke.json`, `cpu-smoke-offline.json`: two FP32 choice-scoring validations.
- `recreation-check.json`, `recreation-install.json`: second environment checks and installation provenance.
- `torch-install.json`, `dependencies-install.json`, `wheel-manifest.json`: package sources/builds and cached wheel hashes.
- `validation-summary.json`: final checks and owner-action status.
- `step-001-setup.tar.gz`, `.sha256`: compact setup backup, locally verified; owner confirmed Mac copy and verification October 3, 2026.

At the end of setup, the managed filesystem had approximately 143 GiB free
before packaging (model/data caches, wheels, and both venvs retained). A cached
resource directory or an on-VM archive is not an independent backup.

The verified backup archive preserves the initial setup records, which still
describe the owner actions as pending at archive creation. It has not been
regenerated after verification. The completion confirmations above are the
current status; they are recorded in source documentation separately from
that initial snapshot.

## A100 validation record (October 4, 2026)

The owner resized the existing VM. `gpu-host.json` confirms
`Standard_NC24ads_A100_v4`, South Central US, the identical managed OS disk ID,
and the same pinned DSVM image. The repository, `.venv`, pinned resource
snapshots/prepared datasets, CPU reports, and verified CPU backup archive all
survived. `findmnt` confirms repository, venv, Hugging Face cache, and GPU output
paths use `/dev/sda1` ext4 on the managed OS disk. The GPU size also has a
64 GiB temporary resource disk; no authoritative project paths were moved there.

`nvidia-smi` confirms NVIDIA A100 80GB PCIe and driver 535.230.02, reporting CUDA
12.2 support. Project Python remains 3.10.18 and PyTorch remains 2.7.1+cu118
(CUDA 11.8 runtime), with Transformers 4.57.6 and the same 77 resolved versions.
Compute capability is 8.0; CUDA and BF16 tensor operations passed. No NVIDIA,
CUDA, Python, or project dependency changes were required.

The restricted Codex tool shell hides NVIDIA device nodes. GPU checks were run
with approved direct device access; the ordinary SSH shell can run them normally.
A restricted-shell `nvidia-smi` failure is not a driver failure on this host.

### Tested inference and recovery

`config/gpu_validation.json` records CUDA BF16, batch size one, thinking
disabled explicitly, evaluation mode, no gradients, no quantization/sampling,
no KV caching, no YaRN, and answer-position-only vocabulary logits. Confidence
and NLL use FP32 choice `log_softmax`. Model/tokenizer and datasets load locally
with Hugging Face network access disabled, using the same immutable revisions
as CPU setup.

The validation uses MMLU-Pro validation question `0` and ARC-Challenge validation
question `Mercury_SC_407695`, with deterministic, paired WikiText train prefixes
and seed 11. Complete inputs are re-tokenized before inference; the question
and choices are preserved and each answer completion is one token in its exact
prefix context. The no-distractor inputs are 131 and 100 tokens. Every nonzero
condition exactly fills its total budget after reserving one answer token:
2,047 / 4,095 / 8,191 / 16,383 / 32,767 input tokens. All 12 scoring records
passed finite-score/NLL and normalized-probability checks.

Every forward forces PyTorch's supported `SDPBackend.FLASH_ATTENTION` with
model attention set to `sdpa`. A separate 2K profiler pass observed all 36
flash-attention operators, one per model layer. The separate `flash_attn`
package was not installed and math fallback was disabled.

At 32K, forwards including choice scoring took 3.373 and 3.379 seconds. Peak
PyTorch allocated memory was 9.92 GiB, with 11.15 GiB peak reserved. These values
include resident BF16 weights and exclude GPU driver/context allocations.
The timing starts after input transfer and includes CUDA synchronization;
loading, preparation, warm-up, and profiling are excluded. See the full table
in [Step 002](next_steps/STEP_002.md) and the 12-row `context-summary.csv`.

A separate process was terminated with SIGKILL after two atomically saved and
flushed records. Resume in a fresh process validated the immutable plan,
source/package identities, prompt hashes, and prior records; reloaded the cached
model; skipped two successes; and completed ten pending evaluations. All 12
unique records are present, with the prior two byte-identical and no duplicates.
Cached model reload took 1.536 seconds. The one-writer lock and atomic record
format are implemented for this validation harness; the full experiment runner
still requires its own implementation and recovery tests.

### Reproduction and records

Activate `.venv` and source `.env` as above. For a fresh run:

```bash
python scripts/validate_gpu.py --run-dir artifacts/gpu_validation/new-check --recovery-check
```

Resume a compatible interrupted run with:

```bash
python scripts/validate_gpu.py --run-dir artifacts/gpu_validation/a100-initial --resume
```

`requirements.gpu-validated.txt` pins the working GPU environment. Use the same
Python interpreter, official cu118 PyTorch wheel source, and installation
procedure documented for the CPU snapshot, substituting the GPU snapshot.
This dependency snapshot is validated only with the recorded driver/hardware
and direct-scoring settings. Revalidate relevant changes before experiments.

New ignored records include `artifacts/environment/gpu-host.json` and
`gpu-preflight.json`, plus `artifacts/gpu_validation/a100-initial/plan.json`,
`distractor-streams.json`, `report.json`, `attention-check.json`,
`recovery-check.json`, `context-summary.csv`, and the 12 atomic result records
under `records/`. The report records software/hardware identities, persistent
paths, package consistency, inference settings, timings, and peak memory.
Source hashes identify relevant uncommitted implementation files.

A separate `artifacts/environment/step-002-gpu.tar.gz` with archive/member
checksums is prepared for `/Volumes/X9/projects/transformer-calibration-backups`.
Its Mac copy is pending; the owner-verified Step 001 archive is preserved.
Copy/verification commands are in [Step 002](next_steps/STEP_002.md). Personal
Terraform settings and Azure authentication are absent from this remote checkout;
confirm the owner-local `vm_size` matches the active A100 configuration and
retain the reviewed resize plan in that infrastructure checkout.

This is successful hardware/runtime validation, not the full scientific pilot
or main experiment. CPU-to-A100 resize was verified; a later return to CPU has
not yet been performed. Larger batches and KV-cached generation are outside
the tested direct-scoring configuration.
