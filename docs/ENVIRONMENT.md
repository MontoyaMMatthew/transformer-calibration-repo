# Development and experiment environment

Created: October 3, 2026. Status: setup plan and record template; no remote setup or validation has been performed by creating this document.

The Azure VM is already provisioned. Initial development uses `Standard_E8s_v5`. Execute the scoped setup work in [Step 001](next_steps/STEP_001.md). Experiment behavior is defined in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md); infrastructure details are in [RESOURCE_DEFINITION.md](RESOURCE_DEFINITION.md).

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
| Provisioning | Existing Azure VM | Provisioned, owner confirmed October 3, 2026 |
| Initial size | `Standard_E8s_v5`, 8 vCPUs, 64 GiB RAM | Pending host verification |
| OS/image | Ubuntu 22.04 DSVM; planned image `25.06.18` | Pending |
| Architecture | x86-64 | Pending |
| Region | South Central US | Pending |
| Persistent disk | Planned 256 GiB managed SSD | Pending capacity/mount verification |
| Repository path | Existing checkout on persistent disk | Pending |
| Python executable/version | Installed interpreter compatible with selected packages | Pending |
| Virtual environment | `<repo>/.venv` | Pending |
| PyTorch version/build/index | Stable CUDA-enabled wheel; CPU validated first | Pending |
| Transformers version | Stable version supporting Qwen3 | Pending |
| Cache location | `<repo>/cache/huggingface` | Pending |
| Artifact location | `<repo>/artifacts` | Pending |
| Shutdown schedule | Inspect current Azure setting | Pending |
| Independent backup | Directory on the Mac, separate from VM | Pending selection |
| CPU validation date | After Step 001 checks pass | Pending |
| GPU validation | Later A100 step | Not performed |

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
| Model/tokenizer | `Qwen/Qwen3-4B`, exact revision pending selection |
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

## 7. Later GPU validation

After the CPU development stage, validate the existing environment on `Standard_NC24ads_A100_v4` before relying on long-context experiments:

- Verify actual GPU identity, driver, CUDA availability, and BF16 execution.
- Verify persistent files survived the resize and cached resources load without another download.
- Run representative prompts through 32K with the fixed scoring method and memory-efficient attention.
- Record memory, timings, package changes, and the final attention backend.
- Create the GPU-validated dependency snapshot only after these checks pass.

FP32 CPU outputs are development checks, not interchangeable with BF16 GPU study results. Do not silently fall back to CPU when a future experiment configuration requests CUDA.

## Setup history

| Date | Action | Outcome |
| --- | --- | --- |
| 2026-10-03 | Created environment plan; owner confirmed VM provisioned and E8s_v5 development target | Setup and host verification pending |

Append actual setup commands, selected versions, validation evidence, and subsequent environment changes here as work proceeds.
