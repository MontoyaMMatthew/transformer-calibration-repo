# Next steps

Implement the Azure DSVM workflow defined in [docs/RESOURCE_DEFINITION.md](docs/RESOURCE_DEFINITION.md): develop on a CPU VM, retain the same installation and managed disk, and resize to one A100 for GPU validation and experiments. Thinking mode stays disabled throughout.

Status: the existing DSVM has been deployed. Step 001 CPU environment setup, persistent caches, online/offline inference, dependency recreation, shutdown confirmation, and the independent Mac backup are complete. Azure metadata confirms `Standard_NC24ads_A100_v4` and the same managed disk/image; direct `nvidia-smi` confirms one A100 80 GB and driver 535.230.02. On October 4, 2026, BF16 scoring passed for both validation questions across all six contexts through 32K, SDPA flash attention was verified in all 36 layers, and forced-kill recovery passed. See [Step 002](docs/next_steps/STEP_002.md). No new deployment is required. See [ENVIRONMENT.md](docs/ENVIRONMENT.md) and [STEP_001.md](docs/next_steps/STEP_001.md). The quota observations below are historical September 24 records, not current allocation limits.

Quota requests submitted September 24, 2026 (Mountain time) for South Central US:

| Quota | Current limit | Requested limit | Last observed status |
| --- | ---: | ---: | --- |
| Standard ESv5 Family vCPUs | 0 | 8 | InProgress |
| Standard NCADS_A100_v4 Family vCPUs | 0 | 24 | InProgress |
| Total Regional vCPUs | 10 | At least 24 needed | Recheck after family approval |

Request IDs: ESv5 `c4b641d0-6bb1-40ff-878f-c85acfccf712`; A100 `abb5bdd9-fd3a-4ed8-81b6-7782ae62fb51`. View [Azure My quotas](https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/myQuotas), selecting Compute, MatthewPersonalSubscription, and South Central US. Microsoft documents that [family quota approval automatically increases regional quota](https://learn.microsoft.com/en-us/azure/quotas/per-vm-quota-requests); verify the resulting limits before provisioning. Quota approval permits deployment but does not reserve hardware capacity.

## 1. Establish the personal Azure subscription

- [x] Create a personal paid subscription (confirmed by the user).
- [x] Copy the example to `VirtualMachine/terraform/personal.auto.tfvars.json`, enter the personal subscription ID and SSH settings, and sign in with `./cloud az login`. Confirm the selection with `./cloud az account show`.
- [x] Select South Central US as the deployment region.
- [x] Inspect CPU and GPU quotas; both family limits are zero, and the regional limit is 10 vCPUs.
- [x] Submit increases to 8 ESv5 and 24 NCADS_A100_v4 family vCPUs.
- [x] Confirm quota permits at least a 24-vCPU A100 allocation and verify GPU availability. The allocated NC24ads_A100_v4 proves the required regional/family entitlement and current availability. Exact live quota limits/headroom were not queried; future GPU allocation remains subject to capacity.
- [ ] Set a budget and spending alerts that include CPU development, GPU execution, disk storage, networking, and backups. Budget alerts are notifications, not an automatic spending cap.

## 2. Select and validate the DSVM image

- [x] Pin `microsoft-dsvm:ubuntu-2204:2204-gen2:25.06.18`; the regional Azure catalog confirms x64, Generation 2, and no purchase plan.
- [x] Verify access to the deployed pinned image; record driver/CUDA versions. The same DSVM image boots on NC24ads_A100_v4; NVIDIA driver 535.230.02 reports CUDA 12.2 support.
- [x] Verify that the image and VM settings permit CPU-to-A100 resizing in the chosen region. Azure metadata confirms the A100 size in South Central US with the identical managed OS disk and pinned DSVM image.
- [x] Confirm that the Qwen3-compatible project PyTorch/Transformers environment runs with the supplied driver. Existing PyTorch 2.7.1+cu118/Transformers 4.57.6 passed BF16 direct scoring through 32K on the A100 without package or driver changes.

Completion criterion: a concrete image and CPU/GPU configuration are selected. The exact image/driver/project environment passed the early GPU test below. Manual driver installation is not the intended fallback; first seek a compatible preconfigured image if validation fails.

## 3. Update the infrastructure configuration

- [x] Replace the plain Ubuntu image reference with the pinned DSVM image. Its metadata requires no purchase plan.
- [x] Default compute to `Standard_E8s_v5` and persistent Standard SSD LRS storage to 256 GiB.
- [x] Update examples and add the `cloud` helper with an isolated Azure profile and a shared local subscription configuration.
- [x] Retain key-only SSH access, the restricted source address, and the static public IP. Use an SSH tunnel for notebooks.
- [x] Retain daily auto-shutdown and document how to disable it for long experiments. Owner confirmed daily 11pm MST during Step 001.
- [ ] Confirm the owner-local Terraform `vm_size` matches the active A100 size and retain the reviewed resize plan; the remote checkout has no personal Terraform settings or authenticated Azure CLI profile.
- [x] Inspect the existing deployment before changing image or region. The current DSVM and managed OS disk were verified during Step 001, and the setup backup was independently copied and verified on the Mac. No image or region change was made.
- [x] Validate Terraform and review the complete deployment plan, including disk retention and image terms, before provisioning. Re-run the plan before applying if configuration or Azure state changes.

Completion criterion: the plan describes the intended DSVM and costs, with no unexplained replacement of existing resources.

## 4. Prepare persistent storage and the Python environment

- [x] Deploy the CPU DSVM and connect through SSH.
- [x] Place the repository, virtual environment, model cache, dataset cache, and outputs on persistent managed storage. Record the paths and explicitly configure the Hugging Face cache location.
- [x] Create a project virtual environment with a compatible CUDA-enabled PyTorch build that also supports CPU execution. Install Transformers, Datasets, NumPy, pandas, SciPy, and plotting dependencies.
- [x] Pin model/tokenizer and dataset revisions, configurations, and splits; download them once to the persistent cache.
- [x] Configure a separate backup destination for results and reproducibility records. Keep credentials outside Git.
- [x] Record image, OS, and package versions. Avoid altering the DSVM's system driver/CUDA installation as part of routine venv setup.

## 5. Develop and perform CPU smoke tests

- [x] Implement explicit development CPU and GPU configurations. `verify_environment.py --cpu-smoke` uses CPU FP32; `config/gpu_validation.json` uses CUDA BF16. Thinking is disabled in both. Main-study runner configuration remains later work.
- [x] Validate dataset loading, distractor sampling, prompt construction, and actual token counts in the GPU validation harness. Paired WikiText prefixes are deterministic; 0K contains no distractor; nonzero budgets include one reserved answer token, including 32,767 input tokens at 32K. Final experimental sampling remains later work.
- [x] Verify answer-label tokenization and extract answer-position logits. Normalize over valid answer choices and save per-choice probabilities. Passed for two development questions in Step 001, online and offline.
- [x] Run a few short-context Qwen3-4B forward passes on CPU. Verify metrics and output records without treating CPU results as final study measurements. Step 001 validated two FP32 development questions, normalized probabilities, finite NLL, and saved records.
- [x] Implement durable progress saving and resume in the GPU validation harness. A forced SIGKILL after two results retained both byte-identically; resume completed ten pending evaluations without duplicates.
- [ ] Extend configuration, inference records, and resume validation into the full experiment runner before the larger scientific pilot.

## 6. Validate the A100 early

Perform this before substantial development depends on an untested GPU environment.

- [x] Resize the existing VM to `Standard_NC24ads_A100_v4` and restart using the same image and managed disk. Owner performed the resize; metadata confirms retained disk/image. Synchronizing the local Terraform size and verifying the saved resize plan remain owner-side infrastructure records.
- [x] Verify the actual allocated size, GPU identity, and driver with `nvidia-smi`; verify CUDA availability and the device identity in PyTorch. NC24ads_A100_v4, A100 80GB PCIe, driver 535.230.02, PyTorch 2.7.1+cu118/CUDA 11.8, and BF16 tensor operations passed with direct device access.
- [x] Confirm that the repository, venv, cached model/data, and results survived. Reload the cached model without downloading it again.
- [x] Run a short BF16 forward pass, then representative examples at every planned context length, including 32K, initially at batch size one.
- [x] Verify thinking is disabled, output probabilities are valid, and memory-efficient attention is active. Prefer a supported PyTorch attention backend before adding packages that require compiling CUDA extensions.
- [x] Measure peak VRAM and runtime by context length. Test interrupted-run recovery and output persistence.
- [x] Record the validated software/hardware combination and lock dependencies in `requirements.gpu-validated.txt` and `docs/ENVIRONMENT.md`. The working managed disk was not additionally snapshotted.

Completion criterion: the full context range works on the intended A100 with the pinned environment, and saved progress can be resumed. **Met October 4, 2026 for the validation harness** (two held-out questions, all six context conditions, batch size one). The full scientific pilot remains later work. If allocation fails, distinguish quota from regional capacity before changing the configuration.

## 7. Run experiments and control spending

- [ ] Finalize sample sizes, subject sampling, distractor repetitions and placement, metric conventions, and uncertainty estimates.
- [ ] Use pilot throughput to revise the provisional 100-GPU-hour allocation and budget. Include paid GPU time spent loading models and debugging.
- [ ] Adjust auto-shutdown to fit each experiment window. Keep automatic progress saving enabled.
- [ ] Execute the main MMLU-Pro study and ARC-Challenge verification with fixed model and inference settings.
- [ ] Back up results and run manifests, then perform analysis on CPU.
- [ ] Resize back to the CPU configuration for further development, keeping Terraform synchronized. Deallocate entirely when no compute is needed; persistent resources still accrue charges.
- [ ] Revalidate GPU execution after changing the kernel, driver, PyTorch, attention backend, or other relevant dependencies.

## Expected daily workflow

**CPU development:** start the CPU VM, activate the project venv, edit code, and run small tests.

**GPU experiments:** save progress, resize and restart, activate the same venv, select the GPU configuration, load the cached model, and run or resume experiments.

**Return to development:** save and back up outputs, resize back to CPU, and analyze results. No routine driver reinstallation or model redownload should be necessary. GPU capacity remains subject to availability each time it is released and requested again.
