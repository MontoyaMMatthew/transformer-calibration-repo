# Next steps

Implement the Azure DSVM workflow defined in [docs/RESOURCE_DEFINITION.md](docs/RESOURCE_DEFINITION.md): develop on a CPU VM, retain the same installation and managed disk, and resize to one A100 for GPU validation and experiments. Thinking mode stays disabled throughout.

Status: Terraform is configured for the pinned Ubuntu 22.04 DSVM image, E8s_v5, a 256 GiB SSD, and South Central US. The image catalog reports Generation 2/x64 and no purchase plan. The repository-local Azure helper is authenticated to the personal subscription. Terraform validation passed and the deployment plan contains 9 additions, no changes, and no deletions. Deployment is blocked by quotas; resize compatibility and actual GPU capacity remain unverified. Follow [README.md](README.md) for commands.

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
- [ ] Confirm quota approval and at least 24 total regional vCPUs; separately check GPU availability. CPU allocation does not reserve GPU capacity.
- [ ] Set a budget and spending alerts that include CPU development, GPU execution, disk storage, networking, and backups. Budget alerts are notifications, not an automatic spending cap.

## 2. Select and validate the DSVM image

- [x] Pin `microsoft-dsvm:ubuntu-2204:2204-gen2:25.06.18`; the regional Azure catalog confirms x64, Generation 2, and no purchase plan.
- [ ] Recheck image access in the personal subscription and record the installed driver/CUDA versions during the GPU pilot.
- [ ] Verify that the image and VM security/disk settings permit CPU-to-A100 resizing in the chosen region.
- [ ] Confirm that a current Qwen3-compatible PyTorch/Transformers environment can run with the supplied driver. Do not assume the image's preinstalled Python packages support Qwen3.

Completion criterion: a concrete image and CPU/GPU configuration are selected. Compatibility still requires the early GPU test below. Manual driver installation is not the intended fallback; first seek a compatible preconfigured image if validation fails.

## 3. Update the infrastructure configuration

- [x] Replace the plain Ubuntu image reference with the pinned DSVM image. Its metadata requires no purchase plan.
- [x] Default compute to `Standard_E8s_v5` and persistent Standard SSD LRS storage to 256 GiB.
- [x] Update examples and add the `cloud` helper with an isolated Azure profile and a shared local subscription configuration.
- [x] Retain key-only SSH access, the restricted source address, and the static public IP. Use an SSH tunnel for notebooks.
- [x] Retain daily auto-shutdown and document how to disable it for long experiments.
- [ ] Check whether any deployment already exists before changing its image or region. Review the Terraform plan for replacement; back up existing work and plan a migration if necessary.
- [x] Validate Terraform and review the complete deployment plan, including disk retention and image terms, before provisioning. Re-run the plan before applying if configuration or Azure state changes.

Completion criterion: the plan describes the intended DSVM and costs, with no unexplained replacement of existing resources.

## 4. Prepare persistent storage and the Python environment

- [ ] Deploy the CPU DSVM and connect through SSH.
- [ ] Place the repository, virtual environment, model cache, dataset cache, and outputs on persistent managed storage. Record the paths and explicitly configure the Hugging Face cache location.
- [ ] Create a project virtual environment with a compatible CUDA-enabled PyTorch build that also supports CPU execution. Install Transformers, Datasets, NumPy, pandas, SciPy, and plotting dependencies.
- [ ] Pin model/tokenizer and dataset revisions, configurations, and splits; download them once to the persistent cache.
- [ ] Configure a separate backup destination for results and reproducibility records. Keep credentials outside Git.
- [ ] Record image, OS, and package versions. Avoid altering the DSVM's system driver/CUDA installation as part of routine venv setup.

## 5. Develop and perform CPU smoke tests

- [ ] Implement explicit CPU and GPU run configurations. Use FP32 for initial CPU smoke tests and BF16 for the GPU study; disable thinking in both.
- [ ] Validate dataset loading, distractor sampling, prompt construction, and actual token counts. Treat 0K as no distractor and cap the longest total context at 32,768 tokens.
- [ ] Verify answer-label tokenization and extract answer-position logits. Normalize over valid answer choices and save per-choice probabilities.
- [ ] Run a few short-context Qwen3-4B forward passes on CPU. Verify metrics and output records without treating CPU results as final study measurements.
- [ ] Implement progress saving and resume behavior so completed examples do not have to be repeated after shutdown.

## 6. Validate the A100 early

Perform this before substantial development depends on an untested GPU environment.

- [ ] Save work, update the configured VM size, and review the resize plan. Deallocate if required, resize to `Standard_NC24ads_A100_v4`, and restart using the same image and managed disk.
- [ ] Verify the actual allocated size, GPU identity, and driver with `nvidia-smi`; verify CUDA availability and the device identity in PyTorch.
- [ ] Confirm that the repository, venv, cached model/data, and results survived. Reload the cached model without downloading it again.
- [ ] Run a short BF16 forward pass, then representative examples at every planned context length, including 32K, initially at batch size one.
- [ ] Verify thinking is disabled, output probabilities are valid, and memory-efficient attention is active. Prefer a supported PyTorch attention backend before adding packages that require compiling CUDA extensions.
- [ ] Measure peak VRAM and runtime by context length. Test interrupted-run recovery and output persistence.
- [ ] Record the validated software/hardware combination and lock dependencies. Optionally snapshot the working managed disk before later environment changes; account for snapshot charges.

Completion criterion: the full context range works on the intended A100 with the pinned environment, and saved progress can be resumed. If allocation fails, distinguish quota from regional capacity before changing the configuration.

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
