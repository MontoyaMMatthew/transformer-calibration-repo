# Research resource definition

## Purpose and scope

This document defines the resources and execution settings for **Quantifying Transformer Confidence and Accuracy Trajectories Under Non-Ideal Conditions**. The study evaluates how increasing irrelevant context affects the accuracy and confidence of a fixed Qwen3-4B model.

The workload is inference only: no training or fine-tuning is planned. MMLU-Pro is the primary evaluation dataset, WikiText-103 supplies distractor text, and ARC-Challenge provides a verification dataset. The planned context conditions are 0K, 2K, 4K, 8K, 16K, and 32K.

The recommendations below were accepted on September 24, 2026. Hardware sizing and runtime estimates remain subject to a pilot benchmark.

## Selected platform and workflow

Use an **Azure Ubuntu Data Science Virtual Machine (DSVM)** on a **personal paid subscription**. Azure for Students is not part of the deployment plan. Develop on CPU hardware, resize the same VM to one A100 for validation and experiments, and resize back to CPU for further development. Use standard pay-as-you-go compute.

The DSVM image must supply a compatible NVIDIA driver and CUDA software stack so that manual driver or CUDA installation is not part of the normal workflow. Manage project-level Python dependencies in a dedicated virtual environment. This is a preconfigured VM, not an Azure Machine Learning managed compute deployment; operating-system maintenance remains the VM owner's responsibility.

Select and pin an available, supported Ubuntu DSVM image version that is compatible with both VM sizes and the chosen PyTorch build. The exact publisher, offer, SKU, version, Ubuntu release, and any Marketplace plan terms must be verified before provisioning. Do not assume the DSVM uses the plain Ubuntu 22.04 image currently specified in Terraform.

## Selected machine configurations

| Resource | Recommended configuration | Purpose |
| --- | --- | --- |
| CPU development VM | `Standard_E8s_v5`: 8 vCPUs, 64 GiB RAM | Development, data preparation, and short-context CPU smoke tests |
| GPU experiment VM | `Standard_NC24ads_A100_v4`: one A100 80 GB, 24 vCPUs, 220 GiB RAM | GPU validation and full experiments |
| Persistent storage | 256 GiB Standard SSD LRS, retained across resizes; increase if the selected image requires more | Operating system, dependencies, model weights, dataset caches, and results |
| Operating system and image | Supported Generation 2, x86-64 Ubuntu DSVM; exact image version pending verification | Preconfigured GPU software and a persistent Linux environment |
| Remote access | SSH; optional Jupyter access through an SSH tunnel | Development and experiment management |
| Separate backup storage | Initially 25-50 GB | Results, configurations, and reproducibility records |

A single GPU is sufficient for the planned study. The CPU and GPU configurations are two sizes of the same VM used at different times, not two continuously running machines. The previously considered L40S and other cloud providers are not the selected deployment.

Approximately four billion BF16 parameters occupy about 8 GB before runtime allocations. Based on the model architecture, a BF16 key/value cache at 32,768 tokens occupies approximately 4.5 GiB per sequence when caching is enabled. Temporary tensors and batching require additional memory. Measure actual peak VRAM usage during the pilot. CPU smoke tests may use FP32, requiring roughly 16 GB for weights alone plus loading and inference overhead; CPU outputs are development checks, not the main experimental results.

## Region, quota, and deployment

Prefer **Central US (`centralus`)**, with **South Central US (`southcentralus`)** as a fallback selected before deployment. No price listing for the selected A100 SKU was found in West Central US during the September 24, 2026 review; this is not an authenticated capacity check. Changing regions later is a migration, not an ordinary resize.

Confirm CPU-family quota for development and at least 24 available vCPUs in both the GPU-family and total regional quotas for the A100 phase. Verify that the exact image, security settings, disk configuration, and both sizes support the resize path. A price listing or quota approval does not guarantee allocation. Running a CPU VM does not reserve an A100 for later use.

The current Terraform scaffold defaults to a CPU-only `Standard_D2s_v5` VM, a 64 GiB disk, and a plain Ubuntu image. Implementation must update the image configuration, initial CPU size, region, and disk capacity. Review Marketplace terms and any required plan block for the chosen image. This document does not itself change or deploy infrastructure. See [NEXT_STEPS.md](../NEXT_STEPS.md) for the implementation checklist.

### Persistence and resizing

Keep the repository, virtual environment, model and dataset caches, and results on persistent managed storage. A compatible resize preserves the OS and data disks, including installed packages. It restarts the VM and may require deallocation; running processes and in-memory state do not survive. Temporary local disks must not hold authoritative files.

The model must be loaded into RAM or VRAM after each process restart, but its pinned weights and datasets should not need downloading again. Keep the same cache paths and user account across both phases. Install a compatible CUDA-enabled PyTorch build in the project environment that also supports CPU execution. Use explicit CPU and GPU configurations and keep thinking disabled in both.

After the first successful GPU validation, resizing back and forth should not require driver reinstallation. Kernel, image, driver, or package upgrades can require revalidation. Keep Terraform's configured size synchronized with resizes and review the plan to ensure a routine resize does not replace the VM or disks.

Use persistent storage for authoritative results. Configure the existing daily shutdown schedule around experiments, or temporarily disable it for long runs. Save progress frequently so interrupted jobs can resume. Deallocate the VM when idle; persistent disks and other retained resources can continue to incur charges.

## Model and inference settings

| Setting | Definition |
| --- | --- |
| Model | `Qwen/Qwen3-4B` |
| Revision | Pin an exact model and tokenizer revision before the pilot |
| Training | None; keep model weights fixed |
| Precision | BF16 for the main experiment |
| Quantization | Disabled for the main experiment |
| Thinking mode | **Disabled explicitly with `enable_thinking=False` in the tokenizer chat template** |
| Execution mode | Evaluation mode with gradient tracking disabled |
| Initial batch size | One; increase only after measuring memory and throughput |
| Attention | A tested memory-efficient attention implementation, such as supported SDPA or FlashAttention |
| Logit extraction | Compute vocabulary logits only at the answer position, then select valid answer-label logits |
| Confidence | Softmax over valid answer-choice logits, calculated in FP32; prediction is the highest-probability valid choice |
| Sampling | No sampled answer generation is needed for direct answer-choice scoring |

Thinking mode must remain disabled for every context condition and both evaluation datasets. Verify the rendered prompt during the pilot. Keep the prompt template, answer-scoring procedure, model revision, precision, and attention implementation fixed throughout the main experiment.

Verify that each answer label is a single token in the exact answer-prefix context used for scoring. If this is not true, revise the label format or define and validate a sequence-scoring procedure before the main experiment. Record that choice. Confidence is conditional on the supplied answer choices, not the maximum probability over the entire vocabulary. Do not apply sampling temperature, top-k, or top-p transformations to the confidence scores.

## Context-length definition

The native context budget is 32,768 tokens. The longest condition must fit the entire rendered prompt and any answer space inside that budget; it must not contain 32,768 distractor tokens plus the question and instructions.

- `0K` means no distractor text; the question, choices, and fixed instructions are still present.
- The nonzero labels are nominal total-context targets: 2,048, 4,096, 8,192, 16,384, and 32,768 tokens.
- Tokenize with the pinned Qwen tokenizer and subtract fixed prompt content and reserved answer space before allocating distractor tokens.
- Record actual prompt and distractor token counts for each example, including any deviation from a target.
- Predefine handling for questions whose fixed prompt exceeds a target. Do not silently truncate the question or answer choices.
- Keep context-extension methods such as YaRN disabled for the main experiment to avoid introducing another experimental factor.

## Data and software access

The machine needs network access to download the following public model and datasets, after which cached, pinned copies can be used:

| Resource | Source | Use |
| --- | --- | --- |
| Qwen3-4B model and tokenizer | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Fixed inference model |
| MMLU-Pro | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | Primary multiple-choice evaluation |
| WikiText-103 | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) | Distractor corpus; record the selected configuration and split |
| ARC-Challenge | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | Verification evaluation using the ARC-Challenge configuration |

Record dataset revisions, configurations, splits, and sampled example IDs. Preserve the same evaluation questions across context conditions and predefine distractor sampling, placement, and random seeds.

Install and pin a tested environment containing:

- The DSVM-provided NVIDIA driver and CUDA stack, validated on the A100 rather than installed manually as part of project setup.
- Python and a compatible CUDA-enabled PyTorch build in the project virtual environment; pin versions after validation.
- Hugging Face Transformers with Qwen3 support and Hugging Face Datasets.
- NumPy, pandas, SciPy, and plotting libraries such as Matplotlib.
- Git and a reproducible dependency lock file or container specification.
- Optional Jupyter for development and optional experiment tracking; local structured records are sufficient.

A commercial model API, separate database, paid experiment tracker, and multi-GPU cluster are not required. Direct access to model logits is required.

## Results and reproducibility

Save per-example records containing dataset and question IDs, gold answers, distractor source IDs and offsets, random seeds, context placement, actual token counts, answer-choice logits and probabilities, predictions, runtime, and run configuration identifiers. Retain enough information to reconstruct prompts without storing every expanded prompt or all-token vocabulary logits.

Calculate accuracy, mean confidence, expected calibration error (ECE), multiclass Brier score, and predictive entropy for each context condition. Specify ECE bins and the Brier-score convention before the main run. Report dataset results separately and retain per-example records for bootstrap confidence intervals without additional GPU inference. When estimating uncertainty across context conditions, preserve the pairing of questions and account for repeated distractor realizations.

Back up results and configuration records outside the compute VM. Record software versions, hardware, model and dataset revisions, prompt templates, and all inference settings with each run.

## Compute allocation and pilot

Plan an **initial allocation of 100 GPU-hours**, including approximately **5-10 hours for GPU validation and pilot benchmarking**. Perform routine development on the CPU size. This is a provisional resource allocation, not a prediction that the complete study will finish within 100 hours.

The proposal does not yet fix sample counts or the number of distractor realizations. An illustrative workload is:

`(1,000 MMLU-Pro questions + 500 ARC-Challenge questions) × 6 context conditions × 3 distractor seeds = 27,000 evaluations`

This is an example for planning, not a finalized sampling design. Identical deterministic no-distractor evaluations can be reused across distractor seeds. At an illustrative average of 10 seconds per evaluation, 27,000 evaluations would require 75 GPU-hours. The 10-second value is not a measured benchmark; full-dataset evaluation may require substantially more compute.

During the pilot:

1. Verify disabled thinking mode, answer-label tokenization, and probability extraction.
2. Run representative examples at every context length, especially the 32K condition.
3. Measure peak GPU memory, processing time, and safe batch sizes at each length.
4. Check output completeness, resumability, and reproducibility of recorded configurations.
5. Finalize question counts and distractor repetitions, then estimate total time from measured throughput separately for each context length.

Calculate the cloud budget from CPU development hours plus GPU hours at their respective rates, persistent storage, backup, networking, and any selected-image charges. Check current regional and image pricing before deployment.

Pricing checked on September 24, 2026: the selected Linux A100 VM was listed at $4.15/hour in Central US and $4.408/hour in South Central US. A 256 GiB E15 Standard SSD LRS was $19.20/month in either region, with operations charged separately. Thus, 100 A100 hours plus one full month of that disk total $434.20 or $460.00 respectively, before CPU hours, other services, and taxes. These are planning figures, not a guaranteed final bill or a capacity reservation.

## Technical references

- [Azure Data Science Virtual Machine overview and CPU/GPU switching](https://learn.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/overview)
- [Create an Ubuntu DSVM](https://learn.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro)
- [Azure Esv5 CPU machine specifications](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/memory-optimized/esv5-series)
- [Azure VM resizing and limitations](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/resize-vm)
- [Azure Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices)
- [Azure NC A100 v4 machine specifications](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series)
- [Azure vCPU quotas](https://learn.microsoft.com/en-us/azure/virtual-machines/quotas)
- [Qwen3-4B model card and thinking-mode configuration](https://huggingface.co/Qwen/Qwen3-4B)
- [Qwen3-4B architecture configuration](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json)
- [Transformers Qwen3 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3)
