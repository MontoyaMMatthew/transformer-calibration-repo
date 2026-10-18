# Paper development review

Reviewed October 16, 2026. Purpose: identify the additions needed to expand the proposal into an Abstract, Introduction, Related Work, Research Project Problem, and Method. This is a planning review, not a completed manuscript or a report of scientific results.

Updated October 17, 2026: the owner retained the original paper intent and goals, accepted `Background text:` for the scientific prompt, and requested conservative higher-batch-size screening when the pilot begins. These decisions are reflected in the LaTeX draft and experiment plan; no pilot was started.

## Materials and status

Reviewed the two-page [proposal](../../DLT_topic_pick/mmonto16_Topic_Pick.pdf), [experiment plan](EXPERIMENT_PLAN.md), [resource definition](RESOURCE_DEFINITION.md), [environment record](ENVIRONMENT.md), pinned resources, and GPU-validation scoring code. No LaTeX source or bibliography was found in the two project workspaces.

The proposal supplies the motivation, hypothesis, model, datasets, and broad metrics. The experiment plan supplies most of the missing experimental design. The environment record documents successful hardware/runtime validation on two validation questions across six context conditions; that is not the full scientific pilot or evidence for the hypothesis. The main experiment and analysis pipeline are still described as planned.

Treat the documents according to their scope: pinned configuration and environment records document resources already selected or validated; experimental sample sizes and main-run procedures remain provisional. Older resource/setup passages contain earlier status statements. For example, the experiment plan correctly counts 24,000 proposed main-study forward passes after baseline reuse, whereas the resource definition includes an older illustrative 27,000-evaluation workload.

## What each requested section needs

| Section | Existing foundation | Add before calling it complete |
| --- | --- | --- |
| Abstract | Motivation and broad objective | Specific intervention, model, two evaluation datasets, answer-choice confidence definition, and evaluation approach. Keep an interim abstract until actual findings are available; add measured results and a bounded conclusion after the study. |
| Introduction | Nuanced discussion of calibration | Concrete motivation for irrelevant-context robustness, definitions of confidence versus calibration, direct citations for long-context/distraction claims, and an explicit contribution paragraph. |
| Related Work | Three calibration papers embedded in the Introduction | A synthesis of calibration, QA confidence elicitation, uncertainty under input shift, and long-context distraction. Explain differences in tasks and confidence measurement rather than listing papers. |
| Research Project Problem | Broad goal and hypothesis | Precisely scoped research questions, independent/dependent variables, experimental unit, primary comparison, and criteria for interpreting support or mixed evidence. |
| Method | Model/dataset names and proposed metrics | Reproducible sampling, distractor construction, exact prompts, context accounting, answer scoring, metric equations, paired analysis, pilot/freeze procedure, failure handling, and limitations. |

The first four sections and most of Method can be drafted now. The final Method must reflect the implemented, frozen protocol, and the final Abstract must reflect observed results.

## Review of the seven existing references

1. **Desai and Durrett (2020), [Calibration of Pre-trained Transformers](https://aclanthology.org/2020.emnlp-main.21/).** Retain for evidence that calibration depends on model and evaluation distribution. Their experiments use BERT/RoBERTa on classification tasks; explain that these results do not establish calibration for an instruction-tuned causal model using answer-label logits.
2. **Kadavath et al. (2022), [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221).** Retain for format-dependent multiple-choice calibration and self-evaluation. Distinguish the paper's answer probabilities, P(True), and P(IK). This project's maximum answer-choice probability is not interchangeable with every confidence signal studied there.
3. **Jiang et al. (2021), [How Can We Know When Language Models Know?](https://aclanthology.org/2021.tacl-1.57/).** Retain for miscalibration in generative QA and the importance of how probabilities are measured. Explain the difference between open-ended answer confidence and this project's probability distribution over supplied answer choices.
4. **Qwen Team (2025), [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).** Retain as the model-family reference. Also cite the [exact Qwen3-4B model-card revision](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/README.md) for the native 32,768-token context and explicit thinking-mode switch. The selected checkpoint has pretraining and post-training; describe it as a fixed pretrained and post-trained model evaluated without additional training.
5. **Wang et al. (2024), [MMLU-Pro](https://arxiv.org/abs/2406.01574).** Retain for the primary benchmark. Its challenging reasoning questions and expanded choice sets motivate the task, but the paper reports benefits from chain-of-thought. State that this project deliberately evaluates zero-shot direct answering with thinking disabled; its accuracy is not directly comparable to a chain-of-thought leaderboard protocol.
6. **Merity et al. (2016), [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843).** Retain as the WikiText corpus citation. Explain why natural expository text is useful background material. The citation does not establish that every sampled passage is irrelevant to every benchmark question.
7. **Clark et al. (2018), [ARC](https://arxiv.org/abs/1803.05457).** Retain for the second evaluation dataset. Describe ARC-Challenge as a check across a different benchmark, not a guarantee that the findings generalize to other models or real deployments.

These references have useful roles, but four mainly document resources. A dedicated Related Work section needs a stronger methodological and long-context foundation.

## Additional literature to add

| Priority | Source | Role in the manuscript |
| --- | --- | --- |
| High | Guo et al. (2017), [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | Define confidence calibration, motivate reliability diagrams/ECE, and support the general neural-network overconfidence motivation. Discuss temperature scaling as background; the current experiment does not fit a calibrator. |
| High | Liu et al. (2024), [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) | Support the distinction between available context capacity and effective context use. Its evidence-position/retrieval experiments motivate careful placement controls; this project does not reproduce that design. |
| High | Shi et al. (2023), [Large Language Models Can Be Easily Distracted by Irrelevant Context](https://proceedings.mlr.press/v202/shi23a.html) | Direct evidence that irrelevant information can harm problem-solving accuracy. Explain the difference between their GSM-IC arithmetic problems and this project's long WikiText prefixes and calibration trajectories. |
| High for positioning | Chhikara (2025), [Mind the Confidence Gap](https://arxiv.org/html/2502.11028v3) | Published TMLR study of confidence and distractors across factual QA datasets. Its distractors are incorrect answer options, and its prompts elicit numerical confidence. Contrast that with increasing irrelevant passage length and direct answer-label probabilities. Do not equate the two meanings of distractor or claim that confidence and distractors have never been studied together. |
| Useful | Ovadia et al. (2019), [Can You Trust Your Model's Uncertainty?](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html) | General motivation for measuring accuracy and uncertainty together as inputs change. Its dataset-shift results do not prove that this particular text intervention will worsen calibration. |
| Useful | Vaicenavicius et al. (2019), [Evaluating Model Calibration in Classification](https://proceedings.mlr.press/v89/vaicenavicius19a.html) | Explain the distinction between calibration of the predicted class's confidence and calibration of the entire multiclass distribution. Supports careful interpretation of ECE. |

This is a targeted initial review, not an exhaustive novelty search. The defensible contribution is a controlled empirical characterization of a particular model, confidence signal, and context intervention. Any stronger priority claim needs a broader literature review.

Organize Related Work into three connected parts: (1) calibration and uncertainty under changing inputs, (2) language-model QA confidence and elicitation, and (3) irrelevant context and long-context utilization. End by explaining the project's precise intersection: fixed-choice, non-thinking Qwen3-4B evaluated across paired context budgets on two datasets.

## Research problem and hypotheses

Use a bounded central question: **How do answer accuracy and answer-choice confidence calibration change when progressively longer WikiText background is prepended to otherwise fixed multiple-choice questions for Qwen3-4B?**

Define the independent variable as nominal total context budget. Dependent variables are accuracy, maximum answer-choice probability, ECE, NLL, Brier score, and entropy. A unique question is the sampling unit; its evaluations across lengths and distractor seeds are repeated measurements.

Keep the existing primary comparison: mean seed-specific ECE at 32K minus ECE at the no-distractor baseline, calculated separately for each dataset. A positive change supports deterioration for that comparison, subject to uncertainty and ECE sensitivity. The intermediate conditions characterize the trajectory; a positive endpoint difference alone does not establish monotonic deterioration.

Supporting questions concern whether accuracy and confidence move together, whether high-confidence mistakes become more common, and whether the pattern also appears on ARC-Challenge. Falling accuracy alone does not establish worsening calibration. If confidence falls appropriately, calibration may remain stable or improve. NLL and Brier changes concern overall probability quality and should not be attributed solely to calibration.

## Method: required subsections and remaining decisions

### 1. Design and model

State that this is inference-only repeated-measures evaluation, with no additional training, fine-tuning, or fitted temperature scaling. Identify the exact checkpoint/tokenizer revision in `config/resources.json`. Explain the selection of a model with accessible logits and a native context window covering the planned sweep.

Document BF16 model inference, FP32 choice probability calculations, evaluation mode, disabled gradients, disabled thinking, no quantization, and no context extension. The validated implementation uses batch size one, SDPA with its flash-attention backend, no KV cache, and answer-position-only logits. Freeze and report the main-run settings after the pilot rather than assuming every validation setting is already the final protocol.

### 2. Datasets and sampling

Move the proposal's brief Datasets section into a substantial Method subsection, unless the course template requires a separate section. Give resource revisions, configurations, splits, eligibility rules, sample counts, and selection seeds. Explain proportional subject stratification for MMLU-Pro, including allocation rounding, and define ARC sampling explicitly.

The proposed counts are 1,000 MMLU-Pro test questions and 500 ARC-Challenge test questions, with a separate approximately 20-question pilot per dataset. Finalize counts from pilot feasibility and the precision needed for the primary comparison. Keep pilot and main question IDs disjoint; validation splits are available. Exclude questions that cannot fit the smallest nonzero budget from the entire paired sweep, disclose exclusions, and sample from the eligible population.

Use each question's actual number of choices, preserve choice order, normalize dataset label formats consistently, and verify the gold-answer mapping. Report datasets separately because their tasks and choice counts differ.

### 3. Distractor construction

Specify WikiText-103 raw training data, treatment of empty rows/headings, sampling unit, replacement policy, separators, and how source fragments become a sufficiently long stream. The validation code samples nonempty rows with replacement; decide and document whether to retain that rule in the main study. Short fragments, headings, or unrelated article boundaries affect what kind of distraction is being tested.

Use three proposed distractor seeds, 11, 22, and 33. For each question and seed, construct one reproducible text stream, then use progressively longer prefixes at successive budgets. Save source IDs, offsets or character endpoints, and exact reconstruction information. Seeds change distractor material, not model training or sampled answers.

Define what counts as acceptable incidental overlap with the question. Inspect pilot material and freeze any exclusion/screening rules before the main run; never screen using observed accuracy. Describe the corpus as background intended to be irrelevant, since unrelated source selection cannot guarantee semantic irrelevance.

### 4. Prompt and context budgets

Include the exact prompt as a listing or appendix, showing instructions, background delimiters, question, options, chat-template text, and answer prefix. State that the background precedes the question. The accepted scientific prompt uses `Background text:` without explicitly signaling irrelevance. The historical validation harness uses `Irrelevant background:`; preserve those earlier records and revalidate the revised prompt during the pilot.

Define 0K as no distractor, with the fixed prompt still present. Define nonzero labels as total budgets of 2,048, 4,096, 8,192, 16,384, and 32,768 tokens. Account for every rendered token and reserve one answer token. Re-tokenize the complete input, preserve the question/options, and report actual lengths and shortfalls. The 32K condition is not 32,768 distractor tokens plus the task.

The design measures this particular prefix intervention: more background also moves the question to a later absolute position and increases its distance from the initial instruction. It does not isolate pure token length from all positional/content effects. State this limitation; an extra placement or instruction-position control is an optional extension if stronger causal attribution is desired.

### 5. Answer scoring and confidence

For valid answer-label logits z at the scoring position, define:

\[
p_{i\ell sk}=\frac{\exp(z_{i\ell sk})}{\sum_{j=1}^{K_i}\exp(z_{i\ell sj})},\qquad
\hat y_{i\ell s}=\arg\max_k p_{i\ell sk},\qquad
c_{i\ell s}=\max_k p_{i\ell sk}.
\]

Here i denotes question, ell denotes context condition, s denotes distractor realization, and K_i is the valid choice count. Implement with stable log-softmax and define deterministic tie handling.

Explain that this distribution is conditional on the supplied labels. It is not softmax over the full vocabulary, verbalized confidence, the probability of a full answer explanation, or a guarantee of correctness. Validate that every label completion, including its leading-space convention, appends exactly one token to the actual answer prefix. Save the token mapping and inspect exact rendered prompts. No answer sampling, top-k/top-p filtering, or generation temperature is involved.

### 6. Metrics

Include equations and conventions for accuracy, mean confidence, top-label ECE, multiclass Brier score, answer-choice NLL, and predictive entropy. For a fixed dataset/condition/seed with n questions:

\[
\operatorname{ECE}=\sum_{b=1}^{B}\frac{|I_b|}{n}
\left|\operatorname{acc}(I_b)-\operatorname{conf}(I_b)\right|,
\quad
\operatorname{NLL}=-\frac1n\sum_i\log p_{i,y_i},
\]

\[
\operatorname{Brier}=\frac1n\sum_i\sum_{k=1}^{K_i}
(p_{ik}-\mathbb{1}[y_i=k])^2,
\quad H_i=-\sum_k p_{ik}\log p_{ik}.
\]

Use 10 equal-width ECE bins; specify edges and endpoint handling. Give empty bins zero weight. Use 15 and 20 bins as sensitivity checks. Use natural logarithms; do not divide the Brier sum by the class count. Include normalized entropy H_i/log(K_i). Call ECE "expected calibration error," correcting "Expected Calibration score" in the proposal. Cross-entropy and NLL are the same quantity here and need one reported column.

The signed mean-confidence-minus-accuracy gap is descriptive: errors can cancel. ECE assesses binned top-label calibration; NLL and Brier also reflect other aspects of probability quality. Confidence on incorrect predictions and correctness transitions from baseline are useful diagnostics requiring no new inference.

### 7. Aggregation and statistical analysis

Calculate each metric within each dataset, length, and seed, then average the three seed-specific metric values. Compute the baseline once per question. Do not average probability vectors across seeds first, which would assess an ensemble.

Use the planned approximately 2,000 question-level bootstrap resamples, maintaining MMLU-Pro subject strata and all repeated measurements within each sampled question. Recompute the full statistic, including ECE, in each resample and use identical resampled questions for baseline contrasts. Specify the bootstrap RNG seed and interval construction; percentile 95% intervals are a straightforward choice to freeze before the main analysis.

For the primary endpoint, each resample must recompute mean seed-specific ECE at 32K minus baseline ECE. Report seed variation separately. These intervals describe question-sampling uncertainty conditional on the selected distractors, not broad uncertainty over every possible background passage. Treat intermediate-length comparisons and diagnostics as secondary/exploratory unless a separate confirmatory analysis is specified. If making formal claims across both datasets, specify whether ARC is secondary confirmation or adopt an appropriate multiple-testing policy.

### 8. Pilot, completeness, and reproducibility

Separate runtime validation, scientific pilot, and main study. The scientific pilot should cover both datasets, all lengths, and all three distractor seeds; check label scoring, budgets, source construction, runtime, memory, and recovery. Freeze sample IDs, prompts, exclusions, model settings, and analysis conventions before the main run.

During that pilot, assess larger batches cautiously, initially 1, 2, and 4, with repeated 32K examples and at least 25% observed device-memory headroom. Do not search deliberately for an out-of-memory limit. Verify batched scoring against single-example scoring, measure questions per second, and choose a conservative execution policy before the main study. Full operational details are recorded in the experiment plan.

The proposed main workload is (1,000 + 500) x (1 baseline + 5 lengths x 3 seeds) = 24,000 forward passes, excluding pilot and other checks. Use pilot timings by length to finalize the allocation. Two successful validation questions establish feasibility for those examples, not adequate statistical precision or representative study runtime.

Save per-example probabilities/log-probabilities, predictions, gold labels, question IDs, distractor reconstruction records, token counts, configuration identifiers, and failure history. Define bounded retries and report completeness. If failures persist, report a clearly defined paired complete-question analysis and the resulting exclusions rather than silently changing the sample across conditions.

Put hardware, core software versions, immutable resource revisions, and a code version in a compact reproducibility paragraph/table; detailed deployment and cloud administration belong outside the main scientific Method. Add full prompt and sampling details in an appendix if space is limited.

## Immediate manuscript changes

1. Expand the Introduction around the distinction between accurate answers and trustworthy confidence; cite its broad overconfidence claim and its long-context claim directly.
2. Replace the uncited "more recent evaluations" sentence with a specific, accurately scoped citation, or remove it. Fix remaining "ref [n]" prose artifacts using LaTeX citation commands.
3. Create the Related Work synthesis and the precise Research Project Problem section before claiming a novel gap.
4. Transfer the established experiment design into the Method subsections above; mark only actual pending decisions as provisional.
5. Add accuracy and answer-choice NLL to the proposed results table. Plan confidence intervals, reliability diagrams with bin counts, and paired changes from baseline.
6. Create a local LaTeX manuscript and `.bib` bibliography for Overleaf once drafting begins. Use the required course template if one exists; otherwise the proposal's IEEE-style organization is a reasonable starting point. No Overleaf project was accessed or changed in this review.

The main remaining design choices are final sample sizes, the main-run prompt and distractor construction rules, explicit sample allocation, and the last statistical reporting details. Most of the additional writing can proceed using the existing experiment plan.
