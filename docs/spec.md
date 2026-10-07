> For reference, may be updated. Available at https://advanced-ai.eoin.ai/project/spec/

# Project Specification

The Advanced AI project: what to build, what to submit, and how it is assessed.

The project is worth 40% of the grade for this major.

| Milestone | Week | Dates |
|---|---|---|
| Groups formed | 3 | Friday 25 September |
| Proposal gate | 4 | 28 September – 2 October |
| Baseline gate | 5 | 5–9 October |
| Results checkpoint | 6 | 12–16 October |
| Submission and oral defence | 7 | 19–23 October |

## 1 Project overview

### 1.1 Task

Over four weeks, you will work in a group of 2–3 students to conduct one small empirical deep-learning study.

Your project should answer a specific question using:

- an appropriate reference condition or baseline;
- one controlled comparison;
- repeated runs where training is stochastic;
- held-out evaluation;
- reproducible code and results;
- analysis of uncertainty, errors and limitations.

The aim is not to obtain the highest-performing model. You are assessed on the quality of the experiment, the reasoning behind it, and your understanding of the work.

A null result can receive the highest grade.

### 1.2 Core structure

Every project should follow:

> research question → reference condition → controlled change → repeated evidence → conclusion

A small experiment that answers one question well is preferable to a large collection of loosely controlled runs.

### 1.3 Status and amendments

This document is the current specification for the project.

It may be updated to:

- correct errors or ambiguities;
- clarify requirements;
- respond to technical or platform changes;
- resolve issues that could not reasonably have been anticipated when the specification was written;
- provide equivalent arrangements where the original requirement becomes impractical;
- incorporate necessary programme, departmental or institutional requirements.

Material changes affecting the cohort will be clearly announced and recorded in the version history.

A student will not be disadvantaged for work reasonably undertaken in accordance with an earlier published version before a change was announced.

Clarifications that do not change the substance of an assessed requirement may be issued without changing work already completed.

### 1.4 Instructor judgement

This specification cannot anticipate every valid project, technical failure or individual circumstance.

The instructor may exercise reasonable academic judgement where necessary to:

- approve or reject a proposed project or dataset;
- require a project to be narrowed or otherwise modified;
- approve an alternative model, dataset, baseline, metric or experimental design;
- accept equivalent evidence where the specified form is inappropriate;
- adapt a requirement to the characteristics of a particular project;
- resolve ambiguity in how a requirement applies;
- respond to unavailable hardware, discontinued services, broken dependencies or other external constraints;
- adjust operational arrangements such as presentation order, questioning, checkpoint format or submission procedure;
- make reasonable arrangements for extensions, accommodations or other approved circumstances.

Any such judgement should remain consistent with the learning outcomes, academic standard and principles of fairness set out in this specification.

Examples in this document illustrate acceptable approaches. They are not exhaustive requirements and do not prevent another well-justified approach from being approved.

Where a proposed departure from the normal specification is material to your project, obtain approval rather than assuming that it is acceptable.

## 2 Learning outcomes

By completing the project, you should be able to:

- formulate an answerable empirical deep-learning question;
- select and justify an appropriate reference condition;
- implement and explain a deep-learning system;
- design and run a controlled experiment;
- separate training, model selection and final evaluation;
- account for run-to-run variation;
- draw conclusions proportionate to the evidence;
- identify important limitations and threats to validity;
- document compute, provenance and reproducibility;
- account critically for external tools, including AI assistants where used.

## 3 Project tracks

Choose one track. Your exact model, dataset and research question must be approved at the week 4 proposal gate.

Other projects of comparable scope may be approved where they meet the same learning outcomes and experimental requirements.

### 3.1 Parameter-efficient language-model fine-tuning

Fine-tune a small pretrained language model using LoRA, QLoRA or another approved parameter-efficient method.

Possible comparisons include:

- adapter rank;
- adapter placement;
- training-data quantity;
- quantisation configuration.

The reference condition should be appropriate to the task, such as the base model under the same evaluation setup.

### 3.2 Parameter-efficient vision modelling

Adapt a pretrained vision model using LoRA, adapters or another parameter-efficient method.

Possible datasets include:

- Oxford-IIIT Pets;
- EuroSAT;
- a bounded Food-101 subset;
- another approved dataset of comparable scale.

A frozen feature extractor with a linear classification head is one possible reference condition.

Possible comparisons include:

- adapter rank;
- adapter placement;
- data augmentation.

### 3.3 Small transformer trained from scratch

Train a deliberately small transformer on a bounded sequence-modelling task.

Possible reference conditions include:

- an n-gram model;
- a smaller transformer;
- another simple sequence model appropriate to the question.

Possible comparisons include:

- depth;
- number of attention heads;
- context length;
- positional representation.

Training a model from scratch carries no automatic grading advantage.

### 3.4 Small diffusion model

Train or investigate a small diffusion model on a bounded image dataset such as Fashion-MNIST or low-resolution CIFAR-10.

Use an appropriate small diffusion configuration as the reference condition.

Possible comparisons include:

- noise schedule;
- number of sampling steps;
- conditioning strategy;
- another isolated design choice.

You are not required to implement an unrelated VAE or GAN solely to create a baseline.

### 3.5 Graph neural network

Investigate a GNN on a modest graph dataset such as Cora, Citeseer or another approved dataset of comparable scale.

A useful reference condition is an MLP or similar model that uses node features but ignores graph connectivity.

Possible comparisons include:

- message-passing depth;
- aggregation method;
- another isolated architectural choice.

## 4 Experimental requirements

### 4.1 Research question

State the main question precisely enough that the experiment can answer it.

For example:

> Does increasing LoRA rank from 8 to 32 improve held-out performance under a fixed training budget?

Avoid questions that amount only to trying many configurations and reporting the best one.

### 4.2 Reference condition

Your project must include an informative reference condition or baseline.

Explain:

- why it is relevant to the question;
- why it makes the comparison informative;
- which plausible alternatives you considered.

A deliberately weak baseline weakens the experiment even if your main model easily outperforms it.

### 4.3 Controlled comparison

The main experiment should isolate a specific change while keeping other important choices fixed.

An ablation is a specific kind of controlled comparison in which a component is removed or disabled.

Do not use ablation as a general term for every hyperparameter comparison.

### 4.4 Repeated runs

Deep-learning training is stochastic. A difference between two individual runs may therefore be noise rather than an effect of the change being studied.

For the main stochastic comparison:

- use at least three independent training seeds per condition;
- use the same set of seeds across compared conditions where applicable;
- report every run;
- report an appropriate summary of the variation;
- interpret differences in the context of that variation.

Do not report only the best seed.

Three runs provide limited evidence about stability. They do not, by themselves, establish statistical significance.

Deterministic conditions do not require artificial seed replication.

Where the nature or cost of an approved experiment makes this replication strategy inappropriate, an alternative may be agreed in advance. Any limitation this introduces should be reflected in the conclusions.

### 4.5 Data separation

Distinguish between data used for:

- fitting model parameters;
- making modelling or hyperparameter decisions;
- final evaluation.

In the usual case these are:

- training set;
- validation set;
- test set.

Do not use the test set to choose:

- model architecture;
- hyperparameters;
- checkpoints;
- adapter rank;
- stopping point;
- other design decisions.

If a dataset supplies only training and test partitions, normally create a validation split from the original training data.

Alternative evaluation designs may be approved where a conventional train/validation/test split is inappropriate.

### 4.6 Evaluation

State:

- the primary evaluation metric;
- any secondary metrics;
- how checkpoints or models are selected;
- when final evaluation occurs.

The metric should match the research question. More metrics do not automatically make an evaluation stronger.

### 4.7 Interpretation

Your conclusion must match the strength of the evidence.

For example:

> We observed a small advantage for configuration B, but the difference was not consistent relative to the variation across repeated runs.

may be supported where:

> Configuration B is better.

is not.

A null result is a valid result.

## 5 Compute and scope

### 5.1 Compute envelope

You are not expected to purchase compute for this project.

Your project must be scoped to the compute reasonably available through the module or suitable free resources.

Approximately T4-class GPU hardware or equivalent should be treated as the reference scale for project planning unless another arrangement is approved.

Runs should be capable of being checkpointed and resumed where practical.

Cloud hardware type and availability can vary. Scope the project accordingly.

### 5.2 Compute disruption

Hardware availability is an experimental constraint, not a reason to continue indefinitely with an infeasible project.

Where access to expected compute becomes materially unavailable, the instructor may approve changes such as:

- reducing model or dataset size;
- reducing the scope of a comparison;
- substituting equivalent hardware;
- accepting a different replication strategy;
- modifying the project while preserving the learning outcomes.

Raise such problems while they can still be acted upon.

### 5.3 Additional hardware

You may use more powerful hardware if you have access to it.

Additional compute does not earn credit by itself.

Do not increase model scale merely because more hardware is available. Additional capacity is better used for:

- repeated runs;
- robustness checks;
- error analysis;
- reproducibility.

### 5.4 Compute reporting

Report, where applicable:

- hardware used;
- approximate training time;
- number of training runs;
- model size or relevant parameter count;
- major efficiency or memory-saving techniques.

## 6 Groups and individual responsibility

### 6.1 Group size

Projects are completed in groups of 2–3.

Pairs and triples have the same required research scope.

A triple is not required to conduct an additional experiment solely because it has an additional member.

Alternative arrangements may be approved where circumstances justify them.

### 6.2 Shared responsibility

You may divide implementation work within the group.

Every student must nevertheless understand:

- the research question;
- the experimental design;
- the main model;
- the reference condition;
- the evaluation;
- the results;
- the main limitations.

The oral defence tests this understanding individually.

### 6.3 Contribution statement

The final submission must include a concise statement of each member's contribution.

Contribution evidence is considered separately from the academic standard of the project and defence.

## 7 Project schedule

### 7.1 Before week 4

Identify:

- a preferred project track;
- a possible dataset;
- an initial research question;
- your group, formed by Friday 25 September.

### 7.2 Week 4 — proposal gate

Submit:

- research question;
- dataset and provenance;
- dataset licence;
- proposed model;
- proposed reference condition;
- proposed controlled comparison;
- primary metric;
- intended train/validation/test strategy;
- compute plan;
- major foreseeable risks.

The project must fit the available time and compute.

If the initial scope is too large, you will be expected to identify the problem and propose a narrower version.

Approval may include conditions or required changes.

### 7.3 Week 5 — baseline gate

Demonstrate an end-to-end working pipeline.

Evidence should normally include:

- data loading and preprocessing;
- model construction or loading;
- a working training or inference loop;
- metric calculation;
- a completed reference-condition run, or equivalent agreed evidence;
- saved configuration and results.

Where the structure of an approved project makes a conventional baseline run inappropriate, equivalent evidence may be agreed.

### 7.4 Week 6 — results checkpoint

Bring:

- a results table containing individual runs;
- at least one useful figure;
- your current interpretation;
- important anomalies or failure cases.

This checkpoint is formative.

### 7.5 Week 7 — freeze and consolidate

Do not expand the project unless a change is agreed.

Use the final week to:

- verify results;
- complete repeated runs;
- inspect failure cases;
- complete the analysis;
- check reproducibility;
- prepare the presentation;
- prepare for the oral defence.

## 8 Project gates

### 8.1 Gate outcomes

The week 4 and week 5 gates are recorded as:

- Met
- Revision required
- Missed

**Revision required** means that a genuine on-time attempt was made but a specified correction is needed.

The gate counts as Met if the correction is completed by the stated correction deadline or another deadline agreed by the instructor.

A gate is **Missed** when the required evidence is not produced by the applicable deadline and no approved extension or alternative arrangement applies.

The instructor may determine what constitutes equivalent evidence where the normal gate requirements do not fit an approved project.

### 8.2 Grade consequences

The gates cap the maximum final grade:

- one missed gate → maximum A2;
- two missed gates → maximum B1.

Approved extensions, accommodations, technical contingencies and instructor-agreed changes do not count as missed gates.

These caps remain in force after any contribution adjustment.

## 9 Submission

The group submission consists of:

- code and reproducibility material;
- written analysis;
- presentation material;
- contribution statement.

The submission mechanism, file formats and deadlines will be specified separately where not included in this document.

### 9.1 Code and reproducibility

The repository must contain enough information for a technically competent reader to understand and reproduce the main experiment.

Include:

- source code or notebooks;
- dependency or environment information;
- experiment configuration;
- random seeds where applicable;
- instructions for running the main experiment;
- results supporting the submitted analysis;
- a concise README.

The expectation is reproduction-ready practice, not guaranteed bit-for-bit identity across all hardware and software environments.

Avoid undocumented manual steps where practical.

Do not submit:

- passwords;
- access tokens;
- API keys;
- other secrets.

### 9.2 Written analysis

The written analysis should cover:

- research question;
- motivation;
- reference condition;
- controlled comparison;
- dataset and split strategy;
- evaluation method;
- repeated-run results;
- error or failure analysis;
- interpretation;
- limitations and threats to validity;
- compute;
- provenance and licensing;
- ethical considerations where relevant;
- use and verification of external tools.

Distinguish clearly between:

- what you observed;
- what you infer;
- what remains uncertain.

The order and presentation may vary where another structure communicates the work more effectively.

### 9.3 Presentation

The presentation should explain:

- the research question;
- the experimental design;
- the main evidence;
- the conclusion;
- the most important limitation.

Do not spend most of the presentation reviewing generic deep-learning concepts.

Presentation polish is not a separate grading criterion.

## 10 AI assistants and external tools

### 10.1 Permitted use

AI assistants may be used for:

- ideation;
- explanation;
- coding;
- debugging;
- analysis support;
- editing and writing assistance.

Use is permitted, not required.

### 10.2 Responsibility

You remain responsible for all submitted work.

Do not submit code, claims or analysis that you cannot explain.

### 10.3 Reflection

If you use an AI assistant, identify specific examples of:

- where it helped;
- where its output required correction, rejection or verification;
- how you checked the result.

A raw prompt log is not required.

If you do not use an AI assistant, state this briefly and discuss another important verification step in your workflow.

### 10.4 Oral defence

AI assistance is not available during the oral defence.

## 11 Provenance, licensing and responsible use

For each important dataset and pretrained model, identify:

- source;
- relevant licence or usage terms where available;
- restrictions relevant to the project.

Do not assume that publicly downloadable material is unrestricted.

Projects involving personal, confidential, sensitive or otherwise restricted data require prior approval and may not be suitable for this assessment.

Identify important ethical limitations associated with the task, data, model or proposed use where relevant.

The instructor may prohibit or require changes to a proposed dataset, model or task where legal, ethical, licensing, privacy, safety or institutional concerns arise.

## 12 Presentation and oral defence

### 12.1 Format

The normal format is:

- up to 8 minutes for the group presentation;
- brief questions on the group experiment;
- approximately 6 minutes of individual questioning per student;
- follow-up questions where needed to resolve the evidence.

Pairs and triples therefore receive comparable individual questioning time.

Exact timings may be adjusted where necessary while preserving a reasonable opportunity for each student to demonstrate the learning outcomes.

### 12.2 Conditions

During the defence you may inspect your:

- submitted code;
- submitted results;
- submitted presentation.

You may:

- pause to think;
- inspect relevant code;
- ask for a question to be repeated or clarified.

During individual questioning, do not use:

- external web search;
- AI assistants;
- messaging or communication with another person;
- sources outside the submitted project artefacts,

unless an approved accommodation or specific instruction provides otherwise.

The defence assesses understanding and reasoning.

It does not assess:

- accent;
- speaking speed;
- conversational fluency;
- confidence;
- presentation style.

Approved assessment accommodations apply.

### 12.3 Question areas

Each student should expect questions from the following areas.

#### 12.3.1 Mechanism

Examples:

- Walk through one training batch from input to loss and parameter update.
- Show where a key architectural operation is implemented and explain it.

#### 12.3.2 Experimental design

Examples:

- Why is this an appropriate reference condition?
- What alternative baseline did you consider?

#### 12.3.3 Evidence and uncertainty

Examples:

- How large is the run-to-run variation?
- What evidence supports the claimed difference between conditions?

#### 12.3.4 Counterfactual reasoning

Examples:

- What would you expect if the amount of training data doubled?
- What would you expect if this architectural choice changed?
- What competing effects might make the outcome uncertain?

A good answer may be that the direction cannot be predicted confidently from the available evidence, provided you can explain why.

#### 12.3.5 Limitations

Examples:

- What is the main threat to your conclusion?
- What additional experiment would most reduce the remaining uncertainty?

#### 12.3.6 Individual contribution

Examples:

- Which part did you work on most directly?
- Show an implementation or design decision that you personally investigated.

The exact questions are selected according to the submitted project and the evidence already demonstrated. Follow-up questions may be used to clarify the level of understanding.

## 13 Null results

A well-designed experiment can find no clear difference between the conditions tested.

That is not a failed experiment.

For example:

> Across the repeated runs, the observed difference between rank 8 and rank 32 was small relative to run-to-run variation. Under the conditions tested, we therefore found no convincing evidence that the higher rank improved the primary metric.

A conclusion of this kind can support an A-level project when the design, execution and interpretation warrant it.

Do not turn an uncertain or null result into a stronger claim simply because a positive result appears more interesting.

## 14 Final checklist

Before submission, check that:

- [ ] the research question can be stated in one sentence;
- [ ] the reference condition is informative;
- [ ] the main comparison isolates the intended change;
- [ ] important confounding choices are held fixed;
- [ ] stochastic conditions use the agreed repeated runs;
- [ ] the same seed set is used across compared conditions where applicable;
- [ ] individual run results are retained and reported;
- [ ] validation data are used for model-selection decisions where applicable;
- [ ] final evaluation data were not repeatedly inspected during development;
- [ ] the conclusion follows from the evidence;
- [ ] important uncertainty and limitations are stated;
- [ ] the repository contains the information needed to reproduce the main experiment;
- [ ] dataset and model provenance are documented;
- [ ] relevant licences or restrictions are documented;
- [ ] compute is reported;
- [ ] each group member's contribution is stated;
- [ ] every group member can explain the full experiment;
- [ ] external and AI assistance has been accounted for appropriately.

Do not add another experiment merely to make the project look larger while one of these requirements remains unresolved.

## 15 Version history

| Version | Date | Changes |
|---|---|---|
| 0.9 | 24 September 2026 | Initial student draft. |
