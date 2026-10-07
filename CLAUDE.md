# CLAUDE.md

## How to respond

- **All responses:** follow the `i-have-adhd:i-have-adhd` skill. Next action first, numbered steps, concrete time estimates, restate where we are each turn, end with one action under 2 minutes. Load the skill at the start of a session if it isn't already active.
- **All code** (writing, editing, reviewing): follow the `ponytail:ponytail` skill (full). Stdlib and installed deps first, shortest working diff, one runnable check for non-trivial logic.
- **Every piece of code must be defensible in the oral** (spec 10.2, 12). After writing code, add 1–3 plain-English lines on what it does and why, so the owner can explain it without AI.
- **.txt files:** clean plain text with "Heading:" lines. No `=====` banners, ALL-CAPS banner headings or dotted leaders.
- Never use the test set to choose anything. If a request would do that, say so before acting.

## Decision log

Record **important or semi-important** project decisions in `docs/decision-log/<name>.md` (`dylan.md`, `eddie.md`, `paddy.md`). A personal decision goes in the current user's log (see "Who the user is"). A team decision goes in the log of whoever owns that area. Only log a suggestion from Claude once the person has accepted it.

Log it only if it could change the results, or the write-up or oral would need to justify it:
- **Log:** data source, split method and parameters (cell size, buffer, N), model or LoRA settings, metrics, seeds, hyperparameter choices, scope changes, fixes to the experimental design
- **Don't log:** housekeeping such as `.gitignore`, file formats, folder names, editor settings, tooling, or doc layout

- Markdown, newest entry at the top, written ADHD-short and direct: no filler, one fact per line
- Each entry is exactly this:

```markdown
## 7 Oct 2026 - <decision in a few words>
- **Decision:** <what was decided, 1-2 lines>
- **Why:** <the reason, 1-2 lines>
- **Affects:** <files, steps or plan sections it changes>
```

## The project

4th-year Advanced AI group project (40% of the module). Track 3.2, parameter-efficient vision modelling. Team: **Dylan** (data + spatial split), **Eddie** (model + LoRA training), **Paddy** (frozen reference, metrics, analysis).

**Who the user is:** check at the start of each session with `git config user.name`, then use that person's name, area and decision log.
- `Dylan Teehan` → Dylan
- `eddiephelanfortyfive` → Eddie
- Anything else (e.g. Paddy, not yet recorded) → ask who it is before logging decisions, then add their git name to this list

**Research question:** With limited labelled data and a spatial split, does increasing LoRA rank beyond a small value improve held-out predictive performance on EuroSAT, measured by cross-entropy, or does performance level off?

**Hypothesis:** the ViT only needs a small adjustment for satellite images. Supported if rank 4 ≈ rank 32 and both beat frozen. "≈" (fixed in advance): mean paired r32 − r4 test CE difference is smaller than the std of the 5 paired differences.

| | |
|---|---|
| Data | EuroSAT RGB, 27,000 Sentinel-2 tiles, 64×64 px at 10 m/px, 10 classes (2,000–3,000 per class). MIT licence. Zenodo record 7711810 |
| Model | `google/vit-base-patch16-224-in21k` (ImageNet-21k, no satellite data, Apache 2.0) |
| Conditions | Frozen ViT + linear head (reference) · LoRA r=4 · LoRA r=32, on query + value in all 12 blocks |
| Seeds | 5 per condition, paired (seed s → same training subset, data order, head init in every condition) = 15 runs |
| Primary metric | Test cross-entropy (nats). Secondary: balanced accuracy. Report every run + mean ± std |
| Selection | Checkpoint = lowest validation CE (early stopping). Test evaluated once per run, after design freeze |
| Compute | Colab free T4, three accounts in parallel, checkpoints to Drive after each run |

**Locked decisions** (detail in `docs/project-plan-v2.txt`):
1. `lora_alpha = r`, so rank doesn't change the effective learning rate
2. Frozen reference caches all 8 rotation/flip views and samples one per epoch
3. Training subset changes with the seed, shared across conditions
4. Per-condition LR sweep on validation, same grid. Batch size, steps, weight decay, data held identical
5. No label smoothing. Plain CE for training and reporting
6. Baseline gate shows validation numbers only
7. A class is "uneven" in test if below half its own expected share (15% of its class size)
8. Preprocessing: bicubic 64→224, mean/std 0.5. Train only: random 90° rotation + flip. No crops, no colour jitter

**Spatial split (Dylan):** tile centres from the MS GeoTIFFs → EPSG:3035 (`data/coords.csv`). Cell size L = distance where same-label fingerprint similarity falls to background (~0.43). Whole cells go to train/val/test (~70/15/15) via StratifiedGroupKFold, a buffer of train cells touching val/test is removed, and code checks for no shared cells. Output `data/splits/split_v1.csv` with the same columns and format as `split_temp.csv`. Training subset = N per class via `make_subset(pool, N, seed)`, with N chosen in a validation-only pilot.

## Deadlines

| Gate | Dates | Needs |
|---|---|---|
| Baseline | 5–9 Oct | End-to-end pipeline, completed frozen reference run, saved config + results |
| Results checkpoint | 12–16 Oct | Table of every run, ≥1 figure, interpretation, anomalies |
| Submission + oral | 19–23 Oct | Code, write-up, ≤8 min slides, contribution statement, individual oral |

Design freeze (git tag `design-freeze`) before any test evaluation. One missed gate caps the grade at A2, two at B1.

## What the spec requires (`docs/spec.md`)

- **Structure:** research question → reference condition → controlled change → repeated evidence → conclusion. No "try many configs, report the best"
- **Seeds:** ≥3 per condition, the same seeds across conditions, report every run plus the variation. Never only the best seed. Don't claim significance from a few seeds
- **Data separation:** the test set is never used for architecture, hyperparameters, checkpoints, rank, stopping or any other design choice
- **Conclusions** proportionate to the evidence. A null result is valid and can score top marks (spec 13)
- **Reproducibility:** code, pinned env, configs, seeds, run instructions, results, README, no secrets. Report hardware, train time, number of runs and parameter counts
- **Write-up:** separates observed / inferred / uncertain. Covers limitations, provenance, licences and ethics. Includes an AI-use reflection with specific examples of help, corrections and how they were checked
- **Oral:** every member must explain the whole experiment unaided (mechanism, design, evidence, counterfactuals, limitations, own contribution)
- **Week 7:** freeze. Don't add experiments while any spec 14 checklist item is open

## Repo map

- `docs/spec.md` (spec), `docs/project-context.md`, `docs/project-plan-v2.txt` (current plan, pitfalls, oral prep), `docs/status.md` (3-line status per person)
- `src/` (pipeline code), `scripts/` (one-off builders), `configs/` (one YAML per condition), `results/runs.csv` (one row per run)
- `data/raw/` (EuroSAT zips + unzipped, git-ignored), `data/coords.csv`, `data/fingerprints/` (Paddy's frozen CLS fingerprints, `.npy` git-ignored, see the explainer), `data/splits/`
- Python env: `.venv/`, deps pinned in `requirements.txt`
