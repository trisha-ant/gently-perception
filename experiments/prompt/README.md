# Prompt Ablation Experiments

Exploring dimension 1 of the optimization landscape: how prompt text affects stage classification accuracy.

All variants use the same representation (three-view MIP), same reference examples (2 per stage), and same model (Claude Opus 4.6). Only the prompt text and strategy differ.

## Results

Accuracy on hard stages (1.5fold, 2fold, pretzel).
All variants achieve 100% adjacent accuracy (within 1 stage of ground truth).

| Variant | Exact | 1.5fold (n=49) | 2fold (n=79) | Pretzel (n=193) | Approach |
|---------|-------|----------------|--------------|-----------------|----------|
| **hybrid** | **83.2%** | 59% | 70% | 95% | Stage-adaptive prompt switching |
| scientific | 82.6% | 55% | 76% | 92% | Eggshell fill fraction + body segment counting |
| duration_aware | 81.3% | 65% | 77% | 87% | Duration-aware temporal prior |
| temporal | 81.0% | 63% | 58% | 95% | Soft temporal anchoring, "prefer earlier stage" |
| unified | 78.8% | 53% | 67% | 90% | Merged temporal + scientific (single prompt) |
| ensemble | 79.8% | 57% | 59% | 94% | 3x majority vote with temperature=0.3 |
| compare | 52.0% | 63% | 57% | 47% | Previous timepoint image comparison |
| minimal | 48.5%\* | 12% | 82% | 29% | Stage names only (Sonnet 4.5 baseline) |
| descriptive | 48.0%\* | 18% | 46% | 33% | Projection-grounded descriptions (Sonnet 4.5 baseline) |
| contrastive | — | 25% | 30% | — | Detailed transition descriptions (aborted) |

\* Sonnet 4.5 baselines measured on all stages (n=769), not directly comparable.

## Key Findings

- **Temporal anchoring** ("prefer earlier stage when uncertain") was the single biggest prompt improvement
- **Eggshell fill fraction** is the most discriminative visual criterion for fold stages
- **Annotation quality matters**: the original ground truth was missing `hatched` transitions; fixing this improved measured accuracy by +35pp
- **Ensemble/majority voting doesn't help**: boundary errors are systematic, not stochastic
- **Previous image comparison** helps 1.5fold but catastrophically hurts pretzel (embryo movement != stage change)
- **Duration-aware priors** improve boundary accuracy but trade off pretzel retention
- **No single prompt wins everywhere**: per-stage differences reach 30-70pp across variants at similar overall accuracy

## Variant Descriptions

### Production-ready

- **hybrid.py** — Switches between temporal prompt (for most stages) and scientific prompt (for 2fold boundary). Best overall.
- **scientific.py** — Eggshell fill fraction as primary discriminator. Best for 2fold (76%).
- **temporal.py** — Reference-focused with strong anchoring ("stages change slowly, default to previous"). Best for pretzel retention (95%).
- **duration_aware.py** — Uses stage duration priors to gate transitions. Best for 1.5fold (65%).

### Baselines

- **minimal.py** — Stage names + reference images only. No morphological descriptions.
- **descriptive.py** — One-line projection-grounded description per stage.

### Experimental (didn't improve over baselines)

- **contrastive.py** — Describes transitions between stages rather than stages themselves. Aborted.
- **compare.py** — Shows previous timepoint image for comparison. Hurts pretzel catastrophically.
- **changegate.py** — Change detection gating.
- **ensemble.py** — 3x majority vote. Same errors repeated, not averaged out.
- **unified.py** — Merged temporal + scientific into single prompt. Worse than either alone.

### Multi-turn

- **minimal_multishot.py** — Multi-turn variant of minimal.
- **descriptive_multishot.py** — Multi-turn variant of descriptive.

## Ground Truth Annotation Fix

The original annotations (`data/ground_truth/59799c78_original.json`) covered stages up through pretzel but did not include hatching/hatched transitions. ~240 post-hatching timepoints were implicitly labeled as pretzel.

Extended by identifying hatched transition timepoints using two independent VLM variants (temporal and scientific), which consistently agreed:

| Embryo | Hatched at | Pretzel duration |
|--------|-----------|-----------------|
| embryo_1 | T139 | 49 timepoints |
| embryo_2 | T123 | 43 timepoints |
| embryo_3 | T110 | 41 timepoints |
| embryo_4 | T157 | 60 timepoints |

Corrected annotations: `data/ground_truth/59799c78.json`. Verify with `make_filmstrip.py`.
