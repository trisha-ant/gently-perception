# Perception Agent Development

You are developing VLM-based perception functions that classify *C. elegans*
embryo developmental stages from 3D light-sheet microscopy images.

## The Task

Given a three-view orthogonal projection (XY, YZ, XZ max-intensity projections)
of a developing embryo, classify it into one of 8 stages:

**early → bean → comma → 1.5fold → 2fold → pretzel → hatching → hatched**

Reference images for each stage are provided in the prompt. The model also
receives temporal context (what stages were observed at recent timepoints).

## How to Iterate

1. **Read current results** in `data/results/` to understand what works and what doesn't
2. **Modify or create** a perception function in `perception/`
3. **Register it** in `perception/__init__.py`
4. **Run evaluation**: `python run.py --variant your_variant --quick`
5. **Analyze failures** from the output (per-stage accuracy, confusion patterns)
6. **Repeat**

## Files You Should Modify

- `perception/*.py` — perception function variants (this is your workspace)
- `perception/__init__.py` — register new variants here

## Files You Should NOT Modify

- `perception/_base.py` — shared API wrapper, parsing, prompt builders (fixed infrastructure)
- `benchmark/` — evaluation harness (testset, metrics, ground truth)
- `run.py` — evaluation entry point
- `data/` — test data and reference images

## The Metric

**Primary**: Exact accuracy (predicted stage == ground truth stage)
**Secondary**: Adjacent accuracy (within 1 stage)

Use `--quick` for fast iteration (~120 predictions, ~2 min per variant).
Use full runs for final evaluation (~769 predictions).

## Perception Function Signature

Every perception function must have this exact signature:

```python
async def perceive_xxx(
    image_b64: str,              # Base64 JPEG of three-view projection
    references: dict[str, list[str]],  # stage -> [base64 reference images]
    history: list[dict],         # previous timepoints: [{timepoint, stage}]
    timepoint: int,              # current timepoint number
    midplane_b64: str | None = None,  # optional: single XY z-slice at z=Z//2
) -> PerceptionOutput:           # from perception._base
```

`midplane_b64` is only passed if your function declares it (or `**kwargs`).
The runner inspects the signature, so existing variants don't need to change.

## Available Utilities (from `perception._base`)

- `call_claude(system, content)` — single API call, returns response text
- `call_claude_conversation(system, messages)` — multi-turn, returns response text
- `response_to_output(raw)` — parse JSON response into PerceptionOutput
- `build_reference_content(references)` — build cached reference image blocks
- `build_history_text(history)` — format temporal context
- `STAGES` — ordered list of valid stage names
- `DEFAULT_MODEL` — current model ID

## What We Know So Far

### Baselines (single-shot, no tools, Sonnet 4.5 — re-run with Opus 4.6)

| Variant | Exact | Best stage | Worst stage |
|---------|-------|------------|-------------|
| minimal (stage names only) | 48.5% | early 98% | 1.5fold 12% |
| descriptive (projection descriptions) | 48.0% | early 98% | 1.5fold 18% |

**Model**: Now using `claude-opus-4-6` (`perception/_base.py`).

### Failed Experiments

**Multi-shot reconsideration** (annexure 2): Asking the model to "look again"
after initial classification reduces accuracy from 48% to 13-15%. The model
treats follow-up as implicit negative feedback and changes correct answers.
Early stage drops from 98% to 5%.

**Tool use** (from gently baseline): Letting the model request 3D rotations
and previous timepoints hurts accuracy (23% with tools vs 35% without).

### Key Challenges

- **Pretzel** (29-33% accuracy, 56% of test data) — most common and hardest
- **1.5-fold** (12-18%) — confused with comma and 2-fold
- **Later stages** share subtle morphological differences
- **Early stage** is easy (98%) but dominates the dataset

### Promising Directions to Explore

- Different prompt structures (contrastive, stage-pair comparisons)
- Ensemble/majority vote across independent calls (avoids the "are you sure" bias)
- Stage-specific prompts for confusable pairs
- Different image representations (separate TOP/SIDE views instead of combined)
- Temperature > 0 with self-consistency
- Chain-of-thought that focuses on specific morphological features
- Two-pass: coarse (early/mid/late) then fine classification
