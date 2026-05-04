# gently-perception

VLM-based perception harness for microscopy stage classification.

Two-layer architecture: a **harness** (pip-installable library) provides fixed infrastructure, and **experiments** use the harness to explore different perception strategies organized by the dimension of the optimization landscape they vary (prompt, representation, tools).

## Usage

### From gently (production)

```python
from gently_perception import Perceiver

perceiver = Perceiver()  # self-contained: loads its own examples and default strategy

result = await perceiver(embryo_id, timepoint, image_b64, timestamp)
result.stage      # "pretzel"
result.reasoning  # "dense coiled mass filling eggshell"
```

### Standalone (benchmarks)

```bash
pip install -e .
python setup_data.py           # download volumes from HuggingFace (~35 GB)
cd experiments
python run.py --variant hybrid --stages pretzel 2fold 1.5fold --force
```

## Results

`hybrid` on opus-4-6 (master config: temperature=0, no thinking) achieves **81.7 ± 2.4%** exact accuracy on hard stages (N=3 replicated, n=233 frames, embryos 1–3). See [experiments/prompt/README.md](experiments/prompt/README.md) for the per-variant table and [RESEARCH.md](RESEARCH.md) for the full investigation.

### Key findings (April–May 2026 investigation)

~50 configurations tested across z-slice subagents, opus-4.7 migration, prompt variants, ensembles, and routing. **Nothing defensibly beats `hybrid` on master.**

| Config | Exact (N≥3) | Notes |
|---|---|---|
| **hybrid@4.6 (master)** | **81.7 ± 2.4** | best replicated; 1× cost |
| sequential judge ensemble | 83.3 ± 0.7 | +1.6pp not significant; 4.2× cost |
| vote3_mm@4.7 | 80.8 ± 0.5 | best 2fold (97.7%); 3× cost |
| multimeasure@4.7 | 80.3 ± 4.8 | 1× cost |
| fillpct@4.6 / @4.7 | 77.5 / 76.1 | 4.6 ≈ 4.7 head-to-head |

**The bottleneck:** errors are contiguous 8–17-frame blocks at stage *transition boundaries*, not per-frame noise. Every routing/ensembling strategy tested (judge, transition-refinement, timepoint-based routing) fails the same way — the signal needed to route correctly *is* the transition timing, which is the thing being estimated. Breaking the ceiling (~93.6% oracle) requires a non-circular external signal: wall-clock developmental time, classical-CV body-length measurement, or sparse manual anchor frames.

**Methodology lessons:**
- Adaptive thinking adds ~5pp run-to-run variance on both 4.6 and 4.7. **Always run N≥3 and report mean±std.** Single-run results are not defensible.
- **Ablate before reporting.** A one-line GT-history leak in the judge ensemble inflated a null result (83.3%) to a "breakthrough" (90.6%).
- embryo_4's volumes are missing on disk (1 of ~190 .tif files). All recent evals are silently on n=233 (embryos 1–3) instead of n=321.

## Architecture

```
gently-perception/
|
+-- gently_perception/          THE HARNESS (pip install, fixed infrastructure)
|   |-- perceiver.py            Perceiver + Session + Observation
|   |-- api.py                  call_claude, parsing, content builders
|   |-- temporal.py             analyze_temporal() pure function
|   |-- examples.py             load reference images from package data
|   |-- organism.py             OrganismConfig + C. elegans defaults
|   +-- types.py                PerceptionOutput(stage, reasoning)
|
+-- experiments/                AUTORESEARCH WORKSPACE (organized by dimension)
|   |-- CLAUDE.md               agent instructions
|   |-- program.md              experiment history & findings
|   |-- run.py                  benchmark runner
|   |
|   |-- prompt/                 dimension 1: prompt ablation (15 variants)
|   |   |-- hybrid.py           83.2% - stage-adaptive prompt switching
|   |   |-- temporal.py         81.0% - temporal anchoring
|   |   |-- scientific.py       82.6% - eggshell fill fraction
|   |   +-- ...
|   |
|   |-- representation/         dimension 2: volume-to-image (future)
|   +-- tools/                  dimension 5: agentic workflows (future)
|
+-- benchmark/                  FIXED evaluation infrastructure
|   |-- testset.py              volume loading, 3-view projections
|   |-- ground_truth.py         stage transition annotations
|   +-- metrics.py              accuracy, confusion matrix
|
+-- data/
    |-- examples/               reference images per stage (ships with package)
    |-- volumes/                3D light-sheet data (downloaded)
    |-- ground_truth/           stage annotations
    +-- results/                experiment outputs
```

## Adding a New Experiment

1. Create `experiments/prompt/my_experiment.py`:
   ```python
   from gently_perception.api import call_claude, build_reference_content, response_to_output

   MY_PROMPT = "..."

   async def perceive_my_experiment(image_b64, references, history, timepoint, **kw):
       content = build_reference_content(references)
       content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}})
       raw = await call_claude(system=MY_PROMPT, content=content)
       return response_to_output(raw)
   ```
2. Run: `cd experiments && python run.py --variant my_experiment --stages pretzel 2fold 1.5fold --force`

See `experiments/program.md` for detailed experiment history and promising directions.

## Related

- [gently](https://github.com/pskeshu/gently) — the microscopy agent framework
- [benchmark dataset](https://huggingface.co/datasets/pskeshu/gently-perception-benchmark) — volumes, ground truth, paper
- [autoresearch](https://github.com/karpathy/autoresearch) — inspiration for this framework pattern
