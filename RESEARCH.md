# Research investigation: harness modifications for overall accuracy

Date: 2026-05-01. Branch: `trisha/gently-opus47-experiments` @ `478f2ae`.

## Motivation

After ~40 replicated configurations (PR #6), the two best harnesses are
statistically tied (hybrid@4.6: 81.7 ± 2.4, vote3_mm@4.7: 80.8 ± 0.5) and
every prompt change trades one stage for another. This investigation asks:
**what headroom exists above ~82%, and what harness changes can capture it?**

## Analysis findings (computed from archived run JSONs, no new API calls)

All numbers on n=233 hard-stage frames (1.5fold+2fold+pretzel, embryos 1-3).

### 1. Oracle ensemble ceiling is 93.6 ± 0.6%

If we could pick the better of hybrid@4.6 and vote3_mm@4.7 per frame, we'd
get 93.6% — **+11.9pp over hybrid alone.** The two models are highly
complementary: hybrid wins 1.5fold (65.9 vs 44.7) and pretzel (92.5 vs 84.5);
vote3_mm wins 2fold (97.7 vs 68.4). Averaged over all 3x3 run pairs.

### 2. Agreement is high-precision

The two models agree on 77.7% of frames with 91.2% accuracy. On the 22.3%
of frames where they disagree, *exactly one is right in almost every case*
(the oracle reaches 93.6%, not 100%, so both-wrong is ~6%).

Disagreement patterns (run 1 × run 1, n=52):

| GT | hybrid | vote3 | n | right |
|---|---|---|---|---|
| pretzel | **pretzel** | 2fold | 16 | hybrid |
| 2fold | pretzel | **2fold** | 11 | vote3 |
| 2fold | 1.5fold | **2fold** | 10 | vote3 |
| 1.5fold | **1.5fold** | 2fold | 8 | hybrid |

The disagreements are mostly adjacent-stage pairs. No single heuristic wins:
"more advanced" gets 83.3 ± 1.9, "less advanced" 79.3 ± 0.9. The +1.6pp from
"more advanced" is marginal (t~1.0).

### 3. Errors are transition-timing misses, not per-frame noise

Ground truth is **100% monotone** per embryo, and both models' predictions
are >99% monotone (cascade via history keeps them consistent). Errors come
in **contiguous blocks of 8-17 frames** — the model places the transition
boundary 8-17 frames early/late.

Transition-point alignment (run 1):

| embryo | GT transitions | hybrid | vote3 |
|---|---|---|---|
| 1 | 1.5f→2f @ T70, 2f→pretzel @ T90 | T60 (−10), T80 (−10) | T60 (−10), T89 (−1) |
| 2 | T60, T80 | T60 (0), T78 (−2) | T61 (+1), T82 (+2) |
| 3 | T50, T69 | T60 (+10), T72 (+3), spurious T109 | T42 (−8), T86 (+17) |

**The task effectively reduces to estimating 6 integers.** Each 1-frame
transition error = 1 wrong prediction. hybrid's errors sum to ~40 = exactly
the observed error count.

### 4. vote3's votes are not a usable posterior

97.9% of vote3's 3-way votes are unanimous. Adaptive thinking doesn't inject
enough diversity for the vote distribution to act as a soft posterior, so
Viterbi/HMM decoding on vote3's votes would have almost nothing to smooth.

## Harness modifications, ranked by expected impact

### A. Judge ensemble (highest ROI)

Run both models; on agreement (78% of frames, 91.2% acc) use the agreement.
On disagreement (22%), send a **judge call** with the image, history, and
both experts' predictions + reasoning, and ask the judge to pick.

Expected accuracy as a function of judge accuracy on disagreements:
- Judge @ 50% (coin flip): 82.0%
- Judge @ 65%: 85.4% (+3.7 vs hybrid)
- Judge @ 80%: 88.7% (+7.0)
- Judge @ 100% (oracle): 93.1%

Cost: ~1 extra call per disagreement = ~22% more calls total (on top of
hybrid's 1 + vote3's 3 = 4 calls/frame baseline). Note: simpler variant
with hybrid + 1-vote multimeasure (2 calls/frame + judge) may be sufficient.

The judge has a huge advantage over cold classification: it knows which two
stages are under consideration, can see both experts' reasoning, and only
needs to resolve a binary choice on a specific image.

### B. Transition refinement (highest potential, higher complexity)

Reframe the task: instead of classifying 233 frames, **estimate ~6
transition points.** Two-pass harness:

1. Coarse pass: run per-frame classifier, find predicted transition windows
2. Refine pass: for each detected transition T, sample a strip of frames
   around T and ask the model "which of these 7 frames is the LAST one
   that is still stage X?" — a temporal comparison task the model has never
   been given.
3. Snap all frame labels to their side of the refined boundary.

This uses the model's vision in a fundamentally new way — *temporal
comparison*, not per-frame classification — and may sidestep the boundary
ambiguity entirely.

Cost: coarse pass + ~3 extra multi-frame calls per embryo = cheap.
Risk: if the model can't do multi-frame comparison, falls back to coarse.

### C. Temporal window classification

Show the model current + previous 2 + next 2 frames (or current + prev 2 if
causal). "Classify the middle frame." Similar idea to B but applied
per-frame. The `compare` variant (52% on 4.6) sent a single previous image
and hurt, but sending a strip with explicit position markers is different.

Cost: 1 call/frame but with 3-5x images. Moderate.

### D. "More advanced" tiebreak (free, small gain)

When hybrid and vote3 disagree, pick the more-advanced prediction. Already
computed: 83.3 ± 1.9 (+1.6 vs hybrid, not significant). Zero cost. Worth
committing as a post-processing option but not the main path.

## Experimental results

### Judge ensemble — ablation found a GT leak; corrected result below

Initial test reported 90.6 ± 1.3% (N=3 pairs). **An ablation revealed this
was inflated by a ground-truth leak:** the judge's history context was
built from GT stages for all frames, but the real harness uses the model's
own *predicted* stages for scored frames. GT and predicted history differ
on 56% of disagreement frames — exactly where the judge operates.

**Ablation (pair 1, 52 disagreement frames):**

| Config | judge acc on disagree | ensemble |
|---|---|---|
| GT history + reasoning + refs (leaky) | 90.4% | 91.0% |
| No history | 50.0% | 82.0% |
| GT history, no reasoning | 84.6% | 89.7% |
| GT history, no references | 80.8% | 88.8% |
| **Predicted (hybrid's) history** | 63.5% | 85.0% |
| "More advanced" heuristic (no judge) | 61.5% | 84.5% |

History is the dominant signal (B: no history = coin flip). GT history
gives the judge a ~27pp boost it wouldn't have in production. Reasoning
helps +5.8pp and references +9.6pp — both legitimate, non-leaky.

**Corrected (sequential, honest history) judge ensemble: 83.3 ± 0.7%**

The production-realistic configuration runs the judge sequentially per
embryo, feeding it the *ensemble's own* predicted history built
frame-by-frame. N=3 pairs:

| pair | hybrid | vote3_mm | judge on disagree | ensemble |
|---|---|---|---|---|
| 1 | 82.8 | 81.1 | 51.9 | 82.4 |
| 2 | 83.3 | 80.3 | 58.2 | 83.7 |
| 3 | 79.0 | 81.1 | 63.1 | 83.7 |
| **mean ± std** | 81.7 ± 2.4 | 80.8 ± 0.5 | **57.7 ± 5.6** | **83.3 ± 0.7** |

Per-stage: 1.5fold 56.9 ± 7.5, 2fold 95.5 ± 2.0, pretzel 86.0 ± 0.4.

**+1.6pp vs hybrid, t≈1.1 — not significant.** The judge is only 57.7%
accurate on disagreements (vs 90.4% with GT history). The ensemble's gain
comes almost entirely from agreement filtering, not from the judge.

The cascade problem applies to the judge too: if the judge errs at T70,
that corrupted history feeds the judge at T71 and beyond. The sequential
result (83.3%) is *lower* than the parallel proxy with hybrid-only history
(85.0%) because the ensemble's own history becomes more corrupted as
judge errors accumulate. This is the same dynamic the classifiers have.

The one clear win: variance drops from ±2.4 to ±0.7. The ensemble gives a
tighter estimate, useful for benchmark comparisons, but not a higher one.

### Transition refinement: 76.8% (−6.0pp) ❌

Sent 8-frame windows around coarse transitions with "find the first frame
showing stage B." The model moved some transitions closer (embryo_1:
T60→T63 toward GT T70) but moved others *away* (embryo_2: T60→T66 when GT
is T60; embryo_3: T72→T78 when GT is T69). Net negative. Adjacent stays
100% so it's not catastrophic, but the multi-frame comparison task doesn't
reliably outperform the per-frame classifier on boundary detection.

Hypothesis for why it failed: the window (±12 frames around coarse) may not
contain the true boundary when coarse is badly off, and the model has a
systematic "pick later" bias in ordered sequences.

## Conclusion and recommendation

**The judge ensemble does not meaningfully improve accuracy once the GT
leak is removed: 83.3 ± 0.7 vs hybrid@4.6's 81.7 ± 2.4, t≈1.1, not
significant.** The 90.6% headline was an artifact of the GT history leak.

The oracle ceiling of 93.6% is real, but it turns out to require an
information source the production system doesn't have. The judge with GT
history reaches it because GT history disambiguates the two candidates
perfectly (stages are monotone, so knowing the recent history constrains
the current stage tightly). Without it, the judge is barely better than a
coin flip (57.7%).

**What the investigation actually established:**

1. The errors are transition-timing blocks, not per-frame noise.
2. hybrid@4.6 and vote3_mm@4.7 are highly complementary (oracle 93.6%).
3. Reaching that ceiling requires history accuracy near 100%, which is
   circular — the history IS the ensemble's output.
4. The judge, reasoning, and references all contribute — but only when the
   history is already accurate. They don't help the judge overcome a
   corrupted history.
5. The ensemble's one real benefit is variance reduction (±0.7 vs ±2.4).

**Lessons about evaluation methodology:**
- **Always ablate new harnesses before reporting.** The GT history leak was
  a single line of code (`tc.ground_truth_stage` vs model prediction) and
  inflated the result by ~6pp — enough to turn a null result into a
  "breakthrough."
- History/temporal-context signals are particularly leak-prone because the
  harness uses GT for skipped frames (by design) and the line between
  "legitimate lead-in" and "oracle leakage" is subtle.
- Ablation also quantified the contribution of reasoning (+5.8pp) and
  references (+9.6pp) to the judge's accuracy, conditional on GT history.

**Recommendation: stick with `hybrid` on opus-4-6 (what's on master).**
It's 81.7 ± 2.4 at 1× cost. Nothing tested in this investigation or the
earlier experiment loops defensibly beats it. The judge ensemble ties it
at 4.2× cost.

Further directions if pursuing this more:
- Restore embryo_4 (88 frames missing from all evals)
- The true bottleneck is transition-timing. A harness that estimated
  transition points directly rather than classifying per-frame could
  theoretically do much better — but the transition-refinement experiment
  (multi-frame window queries) failed (−6pp). A different approach is needed.
- Cross-modal help: the GT-history result (90.6%) shows that if the
  classifier had an oracle-quality temporal signal, it would nearly solve
  the task. Is there an external temporal signal (wall-clock, imaging
  metadata, embryo size) that correlates with GT transitions?

## Not pursuing

- More prompt variants for per-frame classification (exhausted trade-off space)
- HMM/Viterbi on vote distribution (votes are too unanimous — no soft signal)
- Prediction-based routing (failed in hybrid_fillpct)
- Restore embryo_4 (data task, flagged separately)
