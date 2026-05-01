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

### Judge ensemble: 90.6 ± 1.3% (N=3 pairs) ✅

Implemented as a post-processing step on archived hybrid@4.6 × vote3_mm@4.7
runs. The judge (opus-4-7, effort=xhigh, adaptive thinking) gets the image,
history, and both experts' predictions + reasoning, and must pick one.
Falls back to "more advanced" on parse failure.

| pair | hybrid | vote3_mm | judge on disagree | ensemble |
|---|---|---|---|---|
| 1 | 82.8 | 81.1 | 88.5 | 90.6 |
| 2 | 83.3 | 80.3 | 81.8 | 89.3 |
| 3 | 79.0 | 81.1 | 92.3 | 91.8 |
| **mean ± std** | 81.7 ± 2.4 | 80.8 ± 0.5 | **87.5 ± 5.3** | **90.6 ± 1.3** |

Per-stage (pair 1): 1.5fold 65.9, 2fold 94.9, pretzel 96.2. **The first
harness change that lifts all three stages.** 1.5fold matches hybrid's
best, 2fold near vote3_mm's best, pretzel beats both.

vs hybrid@4.6: +8.9pp, t≈5.6 — highly significant.
vs oracle ceiling (93.6): captures ~76% of the available headroom.

Cost: 1 (hybrid) + 3 (vote3_mm) + ~0.22 (judge on ~22% of frames) ≈ 4.2
calls/frame. A cheaper variant pairing hybrid + multimeasure (1 call) gives
the same oracle ceiling (93.7%) and should work at ~2.2 calls/frame — not
yet tested.

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

**The judge ensemble is the best harness found: 90.6 ± 1.3%, +8.9pp over
the previous best.** It works because hybrid@4.6 and vote3_mm@4.7 are
complementary (different strengths per stage), agreement is high-precision
(91.2%), and a cold judge with both arguments correctly arbitrates 87.5%
of disagreements.

Production notes:
- Cost is ~4.2× a single classifier. The cheaper hybrid+multimeasure
  pairing (~2.2×) is worth testing before deployment.
- The judge prompt sees no ground truth; it receives only the image,
  history, and the two competing predictions + reasoning.
- Remaining gap to 93.6% oracle: the ~13% of disagreements the judge gets
  wrong, plus the ~9% of agreements that are both-wrong.

Further directions:
- Test hybrid + multimeasure pairing (half the cost)
- Test a 2-of-3 committee (hybrid@4.6, multimeasure@4.7, a third variant)
  with majority vote, using the judge only on 3-way splits
- The remaining errors are almost all at stage boundaries — a monotone
  smoothing pass on the ensemble output might recover another 1-2pp
- Restore embryo_4 (88 frames missing from all evals)

## Not pursuing

- More prompt variants for per-frame classification (exhausted trade-off space)
- HMM/Viterbi on vote distribution (votes are too unanimous — no soft signal)
- Prediction-based routing (failed in hybrid_fillpct)
- Restore embryo_4 (data task, flagged separately)
