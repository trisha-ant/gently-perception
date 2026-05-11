# gently-perception harness design — diagrams

Generated 2026-05-11 from branch `trisha/gently-opus47-experiments` @ `4db3452`.

---

## 1. Current architecture — three-way duplication

```mermaid
graph TB
    subgraph root["repo root"]
        RUN1["run.py<br/>(runner v1)"]
        PROG1[program.md]
        CLAUDE1[CLAUDE.md]
    end

    subgraph perc["perception/ (legacy workspace)"]
        BASE["_base.py<br/>model=opus-4-7<br/>thinking=adaptive"]
        REG["__init__.py<br/>29-line manual registry"]
        V1[29 variant files<br/>minimal, hybrid, mm_v4...]
    end

    subgraph gp["gently_perception/ (pip library)"]
        API["api.py<br/>model=opus-4-6<br/>NO thinking"]
        PERCEIVER[perceiver.py<br/>Session state]
        TYPES[types.py]
    end

    subgraph exp["experiments/ (new workspace)"]
        RUN2["run.py<br/>(runner v2, ~95% dup of v1)"]
        PROG2["program.md (stale copy)"]
        CLAUDE2["CLAUDE.md (stale copy)"]
        PROMPT["prompt/<br/>15 variant files<br/>(copies of perception/)"]
        JUDGE["judge_ensemble/<br/>6 ad-hoc scripts<br/>hardcoded /tmp/ paths"]
    end

    subgraph bench["benchmark/"]
        TESTSET["testset.py<br/>volume→5 images<br/>recomputed every run"]
        GT[ground_truth.py]
        METRICS["metrics.py<br/>DEAD: imports nonexistent<br/>BenchmarkReport"]
    end

    RESULTS[("data/results/<br/>{variant}_{stages}.json<br/>OVERWRITTEN on --force")]

    RUN1 --> REG
    RUN1 --> TESTSET
    RUN1 --> RESULTS
    REG --> V1
    V1 --> BASE

    RUN2 --> PROMPT
    RUN2 --> TESTSET
    RUN2 --> RESULTS
    PROMPT --> API

    PERCEIVER -.->|"sys.path hack"| PROMPT
    JUDGE -.->|"sys.path.insert"| BASE
    JUDGE -.->|"sys.path.insert"| RUN1
    JUDGE -.-> TESTSET
    JUDGE --> TMP[("/tmp/*.json<br/>ephemeral")]

    BASE -.-|"DUPLICATE<br/>different defaults!"| API
    RUN1 -.-|"DUPLICATE"| RUN2
    V1 -.-|"COPIES"| PROMPT
    PROG1 -.-|"STALE COPY"| PROG2
    CLAUDE1 -.-|"STALE COPY"| CLAUDE2

    classDef dup fill:#ffe0e0,stroke:#c00
    classDef dead fill:#ddd,stroke:#888,stroke-dasharray:4
    classDef escape fill:#fff0d0,stroke:#e90
    class BASE,API,RUN1,RUN2,V1,PROMPT,PROG1,PROG2,CLAUDE1,CLAUDE2 dup
    class METRICS dead
    class JUDGE,TMP escape
```

**Red** = duplicated · **Grey dashed** = dead code · **Yellow** = escapes the harness

---

## 2. Current eval data flow — where the GT leak lives

```mermaid
sequenceDiagram
    autonumber
    participant R as run.py
    participant TS as OfflineTestset
    participant GT as ground_truth.json
    participant H as history[]
    participant P as perceive_fn
    participant M as Claude API

    loop each timepoint T
        TS->>GT: get_stage_at(embryo, T)
        GT-->>TS: gt_stage
        TS-->>R: TestCase(image, gt_stage)

        alt T not in --stages filter (skipped)
            rect rgb(255, 235, 200)
            R->>H: append({T, stage: gt_stage})
            Note over R,H: BY DESIGN: GT used as<br/>lead-in context for early frames
            end
        else T is scored
            R->>P: perceive(image, history=H, T)
            P->>M: system + refs + history + image
            M-->>P: {"stage": pred}
            P-->>R: PerceptionOutput(pred)
            R->>H: append({T, stage: pred})
            R->>R: score: pred == gt_stage ?
        end
    end

    rect rgb(255, 210, 210)
    Note over R,M: judge_replicate.py:131 built history from gt_stage<br/>for ALL frames including scored ones →<br/>judge saw oracle context → 90.6% (false) vs 83.3% (true)
    end
```

The orange band is the **legitimate** GT injection (lead-in for skipped frames). The red band is the **leak**: nothing structurally separates "GT for skipped frames" from "GT for scored frames" — both flow through the same `history` list. One-off scripts that rebuild `history` themselves (judge_replicate.py) get this wrong.

---

## 3. Current result lifecycle — why replication is manual

```mermaid
graph LR
    A["python run.py<br/>--variant fillpct --force"] --> B[("results/fillpct_*.json<br/>83.3%")]
    B -->|"--force again"| B2[("results/fillpct_*.json<br/>71.2%<br/>(83.3% GONE)")]
    B2 -.->|manual| C["cp to /tmp/fillpct_r1.json"]
    A2[run again] -.->|manual| C2["cp to /tmp/fillpct_r2.json"]
    A3[run again] -.->|manual| C3["cp to /tmp/fillpct_r3.json"]
    C & C2 & C3 -.->|hand-written script| D["mean±std in<br/>commit message"]

    classDef lost fill:#fcc,stroke:#c00
    classDef manual fill:#ffd,stroke:#ca0,stroke-dasharray:3
    class B2 lost
    class C,C2,C3,D manual
```

No run ID, no model recorded, no git SHA. The `/tmp/` copies are gone; the only provenance is the commit message.

---

## 4. Proposed architecture

```mermaid
graph TB
    subgraph gp["gently_perception/  (single source of truth)"]
        API["api.py<br/>call_claude(model=, thinking=)<br/>model is a PARAMETER"]
        RENDER["render.py<br/>volume→image + disk cache"]
        PERCEIVER[perceiver.py]
        TYPES[types.py]
    end

    subgraph bench["benchmark/"]
        TESTSET["testset.py<br/>thin: discover + pair w/ GT"]
        GTM[ground_truth.py]
        METRICS["metrics.py<br/>per-stage acc + confusion<br/>+ transition_mae"]
        STATS["stats.py<br/>mean±std, Welch t,<br/>bootstrap CI"]
    end

    subgraph exp["experiments/  (only workspace)"]
        RUN["run.py  (only runner)<br/>--n-runs 3 --model M<br/>--baseline hybrid"]
        VAR["variants/<br/>auto-discovered<br/>perceive(FrameInput)→stage"]
        PIPE["pipelines/<br/>run_pipeline(testset)→preds<br/>judge_ensemble, routing"]
        ANALYSIS["analysis/<br/>offline, no API<br/>oracle_ceiling, summarize"]
    end

    CACHE[("data/cache/images/<br/>{sha1}.jpg")]
    RESULTS[("data/results/{variant}/<br/>{model}/{git8}_{seed}.json<br/>APPEND-ONLY")]

    RUN --> VAR
    RUN --> PIPE
    RUN --> TESTSET
    RUN --> METRICS
    RUN --> STATS
    RUN --> RESULTS
    VAR --> API
    PIPE --> API
    PIPE --> VAR
    TESTSET --> RENDER
    TESTSET --> GTM
    RENDER --> CACHE
    ANALYSIS --> RESULTS
    PERCEIVER --> API

    classDef new fill:#d4f4d4,stroke:#2a2
    classDef store fill:#e0ecff,stroke:#36c
    class RENDER,STATS,PIPE,ANALYSIS,CACHE new
    class RESULTS,CACHE store
```

**Green** = new · **Blue** = persistent storage. One API wrapper, one runner, one variant directory. `perception/` and root `run.py` deleted.

---

## 5. Proposed eval data flow — leak guard + replication

```mermaid
sequenceDiagram
    autonumber
    participant CLI as run.py --n-runs 3
    participant R as run_variant
    participant TS as Testset
    participant H as history[]
    participant FI as FrameInput<br/>(frozen, no GT field)
    participant P as perceive_fn
    participant S as stats.py

    CLI->>CLI: capture config:<br/>{variant, model, thinking,<br/>git_sha, dirty, stages, embryos}

    par seed=0
        CLI->>R: run(seed=0)
    and seed=1
        CLI->>R: run(seed=1)
    and seed=2
        CLI->>R: run(seed=2)
    end

    loop each timepoint T (within a seed)
        TS-->>R: TestCase(image, gt_stage)
        alt T skipped (lead-in)
            R->>H: append({T, gt_stage, source:"gt"})
        else T scored
            rect rgb(212, 244, 212)
            R->>R: ASSERT no entry in H with<br/>source=="gt" AND tp ≥ first_scored
            R->>FI: FrameInput(image,<br/>history=strip_source(H), T)
            Note over FI: gt_stage NOT a field —<br/>variant cannot reach it
            end
            R->>P: perceive(FI)
            P-->>R: pred
            R->>H: append({T, pred, source:"pred"})
            R->>R: score(pred, gt_stage)
        end
    end

    R-->>CLI: results/{variant}/{model}/{git8}_{seed}.json
    CLI->>S: aggregate(seed 0..2)
    S-->>CLI: 81.7 ± 2.4, t=1.1 vs baseline<br/>transition_mae=7.3 frames
```

Green band is the **structural leak guard**: GT lives only in the runner's scoring scope; `FrameInput` is a frozen dataclass with no GT field, so variants and pipelines physically can't read it.

---

## 6. Problems → fixes traceability

```mermaid
graph LR
    subgraph bugs["Observed cost (from RESEARCH.md / git log)"]
        B1["fillpct 83.3% outlier<br/>→ 76.1±4.7 after N=5"]
        B2["judge ensemble 90.6%<br/>→ 83.3% after leak fix"]
        B3["model 4.6 vs 4.7 confusion<br/>CLAUDE.md says 4.6, _base.py is 4.7"]
        B4["t-tests hand-computed<br/>in every commit msg"]
        B5["~3 min/run mostly<br/>TIFF re-rendering"]
        B6["judge_ensemble/ scripts<br/>reimplement metrics, /tmp/ paths"]
        B7["per-frame acc hides that errors<br/>are 6 transition offsets"]
    end

    subgraph fixes["Proposed change"]
        F1["--n-runs default 3<br/>append-only results"]
        F2["FrameInput leak guard<br/>+ history-source assertion"]
        F3["model/thinking as CLI flags<br/>recorded in config block"]
        F4["benchmark/stats.py<br/>auto Welch-t vs --baseline"]
        F5["render.py disk cache<br/>+ lazy per-signature render"]
        F6["pipelines/ as first-class<br/>runs through same harness"]
        F7["transition_mae metric<br/>in metrics.py"]
    end

    B1 --> F1
    B1 --> F4
    B2 --> F2
    B2 --> F6
    B3 --> F3
    B4 --> F4
    B5 --> F5
    B6 --> F6
    B6 --> F2
    B7 --> F7
```

---

## 7. Migration order (lowest-risk first)

```mermaid
graph TD
    P0["Phase 0: Dead-code cleanup<br/>delete experiments/CLAUDE.md, program.md<br/>fix benchmark/metrics.py import"]
    P1["Phase 1: Consolidate<br/>perception/* → experiments/variants/<br/>delete perception/_base.py, root run.py<br/>variants import from gently_perception.api"]
    P2["Phase 2: Provenance + replication<br/>--n-runs, --model, --thinking flags<br/>append-only results/{variant}/{model}/{sha}_{seed}.json<br/>full config block"]
    P3["Phase 3: Stats + metrics<br/>benchmark/stats.py (mean±std, Welch t)<br/>transition_mae in metrics.py"]
    P4["Phase 4: Leak guard<br/>FrameInput dataclass<br/>history-source assertion"]
    P5["Phase 5: Performance<br/>render.py + disk cache<br/>embryo×seed parallelism"]
    P6["Phase 6: Pipelines<br/>experiments/pipelines/<br/>port judge_ensemble through harness"]

    P0 --> P1 --> P2 --> P3
    P2 --> P4
    P2 --> P5
    P3 --> P6
    P4 --> P6

    classDef quick fill:#d4f4d4
    classDef core fill:#e0ecff
    class P0,P1 quick
    class P2,P3,P4 core
```

**Green** = mechanical/safe · **Blue** = changes runner contract (re-baseline after)
