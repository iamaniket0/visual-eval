# Architecture

## T2I Evaluation Pipeline

```mermaid
flowchart LR
    subgraph Prompts
        L1[Layer 1\nPublic Benchmarks]
        L2[Layer 2\nProprietary]
        L3[Layer 3\nTheme-Generated]
    end

    subgraph Generation
        G[Generator Registry]
        M1[FLUX 2 Max]
        M2[GPT-Image]
        M3[Imagen 3]
        M4[Aurora]
        Mn[... 11 models]
    end

    subgraph Judging
        J[MLLM Judge\nQwen3.5-397B]
        LP[logprob\nP Yes extraction]
    end

    subgraph Scoring
        AG[Aggregator]
        AM[AM: arithmetic mean]
        GM[GM: geometric mean]
    end

    L1 & L2 & L3 --> G
    G --> M1 & M2 & M3 & M4 & Mn
    M1 & M2 & M3 & M4 & Mn --> J
    J --> LP --> AG
    AG --> AM & GM
    GM --> LB[Leaderboard CSV]
```

## Edit Evaluation Pipeline

```mermaid
flowchart LR
    subgraph Input
        SI[Source Images]
        EI[Edit Instructions\n3 axes]
    end

    subgraph Editing
        ER[Editor Registry]
        E1[FLUX Fill]
        E2[Firefly]
        E3[Bria Eraser]
        En[... 7 models]
    end

    subgraph Judging
        DJ[Dual-Image Judge\nsource + edited]
        DLP[logprob\nP Yes extraction]
    end

    subgraph Scoring
        DAG[Aggregator]
        DIM[Per-Dimension\nScores]
    end

    SI & EI --> ER
    ER --> E1 & E2 & E3 & En
    E1 & E2 & E3 & En --> DJ
    DJ --> DLP --> DAG
    DAG --> DIM --> DLB[Leaderboard CSV]
```

## Soft-TIFA Scoring

```mermaid
flowchart TD
    P[Prompt] --> D[Decompose into\nN sub-questions]
    D --> Q1[Q1: Is there a cat?]
    D --> Q2[Q2: Is it orange?]
    D --> Qn[Qn: ...]

    Q1 -->|P_Yes=0.92| S1[s₁ = 0.92]
    Q2 -->|P_Yes=0.78| S2[s₂ = 0.78]
    Qn -->|P_Yes=0.85| Sn[sₙ = 0.85]

    S1 & S2 & Sn --> AM["AM = (Σsᵢ)/N"]
    S1 & S2 & Sn --> GM["GM = (Πsᵢ)^(1/N)"]

    GM -->|Primary| Rank[Model Ranking]
    AM -->|Diagnostic| Rank
```

## System Architecture

```mermaid
graph TB
    subgraph CLI["CLI (typer)"]
        TC[visual-eval t2i ...]
        EC[visual-eval edit ...]
        DC[visual-eval dashboard]
    end

    subgraph Core["src/core/"]
        REG[Registry\n@register decorator]
        CT[CostTracker\nthread-safe, hard cap]
        JDG[Judge\nMLLM + logprob]
        UTL[Utils\nIO, retry, config]
    end

    subgraph T2I["src/t2i/"]
        TG[Generators\n11 adapters]
        TA[Aggregator]
        TR[Report]
        TP[Prompt Loader]
    end

    subgraph Edit["src/edit/"]
        EE[Editors\n7 adapters]
        EA[Aggregator]
        ER[Report]
    end

    subgraph Dashboard["dashboard/"]
        ST[Streamlit App]
        PL[Plotly Charts]
    end

    TC --> TG & TA & TR & TP
    EC --> EE & EA & ER
    DC --> ST --> PL

    TG & EE --> REG
    TG & EE --> CT
    TG & EE --> JDG
    TG & EE --> UTL
```
