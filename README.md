<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Models-9%20evaluated-FF6F61?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Scoring-Soft--TIFA-FFD700?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Tests-51%20passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">Visual Eval</h1>

<p align="center">
  <strong>Unified evaluation pipeline for frontier text-to-image generation and image editing models</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> · <a href="#results">Results</a> · <a href="#scoring-methodology">Scoring</a> · <a href="#quick-start">Quick Start</a> · <a href="#cli">CLI</a> · <a href="#dashboard">Dashboard</a> · <a href="#architecture">Architecture</a>
</p>

---

## Overview

Visual Eval benchmarks **T2I generation models** and **image editing models** on compositional faithfulness using [Soft-TIFA](https://arxiv.org/abs/2512.16853) scoring with an MLLM judge. Multi-layer prompt design (public benchmark gold + proprietary adversarial), per-model scorecards, and human-in-the-loop validation.

### Why This Exists

Frontier image models score >90% on standard benchmarks, but fail catastrophically on compositional prompts — counting objects, spatial reasoning, multi-constraint scenes. This pipeline measures exactly where they break.

---

## Pipeline at a Glance

```
                                    Visual Eval Pipeline
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
    │   │  Prompt   │───▶│ Generate │───▶│  Judge   │───▶│ Aggregate│     │
    │   │  Loader   │    │ / Edit   │    │  (MLLM)  │    │ & Report │     │
    │   └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
    │        │               │               │               │           │
    │   L1: Gold        5 T2I models    GPT-4o /        Soft-TIFA       │
    │   L2: Proprietary  4 Editors     Qwen3.5-397B    AM + GM         │
    │   L3: Adversarial  async+retry    logprob P(Yes)  PDF reports     │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## Results

Evaluated on **50 hard adversarial prompts** (L3) for T2I and **24 hard prompts** for editing. These prompts test compositional understanding: attribute binding, counting, negation, spatial reasoning, causal physics, and rare combinations.

### T2I Generation Leaderboard (Hard Prompts)

| Rank | Model | GM | AM | Coverage |
|:-----|:------|:---|:---|:---------|
| 1 | Aurora (xAI) | **0.821** | 0.946 | 100% |
| 2 | GPT Image 2 | 0.810 | 0.950 | 100% |
| 3 | GPT Image 1.5 | 0.785 | 0.943 | 100% |
| 4 | Bria FIBO | 0.721 | 0.906 | 100% |
| 5 | FLUX 2 Max | 0.748* | 0.938* | 73% |

*\*Covered prompts only — FLUX 2 Max filtered 27% of adversarial prompts (content moderation + timeouts).*

> **Key finding**: Aurora leads on hard compositional prompts with **0.82 GM** across 50 prompts (95% CI: 0.78–0.86). All models maintain >0.90 AM — the gap between AM and GM reveals that models fail *completely* on specific atoms rather than performing poorly across the board. FLUX 2 Max filtered 27% of adversarial prompts via content moderation. Confidence intervals are bootstrap-resampled over prompts (10k iterations).

### Image Editing Leaderboard (Hard Prompts)

| Rank | Model | Instruction Following (AM) | Visual Consistency (AM) | Detail Preservation (AM) |
|:-----|:------|:---------------------------|:------------------------|:-------------------------|
| 1 | Flux2 Flex | 0.708 | 0.316 | 0.574 |
| 2 | Bria Edit | 0.615 | 0.345 | 0.275 |
| 3 | Flux Kontext | 0.344 | 0.411 | 0.312 |
| 4 | Picsart | 0.000 | 0.785 | 0.083 |

> **Key finding**: No model excels at all three dimensions simultaneously. Flux2 Flex leads on instruction following but struggles with visual consistency. Picsart preserves consistency but cannot follow edit instructions — a fundamental tension in current editing architectures. Rankings are directional — overlapping CIs at this sample size (24 prompts) mean differences between adjacent models may not be significant.

---

## Models Supported

### T2I Generation

| Model | Provider | Type |
|:------|:---------|:-----|
| FLUX 1.1 Pro Ultra | BFL | Diffusion |
| FLUX 2 Max | BFL | Diffusion |
| Stable Diffusion 3.5 | Stability AI | Diffusion |
| GPT Image 1.5 | OpenAI | Autoregressive |
| GPT Image 2 | OpenAI | Autoregressive |
| Firefly Image 3 | Adobe | Diffusion |
| Bria 2.3 | Bria AI | Diffusion |
| Aurora | xAI | Diffusion |
| Imagen 3 | Google | Diffusion |

### Image Editing

| Model | Provider | Features |
|:------|:---------|:---------|
| Flux Kontext | BFL | Text-guided editing |
| Flux2 Flex | BFL | Mask-based editing |
| Bria Edit | Bria AI | Instruction following |
| Picsart | Picsart | Creative editing |
| Firefly Edit | Adobe | Multi-turn editing |
| PhotoRoom | PhotoRoom | Background editing |

---

## Scoring Methodology

<table>
<tr>
<td width="50%">

### Soft-TIFA Scoring

Based on [GenEval 2](https://arxiv.org/abs/2512.16853) (Kamath et al., Dec 2025):

1. Decompose each prompt into **atomic binary questions**
2. Judge each question via **MLLM** (GPT-4o or Qwen3.5-397B)
3. Extract **P(Yes)** from first-token logprobs
4. Aggregate two ways:

</td>
<td width="50%">

```
         Scoring Formula

  AM = (1/n) Σ pᵢ          ← partial credit
                              (arithmetic mean)

  GM = exp((1/n) Σ log pᵢ) ← strict
                              (geometric mean)
                              one miss collapses

  Primary metric: GM on covered prompts
```

</td>
</tr>
</table>

> **GM vs AM**: AM gives partial credit — a prompt with 4/5 atoms at 0.95 and 1 at 0.10 scores 0.78. GM scores the same prompt 0.53. GM correlates better with human judgment (AUROC 94.5% vs 91.6% for legacy TIFA).

### Edit Evaluation — 3-Axis Scoring

| Dimension | What It Measures | Example Question |
|:----------|:-----------------|:-----------------|
| **Instruction Following** | Did the requested edit happen? | "Is there now a red hat on the person?" |
| **Visual Consistency** | Are unedited regions preserved? | "Is the background unchanged?" |
| **Detail Preservation** | Are fine details intact? | "Is the text on the sign still readable?" |

---

## Quick Start

### Setup

```bash
git clone git@github.com:iamaniket0/visual-eval-.git
cd visual-eval-
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Fill in: TOGETHER_API_KEY, OPENAI_API_KEY, BFL_API_KEY, etc.
```

### T2I Evaluation

```bash
# Using the CLI (recommended)
visual-eval t2i prompts                          # Build prompt set
visual-eval t2i generate --models sanity --dry-run  # Sanity check
visual-eval t2i generate --models full            # Full run
visual-eval t2i judge                             # Judge all images
visual-eval t2i aggregate                         # Aggregate scores
visual-eval t2i report                            # Generate PDF reports

# Or via Python modules directly
python -m scripts.t2i.run_generation --models full
```

### Edit Evaluation

```bash
visual-eval edit download-images                  # Download source images
visual-eval edit run --models sanity --dry-run    # Sanity check
visual-eval edit run --models full                # Full run
visual-eval edit judge                            # Judge edits
visual-eval edit aggregate                        # Aggregate scores
visual-eval edit report                           # Generate report
```

### Docker

```bash
# Run the dashboard
docker compose up dashboard

# Run pipeline commands
docker compose --profile cli run pipeline t2i generate --models sanity
```

---

## CLI

The `visual-eval` CLI wraps all pipeline scripts into a single entry point:

```
visual-eval
├── t2i
│   ├── generate    Generate images across T2I models
│   ├── judge       Run MLLM judge on generated images
│   ├── aggregate   Aggregate scores into leaderboard
│   ├── report      Generate PDF scorecards
│   ├── prompts     Build the prompt set (L1+L2+L3)
│   └── hitl        Launch HITL validation web UI
├── edit
│   ├── run             Run edits across editing models
│   ├── judge           Dual-image MLLM judge
│   ├── aggregate       Aggregate edit scores
│   ├── report          Generate edit report
│   └── download-images Download source images
├── dashboard           Launch Streamlit dashboard
└── test                Run the test suite
```

---

## Dashboard

Interactive Streamlit dashboard for exploring results:

```bash
visual-eval dashboard
# or directly:
streamlit run dashboard/app.py
```

Features:
- **T2I Leaderboard** — ranked bar chart + data table with GM/AM scores
- **Sub-Category Breakdown** — grouped bars + radar chart per category
- **Layer Comparison** — public benchmark vs proprietary prompt performance
- **Theme Analysis** — per-model theme score heatmap
- **Edit Leaderboard** — ranked by overall score with dimension heatmap
- **Cross-Pipeline Comparison** — box plot distributions, T2I vs Edit

---

## Architecture

```
visual-eval/
├── src/
│   ├── core/                          # Shared infrastructure
│   │   ├── scoring.py                 # Soft-TIFA AM/GM math
│   │   └── utils.py                   # CostTracker, JSONL I/O, logging
│   │
│   ├── t2i/                           # Text-to-Image evaluation
│   │   ├── generators/                # Model adapters (@register pattern)
│   │   │   ├── base.py                # BaseGenerator: async, retry, filter detection
│   │   │   ├── openai_gen.py          # GPT Image 1.5 / 2
│   │   │   ├── flux.py                # FLUX 1.1 Pro / 2 Max
│   │   │   └── ...                    # bria, stability, xai, etc.
│   │   ├── judge.py                   # MLLM judge backends (Soft-TIFA)
│   │   ├── aggregator.py              # Per-model/category/theme scoring
│   │   ├── prompt_loader.py           # Multi-layer prompt management
│   │   ├── report.py                  # PDF scorecards + charts
│   │   └── hitl.py / hitl_webui.py    # Human-in-the-loop validation
│   │
│   └── edit/                          # Image Editing evaluation
│       ├── editors/                   # Editor adapters (@register pattern)
│       │   ├── base.py                # BaseEditor: mask, multi-turn support
│       │   ├── flux_kontext.py        # Flux Kontext
│       │   └── ...                    # bria, picsart, etc.
│       ├── judge.py                   # Dual-image judge (source + edited)
│       ├── aggregator.py              # 3-axis dimension scoring
│       └── prompt_loader.py           # Edit prompt management
│
├── config/
│   ├── t2i/                           # T2I model configs + settings
│   └── edit/                          # Edit model configs + taxonomy
│
├── scripts/
│   ├── t2i/                           # T2I pipeline CLI scripts
│   └── edit/                          # Edit pipeline CLI scripts
│
├── prompts/
│   ├── t2i/                           # T2I benchmark prompts (L1+L2+L3)
│   └── edit/                          # Edit prompts + source images
│
├── dashboard/                         # Streamlit interactive dashboard
│
└── tests/                             # 51 tests across all modules
    ├── test_core/                     # Scoring math, CostTracker
    ├── test_t2i/                      # Generators, judge, aggregator
    └── test_edit/                     # Editors, judge, aggregator
```

---

## Key Design Decisions

| Decision | Rationale |
|:---------|:----------|
| **Atomic binary decomposition** (not layered rubrics) | Multi-step judgments cause MLLMs to hallucinate failures |
| **GM as primary metric** | Collapses on single weak atom, correlates best with human judgment (94.5% AUROC) |
| **FILTERED != retried** | Content-policy blocks scored 0, never retried with modified prompts — preserves benchmark integrity |
| **Hard cost cap** | CostTracker with 80% alert threshold and hard cutoff |
| **Resume-friendly** | Generation/editing skips prompts whose output already exists |
| **Scaffold-friendly** | Missing API keys -> `SKIPPED`, not crashes |

---

## Outputs

```
outputs/
├── t2i/
│   ├── generations/{model}/{prompt_id}.png   # not tracked in git
│   ├── judgments/{model}.jsonl
│   ├── scores/  (leaderboard.csv, per_subcategory.csv, layer_comparison.csv)
│   └── reports/ (aggregate_report.pdf, {model}_card.pdf)
└── edit/
    ├── edits/{model}/{prompt_id}.png          # not tracked in git
    ├── judgments/{model}.jsonl
    └── scores/  (leaderboard.csv, per_dimension.csv, per_subcategory.csv)
```

> Generated images and metadata logs are excluded from git (reproducible via the pipeline). For a lightweight clone: `git clone --filter=blob:none`.

</details>

<details>
<summary><strong>Edit Config</strong> — <code>config/edit/</code></summary>

```yaml
# models.yaml — editor endpoints with mask/multi-turn support
models:
  flux_kontext:
    provider: bfl
    supports_mask: false
    supports_multi_turn: false

# settings.yaml — 3 evaluation dimensions, 12 sub-categories
dimensions:
  - instruction_following
  - visual_consistency
  - detail_preservation
```

</details>

---

## Testing

```bash
# Run all 51 tests
pytest tests/ -v
visual-eval test                # or via CLI

# By domain
pytest tests/test_core/ -v     # Shared scoring math (12 tests)
pytest tests/test_t2i/ -v      # T2I pipeline (21 tests)
pytest tests/test_edit/ -v     # Edit pipeline (18 tests)
```

---

## References

- **Soft-TIFA / GenEval 2**: [arXiv 2512.16853](https://arxiv.org/abs/2512.16853) — Kamath et al., Dec 2025
- **T2I-CompBench++**: [GitHub](https://github.com/Karine-Huang/T2I-CompBench) — NeurIPS 2023, TPAMI 2025
- **SpatialGenEval**: [arXiv 2501.09652](https://arxiv.org/abs/2501.09652) — ICLR 2026
- **R2I-Bench**: [GitHub](https://github.com/PLUM-Lab/R2I-Bench) — EMNLP 2025
- **GEditBench v2**: NTU 2026
- **Complex-Edit**: UCSC 2025
- **CompAlign / CompQuest**: [arXiv 2505.11178](https://arxiv.org/abs/2505.11178)

---

<p align="center">
  <sub>Built for evaluating frontier visual AI — where benchmarks saturate but compositional understanding doesn't.</sub>
</p>
