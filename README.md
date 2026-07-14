<p align="center">
  <a href="https://github.com/your-org/visual-eval/actions/workflows/ci.yml"><img src="https://github.com/your-org/visual-eval/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Models-17+-FF6F61?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Judge-Qwen3.5--397B-00D4AA?style=for-the-badge&logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/Scoring-Soft--TIFA-FFD700?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Tests-51%20passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
</p>

<h1 align="center">Visual Eval</h1>

<p align="center">
  <strong>Unified evaluation pipeline for frontier text-to-image generation and image editing models</strong>
</p>

<p align="center">
  <a href="#-t2i-evaluation">T2I Eval</a> · <a href="#-edit-evaluation">Edit Eval</a> · <a href="#-scoring-methodology">Scoring</a> · <a href="#-quick-start">Quick Start</a> · <a href="#-cli">CLI</a> · <a href="#-dashboard">Dashboard</a> · <a href="#-architecture">Architecture</a>
</p>

---

## Overview

Visual Eval benchmarks **10+ T2I generation models** and **7 image editing models** on compositional faithfulness using [Soft-TIFA](https://arxiv.org/abs/2512.16853) scoring with an open-source MLLM judge. Two-layer prompt design (public benchmark gold + proprietary adversarial), multi-seed variance analysis, per-model scorecards, and human-in-the-loop validation.

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
    │   L1: Gold        10+ T2I         Qwen3.5-397B    Soft-TIFA       │
    │   L2: Proprietary  7 Editors      4 backends      AM + GM         │
    │   L3: Adversarial  async+retry    logprob P(Yes)  PDF reports     │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## Models Benchmarked

### T2I Generation

| Model | Provider | Type |
|:------|:---------|:-----|
| FLUX 1.1 Pro Ultra | BFL | Diffusion |
| FLUX 2 Max | BFL | Diffusion |
| Stable Diffusion 3.5 | Stability AI | Diffusion |
| GPT Image 1.5 | OpenAI | Autoregressive |
| Firefly Image 3 | Adobe | Diffusion |
| Midjourney v8 | Midjourney | Diffusion |
| Bria 2.3 | Bria AI | Diffusion |
| Leonardo Phoenix | Leonardo.ai | Diffusion |
| Aurora | xAI | Diffusion |
| Imagen 3 | Google | Diffusion |
| Freepik Mystic | Freepik | Diffusion |

### Image Editing

| Model | Provider | Features |
|:------|:---------|:---------|
| Flux Kontext | BFL | Text-guided editing |
| Flux2 Flex | BFL | Mask-based editing |
| Bria FIBO | Bria AI | Instruction following |
| Firefly Edit | Adobe | Multi-turn editing |
| PhotoRoom | PhotoRoom | Background editing |
| Picsart | Picsart | Creative editing |
| Canva/Leonardo | Canva | Style-aware editing |

---

## Scoring Methodology

<table>
<tr>
<td width="50%">

### Soft-TIFA Scoring

Based on [GenEval 2](https://arxiv.org/abs/2512.16853) (Kamath et al., Dec 2025):

1. Decompose each prompt into **atomic binary questions**
2. Judge each question via **Qwen3.5-397B-A17B** (open-source, no self-bias)
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

Install and use:

```bash
pip install -e .
visual-eval --help
visual-eval t2i generate --models sanity --dry-run
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
│   │   ├── generators/                # 10+ model adapters (@register pattern)
│   │   │   ├── base.py                # BaseGenerator: async, retry, filter detection
│   │   │   ├── openai_gen.py          # GPT Image 1.5
│   │   │   ├── flux.py                # FLUX 1.1 Pro / 2 Max
│   │   │   └── ...                    # adobe, bria, stability, etc.
│   │   ├── judge.py                   # 4 MLLM judge backends
│   │   ├── aggregator.py              # Per-model/category/theme scoring
│   │   ├── prompt_loader.py           # Multi-layer prompt management
│   │   ├── report.py                  # PDF scorecards + charts
│   │   └── hitl.py / hitl_webui.py    # Human-in-the-loop validation
│   │
│   └── edit/                          # Image Editing evaluation
│       ├── editors/                   # 7 editor adapters (@register pattern)
│       │   ├── base.py                # BaseEditor: mask, multi-turn support
│       │   ├── flux_kontext.py        # Flux Kontext
│       │   └── ...                    # bria, firefly, photoroom, etc.
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
│   ├── t2i/                           # T2I benchmark prompts
│   └── edit/                          # Edit prompts + source images
│
└── tests/                             # 51 tests across all modules
    ├── test_core/                     # Scoring math, CostTracker
    ├── test_t2i/                      # Generators, judge, aggregator
    └── test_edit/                     # Editors, judge, aggregator
```

---

## Quick Start

### Setup

```bash
git clone git@github.com:your-org/visual-eval.git
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

## Key Design Decisions

| Decision | Rationale |
|:---------|:----------|
| **Atomic binary decomposition** (not layered rubrics) | Multi-step judgments cause MLLMs to hallucinate failures |
| **Open-source judge** (Qwen3.5-397B) | Avoids self-bias when judging GPT Image outputs |
| **FILTERED != retried** | Content-policy blocks scored 0, never retried with modified prompts — preserves benchmark integrity |
| **GM as primary metric** | Collapses on single weak atom, correlates best with human judgment (94.5% AUROC) |
| **Hard cost cap** | CostTracker with 80% alert threshold and hard cutoff |
| **Resume-friendly** | Generation/editing skips prompts whose output already exists |
| **Scaffold-friendly** | Missing API keys → `SKIPPED`, not crashes |

---

## Outputs

```
outputs/
├── t2i/
│   ├── generations/{model}/{prompt_id}.png    # Generated images
│   ├── metadata/generation_log.jsonl          # Generation metadata + costs
│   ├── judgments/{model}.jsonl                 # Per-image judge results
│   ├── scores/
│   │   ├── leaderboard.csv                    # Overall model ranking
│   │   ├── per_subcategory.csv                # By numeracy/spatial/complex
│   │   ├── layer_comparison.csv               # L1 vs L2 divergence
│   │   └── theme_breakdown.csv                # Fine-grained theme analysis
│   └── reports/
│       ├── aggregate_report.pdf               # Full benchmark report
│       └── {model}_card.pdf                   # Per-model scorecards
│
└── edit/
    ├── edits/{model}/{prompt_id}.png          # Edited images
    ├── metadata/edit_log.jsonl                # Edit metadata + costs
    ├── judgments/{model}.jsonl                 # Per-image judge results
    └── scores/
        ├── leaderboard.csv                    # Overall model ranking
        └── per_dimension.csv                  # Instruction/visual/detail axes
```

---

## Configuration

<details>
<summary><strong>T2I Config</strong> — <code>config/t2i/</code></summary>

```yaml
# models.yaml — model endpoints, costs, concurrency limits
models:
  flux2_max:
    provider: bfl
    model_id: flux-pro-1.1-ultra
    cost_per_image: 0.06
    max_concurrent: 5
  gpt_image_15:
    provider: openai
    model_id: gpt-image-1
    cost_per_image: 0.04

# settings.yaml — judge and pipeline config
judge:
  backend: qwen_together_soft
  model_slug: "Qwen/Qwen3.5-397B-A17B"
seeds_per_prompt: 3
hard_cap: 300  # USD
```

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
  firefly:
    provider: adobe
    supports_mask: true
    supports_multi_turn: true

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a new model adapter, run tests, and submit PRs.

For detailed architecture diagrams (Mermaid), see [docs/architecture.md](docs/architecture.md).

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
