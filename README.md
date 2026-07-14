<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Models-17+-FF6F61?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Judge-Qwen3.5--397B-00D4AA?style=for-the-badge&logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/Scoring-Soft--TIFA-FFD700?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Tests-51%20passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white" />
</p>

<h1 align="center">Visual Eval</h1>

<p align="center">
  <strong>Unified evaluation pipeline for frontier text-to-image generation and image editing models</strong>
</p>

<p align="center">
  <a href="#-t2i-evaluation">T2I Eval</a> · <a href="#-edit-evaluation">Edit Eval</a> · <a href="#-scoring-methodology">Scoring</a> · <a href="#-quick-start">Quick Start</a> · <a href="#-architecture">Architecture</a>
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
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Fill in: TOGETHER_API_KEY, OPENAI_API_KEY, BFL_API_KEY, etc.
```

### T2I Evaluation

```bash
# 1. Build prompt set (L1 gold + L2 proprietary + L3 adversarial)
python -m scripts.t2i.run_prompt_set

# 2. Generate images (sanity check first, then full run)
python -m scripts.t2i.run_generation --models sanity --dry-run
python -m scripts.t2i.run_generation --models full

# 3. Judge all generated images
python -m scripts.t2i.run_judge

# 4. Aggregate scores and generate reports
python -m scripts.t2i.run_aggregate
python -m scripts.t2i.run_report
```

### Edit Evaluation

```bash
# 1. Download source images
python -m scripts.edit.download_source_images

# 2. Run edits across all models
python -m scripts.edit.run_edit --models sanity --dry-run
python -m scripts.edit.run_edit --models full

# 3. Judge and aggregate
python -m scripts.edit.run_judge
python -m scripts.edit.run_aggregate
python -m scripts.edit.run_report
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
