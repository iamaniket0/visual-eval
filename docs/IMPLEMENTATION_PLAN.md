# Implementation Plan

**Tracks:** [DESIGN.md](DESIGN.md)  
**Approach:** Test-driven — write tests first, implement to pass, refactor.

---

## Phase 1: Core Infrastructure

Build the shared scaffolding that both T2I and edit pipelines depend on.

- [x] Soft-TIFA scoring math (`src/core/scoring.py`)
  - `soft_tifa_am(probabilities)` — arithmetic mean
  - `soft_tifa_gm(probabilities, logprob_floor)` — geometric mean with floor
  - `extract_yes_probability(logprobs)` — extract P(Yes) from logprob dict
  - `probabilities_from_answers(answers)` — handle mixed soft/hard answers
  - Tests: `tests/test_core/test_scoring.py`
    - AM basic (all 1s, all 0s, mixed)
    - GM collapses on confident miss
    - GM <= AM invariant (fuzz with random inputs)
    - Extract probability from various logprob formats
- [x] CostTracker (`src/core/utils.py`)
  - Running total by model and stage
  - 80% alert threshold
  - Hard cap enforcement
  - Tests: `tests/test_core/test_utils.py`
- [x] JSONL I/O helpers, logging setup

## Phase 2: T2I Generation

Model adapters for image generation.

- [x] Base generator with async, retry, content filter detection (`src/t2i/generators/base.py`)
- [x] Model adapters — one per provider, registered via `@register` decorator:
  - [x] OpenAI (GPT Image 1, GPT Image 2) — synchronous, returns base64
  - [x] BFL (FLUX 1.1 Pro, FLUX 2 Max) — async polling (202 → poll → COMPLETED)
  - [x] Bria (FIBO, 2.3) — async polling similar to BFL
  - [x] xAI (Aurora) — OpenAI-compatible endpoint
  - [x] Stability AI, Adobe Firefly, Google Imagen — adapters wired, need API keys
- [x] Tests: `tests/test_t2i/test_generators.py`
  - All models registered in registry
  - Filter detection (positive + negative)
  - Missing API key → SKIPPED (not crash)

## Phase 3: T2I Judging

MLLM judge with Soft-TIFA logprob extraction.

- [x] Judge backends (`src/t2i/judge.py`):
  - [x] `TogetherQwen35SoftJudge` — primary, open-source, no self-bias
  - [x] `GPT4oSoftJudge` — fallback via OpenRouter
  - [x] `GPT4oHardJudge` — legacy TIFA (binary yes/no), kept for reproducibility
  - [x] Factory function with backend selection
- [x] Prompt loader with multi-layer support (`src/t2i/prompt_loader.py`)
  - Stratified sampling across sub-categories
  - Layer filtering (L1/L2/L3)
  - Atomic decomposition integration
- [x] Tests: `tests/test_t2i/test_judge.py`
  - Extract JSON from fenced/unfenced responses
  - Format questions output shape
  - Factory picks correct backend
  - Judge result scoring (AM/GM from results)
- [x] Tests: `tests/test_t2i/test_prompt_loader.py`
  - Stratified sample size and determinism
  - Prompt ID format validation
  - JSON extraction edge cases

## Phase 4: T2I Aggregation & Reporting

- [x] Aggregator (`src/t2i/aggregator.py`):
  - Per-model leaderboard (overall AM, GM, coverage rate)
  - Per-subcategory breakdown
  - Layer comparison (L1 vs L2 vs L3 divergence)
  - Theme breakdown (from prompt_themes.json)
  - Failure analysis and filter rates
- [x] PDF report generator (`src/t2i/report.py`)
  - Aggregate report with leaderboard chart
  - Per-model scorecards
- [x] Tests: `tests/test_t2i/test_aggregator.py`
  - Aggregation produces expected output files
  - Layer comparison detects score divergence

## Phase 5: Edit Pipeline

Mirror the T2I pipeline for image editing evaluation.

- [x] Edit model adapters (`src/edit/editors/`):
  - [x] Flux Kontext — text-guided, BFL async polling
  - [x] Flux2 Flex — mask-based editing
  - [x] Bria Edit — instruction following
  - [x] Picsart — creative editing
  - [x] Firefly, PhotoRoom — wired, need API keys
- [x] Dual-image judge (`src/edit/judge.py`)
  - Takes source image + edited image
  - Evaluates on 3 dimensions: instruction following, visual consistency, detail preservation
  - Same Soft-TIFA logprob extraction as T2I
- [x] Edit aggregator (`src/edit/aggregator.py`)
  - Leaderboard with per-dimension breakdown
  - Per-subcategory scores (12 edit categories)
- [x] Tests: `tests/test_edit/`
  - Editor registry and count
  - Judge probability extraction
  - AM/GM invariant (GM <= AM)

## Phase 6: CLI, Dashboard, CI

- [x] Unified CLI entry point (`src/cli.py`) via Typer
- [x] Streamlit dashboard (`dashboard/app.py`)
  - T2I leaderboard tab
  - Edit leaderboard tab
  - Cross-pipeline comparison
- [x] GitHub Actions CI (`.github/workflows/ci.yml`)
  - pytest, ruff lint, ruff format, mypy
- [x] Docker setup for dashboard + pipeline execution

## Phase 7: Evaluation Run

- [x] Generate images: 50 hard prompts × 5 T2I models
- [x] Generate edits: 24 hard prompts × 4 edit models
- [x] Judge all results with Soft-TIFA
- [x] Aggregate scores, generate leaderboards
- [x] Generate PDF reports
- [x] HITL validation webapp for spot-checking

## Testing Strategy

Tests are written *before* implementation, following Red-Green-Refactor:

1. **Red** — Write a test for the next piece of behavior. Run it. Watch it fail.
2. **Green** — Write the minimum code to make the test pass.
3. **Refactor** — Clean up without changing behavior. Tests stay green.

Test categories:
- **Unit tests** — scoring math, probability extraction, factory functions
- **Integration tests** — aggregator end-to-end (mock judgments → CSV outputs)
- **Registry tests** — all models registered, correct count
- **Invariant tests** — GM <= AM (randomized inputs), deterministic sampling

No mocks for scoring math — these are pure functions, test with real values.
Mocks only where unavoidable (API calls in generator tests → test missing key behavior instead).

## Dependencies

```
openai>=1.30       # OpenAI + OpenRouter API client
httpx>=0.27        # Async HTTP for BFL, Bria polling
pyyaml>=6.0        # Config loading
pandas>=2.2        # Score aggregation
tqdm>=4.66         # Progress bars
reportlab>=4.2     # PDF generation
streamlit>=1.38    # Dashboard
plotly>=5.24       # Interactive charts
typer>=0.12        # CLI framework
pytest>=8.0        # Testing
pytest-asyncio     # Async test support
ruff>=0.5          # Linting + formatting
```
